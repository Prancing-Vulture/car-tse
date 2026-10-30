#!/usr/bin/env python3
"""Replay the committed stream recording, make its two figures, and save metrics.

The same configured PyTorch pipeline is used to create both the visual timeline
and target/residual plot. Mono or stereo recordings are downmixed and repeated
to four model channels, matching the repository's existing demo fallback.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_stagewise_latency import load_pipeline, repo_path, synchronize  # noqa: E402
from car_tse.config import load_stage_config  # noqa: E402
from car_tse.pipeline.in_car_pipeline import PipelineState  # noqa: E402
from plot_tse_stream_figures import paper_style, plot_stagewise, plot_tse  # noqa: E402


CSV_FIELDS = [
    "window_index",
    "window_start_seconds",
    "window_end_seconds",
    "inference_latency_ms",
    "schedule_lag_ms",
    "hop_deadline_ms",
    "deadline_miss",
    "accepted",
    "presence_probability",
    "speaker_similarity",
    "wake_probability",
    "wake_detected",
    "command_pending",
    "intent_ready",
    "intent_id",
    "action",
    "intent_confidence",
    "command_suppressed",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL
        ).decode("ascii", "replace").strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _read_audio(path: Path, sample_rate: int) -> np.ndarray:
    audio, source_rate = sf.read(str(path), always_2d=True, dtype="float32")
    if int(source_rate) != int(sample_rate):
        raise ValueError(
            "{} is {} Hz; the configured pipeline expects {} Hz. Resample it first.".format(
                path, source_rate, sample_rate
            )
        )
    if audio.shape[0] == 0 or not np.isfinite(audio).all():
        raise ValueError("Audio is empty or contains non-finite samples: {}".format(path))
    return np.asarray(audio, dtype=np.float32)


def _summary(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
        }
    samples = np.asarray(values, dtype=np.float64)
    return {
        "count": int(samples.size),
        "mean_ms": float(samples.mean()),
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "p99_ms": float(np.percentile(samples, 99)),
        "max_ms": float(samples.max()),
    }


def _hash_checkpoints(config: Dict[str, Any], base_checkpoint: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "base_spex": {"path": str(base_checkpoint), "sha256": _sha256(base_checkpoint)}
    }
    checkpoint_root = repo_path(config["paths"]["checkpoint_root"])
    for name, value in (config.get("checkpoints", {}) or {}).items():
        checkpoint = Path(str(value)).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = repo_path(checkpoint_root / checkpoint)
        result[str(name)] = {
            "path": str(checkpoint),
            "exists": checkpoint.is_file(),
            "sha256": _sha256(checkpoint) if checkpoint.is_file() else None,
        }
    return result


def _make_state(pipeline: Any, enrollment_embedding: Optional[torch.Tensor]) -> PipelineState:
    state = PipelineState()
    if enrollment_embedding is not None:
        state.reference_speaker_embedding = enrollment_embedding
        if getattr(pipeline.presence, "requires_enrollment", False):
            state.presence_enrollment_embedding = enrollment_embedding
        if getattr(pipeline, "target_kws", None) is not None:
            state.target_kws_enrollment_embedding = enrollment_embedding
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio", type=Path, default=REPO_ROOT / "stream_rec/stream_trimmed_for_alignment.wav"
    )
    parser.add_argument("--enrollment", type=Path, default=REPO_ROOT / "data/enrollment.wav")
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/eval_end_to_end_cavemove.yaml"
    )
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.yaml")
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints/car_tse_spexplus_from_stage1.pt",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--hop-seconds", type=float, default=0.25)
    parser.add_argument("--replay-mode", choices=("paced", "fast"), default="paced")
    parser.add_argument("--command-delay-calls", type=int, default=5)
    parser.add_argument("--command-timeout-calls", type=int, default=12)
    parser.add_argument("--kws-threshold", type=float, default=0.4)
    parser.add_argument("--mode", choices=("all", "figures", "metrics"), default="all")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "stream_rec/reproduced")
    parser.add_argument(
        "--metrics-output", type=Path, default=REPO_ROOT / "logs/stream_recording_metrics.json"
    )
    parser.add_argument(
        "--windows-csv", type=Path, default=REPO_ROOT / "logs/stream_recording_windows.csv"
    )
    args = parser.parse_args()
    if args.hop_seconds <= 0 or args.command_delay_calls < 0 or args.command_timeout_calls < 1:
        parser.error("hop must be positive, command delay non-negative, and timeout at least 1")

    audio_path = repo_path(args.audio)
    enrollment_path = repo_path(args.enrollment)
    config_path = repo_path(args.config)
    paths_file = repo_path(args.paths)
    if not paths_file.is_file():
        paths_file = REPO_ROOT / "configs/paths.example.yaml"
    base_checkpoint = repo_path(args.base_checkpoint)
    for path, label in (
        (audio_path, "stream recording"),
        (enrollment_path, "enrollment audio"),
        (config_path, "pipeline config"),
        (paths_file, "paths config"),
        (base_checkpoint, "base SpEx+ checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError("Missing {}: {}".format(label, path))

    config = load_stage_config(config_path, paths_file)
    sample_rate = int(config.get("sample_rate", 16000))
    model_channels = int(config.get("num_mics", 4))
    if model_channels != 4:
        raise ValueError("This demo's input preparation expects the configured four-microphone model")
    source_audio = _read_audio(audio_path, sample_rate)
    enrollment_audio = _read_audio(enrollment_path, sample_rate).mean(axis=1)
    input_channels = int(source_audio.shape[1])
    if input_channels not in (1, 2, 4):
        raise ValueError("Expected mono, stereo, or four-channel input; found {} channels".format(input_channels))
    if input_channels == 4:
        multichannel = source_audio
    else:
        mono_source = source_audio.mean(axis=1).astype(np.float32)
        multichannel = np.repeat(mono_source[:, None], model_channels, axis=1)
    mono = multichannel.mean(axis=1).astype(np.float32)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    model_load_started = time.perf_counter()
    pipeline = load_pipeline(config, paths_file, base_checkpoint, device)
    synchronize(device)
    model_load_ms = (time.perf_counter() - model_load_started) * 1000.0
    if not hasattr(pipeline, "stage23_fusion"):
        pipeline.stage23_fusion = None
    if hasattr(pipeline, "kws_threshold"):
        pipeline.kws_threshold = float(args.kws_threshold)
    if hasattr(pipeline, "command_delay_calls"):
        pipeline.command_delay_calls = int(args.command_delay_calls)
    if hasattr(pipeline, "command_timeout_calls"):
        pipeline.command_timeout_calls = int(args.command_timeout_calls)

    enrollment_tensor = torch.from_numpy(np.ascontiguousarray(enrollment_audio[None, :])).to(device)
    enrollment_embedding = None
    enrollment_embedding_ms = None
    speaker_embedder = getattr(pipeline.stage1, "speaker_embedding", None)
    if callable(speaker_embedder):
        synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            enrollment_embedding = speaker_embedder(enrollment_tensor)
        synchronize(device)
        enrollment_embedding_ms = (time.perf_counter() - started) * 1000.0
        if enrollment_embedding is not None:
            enrollment_embedding = enrollment_embedding.detach()

    window_samples = int(round(float(config.get("segment_seconds", 3.0)) * sample_rate))
    hop_samples = max(1, int(round(args.hop_seconds * sample_rate)))
    if len(mono) < window_samples:
        raise ValueError("The stream recording is shorter than one configured model window")
    ends = list(range(window_samples, len(mono) + 1, hop_samples))
    if not ends or ends[-1] != len(mono):
        ends.append(len(mono))

    # Warm the selected path once; this separate state is excluded from timing.
    first_end = ends[0]
    first_start = max(0, first_end - window_samples)
    first_mc = multichannel[first_start:first_end]
    first_mix = mono[first_start:first_end]
    with torch.inference_mode():
        pipeline.forward(
            torch.from_numpy(np.ascontiguousarray(first_mix[None, :])).to(device),
            enrollment_tensor,
            _make_state(pipeline, enrollment_embedding),
            multichannel=torch.from_numpy(
                np.ascontiguousarray(first_mc.T[None, :, :], dtype=np.float32)
            ).to(device),
            audio_start_sample=first_start,
        )
    synchronize(device)

    state = _make_state(pipeline, enrollment_embedding)
    latency_ms: List[float] = []
    windows: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    window = int(window_samples)
    hop = int(hop_samples)
    window_weight = 0.05 + 0.95 * np.hanning(window + 2)[1:-1].astype(np.float32)
    accumulated = np.zeros(len(mono), dtype=np.float64)
    weight_sum = np.zeros(len(mono), dtype=np.float64)
    replay_started = time.perf_counter()

    for index, end in enumerate(ends):
        scheduled_start = replay_started + (float(end) / sample_rate)
        if args.replay_mode == "paced":
            delay = scheduled_start - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        call_started = time.perf_counter()
        schedule_lag_ms = max(0.0, (call_started - scheduled_start) * 1000.0)
        start = max(0, end - window)
        frame = multichannel[start:end]
        if len(frame) < window:
            pad = window - len(frame)
            frame = np.pad(frame, ((pad, 0), (0, 0)))
        mix_frame = mono[start:end]
        if len(mix_frame) < window:
            mix_frame = np.pad(mix_frame, (window - len(mix_frame), 0))
        mix_tensor = torch.from_numpy(np.ascontiguousarray(mix_frame[None, :])).to(device)
        mc_tensor = torch.from_numpy(
            np.ascontiguousarray(frame.T[None, :, :], dtype=np.float32)
        ).to(device)
        synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            output, state = pipeline.forward(
                mix_tensor,
                enrollment_tensor,
                state,
                multichannel=mc_tensor,
                audio_start_sample=start,
            )
        synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latency_ms.append(elapsed_ms)

        estimate_value = getattr(output, "s1", None)
        if estimate_value is None:
            raise RuntimeError("Pipeline returned no Stage 1 estimate for window {}".format(index))
        estimate = np.asarray(estimate_value.detach().float().cpu().numpy()).reshape(-1)
        if len(estimate) < window:
            estimate = np.pad(estimate, (0, window - len(estimate)))
        estimate = estimate[:window]
        count = min(window, len(mono) - start)
        accumulated[start : start + count] += estimate[:count].astype(np.float64) * window_weight[:count]
        weight_sum[start : start + count] += window_weight[:count]

        action_value = getattr(output, "action", None)
        action = getattr(action_value, "action", action_value)
        intent_id = getattr(action_value, "intent_id", None)
        end_s = float(end) / sample_rate
        start_s = float(start) / sample_rate
        row = {
            "window_index": index,
            "window_start_seconds": start_s,
            "window_end_seconds": end_s,
            "inference_latency_ms": elapsed_ms,
            "schedule_lag_ms": schedule_lag_ms,
            "hop_deadline_ms": args.hop_seconds * 1000.0,
            "deadline_miss": elapsed_ms > args.hop_seconds * 1000.0,
            "accepted": bool(getattr(output, "accepted", False)),
            "presence_probability": getattr(output, "presence_probability", None),
            "speaker_similarity": getattr(output, "speaker_similarity", None),
            "wake_probability": getattr(output, "wake_probability", None),
            "wake_detected": bool(getattr(output, "wake_detected", False)),
            "command_pending": bool(getattr(output, "command_pending", False)),
            "intent_ready": bool(getattr(output, "intent_ready", False)),
            "intent_id": None if intent_id is None else str(intent_id),
            "action": None if action is None else str(action),
            "intent_confidence": getattr(output, "intent_confidence", None),
            "command_suppressed": bool(getattr(output, "command_suppressed", False)),
        }
        windows.append(row)
        if row["command_suppressed"]:
            phase = "command_suppressed"
        elif row["intent_ready"]:
            phase = "intent"
        elif row["wake_detected"]:
            phase = "wake"
        elif row["command_pending"]:
            phase = "command_pending"
        elif row["accepted"]:
            phase = "target_audio"
        else:
            phase = "background_or_gate_rejected"
        row["phase"] = phase
        if row["wake_detected"] or row["intent_ready"] or row["command_suppressed"]:
            events.append(
                {
                    "window_end_seconds": end_s,
                    "phase": phase,
                    "wake_probability": row["wake_probability"],
                    "intent_confidence": row["intent_confidence"],
                    "intent_id": row["intent_id"],
                    "action": row["action"],
                    "command_suppressed": row["command_suppressed"],
                }
            )

    inference_wall_seconds = float(sum(latency_ms) / 1000.0)
    replay_wall_seconds = time.perf_counter() - replay_started
    extracted = (accumulated / np.maximum(weight_sum, 1.0e-12)).astype(np.float32)
    power = np.square(mono, dtype=np.float64)
    rms_window = max(1, int(round(0.02 * sample_rate)))
    pad_left = rms_window // 2
    pad_right = rms_window - 1 - pad_left
    padded = np.pad(power, (pad_left, pad_right), mode="edge")
    integral = np.concatenate(([0.0], np.cumsum(padded, dtype=np.float64)))
    smoothed_rms = np.sqrt((integral[rms_window:] - integral[:-rms_window]) / rms_window)
    activity_threshold = max(1.0e-4, 0.10 * float(np.percentile(smoothed_rms, 95)))
    active_mask = smoothed_rms >= activity_threshold
    if int(active_mask.sum()) < sample_rate // 4:
        active_mask = smoothed_rms >= max(1.0e-5, 0.03 * float(smoothed_rms.max()))
    scale_denominator = float(np.dot(extracted[active_mask], extracted[active_mask]))
    if scale_denominator <= 1.0e-16:
        raise RuntimeError("Stage 1 returned a silent estimate on the active recording")
    display_gain = float(np.dot(mono[active_mask], extracted[active_mask]) / scale_denominator)
    extracted = (extracted * display_gain).astype(np.float32)
    residual = (mono - extracted).astype(np.float32)
    time_s = np.arange(len(mono), dtype=np.float64) / sample_rate

    def rms(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))

    active_mixture_rms = rms(mono[active_mask])
    active_target_rms = rms(extracted[active_mask])
    active_residual_rms = rms(residual[active_mask])
    residual_reduction_db = 20.0 * np.log10(
        max(active_mixture_rms, 1.0e-12) / max(active_residual_rms, 1.0e-12)
    )
    misses = sum(bool(row["deadline_miss"]) for row in windows)
    duration = len(mono) / float(sample_rate)
    timeline = {
        "sample_rate": sample_rate,
        "input_channels": input_channels,
        "model_channels": model_channels,
        "stereo_fallback": input_channels != 4,
        "duration_seconds": duration,
        "window_seconds": window / float(sample_rate),
        "hop_seconds": hop / float(sample_rate),
        "kws_threshold": float(getattr(pipeline, "kws_threshold", args.kws_threshold)),
        "events": events,
        "steps": windows,
    }

    metrics_path = repo_path(args.metrics_output)
    csv_path = repo_path(args.windows_csv)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "benchmark": "single_stream_recording_replay",
        "measurement_scope": "Synchronized wall time of full InCarPipeline.forward calls on rolling windows, with input windows scheduled at their audio end times in paced mode; excludes model loading, enrollment embedding setup, plotting, WAV I/O, microphone capture, and ring-buffer overhead.",
        "audio": {
            "path": str(audio_path),
            "sha256": _sha256(audio_path),
            "sample_rate_hz": sample_rate,
            "input_channels": input_channels,
            "model_channels": model_channels,
            "mono_replicated_to_four_channels": input_channels != 4,
            "duration_seconds": duration,
        },
        "runtime": {
            "device": str(device),
            "torch_version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "git_revision": _git_revision(),
            "config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "paths_config": {"path": str(paths_file), "sha256": _sha256(paths_file)},
            "checkpoints": _hash_checkpoints(config, base_checkpoint),
        },
        "setup": {
            "model_load_ms": model_load_ms,
            "enrollment_embedding_ms": enrollment_embedding_ms,
        },
        "stream": {
            "replay_mode": args.replay_mode,
            "window_seconds": window / float(sample_rate),
            "hop_seconds": hop / float(sample_rate),
            "command_delay_calls": args.command_delay_calls,
            "command_timeout_calls": args.command_timeout_calls,
            "kws_threshold": timeline["kws_threshold"],
            "processed_windows": len(windows),
            "inference_latency": _summary(latency_ms),
            "hop_deadline_ms": args.hop_seconds * 1000.0,
            "deadline_miss_count": misses,
            "deadline_miss_fraction": misses / len(windows) if windows else None,
            "late_window_count": sum(float(row["schedule_lag_ms"]) > 1.0 for row in windows),
            "maximum_schedule_lag_ms": max(
                (float(row["schedule_lag_ms"]) for row in windows), default=0.0
            ),
            "mean_latency_to_hop_ratio": (
                float(np.mean(latency_ms)) / (args.hop_seconds * 1000.0) if latency_ms else None
            ),
            "inference_wall_seconds": inference_wall_seconds,
            "paced_replay_wall_seconds": replay_wall_seconds,
            "wake_events": len([row for row in windows if row["wake_detected"]]),
            "intent_events": len([row for row in windows if row["intent_ready"]]),
            "final_action": next((row["action"] for row in reversed(windows) if row["intent_ready"]), None),
        },
        "signal_summary": {
            "active_sample_count": int(active_mask.sum()),
            "display_gain_applied_to_stage1_estimate": display_gain,
            "active_mixture_rms": active_mixture_rms,
            "active_target_estimate_rms": active_target_rms,
            "active_residual_rms": active_residual_rms,
            "active_residual_amplitude_reduction_db": float(residual_reduction_db),
        },
        "interpretation_notes": [
            "This input recording is stereo and was replicated into four model channels; it is not a four-microphone spatial-separation result.",
            "The displayed target estimate is gain-aligned for visualization. Residual amplitude reduction is not SI-SDR improvement or proof that the residual contains noise only.",
            "Paced mode schedules file windows by their audio end time; microphone capture and StreamingRunner ring-buffer overhead are excluded.",
        ],
        "windows": windows,
        "csv_path": str(csv_path),
    }
    metrics_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in CSV_FIELDS} for row in windows])

    if args.mode != "metrics":
        output_dir = repo_path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paper_style()
        plot_tse(
            time_s,
            mono,
            extracted,
            residual,
            active_mask,
            timeline,
            duration,
            output_dir / "tse_speaker_and_removed_component.png",
        )
        plot_stagewise(time_s, mono, timeline, output_dir / "stagewise_processing_paper_style.png")
        print("Figure 1: {}".format(output_dir / "tse_speaker_and_removed_component.png"))
        print("Figure 2: {}".format(output_dir / "stagewise_processing_paper_style.png"))

    print("Full pipeline latency: mean {:.2f} ms; p95 {:.2f} ms; p99 {:.2f} ms".format(
        float(result["stream"]["inference_latency"]["mean_ms"] or 0.0),
        float(result["stream"]["inference_latency"]["p95_ms"] or 0.0),
        float(result["stream"]["inference_latency"]["p99_ms"] or 0.0),
    ))
    print("Hop deadline misses: {}/{}".format(misses, len(windows)))
    print("Windows starting >1 ms behind schedule: {}".format(
        result["stream"]["late_window_count"]
    ))
    print("Wake events: {}; intent events: {}; final action: {}".format(
        result["stream"]["wake_events"],
        result["stream"]["intent_events"],
        result["stream"]["final_action"],
    ))
    print("Metrics JSON: {}".format(metrics_path))
    print("Per-window CSV: {}".format(csv_path))
    if args.mode == "metrics":
        print("Metrics-only mode: no images were written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
