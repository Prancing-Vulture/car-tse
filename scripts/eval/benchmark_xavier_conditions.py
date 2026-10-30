#!/usr/bin/env python3
"""Measure gated end-to-end CAR-TSE latency on Xavier condition recordings.

Each manifest row is a continuous, four-channel 16 kHz recording. In paced
mode, frames are delivered on a real-time schedule to the same StreamingRunner
used by the live demo. The script records per-window wall latency, deadline
misses against the configured hop, runner lag/drops, wake/intent events, and
optional trial-level outcomes. It does not synthesize noise conditions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import platform
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from car_tse.pipeline.streaming_runner import StreamingRunner  # noqa: E402


CONDITIONS = {
    "engine_noise",
    "passenger_speech",
    "air_conditioning",
    "music_playback",
}
CSV_FIELDS = [
    "condition",
    "recording_id",
    "window_index",
    "window_start_s",
    "window_end_s",
    "inference_latency_ms",
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


def _read_manifest(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON on {}:{}: {}".format(path, line_number, exc)) from exc
        if not isinstance(row, dict):
            raise ValueError("Each JSONL row must be an object ({}:{})".format(path, line_number))
        row.setdefault("recording_id", "trial_{:03d}".format(len(rows) + 1))
        condition = str(row.get("condition", "")).strip().lower()
        if condition not in CONDITIONS:
            raise ValueError(
                "{}:{} condition must be one of {}".format(
                    path, line_number, ", ".join(sorted(CONDITIONS))
                )
            )
        row["condition"] = condition
        for field in ("mixture_wav", "enrollment_wav"):
            if not row.get(field):
                raise ValueError("{}:{} is missing {}".format(path, line_number, field))
        expected_wake = row.get("expected_wake")
        if expected_wake is not None and not isinstance(expected_wake, bool):
            raise ValueError(
                "{}:{} expected_wake must be true, false, or null".format(path, line_number)
            )
        expected_action = row.get("expected_action")
        if expected_action is not None and not isinstance(expected_action, str):
            raise ValueError(
                "{}:{} expected_action must be a string or null".format(path, line_number)
            )
        for field in ("expected_wake_end_s", "expected_command_end_s"):
            value = row.get(field)
            if value is not None and (
                not isinstance(value, (int, float))
                or not np.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError(
                    "{}:{} {} must be a non-negative number or null".format(
                        path, line_number, field
                    )
                )
        details = row.get("condition_details", {})
        if details is not None and not isinstance(details, dict):
            raise ValueError(
                "{}:{} condition_details must be an object or null".format(path, line_number)
            )
        row["condition_details"] = details or {}
        rows.append(row)
    if not rows:
        raise ValueError("No scenario rows found in {}".format(path))
    return rows


def _resolve_input(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_mono(path: Path, sample_rate: int) -> np.ndarray:
    audio, source_rate = sf.read(str(path), always_2d=True, dtype="float32")
    if int(source_rate) != int(sample_rate):
        raise ValueError(
            "{} is {} Hz; save/re-record it at the configured {} Hz before benchmarking".format(
                path, source_rate, sample_rate
            )
        )
    mono = audio.mean(axis=1).astype(np.float32)
    if mono.size == 0 or not np.isfinite(mono).all():
        raise ValueError("Invalid or empty audio: {}".format(path))
    return mono


def _load_multichannel(path: Path, sample_rate: int, channels: int) -> np.ndarray:
    audio, source_rate = sf.read(str(path), always_2d=True, dtype="float32")
    if int(source_rate) != int(sample_rate):
        raise ValueError(
            "{} is {} Hz; save/re-record it at the configured {} Hz before benchmarking".format(
                path, source_rate, sample_rate
            )
        )
    if audio.shape[1] != int(channels):
        raise ValueError(
            "{} has {} channels; this config requires {} simultaneous channels".format(
                path, audio.shape[1], channels
            )
        )
    audio = np.asarray(audio, dtype=np.float32)
    if audio.shape[0] == 0 or not np.isfinite(audio).all():
        raise ValueError("Invalid or empty audio: {}".format(path))
    return audio


def _summary(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None}
    samples = np.asarray(values, dtype=np.float64)
    return {
        "count": int(samples.size),
        "mean_ms": float(samples.mean()),
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "p99_ms": float(np.percentile(samples, 99)),
        "max_ms": float(samples.max()),
    }


def _git_revision() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL
        ).decode("ascii", "replace").strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _platform_info(device: torch.device) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "git_revision": _git_revision(),
    }
    if torch.cuda.is_available():
        result["cuda_version"] = torch.version.cuda
        result["gpu_name"] = torch.cuda.get_device_name(device)
        props = torch.cuda.get_device_properties(device)
        result["gpu_total_memory_bytes"] = int(props.total_memory)
    release_path = Path("/etc/nv_tegra_release")
    if release_path.is_file():
        result["jetpack_release"] = release_path.read_text(encoding="utf-8", errors="replace").strip()
    return result


def _score_condition(trials: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    labelled_wake = [trial for trial in trials if trial.get("expected_wake") is not None]
    positives = [trial for trial in labelled_wake if bool(trial["expected_wake"])]
    negatives = [trial for trial in labelled_wake if not bool(trial["expected_wake"])]
    tp = fn = tn = fp_trials = 0
    false_wake_events = 0
    duplicate_wakes_on_positive_trials = 0
    wake_latencies: List[float] = []
    for trial in labelled_wake:
        predicted = trial["wake_events"]
        expected = bool(trial["expected_wake"])
        detected = bool(predicted)
        if expected and detected:
            tp += 1
            duplicate_wakes_on_positive_trials += max(0, len(predicted) - 1)
            end_s = trial.get("expected_wake_end_s")
            if end_s is not None:
                wake_latencies.append((float(predicted[0]["window_end_s"]) - float(end_s)) * 1000.0)
        elif expected:
            fn += 1
        elif detected:
            fp_trials += 1
            false_wake_events += len(predicted)
        else:
            tn += 1

    expected_commands = [trial for trial in trials if trial.get("expected_action")]
    correct_commands = 0
    command_latencies: List[float] = []
    false_actions = 0
    false_action_events_in_negative_trials = 0
    for trial in trials:
        expected_action = str(trial.get("expected_action") or "").strip().casefold()
        predicted = trial["intent_events"]
        if expected_action:
            matches = [
                event for event in predicted
                if str(event.get("action") or "").strip().casefold() == expected_action
            ]
            if matches:
                correct_commands += 1
                end_s = trial.get("expected_command_end_s")
                if end_s is not None:
                    command_latencies.append(
                        (float(matches[0]["window_end_s"]) - float(end_s)) * 1000.0
                    )
            false_actions += max(0, len(predicted) - (1 if matches else 0))
        elif trial.get("expected_wake") is False:
            false_actions += len(predicted)
            false_action_events_in_negative_trials += len(predicted)

    negative_hours = sum(float(trial["duration_seconds"]) for trial in negatives) / 3600.0
    labelled_command_count = len(expected_commands)
    return {
        "wake_trials": len(labelled_wake),
        "wake_true_positive_trials": tp,
        "wake_false_negative_trials": fn,
        "wake_true_negative_trials": tn,
        "wake_false_positive_trials": fp_trials,
        "target_wake_recall": (tp / len(positives)) if positives else None,
        "negative_trial_specificity": (tn / len(negatives)) if negatives else None,
        "false_wake_events": false_wake_events,
        "false_wakes_per_negative_hour": (false_wake_events / negative_hours) if negative_hours else None,
        "duplicate_wake_events_on_positive_trials": duplicate_wakes_on_positive_trials,
        "wake_latency_from_annotated_end_ms": _summary(wake_latencies),
        "expected_command_trials": labelled_command_count,
        "exact_intent_correct_trials": correct_commands,
        "exact_intent_accuracy": (correct_commands / labelled_command_count) if labelled_command_count else None,
        "false_actions_in_labeled_trials": false_actions,
        "false_actions_per_negative_hour": (
            false_action_events_in_negative_trials / negative_hours if negative_hours else None
        ),
        "command_latency_from_annotated_end_ms": _summary(command_latencies),
        "scoring_note": "Trial-level wake metrics assume at most one expected target wake per recording. Negative-hour false-wake rates require fully annotated target-absent duration.",
    }


def _run_scenario(
    row: Dict[str, Any],
    pipeline: Any,
    config: Dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
    enrollment_embedding: Optional[torch.Tensor],
    enrollment_embedding_ms: Optional[float],
) -> Dict[str, Any]:
    sample_rate = int(config.get("sample_rate", 16000))
    channels = int(config.get("num_mics", 4))
    window_seconds = float(config.get("segment_seconds", 3.0))
    hop_seconds = float(
        args.hop_seconds if args.hop_seconds is not None else config.get("hop_seconds", 0.25)
    )
    mixture_path = _resolve_input(str(row["mixture_wav"]))
    enrollment_path = _resolve_input(str(row["enrollment_wav"]))
    playback_path = _resolve_input(str(row["playback_ref_wav"])) if row.get("playback_ref_wav") else None
    multichannel = _load_multichannel(mixture_path, sample_rate, channels)
    mixture = multichannel.mean(axis=1).astype(np.float32)
    enrollment = _load_mono(enrollment_path, sample_rate)
    playback = _load_mono(playback_path, sample_rate) if playback_path else None
    if playback is not None:
        if len(playback) < len(mixture):
            playback = np.pad(playback, (0, len(mixture) - len(playback)))
        else:
            playback = playback[: len(mixture)]

    initial_state = PipelineState()
    if enrollment_embedding is not None:
        initial_state.reference_speaker_embedding = enrollment_embedding
        if getattr(pipeline.presence, "requires_enrollment", False):
            initial_state.presence_enrollment_embedding = enrollment_embedding

    window_samples = int(round(window_seconds * sample_rate))
    hop_samples = max(1, int(round(hop_seconds * sample_rate)))
    frame_samples = max(1, int(round(args.frame_ms * sample_rate / 1000.0)))
    if len(mixture) < window_samples:
        raise ValueError("{} is shorter than the configured inference window".format(mixture_path))

    window_rows: List[Dict[str, Any]] = []
    runner: Optional[StreamingRunner] = None
    pipeline_supports_timestamp = "audio_start_sample" in inspect.signature(pipeline.forward).parameters

    def timed_forward(mix_tensor: torch.Tensor, enroll_tensor: torch.Tensor, state: Any, **kwargs: Any) -> Any:
        nonlocal runner
        timestamp = kwargs.pop("audio_start_sample", None)
        if timestamp is None and runner is not None:
            timestamp = runner.processed_until + runner.hop_samples - runner.window_samples
        if pipeline_supports_timestamp and timestamp is not None:
            kwargs["audio_start_sample"] = int(timestamp)
        synchronize(device)
        started = time.perf_counter()
        output, next_state = pipeline.forward(mix_tensor, enroll_tensor, state, **kwargs)
        synchronize(device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        window_start_s = max(0, int(timestamp or 0)) / float(sample_rate)
        window_end_s = window_start_s + window_seconds
        action_value = getattr(output, "action", None)
        action = getattr(action_value, "action", action_value)
        intent_id = getattr(action_value, "intent_id", None)
        window_rows.append(
            {
                "condition": row["condition"],
                "recording_id": str(row["recording_id"]),
                "window_index": len(window_rows),
                "window_start_s": window_start_s,
                "window_end_s": window_end_s,
                "inference_latency_ms": latency_ms,
                "hop_deadline_ms": hop_seconds * 1000.0,
                "deadline_miss": latency_ms > hop_seconds * 1000.0,
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
        )
        return output, next_state

    runner = StreamingRunner(
        forward=timed_forward,
        enrollment=enrollment,
        sample_rate=sample_rate,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
        num_mics=channels,
        max_lag_seconds=args.max_lag_seconds,
        overload_policy=args.overload_policy,
        max_windows_per_call=args.max_windows_per_call,
        device=device,
        state=initial_state,
        use_playback_ref=playback is not None,
    )

    replay_start = time.perf_counter()
    if args.replay_mode == "paced":
        done = threading.Event()
        producer_errors: List[BaseException] = []

        def produce_frames() -> None:
            try:
                for start in range(0, len(mixture), frame_samples):
                    end = min(len(mixture), start + frame_samples)
                    deadline = replay_start + start / float(sample_rate)
                    delay = deadline - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    runner.push(
                        mixture[start:end],
                        multichannel=multichannel[start:end],
                        playback_ref=None if playback is None else playback[start:end],
                    )
            except BaseException as exc:
                producer_errors.append(exc)
            finally:
                done.set()

        producer = threading.Thread(target=produce_frames, name="xavier-audio-producer")
        producer.start()
        while not done.is_set():
            if runner.process_available() == 0:
                time.sleep(0.001)
        producer.join()
        if producer_errors:
            raise RuntimeError("Audio replay producer failed") from producer_errors[0]
        runner.process_available()
    else:
        for start in range(0, len(mixture), frame_samples):
            end = min(len(mixture), start + frame_samples)
            runner.push(
                mixture[start:end],
                multichannel=multichannel[start:end],
                playback_ref=None if playback is None else playback[start:end],
            )
            runner.process_available()
    replay_wall_seconds = time.perf_counter() - replay_start
    stats = runner.stats()

    latencies = [float(item["inference_latency_ms"]) for item in window_rows]
    misses = sum(bool(item["deadline_miss"]) for item in window_rows)
    wake_events = [item for item in window_rows if item["wake_detected"]]
    intent_events = [item for item in window_rows if item["intent_ready"]]
    trial = {
        "recording_id": str(row["recording_id"]),
        "condition": str(row["condition"]),
        "duration_seconds": len(mixture) / float(sample_rate),
        "expected_wake": row.get("expected_wake"),
        "expected_wake_end_s": row.get("expected_wake_end_s"),
        "expected_action": row.get("expected_action"),
        "expected_command_end_s": row.get("expected_command_end_s"),
        "wake_events": wake_events,
        "intent_events": intent_events,
    }
    return {
        "recording_id": str(row["recording_id"]),
        "condition": str(row["condition"]),
        "mixture_wav": str(mixture_path),
        "mixture_sha256": _sha256(mixture_path),
        "enrollment_wav": str(enrollment_path),
        "enrollment_sha256": _sha256(enrollment_path),
        "playback_ref_wav": None if playback_path is None else str(playback_path),
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_seconds": trial["duration_seconds"],
        "condition_details": row.get("condition_details", {}),
        "replay_mode": args.replay_mode,
        "replay_wall_seconds": replay_wall_seconds,
        "one_time_enrollment_embedding_ms": enrollment_embedding_ms,
        "windows": int(stats["processed_windows"]),
        "inference_latency": _summary(latencies),
        "hop_deadline_ms": hop_seconds * 1000.0,
        "deadline_miss_count": int(misses),
        "deadline_miss_fraction": (misses / len(latencies)) if latencies else None,
        "real_time_factor": stats["real_time_factor"],
        "runner": stats,
        "wake_event_count": len(wake_events),
        "intent_event_count": len(intent_events),
        "trial_scoring": {
            "expected_wake": trial["expected_wake"],
            "expected_wake_end_s": trial["expected_wake_end_s"],
            "expected_action": trial["expected_action"],
            "expected_command_end_s": trial["expected_command_end_s"],
        },
        "_trial": trial,
        "_windows": window_rows,
    }


def _checkpoint_hashes(config: Dict[str, Any], paths_file: Path, base_checkpoint: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"base_spex": {"path": str(base_checkpoint), "sha256": _sha256(base_checkpoint)}}
    checkpoint_root = Path(str(config.get("paths", {}).get("checkpoint_root", "checkpoints"))).expanduser()
    if not checkpoint_root.is_absolute():
        checkpoint_root = (REPO_ROOT / checkpoint_root).resolve()
    for name, relative in (config.get("checkpoints", {}) or {}).items():
        path = Path(str(relative)).expanduser()
        if not path.is_absolute():
            path = (checkpoint_root / path).resolve()
        result[str(name)] = {
            "path": str(path),
            "sha256": _sha256(path) if path.is_file() else None,
            "exists": path.is_file(),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "configs/xavier_conditions.example.jsonl")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/eval_end_to_end_cavemove.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.yaml")
    parser.add_argument("--base-checkpoint", type=Path, default=REPO_ROOT / "checkpoints/car_tse_spexplus_from_stage1.pt")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--replay-mode", choices=("paced", "fast"), default="paced")
    parser.add_argument("--frame-ms", type=float, default=20.0)
    parser.add_argument("--hop-seconds", type=float, default=None)
    parser.add_argument("--max-lag-seconds", type=float, default=2.0)
    parser.add_argument("--max-windows-per-call", type=int, default=8)
    parser.add_argument("--overload-policy", choices=("drop_oldest", "catch_up"), default="drop_oldest")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "logs/xavier_condition_latency.json")
    parser.add_argument("--windows-csv", type=Path, default=REPO_ROOT / "logs/xavier_condition_windows.csv")
    args = parser.parse_args()
    if args.frame_ms <= 0 or args.max_lag_seconds <= 0 or args.max_windows_per_call < 1:
        parser.error("frame, lag, and window limits must be positive")

    manifest_path = repo_path(args.manifest)
    config_path = repo_path(args.config)
    paths_file = repo_path(args.paths)
    if not paths_file.is_file():
        paths_file = REPO_ROOT / "configs/paths.example.yaml"
    base_checkpoint = repo_path(args.base_checkpoint)
    for path, label in (
        (manifest_path, "condition manifest"),
        (config_path, "pipeline config"),
        (paths_file, "paths config"),
        (base_checkpoint, "base SpEx+ checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError("Missing {}: {}".format(label, path))
    rows = _read_manifest(manifest_path)
    for row in rows:
        for key in ("mixture_wav", "enrollment_wav", "playback_ref_wav"):
            if row.get(key) and not _resolve_input(str(row[key])).is_file():
                raise FileNotFoundError("Missing {} for {}: {}".format(key, row["recording_id"], row[key]))

    config = load_stage_config(config_path, paths_file)
    if int(config.get("num_mics", 4)) != 4:
        raise ValueError("This publication benchmark requires the configured four-microphone pipeline")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type != "cuda":
        print("WARNING: this run is not a Jetson GPU measurement (device={})".format(device), file=sys.stderr)

    model_load_started = time.perf_counter()
    pipeline = load_pipeline(config, paths_file, base_checkpoint, device)
    synchronize(device)
    model_load_ms = (time.perf_counter() - model_load_started) * 1000.0
    # Older checkouts omit this optional member even though forward checks it.
    if not hasattr(pipeline, "stage23_fusion"):
        pipeline.stage23_fusion = None
    if callable(getattr(pipeline, "eval", None)):
        pipeline.eval()

    scenario_results: List[Dict[str, Any]] = []
    all_windows: List[Dict[str, Any]] = []
    trials: List[Dict[str, Any]] = []
    enrollment_times: Dict[str, Optional[float]] = {}
    enrollment_embeddings: Dict[str, Optional[torch.Tensor]] = {}
    for row in rows:
        enrollment_path = _resolve_input(str(row["enrollment_wav"]))
        enrollment = _load_mono(enrollment_path, int(config.get("sample_rate", 16000)))
        enrollment_tensor = torch.from_numpy(np.ascontiguousarray(enrollment[None, :])).to(device)
        enrollment_time = None
        embedding = None
        if callable(getattr(pipeline.stage1, "speaker_embedding", None)):
            synchronize(device)
            started = time.perf_counter()
            with torch.inference_mode():
                embedding = pipeline.stage1.speaker_embedding(enrollment_tensor)
            synchronize(device)
            enrollment_time = (time.perf_counter() - started) * 1000.0
            if embedding is not None:
                embedding = embedding.detach()
        enrollment_times[str(row["recording_id"])] = enrollment_time
        enrollment_embeddings[str(row["recording_id"])] = embedding

    # Warm the actual inference path once using the first scenario, then begin
    # each recorded trial with a fresh state and the same enrollment setup.
    first = rows[0]
    warm_audio = _load_multichannel(
        _resolve_input(str(first["mixture_wav"])),
        int(config.get("sample_rate", 16000)),
        int(config.get("num_mics", 4)),
    )
    warm_enrollment = _load_mono(
        _resolve_input(str(first["enrollment_wav"])), int(config.get("sample_rate", 16000))
    )
    warm_samples = int(round(float(config.get("segment_seconds", 3.0)) * int(config.get("sample_rate", 16000))))
    warm_mc = torch.from_numpy(np.ascontiguousarray(warm_audio[:warm_samples].T[None, :, :])).to(device)
    warm_mix = torch.from_numpy(np.ascontiguousarray(warm_audio[:warm_samples].mean(axis=1)[None, :])).to(device)
    warm_enroll = torch.from_numpy(np.ascontiguousarray(warm_enrollment[None, :])).to(device)
    warm_state = PipelineState()
    warm_embedding = enrollment_embeddings.get(str(first["recording_id"]))
    if warm_embedding is not None:
        warm_state.reference_speaker_embedding = warm_embedding
        if getattr(pipeline.presence, "requires_enrollment", False):
            warm_state.presence_enrollment_embedding = warm_embedding
    with torch.inference_mode():
        pipeline.forward(warm_mix, warm_enroll, warm_state, multichannel=warm_mc)
    synchronize(device)

    for row in rows:
        with torch.inference_mode():
            result = _run_scenario(
                row,
                pipeline,
                config,
                device,
                args,
                enrollment_embeddings[str(row["recording_id"])],
                enrollment_times[str(row["recording_id"])],
            )
        trials.append(result.pop("_trial"))
        all_windows.extend(result.pop("_windows"))
        scenario_results.append(result)

    by_condition: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in scenario_results:
        by_condition[result["condition"]].append(result)
    condition_summaries = []
    for condition in sorted(by_condition):
        condition_rows = by_condition[condition]
        latencies = [
            float(window["inference_latency_ms"])
            for window in all_windows
            if window["condition"] == condition
        ]
        deadline = condition_rows[0]["hop_deadline_ms"]
        miss_count = sum(
            bool(window["deadline_miss"])
            for window in all_windows
            if window["condition"] == condition
        )
        trials_for_condition = [trial for trial in trials if trial["condition"] == condition]
        condition_summaries.append(
            {
                "condition": condition,
                "recording_count": len(condition_rows),
                "total_audio_seconds": sum(float(item["duration_seconds"]) for item in condition_rows),
                "inference_latency": _summary(latencies),
                "hop_deadline_ms": deadline,
                "deadline_miss_count": int(miss_count),
                "deadline_miss_fraction": (miss_count / len(latencies)) if latencies else None,
                "mean_latency_to_hop_ratio": (
                    (float(np.mean(latencies)) / deadline) if latencies and deadline else None
                ),
                "total_windows_processed": sum(int(item["windows"]) for item in condition_rows),
                "total_skipped_audio_seconds": sum(float(item["runner"]["skipped_seconds"]) for item in condition_rows),
                "maximum_lag_seconds": max(float(item["runner"]["max_lag_seconds_seen"]) for item in condition_rows),
                "wake_event_count": sum(int(item["wake_event_count"]) for item in condition_rows),
                "intent_event_count": sum(int(item["intent_event_count"]) for item in condition_rows),
                "annotated_outcomes": _score_condition(trials_for_condition),
            }
        )

    checkpoint_hashes = _checkpoint_hashes(config, paths_file, base_checkpoint)
    result = {
        "benchmark": "xavier_condition_paced_stream_replay",
        "measurement_scope": "Measured full InCarPipeline.forward wall time per StreamingRunner window; includes stage gating and synchronous CUDA completion, excludes audio capture hardware, file reads, model loading, and enrollment setup.",
        "replay_mode": args.replay_mode,
        "model_load_ms": model_load_ms,
        "platform": _platform_info(device),
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "paths_config": {"path": str(paths_file), "sha256": _sha256(paths_file)},
        "condition_manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "checkpoints": checkpoint_hashes,
        "runtime": {
            "sample_rate": int(config.get("sample_rate", 16000)),
            "microphone_channels": 4,
            "window_seconds": float(config.get("segment_seconds", 3.0)),
            "hop_seconds": float(
                args.hop_seconds if args.hop_seconds is not None else config.get("hop_seconds", 0.25)
            ),
            "capture_frame_ms": float(args.frame_ms),
            "overload_policy": args.overload_policy,
            "max_lag_seconds": float(args.max_lag_seconds),
            "max_windows_per_call": int(args.max_windows_per_call),
            "cuda_synchronize_around_each_pipeline_call": device.type == "cuda",
        },
        "conditions": condition_summaries,
        "recordings": scenario_results,
        "per_window_csv": str(repo_path(args.windows_csv)),
        "interpretation_notes": [
            "Paced replay uses saved four-channel recordings and StreamingRunner, not live microphone callbacks; it reports measured Xavier inference time for the recorded conditions.",
            "A deadline miss means measured pipeline forward latency exceeded one configured hop. It is a compute deadline metric, not a wake/intent accuracy result.",
            "Condition-wise accuracy and false-event rates are computed only when expected_wake/expected_action annotations are supplied in the manifest.",
            "Use matched target-positive and target-absent recordings for every condition; do not treat synthetic CAVEMOVE noise or stereo fallback as a four-microphone passenger test.",
        ],
    }
    output_path = repo_path(args.output)
    windows_path = repo_path(args.windows_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    windows_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with windows_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_windows)

    print("Condition-wise Xavier replay results")
    print("condition              windows   mean ms   p95 ms   p99 ms   missed-hop %   wake   intent")
    for summary in condition_summaries:
        stats = summary["inference_latency"]
        percent = 0.0 if summary["deadline_miss_fraction"] is None else 100.0 * summary["deadline_miss_fraction"]
        print(
            "{:<23s} {:>7d} {:>9.2f} {:>8.2f} {:>8.2f} {:>12.2f} {:>6d} {:>8d}".format(
                summary["condition"],
                int(summary["total_windows_processed"]),
                float(stats["mean_ms"] or 0.0),
                float(stats["p95_ms"] or 0.0),
                float(stats["p99_ms"] or 0.0),
                percent,
                int(summary["wake_event_count"]),
                int(summary["intent_event_count"]),
            )
        )
    print("JSON summary: {}".format(output_path))
    print("Per-window CSV: {}".format(windows_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
