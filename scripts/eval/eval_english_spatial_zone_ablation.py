#!/usr/bin/env python
"""Compare the current TSE-first path with English spatial front-end variants.

The benchmark renders English wake words and commands through CAVEMOVE's
declared driver/front-passenger impulse responses, adds real vehicle noise,
and then runs the same stateful wake-to-intent pipeline in three modes:

* ``tse_first``: learned cabin front end + SpEx+ (current reference);
* ``hybrid``: calibrated driver-seat delay-and-sum beamformer followed by
  SpEx+ without the learned cabin front end;
* ``zone_only``: the beamformer output is supplied directly to the existing
  presence/KWS/intent cascade, bypassing per-hop TSE.

This is an English synthetic spatial proxy, not a real-driver evaluation.
The CAVEMOVE impulse responses provide location-dependent multichannel
acoustics, but wake and command clips still come from separate public corpora.
The result is intended to decide whether the proposed zone-first direction is
worth real-driver capture and further training.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy.signal import fftconvolve

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from car_tse.frontend.spatial import delay_and_sum  # noqa: E402
from car_tse.io.audio import read_mono, read_multichannel  # noqa: E402
from car_tse.pipeline.in_car_pipeline import InCarPipeline, PipelineState  # noqa: E402
from car_tse.pipeline.windowing import full_context_window_starts  # noqa: E402
from car_tse.train.utils import get_device, write_metrics  # noqa: E402
from eval_continuous_full_pipeline_proxy import (  # noqa: E402
    fit_start,
    load_jsonl,
    load_pipeline,
    resolve_path,
    rms,
    save_audio,
    speaker_key,
)
from car_tse.config import load_stage_config  # noqa: E402


SAMPLE_RATE = 16_000
CHANNELS = [0, 2, 4, 6]
VARIANTS = ("tse_first", "hybrid", "zone_only")


def _pick_file(directory: Path, preferred: list[str], fallback_prefix: str) -> Path:
    files = {path.stem: path for path in directory.glob("*.wav")}
    for stem in preferred:
        if stem in files:
            return files[stem]
    candidates = sorted(path for path in directory.glob(f"{fallback_prefix}*.wav"))
    if not candidates:
        raise FileNotFoundError(f"No {fallback_prefix!r} impulse response under {directory}")
    return candidates[0]


def load_spatial_calibration(root: Path, car: str) -> dict[str, Any]:
    """Load driver/passenger IRs and derive declared integer steering delays."""

    car_root = root / car / "distributed"
    ir_root = car_root / "IRs"
    driver_path = _pick_file(
        ir_root,
        ["d45_w0", "d50_w0", "d60_w0", "d_w0"],
        "d",
    )
    passenger_path = _pick_file(ir_root, ["pf_w0", "prl_w0", "prr_w0"], "pf")
    driver_full, sr = read_multichannel(driver_path, SAMPLE_RATE)
    passenger_full, passenger_sr = read_multichannel(passenger_path, SAMPLE_RATE)
    if sr != SAMPLE_RATE or passenger_sr != SAMPLE_RATE:
        raise ValueError("CAVEMOVE IRs must be available at 16 kHz")
    driver = driver_full[:, CHANNELS].T
    passenger = passenger_full[:, CHANNELS].T
    driver_peaks = np.argmax(np.abs(driver), axis=1)
    driver_delays = (int(driver_peaks[0]) - driver_peaks).astype(int).tolist()
    return {
        "car": car,
        "driver_ir": driver,
        "passenger_ir": passenger,
        "driver_ir_path": str(driver_path),
        "passenger_ir_path": str(passenger_path),
        "driver_peak_samples": driver_peaks.astype(int).tolist(),
        "driver_delays_samples": driver_delays,
    }


def _render_clip(clip: np.ndarray, impulse_response: np.ndarray) -> np.ndarray:
    """Render a mono English clip through a [channels, taps] IR."""

    clip = np.asarray(clip, dtype=np.float32).reshape(-1)
    ir = np.asarray(impulse_response, dtype=np.float32)
    rendered = np.stack(
        [fftconvolve(clip, ir_channel, mode="full") for ir_channel in ir],
        axis=1,
    ).astype(np.float32)
    # IR recording gain is not a speech level. Preserve the clip RMS while
    # retaining the relative channel gains that encode the source location.
    rendered_rms = float(np.sqrt(np.mean(rendered * rendered) + 1.0e-8))
    if rendered_rms > 1.0e-8:
        rendered *= float(rms(clip) / rendered_rms)
    return rendered


def _add_rendered(
    destination: np.ndarray,
    clip: np.ndarray,
    start: int,
    impulse_response: np.ndarray,
) -> None:
    rendered = _render_clip(clip, impulse_response)
    if start >= len(destination):
        return
    start = max(0, int(start))
    end = min(len(destination), start + len(rendered))
    if end > start:
        destination[start:end] += rendered[: end - start]


def _load_noise(path: Path, total_samples: int) -> np.ndarray:
    raw, sr = read_multichannel(path, SAMPLE_RATE)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Unexpected sample rate for {path}: {sr}")
    noise = raw[:, CHANNELS].astype(np.float32, copy=False)
    if len(noise) < total_samples:
        repeats = int(np.ceil(total_samples / max(1, len(noise))))
        noise = np.tile(noise, (repeats, 1))
    return noise[:total_samples]


def _select_noise(root: Path, car: str, minimum_seconds: float = 20.0) -> list[Path]:
    noise_root = root / car / "distributed"
    candidates: list[Path] = []
    for path in sorted(noise_root.rglob("*.wav")):
        if path.parent.name not in {"noise", "ventilation"}:
            continue
        try:
            if sf.info(str(path)).duration >= minimum_seconds:
                candidates.append(path)
        except (OSError, RuntimeError):
            continue
    if not candidates:
        raise FileNotFoundError(f"No long CAVEMOVE noise files under {noise_root}")
    return candidates


def build_spatial_scene(
    target_wake: np.ndarray,
    target_command: np.ndarray,
    passenger_wake: np.ndarray,
    passenger_nonwake: np.ndarray,
    noise: np.ndarray,
    calibration: dict[str, Any],
    cabin_snr_db: float,
    passenger_snr_db: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    duration_seconds = 14.0
    total_samples = int(round(duration_seconds * SAMPLE_RATE))
    target = np.zeros((total_samples, len(CHANNELS)), dtype=np.float32)
    passenger = np.zeros_like(target)
    wake_start = int(round(1.0 * SAMPLE_RATE))
    command_start = int(round(2.1 * SAMPLE_RATE))
    passenger_wake_start = int(round(6.0 * SAMPLE_RATE))
    passenger_nonwake_start = int(round(9.0 * SAMPLE_RATE))
    _add_rendered(target, target_wake, wake_start, calibration["driver_ir"])
    _add_rendered(target, target_command, command_start, calibration["driver_ir"])
    _add_rendered(passenger, passenger_wake, passenger_wake_start, calibration["passenger_ir"])
    _add_rendered(passenger, passenger_nonwake, passenger_nonwake_start, calibration["passenger_ir"])

    target_level = float(np.sqrt(np.mean(target * target) + 1.0e-8))
    passenger_level = float(np.sqrt(np.mean(passenger * passenger) + 1.0e-8))
    passenger_scale = target_level * (10.0 ** (-float(passenger_snr_db) / 20.0)) / max(
        passenger_level, 1.0e-6
    )
    passenger_scaled = passenger * passenger_scale

    noise = np.asarray(noise, dtype=np.float32)
    noise_level = float(np.sqrt(np.mean(noise * noise) + 1.0e-8))
    noise_scale = target_level * (10.0 ** (-float(cabin_snr_db) / 20.0)) / max(
        noise_level, 1.0e-6
    )
    full = target + passenger_scaled + noise * noise_scale
    mixture = full.mean(axis=1).astype(np.float32)
    truth = {
        "duration_seconds": duration_seconds,
        "target_wake_start_seconds": wake_start / SAMPLE_RATE,
        "target_wake_end_seconds": (wake_start + len(target_wake)) / SAMPLE_RATE,
        "target_command_start_seconds": command_start / SAMPLE_RATE,
        "target_command_end_seconds": (command_start + len(target_command)) / SAMPLE_RATE,
        "passenger_wake_start_seconds": passenger_wake_start / SAMPLE_RATE,
        "passenger_wake_end_seconds": (passenger_wake_start + len(passenger_wake)) / SAMPLE_RATE,
        "passenger_nonwake_start_seconds": passenger_nonwake_start / SAMPLE_RATE,
        "passenger_nonwake_end_seconds": (passenger_nonwake_start + len(passenger_nonwake)) / SAMPLE_RATE,
        "cabin_snr_db": float(cabin_snr_db),
        "passenger_snr_db": float(passenger_snr_db),
    }
    return mixture, full.astype(np.float32), passenger_scaled.astype(np.float32), truth


def _run_variant(
    pipeline: InCarPipeline,
    variant: str,
    mixture: np.ndarray,
    multichannel: np.ndarray,
    beamformed: np.ndarray,
    enrollment: np.ndarray,
    truth: dict[str, Any],
    sample_rate: int,
    stage1_seconds: float,
    hop_seconds: float,
    device: torch.device,
) -> dict[str, Any]:
    window_samples = int(round(stage1_seconds * sample_rate))
    hop_samples = max(1, int(round(hop_seconds * sample_rate)))
    state = PipelineState()
    rows: list[dict[str, Any]] = []
    wake_events: list[dict[str, Any]] = []
    command_events: list[dict[str, Any]] = []
    for start in full_context_window_starts(len(mixture), window_samples, hop_samples):
        end = min(start + window_samples, len(mixture))
        raw_window = np.zeros(window_samples, dtype=np.float32)
        beam_window = np.zeros(window_samples, dtype=np.float32)
        mc_window = np.zeros((multichannel.shape[1], window_samples), dtype=np.float32)
        if end > start:
            raw_window[: end - start] = mixture[start:end]
            beam_window[: end - start] = beamformed[start:end]
            mc_window[:, : end - start] = multichannel[start:end].T
        if variant == "tse_first":
            model_input = raw_window
            kwargs: dict[str, Any] = {"multichannel": torch.from_numpy(mc_window).to(device)}
        elif variant == "hybrid":
            model_input = beam_window
            kwargs = {}
        elif variant == "zone_only":
            model_input = beam_window
            kwargs = {"pre_extracted": torch.from_numpy(beam_window).to(device)}
        else:
            raise ValueError(f"Unknown variant: {variant}")
        output, state = pipeline.forward(
            torch.from_numpy(model_input).to(device),
            torch.from_numpy(enrollment).to(device),
            state,
            **kwargs,
        )
        row: dict[str, Any] = {
            "window_start_seconds": start / sample_rate,
            "window_center_seconds": start / sample_rate + stage1_seconds / 2.0,
            "wake_detected": bool(output.wake_detected),
            "intent_ready": bool(output.intent_ready),
            "accepted": bool(output.accepted),
            "presence_probability": output.presence_probability,
            "speaker_similarity": output.speaker_similarity,
            "wake_probability": output.wake_probability,
            "intent": None,
            "intent_confidence": None,
            "action": None,
        }
        if output.intent_logits is not None:
            probabilities = torch.softmax(output.intent_logits, dim=-1)[0]
            predicted_id = str(int(probabilities.argmax().item()))
            row["intent"] = pipeline.label_map.get(predicted_id, predicted_id) if pipeline.label_map else predicted_id
            row["intent_confidence"] = float(probabilities.max().item())
            row["action"] = output.action.action if output.action is not None else None
        rows.append(row)
        if output.wake_detected:
            wake_events.append(row)
        if output.intent_ready:
            command_events.append(row)

    target_start = float(truth["target_wake_start_seconds"])
    target_end = float(truth["target_wake_end_seconds"])
    target_wake_events = [
        row for row in wake_events
        if target_start - 0.25 <= row["window_center_seconds"] <= target_end + 3.0
    ]
    target_wake_event = target_wake_events[0] if target_wake_events else None
    false_events = [row for row in wake_events if row not in target_wake_events]
    passenger_start = float(truth["passenger_wake_start_seconds"])
    passenger_end = float(truth["passenger_wake_end_seconds"])
    passenger_events = [
        row for row in false_events
        if passenger_start - 0.5 <= row["window_center_seconds"] <= passenger_end + 2.0
    ]
    command_start = float(truth["target_command_start_seconds"])
    command_end = float(truth["target_command_end_seconds"])
    command_truth_events = [
        row for row in command_events
        if command_start - 0.5 <= row["window_center_seconds"] <= command_end + 3.0
    ]
    target_command_event = command_truth_events[0] if command_truth_events else None
    false_command_events = [row for row in command_events if row not in command_truth_events]
    expected_intent = str(truth["expected_intent"])
    intent_correct = bool(target_command_event and target_command_event.get("intent") == expected_intent)
    latency = None if target_wake_event is None else float(
        target_wake_event["window_center_seconds"] - target_start
    )
    return {
        "variant": variant,
        "target_wake_detected": target_wake_event is not None,
        "target_intent_correct": intent_correct,
        "expected_intent": expected_intent,
        "predicted_intent": target_command_event.get("intent") if target_command_event else None,
        "false_wake_events": len(false_events),
        "passenger_wake_false_events": len(passenger_events),
        "false_action_events": len(false_command_events),
        "target_wake_latency_seconds": latency,
        "windows": rows,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    duration = sum(float(row["truth"]["duration_seconds"]) for row in rows)
    wake = sum(bool(row["target_wake_detected"]) for row in rows)
    intent = sum(bool(row["target_intent_correct"]) for row in rows)
    false_wake = sum(int(row["false_wake_events"]) for row in rows)
    passenger_wake = sum(int(row["passenger_wake_false_events"]) for row in rows)
    false_action = sum(int(row["false_action_events"]) for row in rows)
    latencies = [
        float(row["target_wake_latency_seconds"])
        for row in rows
        if row["target_wake_latency_seconds"] is not None
    ]
    by_snr: dict[str, dict[str, Any]] = {}
    for snr in sorted({float(row["snr_db"]) for row in rows}):
        subset = [row for row in rows if float(row["snr_db"]) == snr]
        snr_duration = sum(float(row["truth"]["duration_seconds"]) for row in subset)
        by_snr[f"{snr:+g}dB"] = {
            "scenes": len(subset),
            "wake_recall": sum(bool(row["target_wake_detected"]) for row in subset) / max(len(subset), 1),
            "intent_accuracy": sum(bool(row["target_intent_correct"]) for row in subset) / max(len(subset), 1),
            "false_wakes_per_hour": sum(int(row["false_wake_events"]) for row in subset) * 3600.0 / max(snr_duration, 1.0e-6),
            "passenger_wake_events": sum(int(row["passenger_wake_false_events"]) for row in subset),
        }
    return {
        "scenes": len(rows),
        "duration_seconds": duration,
        "duration_hours": duration / 3600.0,
        "wake_recall": wake / max(len(rows), 1),
        "intent_accuracy_overall": intent / max(len(rows), 1),
        "intent_accuracy_conditioned_on_wake": intent / max(wake, 1),
        "false_wake_events": false_wake,
        "false_wakes_per_hour": false_wake * 3600.0 / max(duration, 1.0e-6),
        "passenger_wake_false_events": passenger_wake,
        "false_action_events": false_action,
        "false_actions_per_hour": false_action * 3600.0 / max(duration, 1.0e-6),
        "wake_latency_mean_seconds": float(np.mean(latencies)) if latencies else None,
        "wake_latency_p95_seconds": float(np.percentile(latencies, 95)) if latencies else None,
        "by_cabin_snr": by_snr,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/eval_end_to_end_cavemove_target_kws_smart_speaker_compact_cabin_smart_thr030_2026-10-12.yaml",
    )
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.yaml")
    parser.add_argument("--wake-manifest", type=Path, default=REPO_ROOT / "manifests/test/stage3_kws_cabin_test.jsonl")
    parser.add_argument("--intent-manifest", type=Path, default=REPO_ROOT / "manifests/official/stage4_intent_fsc_test.jsonl")
    parser.add_argument("--cavemove-root", type=Path, default=REPO_ROOT / "data/raw/cavemove/extracted/cavemove_dataset")
    parser.add_argument("--car", type=str, default="AlfaRomeo_146")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "logs/english_spatial_zone_ablation_2026-10-13.json")
    parser.add_argument("--artifact-dir", type=Path, default=REPO_ROOT / "reports/artifacts/english_spatial_zone_ablation_2026-10-13")
    parser.add_argument("--scenes", type=int, default=8)
    parser.add_argument("--cabin-snr-db", nargs="+", type=float, default=[-5.0, 0.0, 5.0])
    parser.add_argument("--passenger-snr-db", type=float, default=-5.0)
    parser.add_argument("--hop-seconds", type=float, default=0.5)
    parser.add_argument("--presence-threshold", type=float, default=None)
    parser.add_argument("--kws-threshold", type=float, default=None)
    parser.add_argument("--speaker-gate-threshold", type=float, default=None)
    parser.add_argument(
        "--enrollment-mode",
        choices=("same_wake", "separate_same_speaker"),
        default="separate_same_speaker",
        help=(
            "Use the wake clip itself for an exact wake/enrollment control, or "
            "a different Marvin clip from the same Speech Commands speaker for "
            "the stricter wake/enrollment-variation proxy. The command still "
            "comes from Fluent Speech Commands, so neither mode is a same-driver "
            "wake-to-command evaluation."
        ),
    )
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    cfg = load_stage_config(args.config, args.paths)
    if args.presence_threshold is not None:
        cfg["presence_threshold"] = float(args.presence_threshold)
        cfg["command_presence_threshold"] = float(args.presence_threshold)
    if args.kws_threshold is not None:
        cfg["kws_threshold"] = float(args.kws_threshold)
    if args.speaker_gate_threshold is not None:
        cfg["speaker_gate_threshold"] = float(args.speaker_gate_threshold)
        cfg["command_speaker_gate_threshold"] = float(args.speaker_gate_threshold)
    device = get_device()
    pipeline = load_pipeline(cfg, cfg["paths"], device)
    sample_rate = int(cfg.get("sample_rate", SAMPLE_RATE))
    stage1_seconds = float(cfg.get("segment_seconds", 3.0))
    if sample_rate != SAMPLE_RATE:
        raise ValueError("The English spatial proxy currently expects a 16 kHz runtime")

    wake_rows = [
        row for row in load_jsonl(resolve_path(args.wake_manifest))
        if int(row.get("label", 0)) == 1 and not bool(row.get("noise_only", False))
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in wake_rows:
        grouped[speaker_key(resolve_path(row["wav"]))].append(row)
    speaker_groups = [rows for _, rows in sorted(grouped.items()) if rows]
    if len(speaker_groups) < 2:
        raise RuntimeError("Need at least two English Marvin speakers")
    target_groups = speaker_groups[: max(1, int(args.scenes))]
    passenger_group = speaker_groups[-1] if speaker_groups[-1] is not target_groups[0] else speaker_groups[-2]
    passenger_wake_row = passenger_group[0]
    negative_rows = [
        row for row in load_jsonl(resolve_path(args.wake_manifest))
        if int(row.get("label", 0)) == 0 and not bool(row.get("noise_only", False))
    ]
    if not negative_rows:
        raise RuntimeError("No English non-wake clips found")
    passenger_nonwake_row = negative_rows[0]
    intent_rows = load_jsonl(resolve_path(args.intent_manifest))
    by_intent: dict[str, dict[str, Any]] = {}
    for row in intent_rows:
        by_intent.setdefault(str(row["intent"]), row)
    intent_rows = [by_intent[key] for key in sorted(by_intent)]
    if not intent_rows:
        raise RuntimeError("No English intent rows found")

    calibration = load_spatial_calibration(resolve_path(args.cavemove_root), args.car)
    noise_files = _select_noise(resolve_path(args.cavemove_root), args.car)
    print(
        json.dumps(
            {
                "device": str(device),
                "car": args.car,
                "driver_ir": calibration["driver_ir_path"],
                "passenger_ir": calibration["passenger_ir_path"],
                "driver_delays_samples": calibration["driver_delays_samples"],
                "noise_files": len(noise_files),
            },
            indent=2,
        ),
        flush=True,
    )

    scenes: list[dict[str, Any]] = []
    for scene_index, group in enumerate(target_groups):
        wake_row = group[0]
        enrollment_row = (
            group[0]
            if args.enrollment_mode == "same_wake" or len(group) < 2
            else group[1]
        )
        intent_row = intent_rows[scene_index % len(intent_rows)]
        target_wake, _ = read_mono(resolve_path(wake_row["wav"]), sample_rate)
        enrollment, _ = read_mono(resolve_path(enrollment_row["wav"]), sample_rate)
        target_command, _ = read_mono(resolve_path(intent_row["wav"]), sample_rate)
        passenger_wake, _ = read_mono(resolve_path(passenger_wake_row["wav"]), sample_rate)
        passenger_nonwake, _ = read_mono(resolve_path(passenger_nonwake_row["wav"]), sample_rate)
        target_wake = fit_start(target_wake, sample_rate)
        enrollment = fit_start(enrollment, int(round(stage1_seconds * sample_rate / 2.0)))
        target_command = fit_start(target_command, 2 * sample_rate)
        passenger_wake = fit_start(passenger_wake, sample_rate)
        passenger_nonwake = fit_start(passenger_nonwake, int(1.5 * sample_rate))
        for snr_index, cabin_snr_db in enumerate(args.cabin_snr_db):
            noise_path = noise_files[(scene_index + snr_index) % len(noise_files)]
            noise = _load_noise(noise_path, int(round(14.0 * sample_rate)))
            mixture, multichannel, passenger_scaled, truth = build_spatial_scene(
                target_wake,
                target_command,
                passenger_wake,
                passenger_nonwake,
                noise,
                calibration,
                float(cabin_snr_db),
                float(args.passenger_snr_db),
            )
            truth.update(
                {
                    "expected_intent": str(intent_row["intent"]),
                    "target_wake_source": str(resolve_path(wake_row["wav"])),
                    "enrollment_source": str(resolve_path(enrollment_row["wav"])),
                    "target_command_source": str(resolve_path(intent_row["wav"])),
                    "passenger_wake_source": str(resolve_path(passenger_wake_row["wav"])),
                    "passenger_nonwake_source": str(resolve_path(passenger_nonwake_row["wav"])),
                    "noise_source": str(noise_path),
                    "spatial_calibration": {
                        "driver_ir": calibration["driver_ir_path"],
                        "passenger_ir": calibration["passenger_ir_path"],
                        "driver_delays_samples": calibration["driver_delays_samples"],
                    },
                    "english_only": True,
                    "speaker_consistent": False,
                    "same_driver_wake_to_command": False,
                    "production_valid": False,
                }
            )
            beamformed = delay_and_sum(
                multichannel.T,
                calibration["driver_delays_samples"],
            ).astype(np.float32, copy=False)
            scenes.append(
                {
                    "scene_index": scene_index,
                    "snr_db": float(cabin_snr_db),
                    "mixture": mixture,
                    "multichannel": multichannel,
                    "passenger_scaled": passenger_scaled,
                    "beamformed": beamformed,
                    "enrollment": enrollment,
                    "truth": truth,
                }
            )

    original_frontend = pipeline.stage1.frontend
    by_variant: dict[str, list[dict[str, Any]]] = {variant: [] for variant in VARIANTS}
    artifact_saved = False
    try:
        for variant in VARIANTS:
            pipeline.stage1.frontend = original_frontend if variant == "tse_first" else None
            for scene in scenes:
                result = _run_variant(
                    pipeline,
                    variant,
                    scene["mixture"],
                    scene["multichannel"],
                    scene["beamformed"],
                    scene["enrollment"],
                    scene["truth"],
                    sample_rate,
                    stage1_seconds,
                    float(args.hop_seconds),
                    device,
                )
                result.update({"scene_index": scene["scene_index"], "snr_db": scene["snr_db"], "truth": scene["truth"]})
                by_variant[variant].append(result)
                print(
                    f"variant={variant} scene={scene['scene_index']} snr={scene['snr_db']:+.1f} "
                    f"wake={result['target_wake_detected']} intent={result['target_intent_correct']} "
                    f"false_wakes={result['false_wake_events']} passenger_wakes={result['passenger_wake_false_events']}",
                    flush=True,
                )
                if not artifact_saved:
                    artifact_dir = resolve_path(args.artifact_dir)
                    save_audio(artifact_dir / "raw_input_mean.wav", scene["mixture"], sample_rate)
                    save_audio(artifact_dir / "raw_input_4ch.wav", scene["multichannel"], sample_rate)
                    save_audio(artifact_dir / "driver_beamformed.wav", scene["beamformed"], sample_rate)
                    save_audio(artifact_dir / "passenger_spatial.wav", scene["passenger_scaled"], sample_rate)
                    save_audio(artifact_dir / "enrollment.wav", scene["enrollment"], sample_rate)
                    artifact_saved = True
    finally:
        pipeline.stage1.frontend = original_frontend

    report = {
        "benchmark": "english_spatial_zone_ablation",
        "device": str(device),
        "config": str(resolve_path(args.config)),
        "sample_rate": sample_rate,
        "stage1_context_seconds": stage1_seconds,
        "hop_seconds": float(args.hop_seconds),
        "passenger_snr_db": float(args.passenger_snr_db),
        "enrollment_mode": args.enrollment_mode,
        "cavemove_car": args.car,
        "spatial_calibration": {
            "driver_ir": calibration["driver_ir_path"],
            "passenger_ir": calibration["passenger_ir_path"],
            "driver_peak_samples": calibration["driver_peak_samples"],
            "driver_delays_samples": calibration["driver_delays_samples"],
        },
        "english_only": True,
        "speaker_consistent": False,
        "same_driver_wake_to_command": False,
        "production_valid": False,
        "limitation": (
            "English cross-corpus spatial proxy: CAVEMOVE supplies vehicle impulse responses and noise, "
            "but Speech Commands supplies the Marvin wake/enrollment while Fluent Speech Commands supplies "
            "the command from an unrelated speaker. The speaker gate can therefore reject the command even "
            "when wake detection is correct; this is not a same-driver enrollment/wake/command recording. "
            "The integer delay calibration is offline and must not be treated as a production microphone "
            "geometry estimate."
        ),
        "summary": {variant: _summarize(rows) for variant, rows in by_variant.items()},
        "scenes": {variant: rows for variant, rows in by_variant.items()},
        "artifacts": {
            "directory": str(resolve_path(args.artifact_dir)),
            "files": [
                "raw_input_mean.wav",
                "raw_input_4ch.wav",
                "driver_beamformed.wav",
                "passenger_spatial.wav",
                "enrollment.wav",
            ],
        },
    }
    output = resolve_path(args.output)
    write_metrics(output, report)
    print(json.dumps({"output": str(output), "summary": report["summary"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
