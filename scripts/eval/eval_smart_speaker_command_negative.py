#!/usr/bin/env python
"""Evaluate the fixed in-car cascade on continuous non-Marvin commands.

The University of Rochester Smart Speaker Command Dataset contains clips from
seven speakers scripted as ``Hey Alexa, <command>``.  Every clip is therefore
negative for the current ``Marvin`` wake word, while still exercising
speaker-continuous wake-to-command phrasing.  This evaluator is intentionally
diagnostic: the recordings are clean, single-channel studio audio and are
duplicated across the configured microphone channels for interface
compatibility.  It must not be reported as real-cabin or multi-microphone
evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.segment import segment_waveform
from car_tse.io.audio import read_mono
from car_tse.pipeline.in_car_pipeline import PipelineState
from car_tse.pipeline.windowing import full_context_window_starts
from car_tse.train.utils import get_device, write_metrics
from eval_continuous_full_pipeline_proxy import load_pipeline, resolve_path
from eval_aishell5_continuous_kws import evaluate_stream


def evaluate_file(
    pipeline: Any,
    path: Path,
    enrollment: np.ndarray,
    *,
    sample_rate: int,
    num_mics: int,
    stage1_seconds: float,
    hop_seconds: float,
    device: torch.device,
) -> dict[str, Any]:
    wav, source_rate = read_mono(path, sample_rate)
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    multichannel = np.repeat(wav[:, None], num_mics, axis=1)
    window_samples = max(1, int(round(stage1_seconds * sample_rate)))
    hop_samples = max(1, int(round(hop_seconds * sample_rate)))
    enrollment_tensor = torch.from_numpy(enrollment).to(device).unsqueeze(0)
    state = PipelineState()
    wake_events = 0
    action_events = 0
    max_wake_probability = 0.0
    max_presence_probability = 0.0
    windows = 0

    with torch.inference_mode():
        for start in full_context_window_starts(len(wav), window_samples, hop_samples):
            end = min(start + window_samples, len(wav))
            mix_window = np.zeros(window_samples, dtype=np.float32)
            mc_window = np.zeros((num_mics, window_samples), dtype=np.float32)
            if end > start:
                mix_window[: end - start] = wav[start:end]
                mc_window[:, : end - start] = multichannel[start:end].T
            output, state = pipeline.forward(
                torch.from_numpy(mix_window).to(device).unsqueeze(0),
                enrollment_tensor,
                state,
                multichannel=torch.from_numpy(mc_window).to(device).unsqueeze(0),
            )
            windows += 1
            if output.wake_detected:
                wake_events += 1
            if output.action is not None:
                action_events += 1
            if output.wake_probability is not None:
                max_wake_probability = max(max_wake_probability, float(output.wake_probability))
            if output.presence_probability is not None:
                max_presence_probability = max(max_presence_probability, float(output.presence_probability))

    duration_seconds = len(wav) / float(sample_rate)
    return {
        "file": str(path),
        "speaker": path.parent.name,
        "source_sample_rate_hz": int(source_rate),
        "duration_seconds": duration_seconds,
        "windows": windows,
        "false_wake_events": wake_events,
        "false_action_events": action_events,
        "max_wake_probability": max_wake_probability,
        "max_presence_probability": max_presence_probability,
    }


def evaluate_file_batched(
    pipeline: Any,
    path: Path,
    enrollment: np.ndarray,
    *,
    sample_rate: int,
    num_mics: int,
    stage1_seconds: float,
    hop_seconds: float,
    batch_size: int,
    device: torch.device,
    kws_thresholds: list[float] | None = None,
    speaker_gate_thresholds: list[float] | None = None,
) -> dict[str, Any]:
    """Run the wake/presence portion in batches for fast corpus sweeps."""

    wav, source_rate = read_mono(path, sample_rate)
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    multichannel = np.repeat(wav[:, None], num_mics, axis=1)
    metrics = evaluate_stream(
        pipeline.stage1,
        pipeline.presence,
        pipeline.kws,
        pipeline.target_kws,
        pipeline.target_kws_secondary,
        device,
        multichannel,
        enrollment,
        stage1_seconds,
        hop_seconds,
        max(1, int(batch_size)),
        float(pipeline.presence_threshold),
        float(pipeline.kws_threshold),
        pipeline.speaker_gate_threshold,
        int(pipeline.kws_smoothing_window),
        int(pipeline.kws_required_hits),
        int(pipeline.kws_refractory_calls),
        float(pipeline.kws_input_seconds),
        float(pipeline.presence_top_fraction),
        kws_thresholds=kws_thresholds,
        speaker_gate_thresholds=speaker_gate_thresholds,
        speaker_fusion_mode=str(getattr(pipeline, "speaker_fusion_mode", "none")),
        speaker_fusion_center=float(getattr(pipeline, "speaker_fusion_center", 0.0)),
        speaker_fusion_scale=float(getattr(pipeline, "speaker_fusion_scale", 0.10)),
        speaker_fusion_alpha=float(getattr(pipeline, "speaker_fusion_alpha", 0.5)),
        # The batched scorer has its own speaker-score path.  Pass the compact
        # verifier explicitly when that backend is selected; otherwise a
        # compact-configured run would silently fall back to the SpEx cosine
        # scorer while still reporting the compact threshold in its metadata.
        compact_speaker_verifier=(
            getattr(pipeline, "compact_speaker_verifier", None)
            if str(getattr(pipeline, "speaker_gate_backend", "spex")).lower() == "compact"
            else None
        ),
    )
    result = {
        "file": str(path),
        "speaker": path.parent.name,
        "source_sample_rate_hz": int(source_rate),
        "duration_seconds": len(wav) / float(sample_rate),
        "windows": int(metrics["windows"]),
        "false_wake_events": int(metrics["full_path_smoothed_events"]),
        "false_wake_positive_windows": int(metrics.get("full_path_smoothed_positive_windows", 0)),
        "max_wake_probability": float(metrics["max_kws_score"]),
        "max_presence_probability": float(metrics["max_presence_score"]),
        "kws_only_smoothed_events": int(metrics["kws_only_smoothed_events"]),
        "kws_only_positive_windows": int(metrics.get("kws_only_smoothed_positive_windows", 0)),
    }
    if kws_thresholds:
        result["threshold_sweep"] = list(metrics.get("threshold_sweep", []))
    if speaker_gate_thresholds:
        result["speaker_gate_threshold_sweep"] = list(
            metrics.get("speaker_gate_threshold_sweep", [])
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"D:\datasets\smart_speaker_command_26417548\SmartSpeakerCommandDataset_RutowskiEtAl_2023"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "configs"
        / "eval_end_to_end_cavemove_target_kws_dual_domain_stage2_hardneg_aishell5_neg_safety_splitgate_candidate.yaml",
    )
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument(
        "--enrollment-wav",
        type=Path,
        default=REPO_ROOT
        / "data"
        / "raw"
        / "librispeech"
        / "extracted"
        / "LibriSpeech"
        / "test-clean"
        / "1089"
        / "134686"
        / "1089-134686-0014.flac",
    )
    parser.add_argument("--max-files", type=int, default=0, help="0 evaluates every WAV file")
    parser.add_argument("--hop-seconds", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--kws-threshold", type=float, default=None)
    parser.add_argument(
        "--kws-thresholds",
        nargs="+",
        type=float,
        default=None,
        help="Replay these KWS thresholds from one shared forward pass",
    )
    parser.add_argument(
        "--speaker-gate-thresholds",
        nargs="+",
        type=float,
        default=None,
        help="Replay these speaker-gate thresholds from one shared forward pass",
    )
    parser.add_argument("--speaker-gate-threshold", type=float, default=None)
    parser.add_argument(
        "--speaker-gate-backend",
        choices=("spex", "wav2vec", "compact", "auto"),
        default=None,
    )
    parser.add_argument("--speaker-fusion-mode", choices=("none", "min", "product", "mean"), default=None)
    parser.add_argument("--speaker-fusion-center", type=float, default=None)
    parser.add_argument("--speaker-fusion-scale", type=float, default=None)
    parser.add_argument("--speaker-fusion-alpha", type=float, default=None)
    parser.add_argument(
        "--action-sample-files",
        type=int,
        default=14,
        help="Run the exact sequential full pipeline on this many files for action diagnostics; 0 disables it",
    )
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "logs" / "smart_speaker_command_negative_2026-10-11.json")
    args = parser.parse_args()

    root = args.root.resolve()
    files = sorted(root.glob("Speaker*/*.wav"))
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No Speaker*/.wav files found under {root}")

    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    device = get_device()
    pipeline = load_pipeline(cfg, paths, device)
    if args.kws_threshold is not None:
        pipeline.kws_threshold = float(args.kws_threshold)
    if args.speaker_gate_threshold is not None:
        pipeline.speaker_gate_threshold = float(args.speaker_gate_threshold)
        pipeline.command_speaker_gate_threshold = float(args.speaker_gate_threshold)
    if args.speaker_gate_backend is not None:
        pipeline.speaker_gate_backend = str(args.speaker_gate_backend)
    if args.speaker_fusion_mode is not None:
        pipeline.speaker_fusion_mode = str(args.speaker_fusion_mode)
    if args.speaker_fusion_center is not None:
        pipeline.speaker_fusion_center = float(args.speaker_fusion_center)
    if args.speaker_fusion_scale is not None:
        pipeline.speaker_fusion_scale = max(float(args.speaker_fusion_scale), 1.0e-4)
    if args.speaker_fusion_alpha is not None:
        pipeline.speaker_fusion_alpha = min(max(float(args.speaker_fusion_alpha), 0.0), 1.0)
    num_mics = int(cfg.get("num_mics", 4))
    stage1_cfg = load_stage_config(resolve_path(cfg["stage1_config"]), args.paths)
    enrollment, _ = read_mono(resolve_path(args.enrollment_wav), 16000)
    enrollment = segment_waveform(enrollment, 3 * 16000, training=False)

    results: list[dict[str, Any]] = []
    for index, path in enumerate(files, start=1):
        row = evaluate_file_batched(
            pipeline,
            path,
            enrollment,
            sample_rate=16000,
            num_mics=num_mics,
            stage1_seconds=float(stage1_cfg.get("segment_seconds", 3.0)),
            hop_seconds=float(args.hop_seconds),
            batch_size=int(args.batch_size),
            device=device,
            kws_thresholds=[float(value) for value in args.kws_thresholds]
            if args.kws_thresholds
            else None,
            speaker_gate_thresholds=[float(value) for value in args.speaker_gate_thresholds]
            if args.speaker_gate_thresholds
            else None,
        )
        results.append(row)
        if index == 1 or index % 50 == 0 or index == len(files):
            print(
                f"{index}/{len(files)} files, false_wakes={sum(item['false_wake_events'] for item in results)}, "
                f"kws_only_wakes={sum(item['kws_only_smoothed_events'] for item in results)}",
                flush=True,
            )

    duration_seconds = sum(float(row["duration_seconds"]) for row in results)
    duration_hours = duration_seconds / 3600.0
    false_wakes = sum(int(row["false_wake_events"]) for row in results)
    false_wake_positive_windows = sum(int(row.get("false_wake_positive_windows", 0)) for row in results)
    kws_only_wakes = sum(int(row["kws_only_smoothed_events"]) for row in results)
    kws_only_positive_windows = sum(int(row.get("kws_only_positive_windows", 0)) for row in results)

    threshold_sweep = []
    if args.kws_thresholds:
        for index, threshold in enumerate(args.kws_thresholds):
            entries = [
                row["threshold_sweep"][index]
                for row in results
                if len(row.get("threshold_sweep", [])) > index
            ]
            full_events = sum(int(entry["full_path_smoothed_events"]) for entry in entries)
            full_positive_windows = sum(
                int(entry["full_path_smoothed_positive_windows"]) for entry in entries
            )
            kws_events = sum(int(entry["kws_only_smoothed_events"]) for entry in entries)
            kws_positive_windows = sum(
                int(entry["kws_only_smoothed_positive_windows"]) for entry in entries
            )
            threshold_sweep.append(
                {
                    "kws_threshold": float(threshold),
                    "full_path_smoothed_events": full_events,
                    "false_wakes_per_hour": full_events / max(duration_hours, 1.0e-8),
                    "full_path_smoothed_positive_windows": full_positive_windows,
                    "kws_only_smoothed_events": kws_events,
                    "kws_only_false_wakes_per_hour": kws_events / max(duration_hours, 1.0e-8),
                    "kws_only_smoothed_positive_windows": kws_positive_windows,
                }
            )

    speaker_gate_threshold_sweep = []
    if args.speaker_gate_thresholds:
        for index, threshold in enumerate(args.speaker_gate_thresholds):
            entries = [
                row["speaker_gate_threshold_sweep"][index]
                for row in results
                if len(row.get("speaker_gate_threshold_sweep", [])) > index
            ]
            full_events = sum(int(entry["full_path_smoothed_events"]) for entry in entries)
            full_positive_windows = sum(
                int(entry["full_path_smoothed_positive_windows"])
                for entry in entries
            )
            speaker_gate_threshold_sweep.append(
                {
                    "speaker_gate_threshold": float(threshold),
                    "full_path_smoothed_events": full_events,
                    "false_wakes_per_hour": full_events / max(duration_hours, 1.0e-8),
                    "full_path_smoothed_positive_windows": full_positive_windows,
                }
            )

    action_sample_files = []
    if int(args.action_sample_files) > 0:
        # Use one or two files per speaker before filling the remainder so the
        # sequential action check is not accidentally dominated by Speaker1.
        by_speaker: dict[str, list[Path]] = {}
        for path in files:
            by_speaker.setdefault(path.parent.name, []).append(path)
        ordered_action_files: list[Path] = []
        for paths_for_speaker in by_speaker.values():
            ordered_action_files.extend(paths_for_speaker[:2])
        ordered_action_files.extend(path for path in files if path not in ordered_action_files)
        ordered_action_files = ordered_action_files[: int(args.action_sample_files)]
        for path in ordered_action_files:
            action_sample_files.append(
                evaluate_file(
                    pipeline,
                    path,
                    enrollment,
                    sample_rate=16000,
                    num_mics=num_mics,
                    stage1_seconds=float(stage1_cfg.get("segment_seconds", 3.0)),
                    hop_seconds=float(args.hop_seconds),
                    device=device,
                )
            )
    action_duration_seconds = sum(float(row["duration_seconds"]) for row in action_sample_files)
    action_false_wakes = sum(int(row["false_wake_events"]) for row in action_sample_files)
    action_false_actions = sum(int(row["false_action_events"]) for row in action_sample_files)
    action_duration_hours = action_duration_seconds / 3600.0
    report = {
        "benchmark": "smart_speaker_command_nonmarvin_negative",
        "dataset": "University of Rochester Smart Speaker Command Dataset",
        "dataset_source": "https://doi.org/10.60593/ur.d.26417548",
        "dataset_license": "CC BY 4.0",
        "dataset_contract": {
            "speakers": sorted({row["speaker"] for row in results}),
            "files": len(results),
            "wake_phrase": "Hey Alexa",
            "current_wake_phrase": "Marvin",
            "single_channel_duplicated_to_num_mics": num_mics,
            "clean_studio_audio": True,
        },
        "device": str(device),
        "config": str(resolve_path(args.config)),
        "enrollment_wav": str(resolve_path(args.enrollment_wav)),
        "hop_seconds": float(args.hop_seconds),
        "batch_size": int(args.batch_size),
        "kws_threshold": float(pipeline.kws_threshold),
        "speaker_gate_threshold": pipeline.speaker_gate_threshold,
        "speaker_gate_backend": pipeline.speaker_gate_backend,
        "compact_verifier_passed_to_batched_scorer": bool(
            str(getattr(pipeline, "speaker_gate_backend", "spex")).lower() == "compact"
            and getattr(pipeline, "compact_speaker_verifier", None) is not None
        ),
        "speaker_fusion_mode": pipeline.speaker_fusion_mode,
        "speaker_fusion_center": pipeline.speaker_fusion_center,
        "speaker_fusion_scale": pipeline.speaker_fusion_scale,
        "speaker_fusion_alpha": pipeline.speaker_fusion_alpha,
        "duration_seconds": duration_seconds,
        "duration_hours": duration_hours,
        "false_wake_events": false_wakes,
        "false_wakes_per_hour": false_wakes / max(duration_hours, 1.0e-8),
        "false_wake_positive_windows": false_wake_positive_windows,
        "kws_only_smoothed_events": kws_only_wakes,
        "kws_only_false_wakes_per_hour": kws_only_wakes / max(duration_hours, 1.0e-8),
        "kws_only_positive_windows": kws_only_positive_windows,
        "threshold_sweep": threshold_sweep,
        "speaker_gate_threshold_sweep": speaker_gate_threshold_sweep,
        "event_accounting": {
            "smoothed_events": "near-contiguous wake runs merged with a 2-second gap",
            "positive_windows": "raw count of positive smoothed windows, retained as a diagnostic",
        },
        "action_sample": {
            "files": len(action_sample_files),
            "duration_seconds": action_duration_seconds,
            "false_wake_events": action_false_wakes,
            "false_wakes_per_hour": action_false_wakes / max(action_duration_hours, 1.0e-8),
            "false_action_events": action_false_actions,
            "false_actions_per_hour": action_false_actions / max(action_duration_hours, 1.0e-8),
            "results": action_sample_files,
        },
        "files": results,
        "limitation": (
            "This is a clean single-channel, non-Marvin continuous-speech diagnostic. "
            "It is not real-driver, real-cabin, multi-microphone, or positive wake-word evidence."
        ),
    }
    output = resolve_path(args.output)
    write_metrics(output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "files": len(results),
                "duration_hours": duration_hours,
                "false_wakes_per_hour": report["false_wakes_per_hour"],
                "kws_only_false_wakes_per_hour": report["kws_only_false_wakes_per_hour"],
                "action_sample_files": report["action_sample"]["files"],
                "action_sample_false_actions_per_hour": report["action_sample"]["false_actions_per_hour"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
