#!/usr/bin/env python
"""Sweep presence/KWS operating points on the same continuous proxy scenes.

This keeps the expensive Stage 1 pass fixed per scene, then replays the
streaming state machine for each threshold pair. It is an operating-point
ablation, not real-driver evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.io.audio import read_mono, read_multichannel
from car_tse.train.utils import get_device, write_metrics
from eval_continuous_ablation_matrix import (
    VARIANTS,
    collect_window_scores,
    resolve_path,
    speaker_key,
    summarize_variant,
)
from eval_continuous_full_pipeline_proxy import (
    build_scene,
    fit_start,
    load_jsonl,
    load_pipeline,
    select_noise_files,
)
from eval_continuous_kws import prepare_channels


def aggregate(rows: list[dict[str, Any]], duration: float) -> dict[str, Any]:
    wake = [row["target_wake_detected"] for row in rows if row["target_wake_detected"] is not None]
    intents = [bool(row["target_intent_correct"]) for row in rows]
    false_wakes = [row["false_wake_events"] for row in rows if row["false_wake_events"] is not None]
    false_actions = [row["false_action_events"] for row in rows if row["false_action_events"] is not None]
    latencies = [row["target_wake_latency_seconds"] for row in rows if row["target_wake_latency_seconds"] is not None]
    return {
        "scenes": len(rows),
        "target_wake_recall": float(np.mean(wake)) if wake else None,
        "target_intent_accuracy": float(np.mean(intents)) if intents else None,
        "false_wake_events": int(sum(false_wakes)) if false_wakes else None,
        "false_action_events": int(sum(false_actions)) if false_actions else None,
        "false_action_rate_per_hour": (
            float(sum(false_actions) * 3600.0 / max(duration, 1.0e-6)) if false_actions else None
        ),
        "target_wake_latency_mean_seconds": float(np.mean(latencies)) if latencies else None,
        "target_wake_latency_p95_seconds": float(np.percentile(latencies, 95)) if latencies else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/eval_end_to_end_cavemove_target_kws_temporal2_candidate.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.yaml")
    parser.add_argument("--wake-manifest", type=Path, default=REPO_ROOT / "manifests/test/stage3_kws_cabin_test.jsonl")
    parser.add_argument("--intent-manifest", type=Path, default=REPO_ROOT / "manifests/official/stage4_intent_fsc_test.jsonl")
    parser.add_argument("--noise-root", type=Path, default=REPO_ROOT / "data/raw/cavemove/extracted/cavemove_dataset")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "logs/threshold_ablation_sweep.json")
    parser.add_argument("--scenes", type=int, default=2)
    parser.add_argument("--cabin-snr-db", nargs="+", type=float, default=[-5.0, 0.0, 10.0, 20.0])
    parser.add_argument("--passenger-snr-db", type=float, default=0.0)
    parser.add_argument("--presence-thresholds", nargs="+", type=float, default=[0.30, 0.50, 0.70, 0.90])
    parser.add_argument(
        "--command-presence-thresholds",
        nargs="+",
        type=float,
        default=None,
        help="Optional post-wake presence threshold grid; omitted preserves the config value.",
    )
    parser.add_argument("--kws-thresholds", nargs="+", type=float, default=[0.30, 0.40, 0.50, 0.60])
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=["full", "no_speaker_matching", "no_wake_smoothing"])
    parser.add_argument("--hop-seconds", type=float, default=1.0)
    args = parser.parse_args()
    command_presence_thresholds = args.command_presence_thresholds or [None]

    cfg = load_stage_config(args.config, args.paths)
    device = get_device()
    pipeline = load_pipeline(cfg, cfg["paths"], device)
    sample_rate = int(cfg["sample_rate"])
    stage1_seconds = float(cfg.get("segment_seconds", 3.0))

    wake_rows = [
        row for row in load_jsonl(resolve_path(args.wake_manifest))
        if int(row.get("label", 0)) == 1 and not bool(row.get("noise_only", False))
    ]
    wake_rows.sort(key=lambda row: str(row.get("wav", "")))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in wake_rows:
        grouped[speaker_key(resolve_path(row["wav"]))].append(row)
    target_wake_rows = [rows[0] for _, rows in sorted(grouped.items()) if rows][: int(args.scenes)]
    if len(target_wake_rows) < int(args.scenes):
        raise RuntimeError("Not enough Marvin speakers for requested scenes")
    passenger_wake_row = next(
        row for row in wake_rows
        if speaker_key(resolve_path(row["wav"])) != speaker_key(resolve_path(target_wake_rows[0]["wav"]))
    )
    negative_rows = [
        row for row in load_jsonl(resolve_path(args.wake_manifest))
        if int(row.get("label", 0)) == 0 and not bool(row.get("noise_only", False))
    ]
    passenger_nonwake_row = negative_rows[0]
    intent_rows = load_jsonl(resolve_path(args.intent_manifest))
    by_intent: dict[str, dict[str, Any]] = {}
    for row in intent_rows:
        by_intent.setdefault(str(row["intent"]), row)
    intent_rows = [by_intent[key] for key in sorted(by_intent)]
    noise_files = select_noise_files(resolve_path(args.noise_root), 20.0)
    if not noise_files:
        raise RuntimeError("No sufficiently long CAVEMOVE noise streams found")

    cell_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scene_count = 0
    for scene_index in range(min(int(args.scenes), len(target_wake_rows))):
        wake_row = target_wake_rows[scene_index]
        intent_row = intent_rows[scene_index % len(intent_rows)]
        target_wake, _ = read_mono(resolve_path(wake_row["wav"]), sample_rate)
        target_command, _ = read_mono(resolve_path(intent_row["wav"]), sample_rate)
        passenger_wake, _ = read_mono(resolve_path(passenger_wake_row["wav"]), sample_rate)
        passenger_nonwake, _ = read_mono(resolve_path(passenger_nonwake_row["wav"]), sample_rate)
        target_wake = fit_start(target_wake, sample_rate)
        target_command = fit_start(target_command, 2 * sample_rate)
        passenger_wake = fit_start(passenger_wake, sample_rate)
        passenger_nonwake = fit_start(passenger_nonwake, int(1.5 * sample_rate))
        enrollment = fit_start(target_wake, int(round(stage1_seconds * sample_rate / 2.0)))

        for snr_index, cabin_snr_db in enumerate(args.cabin_snr_db):
            noise_path = noise_files[(scene_index + snr_index) % len(noise_files)]
            raw_noise, _ = read_multichannel(noise_path, sample_rate)
            cabin_noise = prepare_channels(
                raw_noise,
                list(cfg.get("cabin_noise_channels", [0, 2, 4, 6])),
                int(cfg.get("num_mics", 4)),
            )
            mixture, multichannel, _, _, truth = build_scene(
                target_wake,
                target_command,
                passenger_wake,
                passenger_nonwake,
                cabin_noise,
                sample_rate,
                float(cabin_snr_db),
                float(args.passenger_snr_db),
            )
            truth.update(
                {
                    "expected_intent": str(intent_row["intent"]),
                    "target_wake_source": str(resolve_path(wake_row["wav"])),
                    "target_command_source": str(resolve_path(intent_row["wav"])),
                    "passenger_wake_source": str(resolve_path(passenger_wake_row["wav"])),
                    "passenger_nonwake_source": str(resolve_path(passenger_nonwake_row["wav"])),
                    "noise_source": str(noise_path),
                    "speaker_consistent": False,
                }
            )
            window_scores = collect_window_scores(
                pipeline,
                mixture,
                multichannel,
                enrollment,
                truth,
                sample_rate,
                stage1_seconds,
                float(args.hop_seconds),
                device,
            )
            scene_count += 1
            for presence_threshold in args.presence_thresholds:
                for kws_threshold in args.kws_thresholds:
                    for command_presence_threshold in command_presence_thresholds:
                        for variant in args.variants:
                            settings = dict(VARIANTS[variant])
                            if command_presence_threshold is not None:
                                settings["command_presence_threshold"] = float(command_presence_threshold)
                            result = summarize_variant(
                                variant,
                                settings,
                                pipeline,
                                window_scores,
                                truth,
                                sample_rate,
                                float(kws_threshold),
                                float(presence_threshold),
                                device,
                            )
                            key = f"presence={float(presence_threshold):.2f}|kws={float(kws_threshold):.2f}|variant={variant}"
                            if command_presence_threshold is not None:
                                key += f"|command_presence={float(command_presence_threshold):.2f}"
                            cell_rows[key].append(result)
            print(f"scene={scene_index} cabin_snr={cabin_snr_db:+.1f}: threshold grid replayed", flush=True)

    duration_seconds = 14.0 * scene_count
    summary = {key: aggregate(rows, duration_seconds) for key, rows in sorted(cell_rows.items())}
    report = {
        "benchmark": "continuous_presence_kws_threshold_ablation_cross_corpus_proxy",
        "device": str(device),
        "config": str(resolve_path(args.config)),
        "checkpoints": cfg.get("checkpoints"),
        "sample_rate": sample_rate,
        "stage1_context_seconds": stage1_seconds,
        "hop_seconds": float(args.hop_seconds),
        "presence_thresholds": [float(value) for value in args.presence_thresholds],
        "command_presence_thresholds": [
            float(value) for value in command_presence_thresholds if value is not None
        ],
        "kws_thresholds": [float(value) for value in args.kws_thresholds],
        "variants": list(args.variants),
        "passenger_snr_db": float(args.passenger_snr_db),
        "scenes": scene_count,
        "duration_seconds": duration_seconds,
        "speaker_consistent": False,
        "production_valid": False,
        "summary": summary,
        "limitation": (
            "Cross-corpus proxy only. Thresholds are replayed on the same generated scenes and are not a "
            "tuning/test result for real drivers."
        ),
    }
    output = resolve_path(args.output)
    write_metrics(output, report)
    print(json.dumps({"output": str(output), "cells": len(summary)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
