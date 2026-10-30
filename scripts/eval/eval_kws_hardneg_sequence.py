#!/usr/bin/env python
"""Evaluate temporal KWS logic on a sequential held-out TSE hard-negative proxy.

The cache contains independent one-second, target-absent speech examples.  We
run them in manifest order as a deterministic sequence to measure the effect
of temporal confirmation.  This is useful for calibration, but it is not a
replacement for a real continuous recording because clip boundaries are
artificial.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.io.manifest import read_jsonl
from car_tse.models.kws_model import KWSModel
from car_tse.pipeline.in_car_pipeline import PipelineState, update_kws_state
from car_tse.train.utils import get_device, load_checkpoint, write_metrics


def count_events(flags: np.ndarray, merge_gap_steps: int) -> int:
    count = 0
    last = -10**9
    for index, flag in enumerate(flags.tolist()):
        if flag and index - last > merge_gap_steps:
            count += 1
            last = index
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "stage3_kws_tse_hybrid.yaml")
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "stage3_kws_tse_hybrid_ft1" / "best.pt")
    parser.add_argument("--cache", type=Path, default=REPO_ROOT / "data" / "cache" / "stage3_kws_tse_hardneg_16k" / "test_cabin_wav_16000hz.npy")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "manifests" / "test" / "stage3_kws_tse_hardneg_test.jsonl")
    parser.add_argument("--threshold", type=float, default=0.29)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--required-hits", type=int, default=2)
    parser.add_argument("--refractory-calls", type=int, default=2)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "logs" / "kws_tse_hardneg_sequence_proxy.json")
    args = parser.parse_args()

    cfg = load_stage_config(args.config, args.paths)
    device = get_device()
    model = KWSModel(
        sample_rate=int(cfg["sample_rate"]),
        n_mels=int(cfg["model"]["n_mels"]),
        channels=list(cfg["model"]["channels"]),
        num_classes=int(cfg["model"]["num_classes"]),
    ).to(device)
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else REPO_ROOT / args.checkpoint
    load_checkpoint(checkpoint, model)
    model.eval()

    cache = np.load(args.cache if args.cache.is_absolute() else REPO_ROOT / args.cache, mmap_mode="r", allow_pickle=False)
    rows = read_jsonl(args.manifest if args.manifest.is_absolute() else REPO_ROOT / args.manifest)
    if len(cache) != len(rows):
        raise ValueError(f"Cache/manifest mismatch: {len(cache)} vs {len(rows)}")

    scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(cache), max(1, int(args.batch_size))):
            wav = torch.from_numpy(np.array(cache[start : start + args.batch_size], dtype=np.float32, copy=True)).to(device)
            scores.extend(torch.softmax(model(wav), dim=-1)[:, 1].cpu().tolist())
    score_array = np.asarray(scores, dtype=np.float32)
    raw_flags = score_array >= float(args.threshold)

    state = PipelineState()
    smoothed_flags = np.zeros(len(score_array), dtype=bool)
    for index, score in enumerate(score_array.tolist()):
        smoothed_flags[index] = update_kws_state(
            state,
            score,
            float(args.threshold),
            int(args.window_size),
            int(args.required_hits),
            int(args.refractory_calls),
        )

    duration_seconds = len(score_array) * 1.0
    raw_events = count_events(raw_flags, merge_gap_steps=2)
    result = {
        "benchmark": "sequential_tse_hard_negative_proxy",
        "device": str(device),
        "checkpoint": str(checkpoint),
        "cache": str(args.cache),
        "manifest": str(args.manifest),
        "examples": int(len(score_array)),
        "duration_seconds_assumed": duration_seconds,
        "threshold": float(args.threshold),
        "raw_positive_windows": int(raw_flags.sum()),
        "raw_positive_window_rate_per_hour": float(raw_flags.sum()) * 3600.0 / duration_seconds,
        "raw_events_with_2s_merge": raw_events,
        "raw_event_rate_per_hour": raw_events * 3600.0 / duration_seconds,
        "smoothing": {
            "window": int(args.window_size),
            "required_hits": int(args.required_hits),
            "refractory_calls": int(args.refractory_calls),
        },
        "smoothed_events": int(smoothed_flags.sum()),
        "smoothed_event_rate_per_hour": float(smoothed_flags.sum()) * 3600.0 / duration_seconds,
        "max_score": float(score_array.max(initial=0.0)),
        "mean_score": float(score_array.mean()),
        "note": (
            "This sequence is formed by concatenating independent one-second "
            "target-absent TSE hard-negative examples in manifest order. It is "
            "a temporal-calibration proxy, not a true continuous acoustic stream."
        ),
    }
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    write_metrics(output, result)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
