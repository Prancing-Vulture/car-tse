#!/usr/bin/env python
"""Evaluate a weighted score fusion of two binary KWS checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.datasets import KWSDataset
from car_tse.models.kws_model import KWSModel
from car_tse.train.utils import get_device, load_checkpoint, write_metrics


def load_scores(
    cfg: dict,
    manifest: Path,
    checkpoint: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    paths = cfg["paths"]
    cache_dir = Path(cfg.get("waveform_cache_dir", REPO_ROOT / "data" / "cache" / "stage3_kws_16k"))
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    cache_stem = "cabin_wav" if cfg.get("waveform_cache_includes_cabin_noise", False) else "wav"
    noise_cache_dir = Path(cfg.get("cabin_noise_cache_dir", "")) if cfg.get("cabin_noise_cache_dir") else None
    if noise_cache_dir is not None and not noise_cache_dir.is_absolute():
        noise_cache_dir = REPO_ROOT / noise_cache_dir
    ds = KWSDataset(
        manifest,
        cfg["sample_rate"],
        cfg["segment_seconds"],
        training=False,
        cabin_noise=bool(cfg.get("cabin_noise", False)),
        cabin_noise_snr_db=cfg.get("cabin_noise_snr_db"),
        cabin_noise_cache_size=int(cfg.get("cabin_noise_cache_size", 32)),
        audio_cache_size=int(cfg.get("audio_cache_size", 2048)),
        cabin_noise_cache_dir=noise_cache_dir,
        waveform_cache_path=cache_dir / f"test_{cache_stem}_{cfg['sample_rate']}hz.npy",
        waveform_cache_is_noisy=bool(cfg.get("waveform_cache_includes_cabin_noise", False)),
    )
    workers = int(cfg["train"].get("num_workers", 0))
    loader = DataLoader(
        ds,
        batch_size=cfg["train"].get("val_batch_size", 512),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )
    model = KWSModel(
        cfg["sample_rate"],
        cfg["model"]["n_mels"],
        cfg["model"]["channels"],
        cfg["model"]["num_classes"],
    ).to(device)
    load_checkpoint(checkpoint, model)
    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["wav"].to(device, non_blocking=True))
            scores.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
            labels.extend(batch["label"].tolist())
    return np.asarray(scores, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def metric(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = scores >= threshold
    tp = int(np.sum(pred & (labels == 1)))
    fp = int(np.sum(pred & (labels == 0)))
    fn = int(np.sum(~pred & (labels == 1)))
    tn = int(np.sum(~pred & (labels == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1.0e-8)
    negatives = max(int(np.sum(labels == 0)), 1)
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "false_accepts_per_hour_if_1s_windows": fp * 3600.0 / negatives,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--alpha-step", type=float, default=0.1)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    args = parser.parse_args()
    cfg = load_stage_config(args.config, args.paths)
    manifest = args.manifest if args.manifest.is_absolute() else REPO_ROOT / args.manifest
    checkpoint_a = args.checkpoint_a if args.checkpoint_a.is_absolute() else REPO_ROOT / args.checkpoint_a
    checkpoint_b = args.checkpoint_b if args.checkpoint_b.is_absolute() else REPO_ROOT / args.checkpoint_b
    device = get_device()
    scores_a, labels = load_scores(cfg, manifest, checkpoint_a, device)
    scores_b, labels_b = load_scores(cfg, manifest, checkpoint_b, device)
    if not np.array_equal(labels, labels_b):
        raise ValueError("Checkpoint evaluations produced different label order")
    rows = []
    alpha = 0.0
    while alpha <= 1.00001:
        fused = alpha * scores_a + (1.0 - alpha) * scores_b
        thresholds = np.arange(args.threshold_step, 1.0, args.threshold_step)
        sweep = [metric(fused, labels, float(t)) for t in thresholds]
        best_f1 = max(sweep, key=lambda row: row["f1"])
        eligible = [row for row in sweep if row["recall"] >= 0.90]
        best_precision_at_recall90 = max(eligible, key=lambda row: row["precision"]) if eligible else None
        rows.append({
            "alpha_checkpoint_a": float(round(alpha, 6)),
            "best_f1": best_f1,
            "best_precision_at_recall90": best_precision_at_recall90,
        })
        alpha += args.alpha_step
    result = {
        "examples": int(len(labels)),
        "checkpoint_a": str(checkpoint_a),
        "checkpoint_b": str(checkpoint_b),
        "rows": rows,
    }
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    write_metrics(output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
