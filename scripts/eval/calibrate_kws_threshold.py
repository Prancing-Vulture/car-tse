#!/usr/bin/env python
"""Sweep binary KWS thresholds and report cabin false-accept trade-offs."""

from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "stage3_kws_cavemove.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--manifest", type=Path, default=None, help="Optional explicit manifest override")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint override")
    parser.add_argument("--output", type=Path, default=None, help="Optional metrics output override")
    args = parser.parse_args()
    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    device = get_device()
    manifest_name = cfg["manifest_val"] if args.split == "val" else cfg["manifest_eval"]
    manifest = args.manifest or (Path(paths["manifest_root"]) / args.split / manifest_name)
    if not manifest.is_absolute():
        manifest = REPO_ROOT / manifest
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
        waveform_cache_path=cache_dir / f"{args.split}_{cache_stem}_{cfg['sample_rate']}hz.npy",
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
    checkpoint = args.checkpoint or (paths["checkpoint_root"] / cfg["name"] / "best.pt")
    if not checkpoint.is_absolute():
        checkpoint = REPO_ROOT / checkpoint
    load_checkpoint(checkpoint, model)
    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["wav"].to(device, non_blocking=True))
            scores.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
            labels.extend(batch["label"].tolist())
    score_array = np.asarray(scores)
    label_array = np.asarray(labels, dtype=np.int64)
    thresholds = np.arange(args.step, 1.0, args.step)
    rows = []
    for threshold in thresholds:
        pred = score_array >= threshold
        tp = int(np.sum(pred & (label_array == 1)))
        fp = int(np.sum(pred & (label_array == 0)))
        fn = int(np.sum(~pred & (label_array == 1)))
        tn = int(np.sum(~pred & (label_array == 0)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        negative_count = max(int(np.sum(label_array == 0)), 1)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
                "false_accepts_per_hour_if_1s_windows": fp * 3600.0 / negative_count,
            }
        )
    best_f1 = max(rows, key=lambda row: row["f1"])
    recall_targets = {}
    for target in (0.95, 0.90, 0.80):
        eligible = [row for row in rows if row["recall"] >= target]
        recall_targets[str(target)] = max(eligible, key=lambda row: row["precision"]) if eligible else None
    result = {
        "split": args.split,
        "wake_word": cfg.get("wake_word", "marvin"),
        "examples": len(labels),
        "best_f1": best_f1,
        "best_precision_at_recall_targets": recall_targets,
        "thresholds": rows,
    }
    output = args.output or (paths["log_root"] / cfg["name"] / f"threshold_sweep_{args.split}.json")
    if not output.is_absolute():
        output = REPO_ROOT / output
    write_metrics(output, result)
    print({"split": args.split, "best_f1": best_f1, "recall_targets": recall_targets, "output": str(output)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
