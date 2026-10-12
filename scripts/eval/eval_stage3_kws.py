#!/usr/bin/env python
"""Evaluate binary wake-word detection on the held-out cabin test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

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
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint override")
    parser.add_argument("--output", type=Path, default=None, help="Optional metrics output override")
    args = parser.parse_args()
    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    device = get_device()
    manifest_name = cfg["manifest_val"] if args.split == "val" else cfg["manifest_eval"]
    manifest = args.manifest or (Path(paths["manifest_root"]) / args.split / manifest_name)
    if not manifest.is_absolute():
        manifest = Path(paths["manifest_root"]) / args.split / manifest
    cache_dir = Path(cfg.get("waveform_cache_dir", REPO_ROOT / "data" / "cache" / "stage3_kws_16k"))
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    cache_stem = "cabin_wav" if cfg.get("waveform_cache_includes_cabin_noise", False) else "wav"
    ds = KWSDataset(
        manifest,
        cfg["sample_rate"],
        cfg["segment_seconds"],
        training=False,
        cabin_noise=bool(cfg.get("cabin_noise", False)),
        cabin_noise_snr_db=cfg.get("cabin_noise_snr_db"),
        cabin_noise_cache_size=int(cfg.get("cabin_noise_cache_size", 32)),
        audio_cache_size=int(cfg.get("audio_cache_size", 2048)),
        waveform_cache_path=cache_dir / f"{args.split}_{cache_stem}_{cfg['sample_rate']}hz.npy",
        waveform_cache_is_noisy=bool(cfg.get("waveform_cache_includes_cabin_noise", False)),
        cabin_noise_cache_dir=(
            (Path(cfg["cabin_noise_cache_dir"]) if Path(cfg["cabin_noise_cache_dir"]).is_absolute() else REPO_ROOT / cfg["cabin_noise_cache_dir"])
            if cfg.get("cabin_noise_cache_dir")
            else None
        ),
    )
    workers = int(cfg["train"].get("num_workers", 0))
    loader = DataLoader(
        ds,
        batch_size=cfg["train"].get("val_batch_size", 256),
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
    correct = total = tp = fp = fn = 0
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["wav"].to(device, non_blocking=True))
            labels = batch["label"].to(device, non_blocking=True)
            predictions = logits.argmax(-1)
            correct += int((predictions == labels).sum())
            total += labels.numel()
            tp += int(((predictions == 1) & (labels == 1)).sum())
            fp += int(((predictions == 1) & (labels == 0)).sum())
            fn += int(((predictions == 0) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    result = {
        "split": args.split,
        "wake_word": cfg.get("wake_word", "marvin"),
        "accuracy": correct / max(total, 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-8),
        "examples": total,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "test_manifest": str(manifest),
    }
    output = args.output or (paths["log_root"] / cfg["name"] / f"metrics_{args.split}.json")
    if not output.is_absolute():
        output = REPO_ROOT / output
    write_metrics(output, result)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
