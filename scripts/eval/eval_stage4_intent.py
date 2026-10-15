#!/usr/bin/env python
"""Evaluate intent accuracy on clean or CAVEMOVE cabin-noise manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from torch.utils.data import DataLoader

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.datasets import IntentDataset
from car_tse.models.intent_classifier import IntentClassifier
from car_tse.train.utils import get_device, load_checkpoint, write_metrics


@torch.no_grad()
def evaluate(
    model: IntentClassifier,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[str, float]:
    model.eval()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    total = top5 = 0
    for batch in loader:
        wav = batch["wav"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(wav)
        pred = logits.argmax(-1)
        confusion.index_put_((labels.cpu(), pred.cpu()), torch.ones_like(labels.cpu()), accumulate=True)
        total += labels.numel()
        top5 += int(
            (logits.topk(min(5, num_classes), dim=-1).indices == labels[:, None])
            .any(dim=1)
            .sum()
        )
    tp = confusion.diag().float()
    support = confusion.sum(dim=1).float()
    predicted = confusion.sum(dim=0).float()
    precision = tp / predicted.clamp_min(1.0)
    recall = tp / support.clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
    return {
        "accuracy": float(tp.sum().item() / max(total, 1)),
        "top5_accuracy": float(top5 / max(total, 1)),
        "macro_precision": float(precision.mean().item()),
        "macro_recall": float(recall.mean().item()),
        "macro_f1": float(f1.mean().item()),
        "examples": float(total),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "stage4_intent_cavemove.yaml",
    )
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--clean", action="store_true", help="Evaluate without CAVEMOVE mixing")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional explicit manifest override")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint override")
    parser.add_argument("--output", type=Path, default=None, help="Optional metrics output path")
    args = parser.parse_args()

    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    device = get_device()
    label_map_path = Path(paths["manifest_root"]) / "meta" / cfg["label_map"]
    num_classes = (
        len(json.loads(label_map_path.read_text(encoding="utf-8")))
        if label_map_path.exists()
        else 31
    )
    manifest = args.manifest
    if manifest is None:
        manifest = Path(paths["manifest_root"]) / args.split / cfg["manifest_eval"]
    elif not manifest.is_absolute():
        manifest = REPO_ROOT / manifest
    use_cabin = bool(cfg.get("cabin_noise", False)) and not args.clean and args.manifest is None
    cache_dir = Path(cfg.get("waveform_cache_dir", REPO_ROOT / "data" / "cache" / "stage4_intent_16k"))
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    noise_cache_dir = Path(cfg.get("cabin_noise_cache_dir", "")) if cfg.get("cabin_noise_cache_dir") else None
    if noise_cache_dir is not None and not noise_cache_dir.is_absolute():
        noise_cache_dir = REPO_ROOT / noise_cache_dir
    cache_stem = "cabin_wav" if cfg.get("waveform_cache_includes_cabin_noise", False) and not args.clean else "wav"
    ds = IntentDataset(
        manifest,
        cfg["sample_rate"],
        cfg["segment_seconds"],
        training=False,
        cabin_noise=use_cabin,
        cabin_noise_snr_db=cfg.get("cabin_noise_snr_db"),
        cabin_noise_cache_size=int(cfg.get("cabin_noise_cache_size", 32)),
        audio_cache_size=int(cfg.get("audio_cache_size", 0)),
        waveform_cache_path=(cache_dir / f"{args.split}_{cache_stem}_{cfg['sample_rate']}hz.npy") if not args.clean and args.manifest is None else None,
        cabin_noise_cache_dir=noise_cache_dir,
        waveform_cache_is_noisy=bool(cfg.get("waveform_cache_includes_cabin_noise", False)) and not args.clean,
    )
    workers = int(cfg["train"].get("num_workers", 0))
    loader_args = {
        "batch_size": int(cfg["train"].get("val_batch_size", cfg["train"]["batch_size"])),
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        loader_args["prefetch_factor"] = int(cfg["train"].get("prefetch_factor", 2))
    loader = DataLoader(ds, **loader_args)
    model = IntentClassifier(
        cfg["sample_rate"],
        cfg["model"]["n_mels"],
        cfg["model"]["hidden"],
        cfg["model"]["num_layers"],
        num_classes,
        bool(cfg["model"].get("normalize_input", False)),
        bidirectional=bool(cfg["model"].get("bidirectional", True)),
    ).to(device)
    checkpoint = args.checkpoint or (Path(paths["checkpoint_root"]) / cfg["name"] / "best.pt")
    if not checkpoint.is_absolute():
        checkpoint = REPO_ROOT / checkpoint
    load_checkpoint(checkpoint, model)
    metrics = evaluate(model, loader, device, num_classes)
    metrics.update({"split": args.split, "clean": args.clean, "checkpoint": str(checkpoint)})
    suffix = f"metrics_{args.split}_{'clean' if args.clean else 'cabin'}.json"
    output_path = args.output or (Path(paths["log_root"]) / cfg["name"] / suffix)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    write_metrics(output_path, metrics)
    print(
        f"Intent {args.split} ({'clean' if args.clean else 'cabin'}): "
        f"acc={metrics['accuracy']:.3f} top5={metrics['top5_accuracy']:.3f} "
        f"macro_f1={metrics['macro_f1']:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
