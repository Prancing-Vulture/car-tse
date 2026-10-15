#!/usr/bin/env python
"""Evaluate the wav2vec intent model on TSE, raw-cabin, or clean audio."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT_LOCAL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT_LOCAL / "src"))
sys.path.insert(0, str(REPO_ROOT_LOCAL / "scripts" / "train"))

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.datasets import IntentDataset
from car_tse.train.utils import get_device, load_checkpoint, write_metrics
from train_stage4_intent_wav2vec import IntentHead, normalize_waveform


@torch.no_grad()
def evaluate(model: IntentHead, encoder: torch.nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> dict[str, float]:
    model.eval()
    encoder.eval()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    total = top5 = 0
    for batch in loader:
        wav = normalize_waveform(batch["wav"].to(device, non_blocking=True))
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(encoder(wav).last_hidden_state.mean(dim=1))
        pred = logits.argmax(-1)
        confusion.index_put_((labels.cpu(), pred.cpu()), torch.ones_like(labels.cpu()), accumulate=True)
        total += labels.numel()
        top5 += int((logits.topk(min(5, num_classes), dim=-1).indices == labels[:, None]).any(dim=1).sum())
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional explicit manifest override")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--encoder", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--head-checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "stage4_intent_wav2vec_ft4_tse_domain" / "best.pt")
    parser.add_argument("--encoder-checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "stage4_intent_wav2vec_ft4_tse_domain" / "encoder_best.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    device = get_device()
    label_map = Path(paths["manifest_root"]) / "meta" / cfg["label_map"]
    num_classes = len(json.loads(label_map.read_text(encoding="utf-8")))
    manifest_root = Path(paths["manifest_root"])
    manifest = args.manifest
    if manifest is None:
        if args.split == "train":
            manifest_name = cfg["manifest_train"]
        elif args.split == "val":
            manifest_name = cfg["manifest_eval"]
        else:
            manifest_name = cfg.get("manifest_test")
            if not manifest_name:
                raise ValueError("--split test requires --manifest or a manifest_test entry in the config")
        manifest = manifest_root / args.split / manifest_name
    elif not manifest.is_absolute():
        manifest = REPO_ROOT / manifest
    if not manifest.exists():
        raise FileNotFoundError(f"Intent manifest not found: {manifest}")
    cache_dir = Path(cfg.get("waveform_cache_dir", ""))
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    # Explicit manifests are used for the official clean FSC benchmark and
    # must be read from their WAV paths rather than an unrelated cache.
    use_cache = not args.clean and args.manifest is None
    cache_stem = "cabin_wav" if cfg.get("waveform_cache_includes_cabin_noise", False) else "wav"
    dataset = IntentDataset(
        manifest,
        cfg["sample_rate"],
        cfg["segment_seconds"],
        training=False,
        cabin_noise=False,
        waveform_cache_path=(cache_dir / f"{args.split}_{cache_stem}_{cfg['sample_rate']}hz.npy") if use_cache else None,
        waveform_cache_is_noisy=use_cache,
        audio_cache_size=0,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    encoder = AutoModel.from_pretrained(args.encoder).to(device)
    encoder_state = torch.load(args.encoder_checkpoint, map_location="cpu", weights_only=False)
    encoder.load_state_dict(encoder_state["model_state_dict"], strict=True)
    head = IntentHead(int(encoder.config.hidden_size), 512, num_classes, 0.2).to(device)
    load_checkpoint(args.head_checkpoint, head)
    metrics = evaluate(head, encoder, loader, device, num_classes)
    metrics.update(
        {
            "split": args.split,
            "clean": args.clean,
            "manifest": str(manifest),
            "head_checkpoint": str(args.head_checkpoint),
            "encoder_checkpoint": str(args.encoder_checkpoint),
        }
    )
    output = args.output
    if output is None:
        output = Path(paths["log_root"]) / "stage4_intent_wav2vec_ft4_tse_domain" / f"metrics_{args.split}_{'clean' if args.clean else 'cabin'}.json"
    if not output.is_absolute():
        output = REPO_ROOT / output
    write_metrics(output, metrics)
    print(f"Intent wav2vec {args.split} ({'clean' if args.clean else 'cached'}): acc={metrics['accuracy']:.3f} top5={metrics['top5_accuracy']:.3f} macro_f1={metrics['macro_f1']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
