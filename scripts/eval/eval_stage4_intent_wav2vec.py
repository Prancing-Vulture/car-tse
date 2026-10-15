#!/usr/bin/env python
"""Evaluate the wav2vec intent model on TSE, raw-cabin, or clean audio."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT_LOCAL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT_LOCAL / "src"))
sys.path.insert(0, str(REPO_ROOT_LOCAL / "scripts" / "train"))

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.datasets import IntentDataset
from car_tse.train.utils import get_device, load_checkpoint, write_metrics
from car_tse.train.wandb_logging import log_evaluation
from train_stage4_intent_wav2vec import IntentHead, normalize_waveform


@torch.no_grad()
def evaluate(model: IntentHead, encoder: torch.nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> dict[str, Any]:
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
        "confusion_matrix": confusion.tolist(),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
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
    parser.add_argument("--wandb", action="store_true", help="Log metrics to W&B (offline when no API key is set)")
    parser.add_argument("--wandb-project", default="car-tse")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default=None)
    parser.add_argument("--wandb-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    device = get_device()
    label_map = Path(paths["manifest_root"]) / "meta" / cfg["label_map"]
    raw_label_map = json.loads(label_map.read_text(encoding="utf-8"))
    id_to_label = {int(value): str(key) for key, value in raw_label_map.items()}
    num_classes = len(raw_label_map)
    manifest_root = Path(paths["manifest_root"])
    manifest = args.manifest
    if manifest is None:
        manifest = manifest_root / args.split / (cfg["manifest_train"] if args.split == "train" else cfg["manifest_eval"])
    elif not manifest.is_absolute():
        manifest = REPO_ROOT / manifest
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
    metrics.update({
        "split": args.split,
        "clean": args.clean,
        "head_checkpoint": str(args.head_checkpoint),
        "encoder_checkpoint": str(args.encoder_checkpoint),
        "manifest": str(manifest),
        "class_names": [id_to_label.get(index, str(index)) for index in range(num_classes)],
    })
    output = args.output
    if output is None:
        output = Path(paths["log_root"]) / "stage4_intent_wav2vec_ft4_tse_domain" / f"metrics_{args.split}_{'clean' if args.clean else 'cabin'}.json"
    if not output.is_absolute():
        output = REPO_ROOT / output
    write_metrics(output, metrics)
    print(f"Intent wav2vec {args.split} ({'clean' if args.clean else 'cached'}): acc={metrics['accuracy']:.3f} top5={metrics['top5_accuracy']:.3f} macro_f1={metrics['macro_f1']:.3f}", flush=True)
    if args.wandb:
        report_scalars = {key: value for key, value in metrics.items() if key not in {"confusion_matrix", "per_class_precision", "per_class_recall", "per_class_f1"}}
        report_scalars.update({
            f"class_precision/{id_to_label.get(index, str(index))}": value
            for index, value in enumerate(metrics["per_class_precision"])
        })
        report_scalars.update({
            f"class_recall/{id_to_label.get(index, str(index))}": value
            for index, value in enumerate(metrics["per_class_recall"])
        })
        log_evaluation(
            project=args.wandb_project,
            run_name=args.wandb_run_name or f"stage4-{args.split}-intent-wav2vec",
            job_type="intent-evaluation",
            config={
                "split": args.split,
                "clean": args.clean,
                "manifest": str(manifest),
                "encoder": args.encoder,
                "head_checkpoint": str(args.head_checkpoint),
                "encoder_checkpoint": str(args.encoder_checkpoint),
                "batch_size": args.batch_size,
                "device": str(device),
                "sample_rate": int(cfg["sample_rate"]),
                "segment_seconds": float(cfg["segment_seconds"]),
                "num_classes": int(num_classes),
                "label_map": str(label_map),
            },
            metrics=report_scalars,
            output_path=output,
            mode=args.wandb_mode,
            directory=args.wandb_dir,
            confusion_matrix=metrics["confusion_matrix"],
            class_names=[id_to_label.get(index, str(index)) for index in range(num_classes)],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
