#!/usr/bin/env python
"""Fast cached-feature Speech Commands benchmark for the Hussein comparison.

The encoder is run once over the official Speech Commands train/validation/
test files, with one pooled WavLM embedding per recording.  A small head is
then trained on the cached embeddings.  This provides a fast, auditable
transfer-learning baseline before attempting expensive encoder fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "train"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

from car_tse.train.utils import get_device, write_metrics
from train_speech_commands_wavlm import Collate, SpeechCommandsDataset, build_rows, metrics_from_confusion


class FeatureDataset(Dataset):
    def __init__(self, path: Path, rows: list[dict[str, Any]]):
        self.features = np.load(path, mmap_mode="r", allow_pickle=False)
        self.labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
        if self.features.shape[0] != self.labels.shape[0]:
            raise ValueError(f"Feature/row mismatch: {self.features.shape[0]} vs {self.labels.shape[0]}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(np.asarray(self.features[index], dtype=np.float32)),
            torch.tensor(int(self.labels[index]), dtype=torch.long),
        )


class Head(nn.Module):
    def __init__(self, hidden: int, classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


@torch.inference_mode()
def cache_features(
    encoder: nn.Module,
    rows: list[dict[str, Any]],
    output_path: Path,
    sample_rate: int,
    max_seconds: float,
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> None:
    hidden = int(encoder.config.hidden_size)
    if output_path.exists():
        try:
            cached = np.load(output_path, mmap_mode="r", allow_pickle=False)
            if tuple(cached.shape) == (len(rows), hidden):
                print(f"cache_reuse={output_path}", flush=True)
                return
        except (OSError, ValueError):
            pass
    dataset = SpeechCommandsDataset(rows, sample_rate, max_seconds, training=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda", collate_fn=Collate())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=(len(rows), hidden))
    encoder.eval()
    offset = 0
    for index, batch in enumerate(loader, start=1):
        values = batch["input_values"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            states = encoder(input_values=values, attention_mask=mask).last_hidden_state
            feature_lengths = mask.sum(dim=-1)
            get_lengths = getattr(encoder, "_get_feat_extract_output_lengths", None)
            if get_lengths is not None:
                feature_lengths = get_lengths(feature_lengths)
            feature_lengths = feature_lengths.to(device).clamp(min=1, max=states.shape[1])
            positions = torch.arange(states.shape[1], device=device).unsqueeze(0)
            feature_mask = positions < feature_lengths.unsqueeze(1)
            pooled = (states * feature_mask.unsqueeze(-1)).sum(dim=1) / feature_mask.sum(dim=1, keepdim=True).clamp_min(1).to(states.dtype)
        values = pooled.float().cpu().numpy()
        output[offset : offset + len(values)] = values
        offset += len(values)
        if index % 50 == 0 or offset == len(rows):
            print(f"cache={output_path.name} rows={offset}/{len(rows)}", flush=True)
    output.flush()
    del output


@torch.no_grad()
def evaluate(model: Head, loader: DataLoader, device: torch.device, classes: int) -> tuple[dict[str, float], torch.Tensor]:
    model.eval()
    confusion = torch.zeros((classes, classes), dtype=torch.long)
    total = top5 = 0
    for features, target in loader:
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = model(features)
        pred = logits.argmax(dim=-1)
        confusion.index_put_((target.cpu(), pred.cpu()), torch.ones_like(target.cpu()), accumulate=True)
        top5 += int((logits.topk(min(5, classes), dim=-1).indices == target[:, None]).any(dim=1).sum())
        total += int(target.numel())
    return metrics_from_confusion(confusion, top5, total), confusion


def plot(history: list[dict[str, Any]], path: Path, run_name: str) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o")
    axes[0].set_title("Training loss")
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], marker="o", label="train")
    axes[1].plot(epochs, [row["val"]["accuracy"] for row in history], marker="o", label="validation")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[2].plot(epochs, [row["val"]["macro_f1"] for row in history], marker="o", label="macro-F1")
    axes[2].plot(epochs, [row["val"]["top5_accuracy"] for row in history], marker="o", label="top-5")
    axes[2].set_title("Validation metrics")
    axes[2].legend()
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    fig.suptitle(run_name)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/speech_commands_v2/extracted"))
    parser.add_argument("--encoder", default="microsoft/wavlm-base-plus")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-seconds", type=float, default=1.0)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--head-batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--head-hidden", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--run-name", default="speech_commands_wavlm_cached_hussein_track_2026-10-18")
    parser.add_argument("--feature-root", type=Path, default=Path("D:/datasets/speech_commands/artifacts/features"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("D:/datasets/speech_commands/artifacts/checkpoints"))
    parser.add_argument("--log-root", type=Path, default=Path("D:/datasets/speech_commands/artifacts/logs"))
    parser.add_argument("--figure-root", type=Path, default=Path("D:/Prawin_New/tse/car-tse-main/reports/figures"))
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=20261018)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    use_amp = bool(args.amp and device.type == "cuda")
    root = args.data_root.resolve()
    train_rows, val_rows, test_rows, labels = build_rows(root)

    encoder = AutoModel.from_pretrained(args.encoder).to(device)
    encoder.config.apply_spec_augment = False
    encoder.config.mask_time_prob = 0.0
    encoder.config.mask_feature_prob = 0.0
    encoder.config.use_cache = False
    encoder.eval()
    feature_dir = args.feature_root.resolve() / args.run_name
    cache_features(encoder, train_rows, feature_dir / "train.npy", args.sample_rate, args.max_seconds, args.feature_batch_size, device, use_amp)
    cache_features(encoder, val_rows, feature_dir / "val.npy", args.sample_rate, args.max_seconds, args.feature_batch_size, device, use_amp)
    cache_features(encoder, test_rows, feature_dir / "test.npy", args.sample_rate, args.max_seconds, args.feature_batch_size, device, use_amp)

    train_set = FeatureDataset(feature_dir / "train.npy", train_rows)
    val_set = FeatureDataset(feature_dir / "val.npy", val_rows)
    test_set = FeatureDataset(feature_dir / "test.npy", test_rows)
    train_loader = DataLoader(train_set, batch_size=args.head_batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=args.head_batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_set, batch_size=args.head_batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = Head( int(encoder.config.hidden_size), len(labels), args.dropout).to(device)
    counts = torch.bincount(torch.tensor([int(row["label"]) for row in train_rows]), minlength=len(labels)).float().clamp_min(1.0)
    weights = ((counts.mean() / counts).pow(0.25) / (counts.mean() / counts).pow(0.25).mean()).clamp(0.5, 3.0).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)

    checkpoint_dir = args.checkpoint_root.resolve() / args.run_name
    log_dir = args.log_root.resolve() / args.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    write_metrics(log_dir / f"run_metadata_{run_id}.json", {
        "run_id": run_id,
        "run_name": args.run_name,
        "device": str(device),
        "encoder": args.encoder,
        "data_root": str(root),
        "feature_root": str(feature_dir),
        "train_examples": len(train_rows),
        "validation_examples": len(val_rows),
        "test_examples": len(test_rows),
        "classes": labels,
        "official_split": True,
        "frozen_encoder": True,
        "seed": args.seed,
    })
    history: list[dict[str, Any]] = []
    best_f1 = float("-inf")
    best_epoch = 0
    bad_epochs = 0
    metrics_path = log_dir / f"epoch_metrics_{run_id}.jsonl"
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        loss_total = correct = total = 0
        for features, target in train_loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = model(features)
            loss = loss_fn(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * target.numel()
            correct += int((logits.argmax(-1) == target).sum())
            total += int(target.numel())
        scheduler.step()
        val, _ = evaluate(model, val_loader, device, len(labels))
        row = {"epoch": epoch, "run_id": run_id, "train_loss": loss_total / max(total, 1), "train_accuracy": correct / max(total, 1), "val": val}
        history.append(row)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(f"epoch={epoch} train_loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.4f} val_acc={val['accuracy']:.4f} val_f1={val['macro_f1']:.4f} val_top5={val['top5_accuracy']:.4f}", flush=True)
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "labels": labels, "encoder": args.encoder, "feature_root": str(feature_dir), "epoch": epoch, "val": val}, checkpoint_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    best = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_metrics, confusion = evaluate(model, test_loader, device, len(labels))
    (log_dir / "metrics_official_test.json").write_text(json.dumps({"best_epoch": best_epoch, "test": test_metrics}, indent=2), encoding="utf-8")
    np.save(log_dir / "confusion_official_test.npy", confusion.numpy())
    (log_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    plot(history, args.figure_root.resolve() / args.run_name, args.run_name)
    print(json.dumps({"best_epoch": best_epoch, "test": test_metrics, "checkpoint": str(checkpoint_dir / "best.pt")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
