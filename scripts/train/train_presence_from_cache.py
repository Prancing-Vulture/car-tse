#!/usr/bin/env python
"""Train a Stage 2 presence head from cached frozen Stage 1 outputs."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import REPO_ROOT
from car_tse.losses.presence import presence_loss
from car_tse.models.presence_head import PresenceHead
from car_tse.train.utils import get_device, load_checkpoint, save_checkpoint, write_metrics


def resolve_path(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


class EmbeddingDataset(Dataset):
    def __init__(self, cache_path: Path, labels_path: Path):
        self.cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        self.labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
        if len(self.cache) != len(self.labels):
            raise ValueError(f"Cache/label length mismatch: {cache_path} vs {labels_path}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(np.array(self.cache[index], dtype=np.float32, copy=True)),
            torch.tensor(float(self.labels[index]), dtype=torch.float32),
        )


@torch.no_grad()
def collect_scores(head: PresenceHead, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    head.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for embeddings, batch_labels in loader:
        logits = head(embeddings.to(device, non_blocking=True))
        scores.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def metrics_at(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = scores >= threshold
    truth = labels >= 0.5
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    tn = int(np.logical_and(~pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / max(len(labels), 1),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "far": fp / max(fp + tn, 1),
        "frr": fn / max(fn + tp, 1),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--val-labels", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--absent-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    train_ds = EmbeddingDataset(resolve_path(args.train_cache), resolve_path(args.train_labels))
    val_ds = EmbeddingDataset(resolve_path(args.val_cache), resolve_path(args.val_labels))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda")
    head = PresenceHead(args.hidden).to(device)
    if args.init_checkpoint:
        load_checkpoint(resolve_path(args.init_checkpoint), head)
        print(f"Initialized presence head from {resolve_path(args.init_checkpoint)}", flush=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_f1 = float("-inf")
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        losses: list[float] = []
        for embeddings, labels in train_loader:
            embeddings = embeddings.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = head(embeddings)
                loss, _ = presence_loss(logits, labels, embeddings, args.absent_penalty)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_scores, val_labels = collect_scores(head, val_loader, device)
        sweep = [metrics_at(val_scores, val_labels, i / 100.0) for i in range(1, 100)]
        best = max(sweep, key=lambda item: float(item["f1"]))
        balanced = max(sweep, key=lambda item: (float(item["recall"]) + float(item["specificity"])) / 2.0)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val": metrics_at(val_scores, val_labels, 0.5),
            "best_f1": best,
            "best_balanced_accuracy": balanced,
        }
        history.append(row)
        print(
            f"epoch {epoch} loss={row['train_loss']:.4f} "
            f"val_f1={float(best['f1']):.3f}@{float(best['threshold']):.2f} "
            f"val_balanced={float(balanced['accuracy']):.3f}@{float(balanced['threshold']):.2f}",
            flush=True,
        )
        if float(best["f1"]) > best_f1:
            best_f1 = float(best["f1"])
            save_checkpoint(resolve_path(args.output_checkpoint), head, epoch, row)

    output_metrics = resolve_path(args.output_metrics)
    write_metrics(
        output_metrics,
        {
            "device": str(device),
            "train_examples": len(train_ds),
            "val_examples": len(val_ds),
            "best_f1": best_f1,
            "history": history,
            "checkpoint": str(resolve_path(args.output_checkpoint)),
        },
    )
    print(f"Done. Best checkpoint: {resolve_path(args.output_checkpoint)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
