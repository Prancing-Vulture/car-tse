#!/usr/bin/env python
"""Evaluate a cached Stage 2 presence checkpoint across thresholds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))

from car_tse.config import REPO_ROOT
from car_tse.models.presence_head import PresenceHead
from car_tse.train.utils import get_device, load_checkpoint, write_metrics
from train_presence_from_cache import EmbeddingDataset, metrics_at, resolve_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    device = get_device()
    dataset = EmbeddingDataset(resolve_path(args.cache), resolve_path(args.labels))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda")
    head = PresenceHead(args.hidden).to(device)
    load_checkpoint(resolve_path(args.checkpoint), head)
    head.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for embeddings, batch_labels in loader:
            scores.append(torch.sigmoid(head(embeddings.to(device, non_blocking=True))).cpu().numpy())
            labels.append(batch_labels.numpy())
    score_array = np.concatenate(scores)
    label_array = np.concatenate(labels)
    sweep = [metrics_at(score_array, label_array, i / 100.0) for i in range(1, 100)]
    result = {
        "device": str(device),
        "examples": len(label_array),
        "positive_examples": int((label_array >= 0.5).sum()),
        "negative_examples": int((label_array < 0.5).sum()),
        "threshold_0.50": metrics_at(score_array, label_array, 0.50),
        "threshold_0.60": metrics_at(score_array, label_array, 0.60),
        "threshold_0.75": metrics_at(score_array, label_array, 0.75),
        "threshold_0.90": metrics_at(score_array, label_array, 0.90),
        "threshold_0.95": metrics_at(score_array, label_array, 0.95),
        "best_f1": max(sweep, key=lambda item: float(item["f1"])),
        "best_balanced_accuracy": max(sweep, key=lambda item: (float(item["recall"]) + float(item["specificity"])) / 2.0),
    }
    write_metrics(resolve_path(args.output), result)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
