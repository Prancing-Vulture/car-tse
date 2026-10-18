#!/usr/bin/env python
"""Calibrate an enrollment-conditioned wake threshold on cached data only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.io.manifest import read_jsonl
from car_tse.models.kws_model import TargetSpeakerKWSModel
from car_tse.train.utils import get_device, load_checkpoint, write_metrics


class CachedDataset(Dataset):
    def __init__(self, manifest: Path, cache_dir: Path, split: str, sample_rate: int):
        self.rows = read_jsonl(manifest)
        self.wav = np.load(cache_dir / f"{split}_wav_{sample_rate}hz.npy", mmap_mode="r", allow_pickle=False)
        self.enrollment = np.load(cache_dir / f"{split}_enrollment_embedding.npy", mmap_mode="r", allow_pickle=False)
        self.extracted = np.load(cache_dir / f"{split}_extracted_embedding.npy", mmap_mode="r", allow_pickle=False)
        if not (len(self.rows) == len(self.wav) == len(self.enrollment) == len(self.extracted)):
            raise ValueError("Target-KWS cache and manifest lengths do not match")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "wav": torch.from_numpy(np.array(self.wav[index], dtype=np.float32, copy=True)),
            "enrollment": torch.from_numpy(np.array(self.enrollment[index], dtype=np.float32, copy=True)),
            "extracted": torch.from_numpy(np.array(self.extracted[index], dtype=np.float32, copy=True)),
            "label": torch.tensor(int(self.rows[index]["label"]), dtype=torch.long),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_stage_config(args.config, args.paths)
    manifest_root = Path(cfg["paths"]["manifest_root"])
    if not manifest_root.is_absolute():
        manifest_root = REPO_ROOT / manifest_root
    manifest = manifest_root / args.split / (cfg["manifest_val"] if args.split == "val" else cfg["manifest_eval"])
    cache_dir = Path(cfg["cache_dir"])
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    device = get_device()
    dataset = CachedDataset(manifest, cache_dir, args.split, int(cfg["sample_rate"]))
    loader = DataLoader(dataset, batch_size=int(cfg["train"].get("val_batch_size", 512)), shuffle=False)
    model = TargetSpeakerKWSModel(
        sample_rate=int(cfg["sample_rate"]),
        n_mels=int(cfg["model"].get("n_mels", 40)),
        channels=cfg["model"].get("channels", [32, 64, 128]),
        num_classes=int(cfg["model"].get("num_classes", 2)),
        speaker_embedding_dim=int(cfg["model"].get("speaker_embedding_dim", 256)),
        hidden=int(cfg["model"].get("hidden", 256)),
    ).to(device)
    load_checkpoint(args.checkpoint if args.checkpoint.is_absolute() else REPO_ROOT / args.checkpoint, model)
    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            logits = model(
                batch["wav"].to(device),
                batch["enrollment"].to(device),
                batch["extracted"].to(device),
            )
            scores.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
            labels.extend(batch["label"].tolist())
    score_array = np.asarray(scores, dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.int64)
    rows = []
    for threshold in np.arange(args.step, 1.0, args.step):
        pred = score_array >= float(threshold)
        tp = int(np.sum(pred & (label_array == 1)))
        fp = int(np.sum(pred & (label_array == 0)))
        fn = int(np.sum(~pred & (label_array == 1)))
        tn = int(np.sum(~pred & (label_array == 0)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        negatives = max(int(np.sum(label_array == 0)), 1)
        rows.append({
            "threshold": float(threshold),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
            "false_accepts_per_hour_if_1s_windows": fp * 3600.0 / negatives,
        })
    best_f1 = max(rows, key=lambda row: row["f1"])
    recall_targets = {}
    for target in (0.95, 0.90, 0.80):
        eligible = [row for row in rows if row["recall"] >= target]
        recall_targets[str(target)] = max(eligible, key=lambda row: row["precision"]) if eligible else None
    result = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "examples": len(labels),
        "positive_examples": int(np.sum(label_array == 1)),
        "negative_examples": int(np.sum(label_array == 0)),
        "best_f1": best_f1,
        "best_precision_at_recall_targets": recall_targets,
        "thresholds": rows,
        "note": "Threshold selection is calibration-only; use a separate held-out continuous stream for deployment false-accept reporting.",
    }
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    write_metrics(output, result)
    print({"split": args.split, "best_f1": best_f1, "recall_targets": recall_targets, "output": str(output)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
