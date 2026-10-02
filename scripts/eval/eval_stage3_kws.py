#!/usr/bin/env python
"""Evaluate KWS accuracy."""

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
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "stage3_kws.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    device = get_device()
    manifest = Path(paths["manifest_root"]) / args.split / cfg["manifest_eval"]
    ds = KWSDataset(manifest, cfg["sample_rate"], cfg["segment_seconds"], training=False)
    loader = DataLoader(ds, batch_size=64)
    model = KWSModel(cfg["sample_rate"], cfg["model"]["n_mels"], cfg["model"]["channels"], cfg["model"]["num_classes"]).to(device)
    load_checkpoint(paths["checkpoint_root"] / cfg["name"] / "best.pt", model)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            wav = batch["wav"].to(device)
            labels = batch["label"].to(device)
            logits = model(wav)
            correct += (logits.argmax(-1) == labels).sum().item()
            total += labels.numel()
    acc = correct / max(total, 1)
    write_metrics(paths["log_root"] / cfg["name"] / f"metrics_{args.split}.json", {"accuracy": acc})
    print(f"KWS accuracy: {acc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
