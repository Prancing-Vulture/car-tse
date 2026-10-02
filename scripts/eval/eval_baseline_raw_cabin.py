#!/usr/bin/env python
"""Baseline: raw cabin mixture -> intent (skip SpEx+)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from torch.utils.data import DataLoader

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.datasets import TripletExtractionDataset
from car_tse.models.intent_classifier import IntentClassifier
from car_tse.train.utils import get_device, load_checkpoint, write_metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    cfg = load_stage_config(REPO_ROOT / "configs" / "eval_end_to_end.yaml", args.paths)
    paths = cfg["paths"]
    device = get_device()
    manifest = Path(paths["manifest_root"]) / args.split / cfg["manifest"]
    ds = TripletExtractionDataset(manifest, cfg["sample_rate"], 2.0, training=False)
    loader = DataLoader(ds, batch_size=8)
    intent_cfg = load_stage_config(REPO_ROOT / "configs" / "stage4_intent.yaml", args.paths)
    label_map_path = Path(paths["manifest_root"]) / "meta" / intent_cfg["label_map"]
    num_classes = len(json.loads(label_map_path.read_text(encoding="utf-8"))) if label_map_path.exists() else 31
    model = IntentClassifier(8000, 80, 256, 2, num_classes).to(device)
    load_checkpoint(paths["checkpoint_root"] / cfg["checkpoints"]["intent"], model)
    model.eval()
    n = 0
    with torch.no_grad():
        for batch in loader:
            wav = batch["mixture"].to(device)
            model(wav)
            n += wav.shape[0]
    write_metrics(paths["log_root"] / "baseline_raw" / "metrics.json", {"clips": n, "note": "raw mixture forward"})
    print(f"Processed {n} raw cabin clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
