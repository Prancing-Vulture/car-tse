#!/usr/bin/env python
"""Evaluate Stage 1 SI-SDR on val/test manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from torch.utils.data import DataLoader

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.datasets import TripletExtractionDataset
from car_tse.metrics.si_sdr import si_sdr
from car_tse.models.stage1_system import Stage1System
from car_tse.train.utils import get_device, load_checkpoint, write_metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "stage1_spex_sc.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    device = get_device()
    manifest = Path(paths["manifest_root"]) / args.split / cfg["train"]["manifest"]
    ds = TripletExtractionDataset(manifest, cfg["sample_rate"], cfg["segment_seconds"], training=False)
    loader = DataLoader(ds, batch_size=4)
    model = Stage1System(
        spex_cfg=cfg["spex"],
        sample_rate=cfg["sample_rate"],
        use_cabin_frontend=cfg.get("use_cabin_frontend", False),
        num_mics=cfg.get("num_mics", 4),
        frontend_cfg=cfg.get("frontend"),
        pretrained_dir=str(paths["spex_pretrained_dir"]),
    ).to(device)
    ckpt = args.checkpoint or (paths["checkpoint_root"] / cfg["name"] / "best.pt")
    if ckpt.exists():
        load_checkpoint(ckpt, model)
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            mix = batch["mixture"].to(device)
            enroll = batch["enrollment"].to(device)
            target = batch["target"].to(device)
            s1, _, _, _ = model(mix, enroll)
            scores.append(float(si_sdr(s1, target)))
    mean_si = sum(scores) / max(len(scores), 1)
    out = paths["log_root"] / cfg["name"] / f"metrics_{args.split}.json"
    write_metrics(out, {"si_sdr": mean_si, "n": len(scores)})
    print(f"SI-SDR {args.split}: {mean_si:.2f} ({len(scores)} batches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
