#!/usr/bin/env python
"""Evaluate presence FAR/FRR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from torch.utils.data import DataLoader

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.datasets import PresenceDataset
from car_tse.models.presence_head import PresenceHead
from car_tse.models.stage1_system import Stage1System
from car_tse.train.utils import get_device, load_checkpoint, write_metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "stage2_presence.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--split", default="val")
    args = parser.parse_args()
    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    device = get_device()
    stage1_cfg = load_stage_config(REPO_ROOT / "configs" / "stage1_spex_mc.yaml", args.paths)
    stage1 = Stage1System(
        spex_cfg=stage1_cfg["spex"],
        sample_rate=stage1_cfg["sample_rate"],
        use_cabin_frontend=stage1_cfg.get("use_cabin_frontend", False),
        num_mics=stage1_cfg.get("num_mics", 4),
        frontend_cfg=stage1_cfg.get("frontend"),
        pretrained_dir=str(paths["spex_pretrained_dir"]),
    ).to(device)
    load_checkpoint(paths["checkpoint_root"] / cfg["spex_checkpoint"], stage1)
    head = PresenceHead(cfg["model"]["hidden"]).to(device)
    load_checkpoint(paths["checkpoint_root"] / cfg["name"] / "best.pt", head)
    stage1.eval()
    head.eval()
    manifest = Path(paths["manifest_root"]) / args.split / cfg["manifest"]
    ds = PresenceDataset(manifest, cfg["sample_rate"], cfg["segment_seconds"], training=False)
    loader = DataLoader(ds, batch_size=8)
    thr = cfg["presence_threshold"]
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for batch in loader:
            mix = batch["mixture"].to(device)
            enroll = batch["enrollment"].to(device)
            labels = batch["target_present"]
            s1, _, _, _ = stage1(mix, enroll)
            prob = torch.sigmoid(head(s1)).cpu()
            for p, y in zip(prob, labels):
                pred = p.item() >= thr
                truth = float(y) >= 0.5
                if truth and pred:
                    tp += 1
                elif truth and not pred:
                    fn += 1
                elif not truth and pred:
                    fp += 1
                else:
                    tn += 1
    far = fp / max(fp + tn, 1)
    frr = fn / max(fn + tp, 1)
    write_metrics(paths["log_root"] / cfg["name"] / f"metrics_{args.split}.json", {"far": far, "frr": frr})
    print(f"FAR={far:.3f} FRR={frr:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
