#!/usr/bin/env python
"""Aggregate end-to-end pipeline metrics."""

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
from car_tse.metrics.si_sdr import si_sdr
from car_tse.models.stage1_system import Stage1System
from car_tse.pipeline.in_car_pipeline import InCarPipeline, PipelineState
from car_tse.train.utils import get_device, load_checkpoint, write_metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "eval_end_to_end.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    args = parser.parse_args()
    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    device = get_device()
    stage1_cfg = load_stage_config(REPO_ROOT / "configs" / "stage1_spex_mc.yaml", args.paths)
    stage1 = Stage1System(
        spex_cfg=stage1_cfg["spex"],
        sample_rate=cfg["sample_rate"],
        use_cabin_frontend=True,
        num_mics=4,
        frontend_cfg=stage1_cfg.get("frontend"),
        pretrained_dir=str(paths["spex_pretrained_dir"]),
    ).to(device)
    load_checkpoint(paths["checkpoint_root"] / cfg["checkpoints"]["spex"], stage1)
    pipeline = InCarPipeline.from_checkpoints(
        stage1,
        paths["checkpoint_root"] / cfg["checkpoints"]["presence"],
        paths["checkpoint_root"] / cfg["checkpoints"]["kws"],
        paths["checkpoint_root"] / cfg["checkpoints"]["intent"],
        presence_threshold=cfg["presence_threshold"],
        kws_threshold=cfg["kws_threshold"],
        device=device,
    )
    manifest = Path(paths["manifest_root"]) / cfg["split"] / cfg["manifest"]
    ds = TripletExtractionDataset(manifest, cfg["sample_rate"], 4.0, training=False)
    loader = DataLoader(ds, batch_size=1)
    accepted = wake = 0
    si_scores = []
    state = PipelineState()
    for batch in loader:
        mix = batch["mixture"][0]
        enroll = batch["enrollment"][0]
        target = batch["target"][0].to(device).unsqueeze(0)
        out, state = pipeline.forward(mix, enroll, state)
        if out.s1 is not None:
            si_scores.append(float(si_sdr(out.s1, target)))
        if out.accepted:
            accepted += 1
        if out.wake_detected:
            wake += 1
    metrics = {
        "si_sdr_mean": sum(si_scores) / max(len(si_scores), 1),
        "accept_rate": accepted / max(len(ds), 1),
        "wake_rate": wake / max(len(ds), 1),
        "n": len(ds),
    }
    write_metrics(paths["log_root"] / "eval_end_to_end.json", metrics)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
