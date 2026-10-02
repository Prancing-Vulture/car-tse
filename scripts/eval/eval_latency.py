#!/usr/bin/env python
"""Per-stage latency / RTF on GPU or CPU."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.models.intent_classifier import IntentClassifier
from car_tse.models.kws_model import KWSModel
from car_tse.models.presence_head import PresenceHead
from car_tse.models.stage1_system import Stage1System
from car_tse.train.utils import get_device, write_metrics


def bench(fn, warmup: int = 3, repeats: int = 20) -> float:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    args = parser.parse_args()
    device = get_device()
    sr = 8000
    t = int(4 * sr)
    mix = torch.randn(1, t, device=device)
    enroll = torch.randn(1, t // 2, device=device)
    stage1_cfg = load_stage_config(REPO_ROOT / "configs" / "stage1_spex_mc.yaml", args.paths)
    paths = stage1_cfg["paths"]
    stage1 = Stage1System(
        spex_cfg=stage1_cfg["spex"],
        sample_rate=sr,
        use_cabin_frontend=True,
        num_mics=4,
        frontend_cfg=stage1_cfg.get("frontend"),
        pretrained_dir=str(paths["spex_pretrained_dir"]),
    ).to(device)
    stage1.eval()
    presence = PresenceHead().to(device)
    kws = KWSModel(sr).to(device)
    intent = IntentClassifier(sr).to(device)
    mc = torch.randn(1, 4, t, device=device)

    def run_stage1():
        with torch.no_grad():
            stage1(mix, enroll, multichannel=mc)

    def run_presence():
        with torch.no_grad():
            presence(mix)

    def run_kws():
        with torch.no_grad():
            kws(mix)

    def run_intent():
        with torch.no_grad():
            intent(mix)

    metrics = {
        "stage1_ms": bench(run_stage1) * 1000,
        "presence_ms": bench(run_presence) * 1000,
        "kws_ms": bench(run_kws) * 1000,
        "intent_ms": bench(run_intent) * 1000,
        "device": str(device),
        "audio_seconds": 4.0,
    }
    metrics["rtf_stage1"] = metrics["stage1_ms"] / 1000 / 4.0
    out = paths["log_root"] / "latency.json"
    write_metrics(out, metrics)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
