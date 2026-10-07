#!/usr/bin/env python
"""Measure latency of the current cabin-adapted Stage 1 and KWS models."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.models.kws_model import KWSModel
from car_tse.models.stage1_system import Stage1System
from car_tse.models.wav2vec_intent import Wav2VecIntentClassifier
from car_tse.train.utils import get_device, load_checkpoint, write_metrics


def bench(fn, warmup: int = 3, repeats: int = 20) -> float:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--stage1-config", type=Path, default=REPO_ROOT / "configs" / "stage1_spex_transfer_mc_cavemove.yaml")
    parser.add_argument("--stage1-checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "stage1_spex_transfer_mc_cavemove_cabin" / "best.pt")
    parser.add_argument("--kws-config", type=Path, default=REPO_ROOT / "configs" / "stage3_kws_cavemove.yaml")
    parser.add_argument("--kws-checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "stage3_kws_cavemove_binary" / "best.pt")
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--intent-encoder", default=None)
    parser.add_argument("--intent-encoder-checkpoint", type=Path, default=None)
    parser.add_argument("--intent-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available")
    device = (
        get_device()
        if args.device == "auto"
        else torch.device(args.device)
    )
    stage1_cfg = load_stage_config(args.stage1_config, args.paths)
    kws_cfg = load_stage_config(args.kws_config, args.paths)
    base_checkpoint = Path(stage1_cfg["pretrained_checkpoint"])
    stage1 = Stage1System(
        spex_cfg=stage1_cfg["spex"],
        sample_rate=stage1_cfg["sample_rate"],
        use_cabin_frontend=True,
        num_mics=stage1_cfg["num_mics"],
        frontend_cfg=stage1_cfg.get("frontend"),
        pretrained_dir=str(base_checkpoint),
        backend=stage1_cfg["spex_backend"],
    ).to(device)
    load_checkpoint(args.stage1_checkpoint, stage1)
    stage1.eval()
    kws = KWSModel(
        kws_cfg["sample_rate"],
        kws_cfg["model"]["n_mels"],
        kws_cfg["model"]["channels"],
        kws_cfg["model"]["num_classes"],
    ).to(device)
    load_checkpoint(args.kws_checkpoint, kws)
    kws.eval()
    sr = stage1_cfg["sample_rate"]
    samples = int(round(args.duration * sr))
    mix = torch.randn(1, samples, device=device)
    enrollment = torch.randn(1, max(samples // 2, 1), device=device)
    multichannel = torch.randn(1, stage1_cfg["num_mics"], samples, device=device)
    kws_input = torch.randn(1, int(round(args.duration * kws_cfg["sample_rate"])), device=device)

    intent = None
    if args.intent_encoder and args.intent_encoder_checkpoint and args.intent_checkpoint:
        intent = Wav2VecIntentClassifier(
            encoder_name=args.intent_encoder,
            encoder_checkpoint=args.intent_encoder_checkpoint,
            head_checkpoint=args.intent_checkpoint,
            num_classes=31,
            input_sample_rate=stage1_cfg["sample_rate"],
        ).to(device)
        intent.eval()
    intent_input = torch.randn(1, int(round(args.duration * stage1_cfg["sample_rate"])), device=device)

    def run_stage1() -> None:
        with torch.inference_mode():
            stage1(mix, enrollment, multichannel=multichannel)

    def run_kws() -> None:
        with torch.inference_mode():
            kws(kws_input)

    def run_intent() -> None:
        with torch.inference_mode():
            intent(intent_input)

    stage1_ms = bench(run_stage1, repeats=args.repeats) * 1000
    kws_ms = bench(run_kws, repeats=args.repeats) * 1000
    intent_ms = bench(run_intent, repeats=args.repeats) * 1000 if intent is not None else None
    metrics = {
        "device": str(device),
        "audio_seconds": args.duration,
        "stage1_ms": stage1_ms,
        "kws_ms": kws_ms,
        "stage1_rtf": stage1_ms / 1000 / args.duration,
        "kws_rtf": kws_ms / 1000 / args.duration,
        "intent_ms": intent_ms,
        "intent_rtf": intent_ms / 1000 / args.duration if intent_ms is not None else None,
        "stage1_checkpoint": str(args.stage1_checkpoint),
        "kws_checkpoint": str(args.kws_checkpoint),
    }
    output = args.output or (stage1_cfg["paths"]["log_root"] / "latency_cavemove.json")
    write_metrics(output, metrics)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
