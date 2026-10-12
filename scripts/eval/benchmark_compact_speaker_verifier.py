#!/usr/bin/env python
"""Benchmark the compact distilled speaker-verifier candidate."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from car_tse.models.speaker_verifier import CompactSpeakerVerifier
from car_tse.train.utils import load_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints/compact_speaker_verifier_gsc_ft1/best.pt",
    )
    parser.add_argument(
        "--enrollment-cache",
        type=Path,
        default=REPO_ROOT / "data/cache/compact_speaker_verifier_gsc_marvin_2026-10-11/test_enrollment_wav.npy",
    )
    parser.add_argument(
        "--extracted-cache",
        type=Path,
        default=REPO_ROOT / "data/cache/stage3_target_kws_gsc_marvin_ft2_stage1_16k/test_wav_16000hz.npy",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "logs/compact_speaker_verifier_latency_2026-10-11.json",
    )
    args = parser.parse_args()

    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = CompactSpeakerVerifier(embedding_dim=128, hidden=128, sample_rate=16000, n_mels=40, n_fft=400, hop=160).to(device).eval()
    load_checkpoint(resolve(args.checkpoint), model)
    enrollment_cache = np.load(resolve(args.enrollment_cache), mmap_mode="r", allow_pickle=False)
    extracted_cache = np.load(resolve(args.extracted_cache), mmap_mode="r", allow_pickle=False)
    enrollment = torch.from_numpy(np.array(enrollment_cache[0], dtype=np.float32, copy=True)).unsqueeze(0).to(device)
    extracted = torch.from_numpy(np.array(extracted_cache[0], dtype=np.float32, copy=True)).unsqueeze(0).to(device)
    with torch.inference_mode():
        for _ in range(max(0, int(args.warmup))):
            model(enrollment, extracted)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        times: list[float] = []
        for _ in range(max(1, int(args.iterations))):
            start = time.perf_counter()
            model(enrollment, extracted)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000.0)
    output = {
        "benchmark": "compact_speaker_verifier_latency",
        "checkpoint": str(resolve(args.checkpoint)),
        "device": str(device),
        "input": {"sample_rate": 16000, "seconds_per_waveform": 1.0, "waveforms_per_call": 2},
        "warmup": int(args.warmup),
        "iterations": int(args.iterations),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "mean_ms": float(np.mean(times)),
        "parameters_m": float(sum(parameter.numel() for parameter in model.parameters()) / 1.0e6),
        "peak_cuda_memory_mb": float(torch.cuda.max_memory_allocated() / 1.0e6) if device.type == "cuda" else None,
        "decision": "diagnostic_only; not integrated into the default pipeline",
    }
    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
