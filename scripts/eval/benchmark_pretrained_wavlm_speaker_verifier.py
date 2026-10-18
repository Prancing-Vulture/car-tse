#!/usr/bin/env python
"""Benchmark the inference cost of the optional pretrained WavLM verifier."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="microsoft/wavlm-base-plus-sv")
    parser.add_argument("--model-cache-dir", type=Path, default=Path("D:/datasets/pretrained_models"))
    parser.add_argument(
        "--input-wav-cache",
        type=Path,
        default=REPO_ROOT / "data/cache/stage3_target_kws_gsc_marvin_ft2_stage1_16k/test_wav_16000hz.npy",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "logs/wavlm_speaker_verifier_latency_2026-10-11.json")
    args = parser.parse_args()

    from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cache_dir = resolve(args.model_cache_dir)
    wav_cache = np.load(resolve(args.input_wav_cache), mmap_mode="r", allow_pickle=False)
    waveform = np.asarray(wav_cache[0], dtype=np.float32)
    processor = AutoFeatureExtractor.from_pretrained(args.model_name, cache_dir=model_cache_dir)
    model = AutoModelForAudioXVector.from_pretrained(args.model_name, cache_dir=model_cache_dir).to(device).eval()
    inputs = processor([waveform], sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        for _ in range(max(0, int(args.warmup))):
            model(**inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        times: list[float] = []
        for _ in range(max(1, int(args.iterations))):
            start = time.perf_counter()
            model(**inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000.0)
    output = {
        "benchmark": "pretrained_wavlm_speaker_verifier_latency",
        "model_name": str(args.model_name),
        "device": str(device),
        "input": {"sample_rate": 16000, "seconds": float(len(waveform) / 16000.0)},
        "warmup": int(args.warmup),
        "iterations": int(args.iterations),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "mean_ms": float(np.mean(times)),
        "parameters_m": float(sum(parameter.numel() for parameter in model.parameters()) / 1.0e6),
        "peak_cuda_memory_mb": (
            float(torch.cuda.max_memory_allocated() / 1.0e6) if device.type == "cuda" else None
        ),
        "decision": "diagnostic_only; not integrated into the production pipeline",
    }
    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
