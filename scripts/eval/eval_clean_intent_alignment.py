#!/usr/bin/env python
"""Inspect clean intent confidence for raw clips and scene-aligned crops."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "eval"))

from car_tse.config import load_stage_config  # noqa: E402
from car_tse.train.utils import get_device  # noqa: E402
from eval_continuous_full_pipeline_proxy import fit_start, load_pipeline, resolve_path  # noqa: E402


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32")
    if int(sr) != int(sample_rate):
        raise ValueError(f"Expected {sample_rate} Hz, got {sr} Hz for {path}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32)


def score(pipeline, waveform: np.ndarray, device: torch.device) -> dict[str, object]:
    tensor = torch.from_numpy(waveform).to(device)
    with torch.inference_mode():
        logits = pipeline.intent(tensor.unsqueeze(0))
        probability = torch.softmax(logits, dim=-1)[0]
        index = int(probability.argmax().item())
    return {"id": str(index), "confidence": float(probability[index].item())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/eval_end_to_end_clean_english_control.yaml")
    parser.add_argument("--command", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()
    cfg = load_stage_config(args.config, REPO_ROOT / "configs/paths.yaml")
    device = get_device()
    pipeline = load_pipeline(cfg, cfg["paths"], device)
    command = fit_start(load_audio(resolve_path(args.command), args.sample_rate), 2 * args.sample_rate)
    rows = [{"source": "command_start", "start_seconds": 0.0, **score(pipeline, command, device)}]
    for start_seconds in (0.05, 0.10, 0.20, 0.50, 1.00):
        start = int(round(start_seconds * args.sample_rate))
        crop = command[start : start + 2 * args.sample_rate]
        if len(crop) < 2 * args.sample_rate:
            crop = np.pad(crop, (0, 2 * args.sample_rate - len(crop)))
        rows.append({"source": "command_crop", "start_seconds": start_seconds, **score(pipeline, crop.astype(np.float32), device)})
    if args.scene is not None:
        scene = load_audio(resolve_path(args.scene), args.sample_rate)
        for start_seconds in (1.5, 2.0, 2.1, 2.2, 2.5):
            start = int(round(start_seconds * args.sample_rate))
            crop = scene[start : start + 2 * args.sample_rate]
            if len(crop) < 2 * args.sample_rate:
                crop = np.pad(crop, (0, 2 * args.sample_rate - len(crop)))
            rows.append({"source": "scene_crop", "start_seconds": start_seconds, **score(pipeline, crop.astype(np.float32), device)})
    print(json.dumps({"command": str(args.command), "scene": str(args.scene) if args.scene else None, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
