#!/usr/bin/env python
"""Evaluate the existing 16 kHz transfer checkpoint on noisy cabin mixtures.

For the local synthetic multichannel set, the baseline collapses the four
microphones by averaging them and then feeds that noisy mono signal to the
pretrained SpEx+ model.  This deliberately excludes the newly initialized
cabin front end, so the result is a useful pretrained-model baseline rather
than a partially trained end-to-end score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from car_tse.data.datasets import TripletExtractionDataset
from car_tse.metrics.si_sdr import si_sdr_per_example
from car_tse.models.stage1_system import Stage1System
from car_tse.train.utils import load_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = Path(r"E:\prawin\snn\runs\spexplus_no_jepa_30\best.pt")
DEFAULT_MANIFEST = REPO_ROOT / "manifests" / "test" / "stage1_mc_triplets.jsonl"


def summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "min": float(array.min()),
        "max": float(array.max()),
        "negative": int((array < 0).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-checkpoint", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--segment-seconds", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--use-cabin-frontend", action="store_true")
    parser.add_argument("--cabin-noise", action="store_true")
    parser.add_argument("--num-mics", type=int, default=4)
    parser.add_argument("--cabin-noise-channels", nargs="+", type=int, default=[0, 2, 4, 6])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if args.device != "auto" else "cpu"
    )

    dataset = TripletExtractionDataset(
        args.manifest,
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
        training=False,
        cabin_noise=args.cabin_noise,
        num_mics=args.num_mics,
        cabin_noise_channels=args.cabin_noise_channels,
        cabin_noise_snr_db=[-5.0, 0.0, 5.0, 10.0, 15.0, 20.0],
    )
    limit = min(max(1, args.limit), len(dataset))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = Stage1System(
        spex_cfg={},
        sample_rate=args.sample_rate,
        use_cabin_frontend=args.use_cabin_frontend,
        num_mics=args.num_mics,
        frontend_cfg={"ec_hidden": 128, "spatial_channels": 64},
        pretrained_dir=str(args.checkpoint),
        backend="transfer_spexplus_16k",
    ).to(device)
    if args.model_checkpoint is not None:
        load_checkpoint(args.model_checkpoint, model)
    model.eval()

    input_scores: list[float] = []
    output_scores: list[float] = []
    improvement_scores: list[float] = []
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if index >= limit:
                break
            mixture = batch["mixture"].to(device)
            enrollment = batch["enrollment"].to(device)
            target = batch["target"].to(device)
            multichannel = batch.get("multichannel")
            if isinstance(multichannel, torch.Tensor) and not args.use_cabin_frontend:
                # The manifest's 4-channel files contain the synthetic cabin
                # noise.  Mean pooling is the transparent no-front-end baseline.
                mixture = multichannel.to(device).mean(dim=1)
            if args.use_cabin_frontend:
                if not isinstance(multichannel, torch.Tensor):
                    raise RuntimeError("Cabin-front-end evaluation requires multichannel input")
                s1, _, _, _ = model(mixture, enrollment, multichannel=multichannel.to(device))
                model_input = model.frontend(multichannel.to(device))
            else:
                s1, _, _, _ = model(mixture, enrollment)
                model_input = mixture
            input_batch = si_sdr_per_example(model_input, target)
            output_batch = si_sdr_per_example(s1, target)
            input_scores.extend(float(x) for x in input_batch.detach().cpu())
            output_scores.extend(float(x) for x in output_batch.detach().cpu())
            improvement_scores.extend(float(x) for x in (output_batch - input_batch).detach().cpu())

    result = {
        "checkpoint": str(args.checkpoint),
        "model_checkpoint": str(args.model_checkpoint) if args.model_checkpoint else None,
        "use_cabin_frontend": args.use_cabin_frontend,
        "cabin_noise": args.cabin_noise,
        "manifest": str(args.manifest),
        "device": str(device),
        "input_si_sdr_db": summary(input_scores),
        "output_si_sdr_db": summary(output_scores),
        "si_sdri_db": summary(improvement_scores),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
