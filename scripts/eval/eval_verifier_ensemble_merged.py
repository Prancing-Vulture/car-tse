#!/usr/bin/env python
"""Compare compact verifier checkpoints and simple score fusions on a cache.

Scores enrollment/extracted pairs with one or more checkpoints and reports the
per-group pass rates for mean/min/product fusion at fixed thresholds.  This is
a diagnostic: no checkpoint or pipeline default is changed here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch

from car_tse.config import REPO_ROOT
from car_tse.data.segment import segment_waveform
from car_tse.io.audio import read_mono
from car_tse.models.speaker_verifier import CompactSpeakerVerifier
from car_tse.train.utils import load_checkpoint
from car_tse.train.utils import get_device


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@torch.no_grad()
def score(
    checkpoint: Path,
    enrollment: np.ndarray,
    extracted: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model = CompactSpeakerVerifier(
        embedding_dim=128, hidden=128, sample_rate=16000, n_mels=40, n_fft=400, hop=160
    ).to(device)
    load_checkpoint(resolve(checkpoint), model)
    model.eval()
    scores: list[np.ndarray] = []
    for start in range(0, len(enrollment), batch_size):
        end = min(len(enrollment), start + batch_size)
        logits, _, _ = model(
            torch.from_numpy(np.array(enrollment[start:end], dtype=np.float32, copy=True)).to(device),
            torch.from_numpy(np.array(extracted[start:end], dtype=np.float32, copy=True)).to(device),
        )
        scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def rates(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    thresholds: list[float],
) -> dict[str, list[float]]:
    output: dict[str, list[float]] = {}
    masks = {
        "gsc_pos": (groups == "stage1_gsc_marvin_cabin") & (labels == 1),
        "libri_pos": (groups == "librispeech_pair") & (labels == 1),
        "libri_neg": (groups == "librispeech_pair") & (labels == 0),
        "aishell": groups == "Eval1",
        "smart": groups == "University of Rochester Smart Speaker Command Dataset",
        "nonwake": groups == "stage1_target_kws_nonwake_hardneg",
    }
    for name, mask in masks.items():
        output[name] = [float((scores[mask] >= t).mean()) if mask.any() else 0.0 for t in thresholds]
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "data/cache/verifier_merged_extracted_2026-10-12",
    )
    parser.add_argument(
        "--enrollment-cache-dir",
        type=Path,
        default=REPO_ROOT / "data/cache/verifier_merged_enrollment_2026-10-12",
    )
    parser.add_argument("--manifest-root", type=Path, default=REPO_ROOT / "manifests")
    parser.add_argument("--manifest-template", default="verifier_merged_{split}.jsonl")
    parser.add_argument("--split", default="test")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "logs/verifier_ensemble_merged_2026-10-12.json",
    )
    args = parser.parse_args()

    device = get_device() if args.device == "auto" else torch.device(args.device)
    cache = resolve(args.cache_dir)
    enrollment_cache = resolve(args.enrollment_cache_dir)
    rows = load_rows(resolve(args.manifest_root) / args.split / str(args.manifest_template).format(split=args.split))
    extracted = np.load(cache / f"{args.split}_wav_16000hz.npy", mmap_mode="r", allow_pickle=False)
    enrollment = np.load(
        enrollment_cache / f"{args.split}_enrollment_wav.npy", mmap_mode="r", allow_pickle=False
    )
    labels = np.asarray([int(row.get("label", 0)) for row in rows], dtype=np.int64)
    groups = np.asarray(
        [
            str(row.get("source") or Path(str(row.get("source_manifest", ""))).stem or "unknown")
            for row in rows
        ],
        dtype=object,
    )
    if not (len(rows) == len(extracted) == len(enrollment)):
        raise ValueError("manifest/cache length mismatch")

    checkpoint_scores = [score(path, enrollment, extracted, device) for path in args.checkpoint]
    fusions: dict[str, np.ndarray] = {}
    if len(checkpoint_scores) == 1:
        fusions["single"] = checkpoint_scores[0]
    else:
        stacked = np.stack(checkpoint_scores, axis=0)
        fusions["mean"] = stacked.mean(axis=0)
        fusions["min"] = stacked.min(axis=0)
        fusions["product"] = stacked.prod(axis=0)

    result: dict[str, Any] = {
        "benchmark": "verifier_ensemble_merged",
        "checkpoints": [str(resolve(path)) for path in args.checkpoint],
        "split": args.split,
        "records": int(len(rows)),
        "thresholds": [float(t) for t in args.thresholds],
        "fusions": {},
    }
    for name, fused in fusions.items():
        result["fusions"][name] = rates(fused, labels, groups, [float(t) for t in args.thresholds])
        print(f"=== {name} ===")
        table = result["fusions"][name]
        header = "  thr | " + " | ".join(f"{key:>9s}" for key in table)
        print(header)
        for index, threshold in enumerate(args.thresholds):
            row = "  %.2f | " % threshold + " | ".join(
                f"{table[key][index]:9.3f}" for key in table
            )
            print(row, flush=True)
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
