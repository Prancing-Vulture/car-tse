#!/usr/bin/env python
"""Evaluate the compact speaker verifier on the cabin-aware target-KWS cache.

The cache built from ``stage3_target_kws_gsc_marvin_hardneg_aishell5`` contains
synthetic CAVEMOVE/GSC-Marvin pairs plus real AISHELL-5 target-absent cabin
windows.  This diagnostic measures whether the compact verifier separates the
target-present rows from both synthetic and real-cabin negatives, and what a
validation-selected operating threshold would look like.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from car_tse.data.segment import segment_waveform  # noqa: E402
from car_tse.io.audio import read_mono  # noqa: E402
from car_tse.models.speaker_verifier import CompactSpeakerVerifier  # noqa: E402
from car_tse.train.utils import load_checkpoint  # noqa: E402
from scripts.eval.calibrate_spex_speaker_verifier import (  # noqa: E402
    choose_threshold,
    roc_summary,
    threshold_metrics,
)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_rows(manifest: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class SourceIndex:
    def __init__(self) -> None:
        self._manifests: dict[str, dict[str, dict[str, Any]]] = {}

    def get(self, manifest: str, source_id: str) -> dict[str, Any] | None:
        path = Path(manifest)
        if not path.exists():
            return None
        if manifest not in self._manifests:
            self._manifests[manifest] = {
                str(row.get("id", "")): row
                for row in load_rows(path)
            }
        row = self._manifests[manifest].get(source_id)
        if row is None:
            raise KeyError(f"Missing source row {source_id} in {manifest}")
        return row


@torch.no_grad()
def score_pairs(
    model: CompactSpeakerVerifier,
    enrollment: np.ndarray,
    extracted: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    for start in range(0, len(enrollment), batch_size):
        end = min(len(enrollment), start + batch_size)
        logits, _, _ = model(
            torch.from_numpy(np.array(enrollment[start:end], dtype=np.float32, copy=True)).to(device),
            torch.from_numpy(np.array(extracted[start:end], dtype=np.float32, copy=True)).to(device),
        )
        scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)


def group_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    threshold: float | None,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        entry: dict[str, Any] = {
            "records": int(mask.sum()),
            "positive_count": int((labels[mask] == 1).sum()),
            "negative_count": int((labels[mask] == 0).sum()),
            "mean_score": float(scores[mask].mean()) if mask.any() else None,
        }
        group_labels = labels[mask]
        if len(set(group_labels.tolist())) > 1:
            entry["roc"] = roc_summary(scores[mask], group_labels)
        if threshold is not None:
            entry["at_threshold"] = threshold_metrics(scores[mask], group_labels, threshold)
        output[str(group)] = entry
    return output


def rate(scores: np.ndarray, mask: np.ndarray, threshold: float) -> float:
    if not mask.any():
        return 0.0
    return float((scores[mask] >= threshold).mean())


def threshold_sweep(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, list[float]]:
    output: dict[str, list[float]] = {
        "thresholds": [float(value) for value in thresholds]
    }
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        output[f"group::{group}"] = [
            rate(scores, mask, float(threshold)) for threshold in thresholds
        ]
        positive_mask = mask & (labels == 1)
        if positive_mask.any():
            output[f"positive::{group}"] = [
                rate(scores, positive_mask, float(threshold)) for threshold in thresholds
            ]
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints/compact_speaker_verifier_gsc_ft1/best.pt",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "data/cache/stage3_target_kws_gsc_marvin_hardneg_aishell5_16k",
    )
    parser.add_argument("--manifest-root", type=Path, default=REPO_ROOT / "manifests")
    parser.add_argument(
        "--manifest-template",
        default="stage3_target_kws_gsc_marvin_hardneg_aishell5_{split}.jsonl",
    )
    parser.add_argument(
        "--fixed-enrollment-wav",
        type=Path,
        default=REPO_ROOT
        / "data/raw/librispeech/extracted/LibriSpeech/test-clean/1089/134686/1089-134686-0014.flac",
    )
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "logs/compact_speaker_verifier_cabin_cache_diagnostic_2026-10-12.json",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    cache_dir = resolve(args.cache_dir)
    manifest_root = resolve(args.manifest_root)
    num_samples = max(1, int(round(float(args.sample_rate) * float(args.seconds))))
    fixed_wav, _ = read_mono(resolve(args.fixed_enrollment_wav), int(args.sample_rate))
    fixed_enrollment = segment_waveform(fixed_wav, num_samples, training=False)

    model = CompactSpeakerVerifier(
        embedding_dim=128,
        hidden=128,
        sample_rate=int(args.sample_rate),
        n_mels=40,
        n_fft=400,
        hop=160,
    ).to(device)
    load_checkpoint(resolve(args.checkpoint), model)

    index = SourceIndex()
    splits: dict[str, Any] = {}
    split_scores: dict[str, np.ndarray] = {}
    split_labels: dict[str, np.ndarray] = {}
    split_groups: dict[str, np.ndarray] = {}
    for split in ("val", "test"):
        manifest = manifest_root / split / str(args.manifest_template).format(split=split)
        rows = load_rows(manifest)
        extracted = np.load(cache_dir / f"{split}_wav_16000hz.npy", mmap_mode="r", allow_pickle=False)
        if len(rows) != len(extracted):
            raise ValueError(f"{split}: manifest/cache lengths differ")
        enrollment = np.zeros((len(rows), num_samples), dtype=np.float32)
        labels = np.zeros(len(rows), dtype=np.int64)
        groups = np.empty(len(rows), dtype=object)
        for position, row in enumerate(rows):
            source = index.get(str(row["source_manifest"]), str(row["source_id"]))
            if source is None:
                enrollment[position] = fixed_enrollment
                groups[position] = str(row.get("source_domain", row["source_manifest"]))
            else:
                wav, _ = read_mono(Path(source["enroll_wav"]), int(args.sample_rate))
                enrollment[position] = segment_waveform(wav, num_samples, training=False)
                groups[position] = Path(str(row["source_manifest"])).stem
            labels[position] = int(row.get("label", 0))
        scores = score_pairs(model, enrollment, extracted, device, int(args.batch_size))
        split_scores[split] = scores
        split_labels[split] = labels
        split_groups[split] = groups
        splits[split] = {
            "records": int(len(rows)),
            "positive_count": int((labels == 1).sum()),
            "negative_count": int((labels == 0).sum()),
            "roc": roc_summary(scores, labels),
        }

    selected = choose_threshold(split_scores["val"], split_labels["val"], float(args.target_fpr))
    test_at_selected = threshold_metrics(
        split_scores["test"], split_labels["test"], selected["threshold"]
    )
    for split in ("val", "test"):
        splits[split]["at_selected_threshold"] = threshold_metrics(
            split_scores[split], split_labels[split], selected["threshold"]
        )
        splits[split]["by_group"] = group_metrics(
            split_scores[split], split_labels[split], split_groups[split], selected["threshold"]
        )
        splits[split]["threshold_sweep"] = threshold_sweep(
            split_scores[split],
            split_labels[split],
            split_groups[split],
            np.arange(0.05, 1.0, 0.05),
        )

    output: dict[str, Any] = {
        "benchmark": "compact_speaker_verifier_cabin_cache_diagnostic",
        "checkpoint": str(resolve(args.checkpoint)),
        "cache_dir": str(cache_dir),
        "device": str(device),
        "target_fpr": float(args.target_fpr),
        "validation_selected_threshold": selected,
        "test_at_validation_operating_point": test_at_selected,
        "splits": splits,
        "data_contract": {
            "source": "stage3_target_kws_gsc_marvin_hardneg_aishell5 cache",
            "real_cabin_evidence": "AISHELL-5 Train/Dev/Eval1 target-absent windows",
            "real_driver_evidence": False,
            "note": "target-absent only; fixed enrollment speaker is absent from AISHELL-5",
        },
        "decision": "diagnostic_only; no checkpoint or pipeline default changed",
    }
    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    summary = {
        "output": str(output_path),
        "val_auc": splits["val"]["roc"]["auc"],
        "test_auc": splits["test"]["roc"]["auc"],
        "val_threshold": selected["threshold"],
        "test_tpr": test_at_selected["tpr"],
        "test_fpr": test_at_selected["fpr"],
        "groups": list(splits["test"]["by_group"].keys()),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
