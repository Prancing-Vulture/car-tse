#!/usr/bin/env python
"""Calibrate frozen SpEx similarity on cached cabin-domain mixtures.

This is a calibration-only study.  It does not retrain Stage 1 and it does
not promote a threshold into the runtime.  The cache is produced by the
Stage-3 target-KWS preparation job and contains an enrollment embedding, an
extracted embedding, and a target-present/target-absent label for each
synthetic cabin mixture.

Compared with clean LibriSpeech or isolated Speech Commands pairs, this
measurement is closer to the runtime path because the second embedding is
computed from the extracted mixture.  It is still not real-driver evidence:
the data are synthetic CAVEMOVE/GSC-Marvin cabin mixtures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.calibrate_spex_speaker_verifier import (  # noqa: E402
    choose_threshold,
    fit_logistic_calibrator,
    roc_summary,
    threshold_metrics,
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _load_split(
    cache_root: Path,
    manifest_root: Path,
    manifest_template: str,
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    enrollment_path = cache_root / f"{split}_enrollment_embedding.npy"
    extracted_path = cache_root / f"{split}_extracted_embedding.npy"
    manifest_path = manifest_root / split / manifest_template.format(split=split)
    if not enrollment_path.exists():
        raise FileNotFoundError(enrollment_path)
    if not extracted_path.exists():
        raise FileNotFoundError(extracted_path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    enrollment = np.asarray(np.load(enrollment_path, mmap_mode="r"), dtype=np.float32)
    extracted = np.asarray(np.load(extracted_path, mmap_mode="r"), dtype=np.float32)
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != len(enrollment) or len(rows) != len(extracted):
        raise ValueError(
            f"{split}: manifest/cache length mismatch: "
            f"rows={len(rows)} enrollment={len(enrollment)} extracted={len(extracted)}"
        )
    enrollment_norm = enrollment / np.maximum(
        np.linalg.norm(enrollment, axis=1, keepdims=True), 1.0e-8
    )
    extracted_norm = extracted / np.maximum(
        np.linalg.norm(extracted, axis=1, keepdims=True), 1.0e-8
    )
    scores = np.sum(enrollment_norm * extracted_norm, axis=1).astype(np.float32)
    labels = np.asarray([int(row.get("label", 0)) for row in rows], dtype=np.float32)
    return scores, labels, enrollment_norm, rows


def _score_summary(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    output: dict[str, Any] = {
        "records": int(len(scores)),
        "positive_count": int((labels == 1).sum()),
        "negative_count": int((labels == 0).sum()),
        "roc": roc_summary(scores, labels),
    }
    for label, key in ((1, "positive"), (0, "negative")):
        subset = scores[labels == label]
        output[key] = {
            "mean": float(subset.mean()) if len(subset) else None,
            "std": float(subset.std()) if len(subset) else None,
            "quantiles": (
                np.quantile(subset, [0.01, 0.05, 0.50, 0.95, 0.99]).astype(float).tolist()
                if len(subset)
                else []
            ),
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=REPO_ROOT / "data/cache/stage3_target_kws_gsc_marvin_ft2_stage1_16k",
    )
    parser.add_argument("--manifest-root", type=Path, default=REPO_ROOT / "manifests")
    parser.add_argument(
        "--manifest-template",
        default="stage3_target_kws_gsc_marvin_ft2_stage1_{split}.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "logs/speaker_verifier_calibration_gsc_cache_stage1_ft2.json",
    )
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--fit-steps", type=int, default=500)
    args = parser.parse_args()

    cache_root = _resolve(args.cache_root)
    manifest_root = _resolve(args.manifest_root)
    split_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]] = {}
    for split in ("train", "val", "test"):
        split_data[split] = _load_split(
            cache_root,
            manifest_root,
            str(args.manifest_template),
            split,
        )

    train_scores, train_labels = split_data["train"][:2]
    val_scores, val_labels = split_data["val"][:2]
    test_scores, test_labels = split_data["test"][:2]
    weight, bias, train_loss = fit_logistic_calibrator(
        train_scores,
        train_labels,
        steps=int(args.fit_steps),
    )
    center = -bias / max(weight, 1.0e-8)
    scale = 1.0 / max(weight, 1.0e-8)

    def calibrate(scores: np.ndarray) -> np.ndarray:
        logits = weight * scores + bias
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

    calibrated_val = calibrate(val_scores)
    calibrated_test = calibrate(test_scores)
    selected = choose_threshold(calibrated_val, val_labels, float(args.target_fpr))
    test_at_selected = threshold_metrics(calibrated_test, test_labels, selected["threshold"])
    output = {
        "benchmark": "speaker_verifier_calibration_cached_cabin_mixtures",
        "cache_root": str(cache_root),
        "manifest_root": str(manifest_root),
        "manifest_template": str(args.manifest_template),
        "dataset": "synthetic CAVEMOVE/GSC-Marvin cabin mixtures",
        "data_contract": {
            "enrollment_embedding": "frozen Stage-1 enrollment branch",
            "extracted_embedding": "frozen Stage-1 embedding of target-KWS extracted mixture",
            "positive_label": "target speaker present",
            "negative_label": "target speaker absent",
            "split_selection": "existing train/val/test manifests; no threshold fit on test",
            "real_driver_evidence": False,
            "real_cabin_evidence": False,
        },
        "calibrator": {
            "form": "sigmoid((cosine-center)/scale)",
            "weight": float(weight),
            "bias": float(bias),
            "center": float(center),
            "scale": float(scale),
            "train_loss": float(train_loss),
        },
        "validation_operating_point": {
            "target_fpr": float(args.target_fpr),
            **selected,
        },
        "test_at_validation_operating_point": test_at_selected,
        "splits": {
            split: _score_summary(scores, labels)
            for split, (scores, labels, _enrollment, _rows) in split_data.items()
        },
        "decision": (
            "calibration_candidate_only; runtime promotion requires a separate "
            "speaker-consistent cabin tuning/test study"
        ),
    }
    output_path = _resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "center": center,
                "scale": scale,
                "val_tpr": selected["tpr"],
                "val_fpr": selected["fpr"],
                "test_tpr": test_at_selected["tpr"],
                "test_fpr": test_at_selected["fpr"],
                "test_auc": output["splits"]["test"]["roc"]["auc"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
