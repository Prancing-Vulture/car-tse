#!/usr/bin/env python
"""Calibrate the frozen SpEx speaker similarity on speaker-disjoint pairs.

This is a bounded speaker-verification study.  It does not retrain Stage 1
and it does not claim cabin-domain performance.  It fits a one-dimensional
logistic calibrator to cosine similarity scores from LibriSpeech speaker
pairs, with train/validation/test speakers kept disjoint.  The resulting
``center`` and ``scale`` can be used by the opt-in late-fusion runtime, where
``sigmoid((cosine-center)/scale)`` is the speaker branch score.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.segment import segment_waveform
from car_tse.io.audio import read_mono
from car_tse.models.stage1_system import Stage1System
from car_tse.train.utils import get_device, load_checkpoint


def _softplus_inverse(value: float) -> float:
    value = max(float(value), 1.0e-6)
    return float(math.log(math.expm1(value)))


def load_or_compute_embeddings(
    split_data: dict[str, Any],
    split_names: tuple[str, ...],
    *,
    sample_rate: int,
    segment_seconds: float,
    utterances_per_speaker: int,
    max_speakers: int,
    batch_size: int,
    seed: int,
    stage1: Stage1System,
    device: torch.device,
) -> list[dict[str, Any]]:
    rng = random.Random(int(seed))
    selected: list[tuple[str, str, str]] = []
    for split in split_names:
        speakers = list(split_data.get(split, []))
        if max_speakers > 0:
            speakers = speakers[: int(max_speakers)]
        for speaker in speakers:
            paths = list(dict.fromkeys(
                str(path) for path in split_data["utterances"].get(str(speaker), [])
            ))
            paths = sorted(paths)
            if len(paths) < 2:
                continue
            rng.shuffle(paths)
            for path in paths[: max(2, int(utterances_per_speaker))]:
                selected.append((split, str(speaker), path))

    if not selected:
        raise RuntimeError("No eligible LibriSpeech utterances were selected")

    segment_samples = max(1, int(round(float(segment_seconds) * sample_rate)))
    records: list[dict[str, Any]] = []
    for offset in range(0, len(selected), max(1, int(batch_size))):
        batch_info = selected[offset : offset + max(1, int(batch_size))]
        waveforms: list[np.ndarray] = []
        valid_info: list[tuple[str, str, str]] = []
        for split, speaker, path in batch_info:
            try:
                wav, _ = read_mono(Path(path), sample_rate)
            except (OSError, RuntimeError, ValueError) as exc:
                print(f"skip={path}: {exc}", file=sys.stderr, flush=True)
                continue
            waveforms.append(segment_waveform(wav, segment_samples, training=False))
            valid_info.append((split, speaker, path))
        if not waveforms:
            continue
        batch = torch.from_numpy(np.stack(waveforms, axis=0)).to(device)
        with torch.inference_mode():
            embeddings = stage1.speaker_embedding(batch)
        if embeddings is None:
            raise RuntimeError("Stage 1 backend does not expose speaker embeddings")
        embeddings = torch.nn.functional.normalize(embeddings.float(), dim=-1).cpu().numpy()
        for (split, speaker, path), embedding in zip(valid_info, embeddings, strict=True):
            records.append(
                {
                    "split": split,
                    "speaker": speaker,
                    "path": path,
                    "embedding": embedding.astype(np.float32),
                }
            )
        if offset == 0 or (offset + len(batch_info)) % 200 == 0 or offset + len(batch_info) >= len(selected):
            print(f"embedded={len(records)}/{len(selected)}", flush=True)
    return records


def build_manifest_split_data(manifest_root: Path, manifest_name: str) -> dict[str, Any]:
    """Build the same split structure from speaker-labeled JSONL manifests."""

    split_data: dict[str, Any] = {"train": [], "val": [], "test": [], "utterances": {}}
    for split in ("train", "val", "test"):
        path = manifest_root / split / manifest_name
        if not path.exists():
            raise FileNotFoundError(path)
        by_speaker: dict[str, set[str]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            speaker = str(row.get("speaker_id", ""))
            audio_path = row.get("enroll_wav") or row.get("wav")
            if not speaker or not audio_path:
                continue
            by_speaker.setdefault(speaker, set()).add(str(audio_path))
        split_data[split] = sorted(by_speaker)
        for speaker, paths in by_speaker.items():
            split_data["utterances"].setdefault(speaker, [])
            split_data["utterances"][speaker].extend(sorted(paths))
    return split_data


def make_pairs(
    records: list[dict[str, Any]],
    *,
    pairs_per_speaker: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(int(seed))
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_speaker.setdefault(str(record["speaker"]), []).append(record)
    speakers = sorted(by_speaker)
    if len(speakers) < 2:
        raise RuntimeError("At least two speakers are required to create verification pairs")

    scores: list[float] = []
    labels: list[int] = []
    for speaker in speakers:
        items = by_speaker[speaker]
        if len(items) < 2:
            continue
        for pair_index in range(max(1, int(pairs_per_speaker))):
            first, second = rng.sample(items, 2)
            scores.append(float(np.dot(first["embedding"], second["embedding"])))
            labels.append(1)

            other_speaker = rng.choice([candidate for candidate in speakers if candidate != speaker])
            other = rng.choice(by_speaker[other_speaker])
            scores.append(float(np.dot(first["embedding"], other["embedding"])))
            labels.append(0)
    if not scores or not any(labels) or all(labels):
        raise RuntimeError("Pair construction did not produce both classes")
    return np.asarray(scores, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def fit_logistic_calibrator(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    steps: int = 500,
    learning_rate: float = 0.05,
) -> tuple[float, float, float]:
    """Fit ``sigmoid(positive_weight * score + bias)`` and return weight/bias/loss."""

    score_tensor = torch.from_numpy(np.asarray(scores, dtype=np.float32))
    label_tensor = torch.from_numpy(np.asarray(labels, dtype=np.float32))
    log_weight = torch.nn.Parameter(torch.tensor(_softplus_inverse(5.0)))
    bias = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.Adam([log_weight, bias], lr=float(learning_rate))
    for _ in range(max(1, int(steps))):
        optimizer.zero_grad(set_to_none=True)
        weight = torch.nn.functional.softplus(log_weight)
        logits = weight * score_tensor + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label_tensor)
        loss.backward()
        optimizer.step()
    weight = float(torch.nn.functional.softplus(log_weight).detach().item())
    bias_value = float(bias.detach().item())
    return weight, bias_value, float(loss.detach().item())


def threshold_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    positives = labels == 1
    negatives = labels == 0
    predicted = scores >= float(threshold)
    tp = int(np.logical_and(predicted, positives).sum())
    fp = int(np.logical_and(predicted, negatives).sum())
    fn = int(np.logical_and(~predicted, positives).sum())
    tn = int(np.logical_and(~predicted, negatives).sum())
    return {
        "threshold": float(threshold),
        "tpr": float(tp / max(1, int(positives.sum()))),
        "fpr": float(fp / max(1, int(negatives.sum()))),
        "precision": float(tp / max(1, tp + fp)),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def roc_summary(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    thresholds = np.unique(np.concatenate((scores, np.asarray([-np.inf, np.inf], dtype=np.float32))))
    rows = [threshold_metrics(scores, labels, float(threshold)) for threshold in thresholds]
    rows.sort(key=lambda row: row["fpr"])
    fpr = np.asarray([row["fpr"] for row in rows], dtype=np.float64)
    tpr = np.asarray([row["tpr"] for row in rows], dtype=np.float64)
    order = np.argsort(fpr, kind="stable")
    fpr = fpr[order]
    tpr = tpr[order]
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    fnr = 1.0 - tpr
    eer_index = int(np.argmin(np.abs(fpr - fnr)))
    return {
        "auc": auc,
        "eer": float((fpr[eer_index] + fnr[eer_index]) / 2.0),
        "eer_threshold": float(rows[eer_index]["threshold"]),
        "positive_count": float((labels == 1).sum()),
        "negative_count": float((labels == 0).sum()),
    }


def choose_threshold(scores: np.ndarray, labels: np.ndarray, max_fpr: float) -> dict[str, float]:
    candidates = np.unique(np.asarray(scores, dtype=np.float32))
    rows = [threshold_metrics(scores, labels, float(threshold)) for threshold in candidates]
    safe = [row for row in rows if row["fpr"] <= float(max_fpr)]
    if not safe:
        return min(rows, key=lambda row: (row["fpr"], -row["tpr"]))
    return max(safe, key=lambda row: (row["tpr"], row["precision"], row["threshold"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("librispeech", "gsc_marvin"), default="librispeech")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.yaml")
    parser.add_argument("--stage1-config", type=Path, default=REPO_ROOT / "configs/stage1_spex_dual_domain_rms010_ft1.yaml")
    parser.add_argument("--stage1-checkpoint", type=Path, default=REPO_ROOT / "checkpoints/stage1_spex_dual_domain_rms010_ft1/best.pt")
    parser.add_argument("--speaker-split", type=Path, default=REPO_ROOT / "manifests/meta/librispeech_speakers.json")
    parser.add_argument("--manifest-root", type=Path, default=REPO_ROOT / "manifests")
    parser.add_argument("--manifest-name", default="stage1_gsc_marvin_cabin.jsonl")
    parser.add_argument("--cache", type=Path, default=REPO_ROOT / "data/cache/speaker_verifier_librispeech_stage1_ft1.npz")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "logs/speaker_verifier_calibration_librispeech_stage1_ft1.json")
    parser.add_argument("--utterances-per-speaker", type=int, default=6)
    parser.add_argument("--pairs-per-speaker", type=int, default=8)
    parser.add_argument("--max-speakers", type=int, default=0, help="0 uses all speakers in each split")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--segment-seconds", type=float, default=3.0)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--fit-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.source == "gsc_marvin":
        manifest_root = args.manifest_root if args.manifest_root.is_absolute() else REPO_ROOT / args.manifest_root
        split_path = manifest_root / "{train,val,test}" / args.manifest_name
        split_data = build_manifest_split_data(manifest_root, args.manifest_name)
    else:
        split_path = args.speaker_split if args.speaker_split.is_absolute() else REPO_ROOT / args.speaker_split
        split_data = json.loads(split_path.read_text(encoding="utf-8"))
    stage1_cfg = load_stage_config(args.stage1_config, args.paths)
    pretrained = Path(stage1_cfg["pretrained_checkpoint"])
    if not pretrained.is_absolute():
        pretrained = REPO_ROOT / pretrained
    checkpoint = args.stage1_checkpoint if args.stage1_checkpoint.is_absolute() else REPO_ROOT / args.stage1_checkpoint
    device = get_device()
    stage1 = Stage1System(
        spex_cfg=stage1_cfg["spex"],
        sample_rate=int(stage1_cfg["sample_rate"]),
        use_cabin_frontend=False,
        pretrained_dir=str(pretrained),
        backend=stage1_cfg.get("spex_backend", "transfer_spexplus_16k"),
    ).to(device)
    load_checkpoint(checkpoint, stage1)
    stage1.eval()

    cache_path = args.cache if args.cache.is_absolute() else REPO_ROOT / args.cache
    if cache_path.exists() and not args.force:
        cached = np.load(cache_path, allow_pickle=True)
        records = [
            {
                "split": str(split),
                "speaker": str(speaker),
                "path": str(path),
                "embedding": embedding.astype(np.float32),
            }
            for split, speaker, path, embedding in zip(
                cached["split"], cached["speaker"], cached["path"], cached["embedding"], strict=True
            )
        ]
        print(f"loaded_cache={cache_path} records={len(records)}", flush=True)
    else:
        records = load_or_compute_embeddings(
            split_data,
            ("train", "val", "test"),
            sample_rate=int(stage1_cfg["sample_rate"]),
            segment_seconds=float(args.segment_seconds),
            utterances_per_speaker=int(args.utterances_per_speaker),
            max_speakers=int(args.max_speakers),
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            stage1=stage1,
            device=device,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            split=np.asarray([row["split"] for row in records]),
            speaker=np.asarray([row["speaker"] for row in records]),
            path=np.asarray([row["path"] for row in records]),
            embedding=np.stack([row["embedding"] for row in records], axis=0),
        )
        print(f"wrote_cache={cache_path} records={len(records)}", flush=True)

    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for record in records:
        by_split.setdefault(str(record["split"]), []).append(record)
    train_scores, train_labels = make_pairs(by_split["train"], pairs_per_speaker=int(args.pairs_per_speaker), seed=int(args.seed))
    val_scores, val_labels = make_pairs(by_split["val"], pairs_per_speaker=int(args.pairs_per_speaker), seed=int(args.seed) + 1)
    test_scores, test_labels = make_pairs(by_split["test"], pairs_per_speaker=int(args.pairs_per_speaker), seed=int(args.seed) + 2)
    weight, bias, train_loss = fit_logistic_calibrator(
        train_scores,
        train_labels,
        steps=int(args.fit_steps),
    )
    center = -bias / max(weight, 1.0e-8)
    scale = 1.0 / max(weight, 1.0e-8)

    calibrated_val = 1.0 / (1.0 + np.exp(-(weight * val_scores + bias)))
    calibrated_test = 1.0 / (1.0 + np.exp(-(weight * test_scores + bias)))
    selected = choose_threshold(calibrated_val, val_labels, float(args.target_fpr))
    test_at_selected = threshold_metrics(calibrated_test, test_labels, selected["threshold"])
    raw_test = roc_summary(test_scores, test_labels)
    calibrated_test_summary = roc_summary(calibrated_test, test_labels)
    output = {
        "benchmark": "speaker_verifier_calibration_frozen_spex_librispeech",
        "device": str(device),
        "stage1_config": str(args.stage1_config),
        "stage1_checkpoint": str(checkpoint),
        "speaker_split": str(split_path),
        "dataset": "LibriSpeech" if args.source == "librispeech" else "Speech Commands v2 Marvin",
        "source": str(args.source),
        "data_contract": {
            "same_speaker_positive_pairs": True,
            "different_speaker_negative_pairs": True,
            "speaker_disjoint_train_val_test": True,
            "cabin_domain": False,
            "real_driver_evidence": False,
        },
        "sampling": {
            "segment_seconds": float(args.segment_seconds),
            "utterances_per_speaker": int(args.utterances_per_speaker),
            "pairs_per_speaker": int(args.pairs_per_speaker),
            "max_speakers": int(args.max_speakers),
            "seed": int(args.seed),
        },
        "record_counts": {
            split: {
                "records": len(rows),
                "speakers": len({str(row["speaker"]) for row in rows}),
            }
            for split, rows in by_split.items()
        },
        "pair_counts": {
            "train": int(len(train_scores)),
            "val": int(len(val_scores)),
            "test": int(len(test_scores)),
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
        "test_raw_cosine": raw_test,
        "test_calibrated_probability": calibrated_test_summary,
        "decision": (
            "calibration_candidate_only; requires same-driver cabin tuning before runtime promotion"
        ),
    }
    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "center": center,
        "scale": scale,
        "val_tpr": selected["tpr"],
        "val_fpr": selected["fpr"],
        "test_tpr": test_at_selected["tpr"],
        "test_fpr": test_at_selected["fpr"],
        "test_auc": calibrated_test_summary["auc"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
