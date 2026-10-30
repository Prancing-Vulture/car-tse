#!/usr/bin/env python
"""Build a small, append-only overlap-augmented intent cache.

The retained Stage-4 cache contains Stage-1 output for clean command examples
with the existing cabin-noise distribution.  This utility adds passenger
speech *after* that frozen Stage-1 representation so an intent candidate can
learn the co-channel failure mode without changing the retained extractor.

This is an engineering adaptation experiment, not a replacement for training
Stage 1 on spatially separated, permitted cabin recordings.  The passenger
speech is drawn from the held-out KWS speech pool and is never used as an
intent label.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from car_tse.config import REPO_ROOT
from car_tse.io.audio import read_mono
from car_tse.io.manifest import read_jsonl, write_jsonl


def balanced_indices(rows: list[dict], limit: int, seed: int) -> list[int]:
    """Select a deterministic, approximately class-balanced subset."""

    if limit <= 0:
        return []
    groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[int(row["label"])].append(index)
    rng = np.random.default_rng(seed)
    for values in groups.values():
        rng.shuffle(values)
    labels = sorted(groups)
    selected: list[int] = []
    cursors = {label: 0 for label in labels}
    while len(selected) < min(limit, len(rows)):
        progressed = False
        for label in labels:
            cursor = cursors[label]
            values = groups[label]
            if cursor >= len(values):
                continue
            selected.append(values[cursor])
            cursors[label] += 1
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return selected


def fit_start(wav: np.ndarray, samples: int) -> np.ndarray:
    out = np.zeros(max(1, int(samples)), dtype=np.float32)
    values = np.asarray(wav, dtype=np.float32).reshape(-1)
    out[: min(len(values), len(out))] = values[: len(out)]
    return out


def add_at(destination: np.ndarray, source: np.ndarray, start: int) -> None:
    start = max(0, int(start))
    if start >= len(destination) or len(source) == 0:
        return
    end = min(len(destination), start + len(source))
    destination[start:end] += source[: end - start]


def rms(wav: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(wav, dtype=np.float32) ** 2) + 1.0e-8))


def load_passenger_pool(path: Path, sample_rate: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rows = read_jsonl(path)
    wake_rows = [row for row in rows if int(row.get("label", 0)) == 1 and not row.get("noise_only", False)]
    nonwake_rows = [row for row in rows if int(row.get("label", 0)) == 0 and not row.get("noise_only", False)]
    if not wake_rows or not nonwake_rows:
        raise ValueError("The passenger manifest must contain speech rows for labels 0 and 1")
    wake = [read_mono(row["wav"], sample_rate)[0].astype(np.float32) for row in wake_rows]
    nonwake = [read_mono(row["wav"], sample_rate)[0].astype(np.float32) for row in nonwake_rows]
    return wake, nonwake


def make_overlap(
    base: np.ndarray,
    passenger_wake: np.ndarray,
    passenger_nonwake: np.ndarray,
    sample_rate: int,
    passenger_snr_db: float,
) -> np.ndarray:
    """Mix two passenger utterances into a fixed two-second command crop."""

    target = np.asarray(base, dtype=np.float32).reshape(-1).copy()
    passenger = np.zeros_like(target)
    add_at(passenger, fit_start(passenger_wake, sample_rate), int(round(0.30 * sample_rate)))
    add_at(passenger, fit_start(passenger_nonwake, int(round(1.50 * sample_rate))), int(round(0.95 * sample_rate)))
    target_rms = rms(target)
    passenger_rms = rms(passenger)
    if target_rms > 1.0e-6 and passenger_rms > 1.0e-6:
        passenger *= target_rms * (10.0 ** (-float(passenger_snr_db) / 20.0)) / passenger_rms
    mixed = target + passenger
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.95:
        mixed *= 0.95 / peak
    return mixed.astype(np.float32)


def build_split(
    split: str,
    rows: list[dict],
    base_cache: np.ndarray,
    output_dir: Path,
    manifest_out: Path,
    passenger_wake: list[np.ndarray],
    passenger_nonwake: list[np.ndarray],
    sample_rate: int,
    clean_limit: int,
    overlap_limit: int,
    passenger_snr_db: float,
    seed: int,
) -> None:
    clean_indices = balanced_indices(rows, clean_limit, seed)
    overlap_indices = balanced_indices(rows, overlap_limit, seed + 17)
    selected = [(index, "stage1_tse_cabin") for index in clean_indices]
    selected.extend((index, "stage1_tse_cabin_passenger_overlap") for index in overlap_indices)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}_overlap_wav_{sample_rate}hz.npy"
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(selected), int(base_cache.shape[1])),
    )
    manifest_rows: list[dict] = []
    for output_index, (source_index, condition) in enumerate(selected):
        base = np.asarray(base_cache[source_index], dtype=np.float32)
        if condition.endswith("overlap"):
            wake = passenger_wake[(output_index + seed) % len(passenger_wake)]
            nonwake = passenger_nonwake[(output_index * 3 + seed) % len(passenger_nonwake)]
            values = make_overlap(base, wake, nonwake, sample_rate, passenger_snr_db)
        else:
            values = base.copy()
        output[output_index] = values
        row = dict(rows[source_index])
        row.update(
            {
                "condition": condition,
                "source_cache_index": int(source_index),
                "passenger_snr_db": float(passenger_snr_db) if condition.endswith("overlap") else None,
                "overlap_passenger_wake": bool(condition.endswith("overlap")),
            }
        )
        manifest_rows.append(row)
        if (output_index + 1) % 512 == 0 or output_index + 1 == len(selected):
            print(f"{split}: wrote {output_index + 1}/{len(selected)}", flush=True)
    output.flush()
    del output
    write_jsonl(manifest_out, manifest_rows)
    print(f"Wrote {split}: {len(manifest_rows)} rows -> {output_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache-dir", type=Path, default=REPO_ROOT / "data/cache/stage4_intent_tse_domain_16k_v4")
    parser.add_argument("--train-manifest", type=Path, default=REPO_ROOT / "manifests/train/stage4_intent_tse_domain_train.jsonl")
    parser.add_argument("--val-manifest", type=Path, default=REPO_ROOT / "manifests/val/stage4_intent_tse_domain_eval.jsonl")
    parser.add_argument("--passenger-manifest", type=Path, default=REPO_ROOT / "manifests/test/stage3_kws_cabin_test.jsonl")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/cache/stage4_intent_overlap_ft1_16k")
    parser.add_argument("--train-manifest-out", type=Path, default=REPO_ROOT / "manifests/train/stage4_intent_overlap_ft1.jsonl")
    parser.add_argument("--val-manifest-out", type=Path, default=REPO_ROOT / "manifests/val/stage4_intent_overlap_ft1.jsonl")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--train-clean-limit", type=int, default=2048)
    parser.add_argument("--train-overlap-limit", type=int, default=2048)
    parser.add_argument("--val-clean-limit", type=int, default=512)
    parser.add_argument("--val-overlap-limit", type=int, default=512)
    parser.add_argument("--passenger-snr-db", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()

    train_rows = read_jsonl(args.train_manifest)
    val_rows = read_jsonl(args.val_manifest)
    train_cache = np.load(args.base_cache_dir / f"train_cabin_wav_{args.sample_rate}hz.npy", mmap_mode="r", allow_pickle=False)
    val_cache = np.load(args.base_cache_dir / f"val_cabin_wav_{args.sample_rate}hz.npy", mmap_mode="r", allow_pickle=False)
    if train_cache.shape[0] != len(train_rows) or val_cache.shape[0] != len(val_rows):
        raise ValueError("Base cache and manifest lengths do not match")
    passenger_wake, passenger_nonwake = load_passenger_pool(args.passenger_manifest, args.sample_rate)
    print(
        f"train_rows={len(train_rows)} val_rows={len(val_rows)} "
        f"passenger_wake={len(passenger_wake)} passenger_nonwake={len(passenger_nonwake)}",
        flush=True,
    )
    build_split(
        "train", train_rows, train_cache, args.output_dir, args.train_manifest_out,
        passenger_wake, passenger_nonwake, args.sample_rate, args.train_clean_limit,
        args.train_overlap_limit, args.passenger_snr_db, args.seed,
    )
    build_split(
        "val", val_rows, val_cache, args.output_dir, args.val_manifest_out,
        passenger_wake, passenger_nonwake, args.sample_rate, args.val_clean_limit,
        args.val_overlap_limit, args.passenger_snr_db, args.seed + 101,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
