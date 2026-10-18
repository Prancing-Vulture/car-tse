#!/usr/bin/env python
"""Build train-only English target/passenger mixtures for Stage-1 adaptation.

This creates a small, speaker-disjoint research candidate set from FSC command
utterances and the train-only cabin speech pool.  The target command is kept as
the supervision signal while passenger wake/non-wake clips are mixed into the
same four-channel input.  All four channels are intentionally identical here:
the data targets co-channel robustness, not spatial separation.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import REPO_ROOT
from car_tse.io.audio import read_mono
from car_tse.io.manifest import read_jsonl, write_jsonl


def fit_clip(wav: np.ndarray, samples: int, start: int = 0) -> np.ndarray:
    out = np.zeros(int(samples), dtype=np.float32)
    values = np.asarray(wav, dtype=np.float32).reshape(-1)
    start = max(0, int(start))
    if start < len(out):
        end = min(len(out), start + len(values))
        out[start:end] = values[: end - start]
    return out


def add_at(destination: np.ndarray, source: np.ndarray, start: int) -> None:
    start = max(0, int(start))
    if start >= len(destination):
        return
    end = min(len(destination), start + len(source))
    destination[start:end] += source[: end - start]


def rms(wav: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(wav, dtype=np.float32) ** 2) + 1.0e-8))


def speaker_id(row: dict) -> str:
    return Path(row["wav"]).parent.name


def select_balanced(rows: list[dict], limit: int, seed: int) -> list[dict]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        groups[int(row["label"])].append(row)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    selected: list[dict] = []
    labels = sorted(groups)
    cursors = {label: 0 for label in labels}
    while len(selected) < min(limit, len(rows)):
        progressed = False
        for label in labels:
            values = groups[label]
            cursor = cursors[label]
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


def build_split(
    rows: list[dict],
    passenger_wake: list[np.ndarray],
    passenger_nonwake: list[np.ndarray],
    output_dir: Path,
    manifest_out: Path,
    split: str,
    sample_rate: int,
    limit: int,
    seed: int,
    passenger_snr_db: float,
) -> None:
    rows = select_balanced(rows, limit, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_speaker[speaker_id(row)].append(row)
    rng = random.Random(seed)
    samples = 3 * sample_rate
    output_rows: list[dict] = []
    for index, row in enumerate(rows):
        target, _ = read_mono(row["wav"], sample_rate)
        same_speaker = [item for item in by_speaker[speaker_id(row)] if item["wav"] != row["wav"]]
        enrollment_row = same_speaker[0] if same_speaker else row
        enrollment, _ = read_mono(enrollment_row["wav"], sample_rate)
        target_clip = fit_clip(target, samples, int(0.35 * sample_rate))
        passenger = np.zeros(samples, dtype=np.float32)
        wake = fit_clip(passenger_wake[(index * 3 + seed) % len(passenger_wake)], int(0.9 * sample_rate))
        nonwake = fit_clip(passenger_nonwake[(index * 7 + seed) % len(passenger_nonwake)], int(1.0 * sample_rate))
        add_at(passenger, wake, int(0.75 * sample_rate))
        add_at(passenger, nonwake, int(1.85 * sample_rate))
        target_rms = rms(target_clip)
        passenger_rms = rms(passenger)
        if target_rms > 1e-6 and passenger_rms > 1e-6:
            passenger *= target_rms * (10.0 ** (-float(passenger_snr_db) / 20.0)) / passenger_rms
        mixture = target_clip + passenger
        peak = float(np.max(np.abs(mixture)))
        if peak > 0.95:
            mixture *= 0.95 / peak
            target_clip *= 0.95 / peak
        enrollment_clip = fit_clip(enrollment, samples // 2)
        stem = f"{split}_{index:05d}"
        mixture_path = output_dir / f"{stem}_mixture.wav"
        multichannel_path = output_dir / f"{stem}_multichannel.wav"
        target_path = output_dir / f"{stem}_target.wav"
        enrollment_path = output_dir / f"{stem}_enrollment.wav"
        sf.write(mixture_path, mixture.astype(np.float32), sample_rate, subtype="FLOAT")
        sf.write(multichannel_path, np.repeat(mixture[:, None], 4, axis=1).astype(np.float32), sample_rate, subtype="FLOAT")
        sf.write(target_path, target_clip.astype(np.float32), sample_rate, subtype="FLOAT")
        sf.write(enrollment_path, enrollment_clip.astype(np.float32), sample_rate, subtype="FLOAT")
        output_rows.append(
            {
                "id": stem,
                "mixture_wav": str(mixture_path),
                "mixture_multich_wav": str(multichannel_path),
                "enroll_wav": str(enrollment_path),
                "target_wav": str(target_path),
                "sample_rate": sample_rate,
                "speaker_id": speaker_id(row),
                "target_present": 1,
                "target_activity_class": 1,
                "condition": "fsc_target_passenger_cochannel",
                "meta": {
                    "target_source": row["wav"],
                    "passenger_snr_db": float(passenger_snr_db),
                    "passenger_pool": "stage3_kws_cabin_train",
                    "spatial_cues": False,
                },
            }
        )
    write_jsonl(manifest_out, output_rows)
    print(f"{split}: wrote {len(output_rows)} rows -> {manifest_out}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", type=Path, default=REPO_ROOT / "manifests/train/stage4_cavemove_intent.jsonl")
    parser.add_argument("--val-source", type=Path, default=REPO_ROOT / "manifests/val/stage4_cavemove_intent.jsonl")
    parser.add_argument("--passenger-manifest", type=Path, default=REPO_ROOT / "manifests/train/stage3_kws_cabin_train.jsonl")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "data/processed/stage1_fsc_passenger_mixtures_2026-10-16")
    parser.add_argument("--train-manifest-out", type=Path, default=REPO_ROOT / "manifests/train/stage1_fsc_passenger_mixtures_2026-10-16.jsonl")
    parser.add_argument("--val-manifest-out", type=Path, default=REPO_ROOT / "manifests/val/stage1_fsc_passenger_mixtures_2026-10-16.jsonl")
    parser.add_argument("--train-limit", type=int, default=768)
    parser.add_argument("--val-limit", type=int, default=256)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--passenger-snr-db", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    train_rows = read_jsonl(args.train_source)
    val_rows = read_jsonl(args.val_source)
    passenger_rows = read_jsonl(args.passenger_manifest)
    # Only a small deterministic pool is needed for augmentation.  Loading all
    # thousands of cabin clips is unnecessarily slow on Windows and does not
    # improve the mixture diversity for this bounded candidate.
    pool_rng = random.Random(args.seed ^ 0x13579BDF)
    wake_rows = [row for row in passenger_rows if int(row.get("label", 0)) == 1 and not row.get("noise_only", False)]
    nonwake_rows = [row for row in passenger_rows if int(row.get("label", 0)) == 0 and not row.get("noise_only", False)]
    pool_rng.shuffle(wake_rows)
    pool_rng.shuffle(nonwake_rows)
    passenger_rows = wake_rows[:128] + nonwake_rows[:256]
    wake = []
    nonwake = []
    for row in passenger_rows:
        if row.get("noise_only", False):
            continue
        wav, _ = read_mono(row["wav"], args.sample_rate)
        (wake if int(row.get("label", 0)) == 1 else nonwake).append(np.asarray(wav, dtype=np.float32))
    if not wake or not nonwake:
        raise RuntimeError("passenger manifest lacks both speech classes")
    build_split(train_rows, wake, nonwake, args.output_root / "train", args.train_manifest_out, "train", args.sample_rate, args.train_limit, args.seed, args.passenger_snr_db)
    build_split(val_rows, wake, nonwake, args.output_root / "val", args.val_manifest_out, "val", args.sample_rate, args.val_limit, args.seed + 101, args.passenger_snr_db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
