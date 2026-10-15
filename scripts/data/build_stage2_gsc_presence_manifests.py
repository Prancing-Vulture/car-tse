#!/usr/bin/env python
"""Prepare speaker-disjoint synthetic Stage 2 presence data from GSC + CAVEMOVE.

Each target speaker contributes paired present/absent scenes. In a present
scene the enrolled GSC speaker talks while a second speaker overlaps; in an
absent scene only the second speaker talks. CAVEMOVE is added later by
PresenceDataset, preserving the clean target track for frame-level labels.
This is a training/development proxy, not a substitute for in-car recordings.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.io.audio import read_mono


SAMPLE_RATE = 16_000
SEGMENT_SECONDS = 3
TARGET_START_SECONDS = 0.65
PASSENGER_START_SECONDS = 0.85


def speaker_key(path: Path) -> str:
    return path.stem.split("_nohash_")[0]


def fit_clip(wav: np.ndarray, samples: int, start: int) -> np.ndarray:
    output = np.zeros(samples, dtype=np.float32)
    clip = np.asarray(wav, dtype=np.float32).reshape(-1)
    end = min(samples, start + len(clip))
    if end > start:
        output[start:end] = clip[: end - start]
    return output


def rms(wav: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(wav, dtype=np.float32) ** 2) + 1e-8))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsc-root", type=Path, required=True)
    parser.add_argument("--noise-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True, help="Dedicated directory for generated WAV files")
    parser.add_argument("--manifest-root", type=Path, required=True, help="Root containing train/val/test")
    parser.add_argument("--speaker-limit", type=int, default=500)
    parser.add_argument("--copies-per-speaker", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()

    gsc_root = args.gsc_root.resolve()
    noise_root = args.noise_root.resolve()
    output_root = args.output_root.resolve()
    manifest_root = args.manifest_root.resolve()
    if not gsc_root.is_dir() or not noise_root.is_dir():
        raise FileNotFoundError(f"Missing GSC or CAVEMOVE root: {gsc_root}, {noise_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty generated-data directory: {output_root}")
    existing_manifests = [
        manifest_root / split / "stage2_presence_gsc_cavemove.jsonl"
        for split in ("train", "val", "test")
        if (manifest_root / split / "stage2_presence_gsc_cavemove.jsonl").exists()
    ]
    if existing_manifests:
        raise FileExistsError(f"Refusing to overwrite existing manifests: {existing_manifests}")

    by_speaker: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for label_dir in sorted(path for path in gsc_root.iterdir() if path.is_dir() and not path.name.startswith("_")):
        for wav_path in sorted(label_dir.glob("*.wav")):
            by_speaker[speaker_key(wav_path)].append((label_dir.name, wav_path))
    eligible = [(speaker, files) for speaker, files in by_speaker.items() if len(files) >= 2]
    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    if args.speaker_limit > 0:
        eligible = eligible[: args.speaker_limit]
    if len(eligible) < 30:
        raise RuntimeError(f"Only {len(eligible)} eligible GSC speakers found; need at least 30")

    noise_paths = sorted(
        path for path in noise_root.rglob("*.wav")
        if "noise" in path.parts or "ventilation" in path.parts
    )
    if not noise_paths:
        raise RuntimeError(f"No CAVEMOVE noise/ventilation WAVs found under {noise_root}")

    train_count = max(1, int(round(0.70 * len(eligible))))
    val_count = max(1, int(round(0.15 * len(eligible))))
    splits = {
        "train": eligible[:train_count],
        "val": eligible[train_count : train_count + val_count],
        "test": eligible[train_count + val_count :],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    silence_by_split: dict[str, Path] = {}
    snrs = (-5.0, 0.0, 5.0, 10.0)
    samples = SAMPLE_RATE * SEGMENT_SECONDS
    target_start = int(TARGET_START_SECONDS * SAMPLE_RATE)
    passenger_start = int(PASSENGER_START_SECONDS * SAMPLE_RATE)

    for split_name, groups in splits.items():
        if not groups:
            continue
        split_output = output_root / split_name
        split_output.mkdir(parents=True, exist_ok=True)
        silence_path = split_output / "silence_3s.wav"
        sf.write(str(silence_path), np.zeros(samples, dtype=np.float32), SAMPLE_RATE, subtype="FLOAT")
        silence_by_split[split_name] = silence_path

        rows: list[dict] = []
        for speaker_index, (target_speaker, target_files) in enumerate(groups):
            # All GSC command classes are treated as ordinary target speech;
            # only the enrolled identity determines presence.
            other_enrollment_files = [item for item in target_files if item[1] != target_files[0][1]]
            passenger_speakers = [item for item in groups if item[0] != target_speaker]
            for copy_index in range(max(1, args.copies_per_speaker)):
                target_label, target_path = target_files[(2 * copy_index) % len(target_files)]
                enroll_index = (2 * copy_index + 1) % len(target_files)
                _, enroll_path = target_files[enroll_index]
                if enroll_path == target_path:
                    enroll_path = other_enrollment_files[0][1]

                passenger_speaker, passenger_files = passenger_speakers[
                    (speaker_index + copy_index) % len(passenger_speakers)
                ]
                passenger_label, passenger_path = passenger_files[(speaker_index + copy_index) % len(passenger_files)]
                target, _ = read_mono(target_path, SAMPLE_RATE)
                passenger, _ = read_mono(passenger_path, SAMPLE_RATE)
                target_clip = fit_clip(target, samples, target_start)
                passenger_clip = fit_clip(passenger, samples, passenger_start)
                snr_db = float(snrs[(speaker_index + copy_index) % len(snrs)])
                passenger_gain = rms(target_clip) * (10.0 ** (-snr_db / 20.0)) / max(rms(passenger_clip), 1e-6)
                passenger_scaled = passenger_clip * passenger_gain
                positive_mix = target_clip + passenger_scaled
                absent_mix = passenger_scaled

                stem = f"{target_speaker}_{speaker_index:04d}_{copy_index}"
                positive_path = split_output / f"{stem}_present_mix.wav"
                target_path_out = split_output / f"{stem}_target.wav"
                absent_path = split_output / f"{stem}_absent_mix.wav"
                sf.write(str(positive_path), positive_mix.astype(np.float32), SAMPLE_RATE, subtype="FLOAT")
                sf.write(str(target_path_out), target_clip.astype(np.float32), SAMPLE_RATE, subtype="FLOAT")
                sf.write(str(absent_path), absent_mix.astype(np.float32), SAMPLE_RATE, subtype="FLOAT")
                noise_path = noise_paths[(speaker_index * max(1, args.copies_per_speaker) + copy_index) % len(noise_paths)]
                common = {
                    "enroll_wav": str(enroll_path),
                    "sample_rate": SAMPLE_RATE,
                    "speaker_id": target_speaker,
                    "cabin_noise_wav": str(noise_path),
                    "cabin_noise_channels": [0, 2, 4, 6],
                    "cabin_noise_snr_db": float(snrs[(speaker_index + copy_index + 1) % len(snrs)]),
                }
                rows.append({
                    **common,
                    "id": f"presence_{split_name}_{stem}_present",
                    "mixture_wav": str(positive_path),
                    "target_wav": str(target_path_out),
                    "target_present": 1,
                    "condition": "gsc_target_plus_other_speaker_with_cavemove",
                    "meta": {
                        "source": "GSC_v2_plus_CAVEMOVE_proxy",
                        "target_word": target_label,
                        "passenger_speaker_id": passenger_speaker,
                        "passenger_word": passenger_label,
                        "passenger_snr_db": snr_db,
                        "speaker_disjoint_split": True,
                    },
                })
                rows.append({
                    **common,
                    "id": f"presence_{split_name}_{stem}_absent",
                    "mixture_wav": str(absent_path),
                    "target_wav": str(silence_by_split[split_name]),
                    "target_present": 0,
                    "condition": "other_speaker_only_with_cavemove",
                    "meta": {
                        "source": "GSC_v2_plus_CAVEMOVE_proxy",
                        "passenger_speaker_id": passenger_speaker,
                        "passenger_word": passenger_label,
                        "speaker_disjoint_split": True,
                    },
                })

        manifest_path = manifest_root / split_name / "stage2_presence_gsc_cavemove.jsonl"
        write_jsonl(manifest_path, rows)
        print(
            f"{split_name}: {len(rows)} rows, {len(groups)} held-out speakers "
            f"({len(rows) // 2} present + {len(rows) // 2} absent) -> {manifest_path}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
