#!/usr/bin/env python
"""Build a compact LibriMix-style corpus from locally downloaded LibriSpeech.

This is a workspace bootstrap generator, not a replacement for the official
LibriMix recipe.  It creates fixed-length 8 kHz two-speaker mixtures and
targets so the repository's manifests and SpEx+ training loop can be started
without downloading the multi-hundred-GB full LibriMix expansion.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import resolve_paths


SUBSETS = {"train": "train-clean-100", "val": "dev-clean", "test": "test-clean"}


def read_8k(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), always_2d=True)
    audio = audio.mean(axis=1).astype(np.float32)
    if sr != 8000:
        audio = resample_poly(audio, 8000, sr).astype(np.float32)
    return audio


def fixed_segment(audio: np.ndarray, n: int, rng: random.Random) -> np.ndarray:
    if len(audio) >= n:
        start = rng.randrange(0, len(audio) - n + 1)
        return audio[start : start + n].copy()
    out = np.zeros(n, dtype=np.float32)
    out[: len(audio)] = audio
    return out


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio * audio) + 1e-8))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=Path(__file__).resolve().parents[2] / "configs" / "paths.yaml")
    parser.add_argument("--split", choices=sorted(SUBSETS), action="append")
    parser.add_argument("--train-rows", type=int, default=4000)
    parser.add_argument("--val-rows", type=int, default=400)
    parser.add_argument("--test-rows", type=int, default=400)
    parser.add_argument("--segment-seconds", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.paths)
    libri_root = paths["corpus_paths"]["librispeech"]
    out_root = paths["corpus_paths"]["librimix"]
    requested = args.split or list(SUBSETS)
    row_limits = {"train": args.train_rows, "val": args.val_rows, "test": args.test_rows}
    n_samples = int(8000 * args.segment_seconds)

    for split in requested:
        subset_root = libri_root / SUBSETS[split]
        files = sorted(subset_root.rglob("*.flac"))
        if not files:
            print(f"No LibriSpeech FLAC files found under {subset_root}")
            return 1
        by_speaker: dict[str, list[Path]] = {}
        for path in files:
            rel = path.relative_to(subset_root).parts
            if len(rel) < 3:
                continue
            by_speaker.setdefault(rel[0], []).append(path)
        speakers = sorted(by_speaker)
        if len(speakers) < 2:
            print(f"Need at least two speakers in {subset_root}")
            return 1

        split_offset = {"train": 0, "val": 101, "test": 202}[split]
        rng = random.Random(args.seed + split_offset)
        subset_out = out_root / {"train": "train-100", "val": "dev", "test": "test"}[split] / "wav8k" / "min"
        mix_dir = subset_out / "mix_both"
        target_dir = subset_out / "s1"
        distractor_dir = subset_out / "s2"
        mix_dir.mkdir(parents=True, exist_ok=True)
        target_dir.mkdir(parents=True, exist_ok=True)
        distractor_dir.mkdir(parents=True, exist_ok=True)

        limit = min(row_limits[split], max(1, len(files)))
        print(f"Building {split}: {limit} rows from {len(files)} utterances / {len(speakers)} speakers")
        for index in range(limit):
            target_spk = speakers[index % len(speakers)]
            distractor_spk = speakers[(index + 1 + (index // len(speakers))) % len(speakers)]
            target_src = rng.choice(by_speaker[target_spk])
            distractor_src = rng.choice(by_speaker[distractor_spk])
            target = fixed_segment(read_8k(target_src), n_samples, rng)
            distractor = fixed_segment(read_8k(distractor_src), n_samples, rng)

            target_level = rms(target)
            distractor_level = rms(distractor)
            if target_level > 1e-5:
                target = target / max(1.0, target_level * 4.0)
            if distractor_level > 1e-5:
                distractor = distractor / max(1.0, distractor_level * 4.0)
            mixture = target + (0.7 + 0.6 * rng.random()) * distractor
            peak = float(np.max(np.abs(mixture)))
            if peak > 0.98:
                scale = 0.98 / peak
                mixture = mixture * scale
                target = target * scale
                distractor = distractor * scale

            stem = f"{target_spk}_{index:06d}"
            sf.write(str(target_dir / f"{stem}.wav"), target.astype(np.float32), 8000)
            sf.write(str(distractor_dir / f"{stem}.wav"), distractor.astype(np.float32), 8000)
            sf.write(str(mix_dir / f"{stem}.wav"), mixture.astype(np.float32), 8000)

        print(f"Wrote {limit} mixtures to {mix_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
