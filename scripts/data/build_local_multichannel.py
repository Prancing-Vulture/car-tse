#!/usr/bin/env python
"""Create paired four-channel bootstrap mixtures from local mono mixtures.

The real AISHELL-5 far-field files are useful for cabin adaptation, but they
do not share the target/enrollment signals in the LibriSpeech mixture
manifests.  This script keeps the supervision paired by applying deterministic
channel gains/delays and optional AISHELL noise to each local mixture.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import resolve_paths


def read_mono_8k(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), always_2d=True)
    audio = audio.mean(axis=1).astype(np.float32)
    if sr == 8000:
        return audio
    n_out = max(1, int(round(len(audio) * 8000 / sr)))
    x_old = np.linspace(0.0, 1.0, len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def shifted(audio: np.ndarray, delay: int) -> np.ndarray:
    out = np.zeros_like(audio)
    if delay >= 0:
        if delay < len(audio):
            out[delay:] = audio[: len(audio) - delay]
    else:
        offset = -delay
        if offset < len(audio):
            out[: len(audio) - offset] = audio[offset:]
    return out


def crop(audio: np.ndarray, length: int, rng: random.Random) -> np.ndarray:
    if len(audio) <= length:
        out = np.zeros(length, dtype=np.float32)
        out[: len(audio)] = audio
        return out
    start = rng.randrange(0, len(audio) - length + 1)
    return audio[start : start + length]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=Path(__file__).resolve().parents[2] / "configs" / "paths.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], action="append")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--out-subdir", default="mix_multich")
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument(
        "--snr-db",
        nargs="+",
        type=float,
        default=[-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 30.0],
    )
    args = parser.parse_args()

    paths = resolve_paths(args.paths)
    root = paths["corpus_paths"]["librimix"]
    noise_root = paths["corpus_paths"].get("aishell5", root / "missing") / "noise"
    noise_files = sorted(noise_root.rglob("*.wav")) if noise_root.exists() else []
    rng = random.Random(args.seed)
    noise_banks: list[np.ndarray] = []
    if noise_files:
        try:
            # Keep one aligned four-channel noise recording in memory.  Crops
            # are randomized per row, giving varied noise while avoiding
            # repeatedly decoding large AISHELL files.
            first_parent = noise_files[0].parent
            grouped = [first_parent / f"noise_DX0{i}C01.wav" for i in range(1, 5)]
            if not all(path.exists() for path in grouped):
                grouped = noise_files[:4]
            noise_banks = [read_mono_8k(path) for path in grouped]
        except (OSError, RuntimeError):
            noise_banks = []
    splits = args.split or ["train", "val", "test"]

    for split in splits:
        subset = {"train": "train-100", "val": "dev", "test": "test"}[split]
        mix_dir = root / subset / "wav8k" / "min" / "mix_both"
        out_dir = root / subset / "wav8k" / "min" / args.out_subdir
        mixes = sorted(mix_dir.glob("*.wav"))
        if args.max_rows:
            mixes = mixes[: args.max_rows]
        if not mixes:
            print(f"No mono mixtures under {mix_dir}")
            return 1
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Building multichannel {split}: {len(mixes)} rows")
        metadata_rows = []
        for index, mix_path in enumerate(mixes):
            out_path = out_dir / mix_path.name
            if out_path.exists():
                metadata_rows.append({"file": mix_path.name, "snr_db": None})
                continue
            mono = read_mono_8k(mix_path)
            if args.randomize:
                snr_db = float(rng.choice(args.snr_db))
                gains = tuple(rng.uniform(0.85, 1.15) for _ in range(4))
                delays = tuple(rng.randint(-16, 16) for _ in range(4))
            else:
                snr_db = 32.0
                gains = (1.00, 0.94, 1.06, 0.89)
                delays = (0, 3, -4, 7)
            channels = []
            for channel, (gain, delay) in enumerate(zip(gains, delays)):
                channel_audio = gain * shifted(mono, delay)
                signal_rms = float(np.sqrt(np.mean(channel_audio * channel_audio) + 1e-8))
                if noise_banks:
                    noise = crop(noise_banks[channel % len(noise_banks)], len(mono), rng)
                    noise_rms = float(np.sqrt(np.mean(noise * noise) + 1e-8))
                    if noise_rms > 1e-5:
                        noise = noise / noise_rms
                        noise_scale = signal_rms * (10.0 ** (-snr_db / 20.0))
                        channel_audio = channel_audio + noise_scale * noise
                gaussian_scale = signal_rms * (10.0 ** (-35.0 / 20.0)) if args.randomize else signal_rms * 0.002
                channel_audio = channel_audio + np.random.default_rng(args.seed + index + channel).normal(
                    0.0, max(gaussian_scale, 1e-5), len(mono)
                ).astype(np.float32)
                channels.append(channel_audio)
            multichannel = np.stack(channels, axis=1)
            peak = float(np.max(np.abs(multichannel)))
            if peak > 0.98:
                multichannel = multichannel * (0.98 / peak)
            sf.write(str(out_path), multichannel.astype(np.float32), 8000)
            metadata_rows.append(
                {
                    "file": mix_path.name,
                    "snr_db": snr_db,
                    "gains": [float(x) for x in gains],
                    "delays_samples": [int(x) for x in delays],
                }
            )
        print(f"Wrote {len(mixes)} paired multichannel mixtures to {out_dir}")
        metadata_path = out_dir.parent / f"{args.out_subdir}_metadata.jsonl"
        with metadata_path.open("w", encoding="utf-8") as handle:
            for row in metadata_rows:
                handle.write(json.dumps(row) + "\n")
        print(f"Wrote augmentation metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
