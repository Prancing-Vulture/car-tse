#!/usr/bin/env python
"""Pre-mix each intent example with its assigned CAVEMOVE recording."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.io.manifest import read_jsonl


def noise_file(cache_dir: Path, source: str) -> Path:
    digest = hashlib.sha1(str(Path(source)).encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.npy"


def build_split(
    split: str,
    manifest_path: Path,
    clean_cache_path: Path,
    output_path: Path,
    noise_cache_dir: Path,
    sample_rate: int,
    segment_seconds: float,
    seed: int,
) -> None:
    rows = read_jsonl(manifest_path)
    num_samples = int(sample_rate * segment_seconds)
    clean = np.load(clean_cache_path, mmap_mode="r", allow_pickle=False)
    expected = (len(rows), num_samples)
    if tuple(clean.shape) != expected:
        raise ValueError(f"Unexpected clean cache shape {clean.shape}; expected {expected}")
    if output_path.exists():
        try:
            existing = np.load(output_path, mmap_mode="r", allow_pickle=False)
            if tuple(existing.shape) == expected:
                print(f"Already cached {split}: {output_path}", flush=True)
                return
        except (OSError, ValueError):
            pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mixed_cache = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=expected,
    )
    for index, row in enumerate(rows):
        wav = np.array(clean[index], dtype=np.float32, copy=True)
        source = row.get("cabin_noise_wav")
        if not source:
            raise ValueError(f"Missing cabin_noise_wav at {manifest_path}:{index + 1}")
        noise_path = noise_file(noise_cache_dir, source)
        if not noise_path.exists():
            raise FileNotFoundError(noise_path)
        noise = np.load(noise_path, mmap_mode="r", allow_pickle=False)
        if len(noise) >= num_samples:
            if split == "train":
                rng = np.random.default_rng(seed + index * 7919)
                start = int(rng.integers(0, len(noise) - num_samples + 1))
            else:
                start = (len(noise) - num_samples) // 2
            noise_chunk = np.asarray(noise[start : start + num_samples], dtype=np.float32)
        else:
            repeats = int(np.ceil(num_samples / max(1, len(noise))))
            noise_chunk = np.tile(np.asarray(noise, dtype=np.float32), repeats)[:num_samples]
        snr_db = float(row.get("cabin_noise_snr_db", 0.0))
        speech_rms = float(np.sqrt(np.mean(wav * wav) + 1e-8))
        noise_rms = float(np.sqrt(np.mean(noise_chunk * noise_chunk) + 1e-8))
        if speech_rms > 1e-6 and noise_rms > 1e-6:
            wav = wav + noise_chunk * (speech_rms * (10.0 ** (-snr_db / 20.0)) / noise_rms)
        peak = float(np.max(np.abs(wav)))
        if peak > 0.95:
            wav = wav * (0.95 / peak)
        mixed_cache[index] = wav
        if (index + 1) % 1000 == 0 or index + 1 == len(rows):
            print(f"{split}: {index + 1}/{len(rows)}", flush=True)
    mixed_cache.flush()
    del mixed_cache
    print(f"Wrote {output_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "stage4_intent_cavemove.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    manifest_root = Path(paths["manifest_root"])
    cache_dir = Path(cfg["waveform_cache_dir"])
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    noise_cache_dir = Path(cfg["cabin_noise_cache_dir"])
    if not noise_cache_dir.is_absolute():
        noise_cache_dir = REPO_ROOT / noise_cache_dir
    for split in ("train", "val"):
        manifest_name = cfg["manifest_train"] if split == "train" else cfg["manifest_eval"]
        build_split(
            split,
            manifest_root / split / manifest_name,
            cache_dir / f"{split}_wav_{cfg['sample_rate']}hz.npy",
            cache_dir / f"{split}_cabin_wav_{cfg['sample_rate']}hz.npy",
            noise_cache_dir,
            int(cfg["sample_rate"]),
            float(cfg["segment_seconds"]),
            args.seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
