#!/usr/bin/env python
"""Create fixed 1-second cabin-mixed waveforms for fast KWS training."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.segment import segment_waveform
from car_tse.io.audio import read_mono
from car_tse.io.manifest import read_jsonl


def cached_noise(cache_dir: Path, source: str) -> np.ndarray | None:
    digest = hashlib.sha1(str(Path(source)).encode("utf-8")).hexdigest()
    path = cache_dir / f"{digest}.npy"
    if not path.exists():
        return None
    try:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError):
        return None


def cache_split(
    split: str,
    manifest_path: Path,
    output_path: Path,
    noise_cache_dir: Path,
    sample_rate: int,
    segment_seconds: float,
    seed: int,
) -> None:
    rows = read_jsonl(manifest_path)
    num_samples = int(sample_rate * segment_seconds)
    expected = (len(rows), num_samples)
    if output_path.exists():
        try:
            existing = np.load(output_path, mmap_mode="r", allow_pickle=False)
            if tuple(existing.shape) == expected:
                print(f"Already cached {split}: {output_path}", flush=True)
                return
        except (OSError, ValueError):
            pass
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=expected)
    for index, row in enumerate(rows):
        source = row["wav"]
        wav = cached_noise(noise_cache_dir, source) if row.get("noise_only", False) else None
        if wav is None:
            wav, _ = read_mono(source, sample_rate)
        wav = segment_waveform(np.asarray(wav, dtype=np.float32), num_samples, training=False)

        cabin_source = row.get("cabin_noise_wav")
        if cabin_source:
            noise = cached_noise(noise_cache_dir, cabin_source)
            if noise is None:
                noise, _ = read_mono(cabin_source, sample_rate)
            noise = np.asarray(noise, dtype=np.float32)
            if len(noise) >= num_samples:
                if split == "train":
                    rng = np.random.default_rng(seed + index * 6151)
                    start = int(rng.integers(0, len(noise) - num_samples + 1))
                else:
                    start = (len(noise) - num_samples) // 2
                noise = noise[start : start + num_samples]
            else:
                noise = np.tile(noise, int(np.ceil(num_samples / max(1, len(noise)))))[:num_samples]
            snr_db = float(row.get("cabin_noise_snr_db", 0.0))
            wav_rms = float(np.sqrt(np.mean(wav * wav) + 1e-8))
            noise_rms = float(np.sqrt(np.mean(noise * noise) + 1e-8))
            if wav_rms > 1e-6 and noise_rms > 1e-6:
                wav = wav + noise * (wav_rms * (10.0 ** (-snr_db / 20.0)) / noise_rms)
        peak = float(np.max(np.abs(wav)))
        if peak > 0.95:
            wav = wav * (0.95 / peak)
        out[index] = wav
        if (index + 1) % 1000 == 0 or index + 1 == len(rows):
            print(f"{split}: {index + 1}/{len(rows)}", flush=True)
    out.flush()
    del out
    print(f"Wrote {output_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "stage3_kws_cavemove_hardneg.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    output_dir = Path(cfg["waveform_cache_dir"])
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    noise_cache_dir = Path(cfg["cabin_noise_cache_dir"])
    if not noise_cache_dir.is_absolute():
        noise_cache_dir = REPO_ROOT / noise_cache_dir
    root = Path(paths["manifest_root"])
    for split in ("train", "val", "test"):
        manifest_name = cfg["manifest_train"] if split == "train" else cfg["manifest_val"] if split == "val" else cfg["manifest_eval"]
        cache_split(
            split,
            root / split / manifest_name,
            output_dir / f"{split}_cabin_wav_{cfg['sample_rate']}hz.npy",
            noise_cache_dir,
            int(cfg["sample_rate"]),
            float(cfg["segment_seconds"]),
            args.seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
