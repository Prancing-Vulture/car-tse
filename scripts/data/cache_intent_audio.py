#!/usr/bin/env python
"""Create fixed-size 16 kHz memmaps for fast repeated intent training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.segment import segment_waveform
from car_tse.io.audio import read_mono
from car_tse.io.manifest import read_jsonl


def cache_split(
    manifest_path: Path,
    output_path: Path,
    sample_rate: int,
    segment_seconds: float,
) -> None:
    rows = read_jsonl(manifest_path)
    num_samples = int(sample_rate * segment_seconds)
    if output_path.exists():
        try:
            existing = np.load(output_path, mmap_mode="r", allow_pickle=False)
            if tuple(existing.shape) == (len(rows), num_samples):
                print(f"Already cached {manifest_path.name}: {output_path}", flush=True)
                return
        except (OSError, ValueError):
            pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), num_samples),
    )
    for index, row in enumerate(rows):
        path = row.get("wav") or row.get("mixture_wav")
        wav, _ = read_mono(path, sample_rate)
        cache[index] = segment_waveform(wav, num_samples, training=False)
        if (index + 1) % 1000 == 0 or index + 1 == len(rows):
            print(f"{manifest_path.name}: {index + 1}/{len(rows)}", flush=True)
    cache.flush()
    del cache
    print(f"Wrote {output_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "stage4_intent_cavemove.yaml",
    )
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    output_dir = args.output_dir or Path(cfg.get("waveform_cache_dir", REPO_ROOT / "data" / "cache" / "stage4_intent_16k"))
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    manifest_root = Path(paths["manifest_root"])
    for split in ("train", "val"):
        source = manifest_root / split / (cfg["manifest_train"] if split == "train" else cfg["manifest_eval"])
        target = output_dir / f"{split}_wav_{cfg['sample_rate']}hz.npy"
        cache_split(source, target, int(cfg["sample_rate"]), float(cfg["segment_seconds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
