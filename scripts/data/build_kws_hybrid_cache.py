#!/usr/bin/env python
"""Combine the cached TSE-domain and cabin hard-negative KWS corpora."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.io.manifest import read_jsonl, write_jsonl


def merge_split(
    split: str,
    manifest_root: Path,
    tse_manifest_name: str,
    hardneg_manifest_name: str,
    tse_cache_dir: Path,
    hardneg_cache_dir: Path,
    output_manifest_name: str,
    output_cache_dir: Path,
    sample_rate: int,
    force: bool,
) -> None:
    tse_manifest = manifest_root / split / tse_manifest_name
    hardneg_manifest = manifest_root / split / hardneg_manifest_name
    output_manifest = manifest_root / split / output_manifest_name
    tse_cache_path = tse_cache_dir / f"{split}_cabin_wav_{sample_rate}hz.npy"
    hardneg_cache_path = hardneg_cache_dir / f"{split}_cabin_wav_{sample_rate}hz.npy"
    output_cache = output_cache_dir / f"{split}_cabin_wav_{sample_rate}hz.npy"

    tse_rows = read_jsonl(tse_manifest)
    hardneg_rows = read_jsonl(hardneg_manifest)
    tse_cache = np.load(tse_cache_path, mmap_mode="r", allow_pickle=False)
    hardneg_cache = np.load(hardneg_cache_path, mmap_mode="r", allow_pickle=False)
    if len(tse_rows) != tse_cache.shape[0] or len(hardneg_rows) != hardneg_cache.shape[0]:
        raise ValueError(f"Manifest/cache mismatch for {split}")
    expected_shape = (len(tse_rows) + len(hardneg_rows), int(sample_rate))
    if output_cache.exists() and output_manifest.exists() and not force:
        try:
            existing = np.load(output_cache, mmap_mode="r", allow_pickle=False)
            if tuple(existing.shape) == expected_shape:
                print(f"Already merged {split}: {output_cache}", flush=True)
                return
        except (OSError, ValueError):
            pass

    output_cache_dir.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_cache, mode="w+", dtype=np.float32, shape=expected_shape
    )
    output[: len(tse_rows)] = tse_cache
    output[len(tse_rows) :] = hardneg_cache
    output.flush()
    del output

    rows = []
    for source, tag in ((tse_rows, "tse_domain"), (hardneg_rows, "hard_negative")):
        for row in source:
            merged = dict(row)
            merged["hybrid_source"] = tag
            rows.append(merged)
    write_jsonl(output_manifest, rows)
    print(f"{split}: {len(rows)} rows -> {output_cache}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--tse-config", type=Path, default=REPO_ROOT / "configs" / "stage3_kws_tse_domain.yaml")
    parser.add_argument("--hardneg-config", type=Path, default=REPO_ROOT / "configs" / "stage3_kws_cavemove_hardneg.yaml")
    parser.add_argument("--output-cache-dir", type=Path, default=REPO_ROOT / "data" / "cache" / "stage3_kws_hybrid_16k")
    parser.add_argument("--output-prefix", default="stage3_kws_hybrid")
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val", "test"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    tse_cfg = load_stage_config(args.tse_config, args.paths)
    hardneg_cfg = load_stage_config(args.hardneg_config, args.paths)
    manifest_root = Path(tse_cfg["paths"]["manifest_root"])
    tse_cache_dir = Path(tse_cfg.get("waveform_cache_dir", ""))
    hardneg_cache_dir = Path(hardneg_cfg.get("waveform_cache_dir", ""))
    output_cache_dir = args.output_cache_dir
    if not tse_cache_dir.is_absolute():
        tse_cache_dir = REPO_ROOT / tse_cache_dir
    if not hardneg_cache_dir.is_absolute():
        hardneg_cache_dir = REPO_ROOT / hardneg_cache_dir
    if not output_cache_dir.is_absolute():
        output_cache_dir = REPO_ROOT / output_cache_dir

    for split in args.splits:
        tse_name = tse_cfg["manifest_train"] if split == "train" else tse_cfg["manifest_val"] if split == "val" else tse_cfg["manifest_eval"]
        hardneg_name = hardneg_cfg["manifest_train"] if split == "train" else hardneg_cfg["manifest_val"] if split == "val" else hardneg_cfg["manifest_eval"]
        merge_split(
            split,
            manifest_root,
            tse_name,
            hardneg_name,
            tse_cache_dir,
            hardneg_cache_dir,
            f"{args.output_prefix}_{split}.jsonl",
            output_cache_dir,
            int(tse_cfg["sample_rate"]),
            args.force,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
