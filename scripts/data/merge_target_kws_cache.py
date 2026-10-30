#!/usr/bin/env python
"""Merge the balanced target/absent cache with extra hard negatives."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import REPO_ROOT
from car_tse.io.manifest import read_jsonl, write_jsonl


def copy_array(destination: np.memmap, source: np.ndarray, offset: int) -> None:
    block = 256
    for start in range(0, len(source), block):
        destination[offset + start : offset + min(len(source), start + block)] = source[
            start : min(len(source), start + block)
        ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, default=REPO_ROOT / "manifests")
    parser.add_argument(
        "--base-cache-dir",
        type=Path,
        default=REPO_ROOT / "data/cache/stage3_target_kws_gsc_marvin_16k",
    )
    parser.add_argument(
        "--hardneg-cache-dir",
        type=Path,
        default=REPO_ROOT / "data/cache/stage3_target_kws_nonwake_hardneg_16k",
    )
    parser.add_argument(
        "--output-cache-dir",
        type=Path,
        default=REPO_ROOT / "data/cache/stage3_target_kws_gsc_marvin_hardneg_16k",
    )
    parser.add_argument("--base-prefix", default="stage3_target_kws_gsc_marvin")
    parser.add_argument("--hardneg-prefix", default="stage3_target_kws_nonwake_hardneg")
    parser.add_argument("--extra-cache-dir", type=Path, nargs="+", default=None)
    parser.add_argument("--extra-prefix", nargs="+", default=["stage3_target_kws_aishell5_negative"])
    parser.add_argument("--output-prefix", default="stage3_target_kws_gsc_marvin_hardneg")
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    manifest_root = args.manifest_root if args.manifest_root.is_absolute() else REPO_ROOT / args.manifest_root
    base_cache = args.base_cache_dir if args.base_cache_dir.is_absolute() else REPO_ROOT / args.base_cache_dir
    hardneg_cache = args.hardneg_cache_dir if args.hardneg_cache_dir.is_absolute() else REPO_ROOT / args.hardneg_cache_dir
    extra_caches: list[Path] = []
    if args.extra_cache_dir is not None:
        if len(args.extra_cache_dir) != len(args.extra_prefix):
            raise ValueError("--extra-cache-dir and --extra-prefix must have the same number of values")
        extra_caches = [
            value if value.is_absolute() else REPO_ROOT / value
            for value in args.extra_cache_dir
        ]
    output_cache = args.output_cache_dir if args.output_cache_dir.is_absolute() else REPO_ROOT / args.output_cache_dir
    output_cache.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        base_manifest = manifest_root / split / f"{args.base_prefix}_{split}.jsonl"
        hardneg_manifest = manifest_root / split / f"{args.hardneg_prefix}_{split}.jsonl"
        base_rows = read_jsonl(base_manifest)
        hardneg_rows = read_jsonl(hardneg_manifest)
        extra_rows: list[dict] = []
        for prefix in args.extra_prefix if extra_caches else []:
            extra_manifest = manifest_root / split / f"{prefix}_{split}.jsonl"
            extra_rows.extend(read_jsonl(extra_manifest))
        arrays = []
        for stem in (
            f"{split}_wav_{args.sample_rate}hz.npy",
            f"{split}_enrollment_embedding.npy",
            f"{split}_extracted_embedding.npy",
        ):
            base = np.load(base_cache / stem, mmap_mode="r", allow_pickle=False)
            hardneg = np.load(hardneg_cache / stem, mmap_mode="r", allow_pickle=False)
            extras = [
                np.load(cache / stem, mmap_mode="r", allow_pickle=False)
                for cache in extra_caches
            ]
            shape = (len(base) + len(hardneg) + sum(len(extra) for extra in extras),) + tuple(base.shape[1:])
            destination = np.lib.format.open_memmap(
                output_cache / stem, mode="w+", dtype=np.float32, shape=shape
            )
            copy_array(destination, base, 0)
            copy_array(destination, hardneg, len(base))
            offset = len(base) + len(hardneg)
            for extra in extras:
                copy_array(destination, extra, offset)
                offset += len(extra)
            destination.flush()
            del destination
        output_manifest = manifest_root / split / f"{args.output_prefix}_{split}.jsonl"
        write_jsonl(output_manifest, base_rows + hardneg_rows + extra_rows)
        print(
            f"{split}: base={len(base_rows)} hardneg={len(hardneg_rows)} extra={len(extra_rows)} "
            f"total={len(base_rows) + len(hardneg_rows) + len(extra_rows)}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
