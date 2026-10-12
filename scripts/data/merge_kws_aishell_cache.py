#!/usr/bin/env python
"""Merge the current hybrid KWS corpus with AISHELL-5 real-cabin negatives."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from car_tse.config import REPO_ROOT
from car_tse.io.manifest import read_jsonl, write_jsonl


def resolve_path(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def merge_split(
    split: str,
    hybrid_dir: Path,
    hybrid_manifest_root: Path,
    hybrid_manifest_prefix: str,
    aishell_dir: Path,
    aishell_manifest_prefix: str,
    output_dir: Path,
    output_manifest_root: Path,
    output_manifest_prefix: str,
    force: bool,
) -> None:
    sample_rate = 16000
    hybrid_cache = hybrid_dir / f"{split}_cabin_wav_{sample_rate}hz.npy"
    aishell_cache = aishell_dir / f"{split}_cabin_wav_{sample_rate}hz.npy"
    hybrid_manifest = hybrid_manifest_root / split / f"{hybrid_manifest_prefix}_{split}.jsonl"
    aishell_manifest = output_manifest_root / split / f"{aishell_manifest_prefix}_{split}.jsonl"
    output_cache = output_dir / f"{split}_cabin_wav_{sample_rate}hz.npy"
    output_manifest = output_manifest_root / split / f"{output_manifest_prefix}_{split}.jsonl"
    hybrid_rows = read_jsonl(hybrid_manifest)
    aishell_rows = read_jsonl(aishell_manifest)
    hybrid_values = np.load(hybrid_cache, mmap_mode="r", allow_pickle=False)
    aishell_values = np.load(aishell_cache, mmap_mode="r", allow_pickle=False)
    if len(hybrid_rows) != hybrid_values.shape[0] or len(aishell_rows) != aishell_values.shape[0]:
        raise ValueError(f"Manifest/cache mismatch for {split}")
    expected = (len(hybrid_rows) + len(aishell_rows), sample_rate)
    if output_cache.exists() and output_manifest.exists() and not force:
        try:
            existing = np.load(output_cache, mmap_mode="r", allow_pickle=False)
            if tuple(existing.shape) == expected:
                print(f"Already merged {split}: {output_cache}", flush=True)
                return
        except (OSError, ValueError):
            pass
    output_dir.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(output_cache, mode="w+", dtype=np.float32, shape=expected)
    output[: len(hybrid_rows)] = hybrid_values
    output[len(hybrid_rows) :] = aishell_values
    output.flush()
    del output
    rows = []
    for row in hybrid_rows:
        merged = dict(row)
        merged["hybrid_source"] = merged.get("hybrid_source", "existing_hybrid")
        rows.append(merged)
    for row in aishell_rows:
        merged = dict(row)
        merged["hybrid_source"] = "aishell5_real_cabin"
        rows.append(merged)
    write_jsonl(output_manifest, rows)
    print(f"{split}: {len(rows)} rows -> {output_cache}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid-dir", type=Path, default=REPO_ROOT / "data" / "cache" / "stage3_kws_tse_hybrid_16k")
    parser.add_argument("--aishell-dir", type=Path, default=REPO_ROOT / "data" / "cache" / "stage3_kws_aishell5_16k")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "cache" / "stage3_kws_tse_aishell_16k")
    parser.add_argument("--manifest-root", type=Path, default=REPO_ROOT / "manifests")
    parser.add_argument("--hybrid-manifest-prefix", default="stage3_kws_tse_hybrid")
    parser.add_argument("--aishell-manifest-prefix", default="stage3_kws_aishell5")
    parser.add_argument("--output-manifest-prefix", default="stage3_kws_tse_aishell")
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val", "test"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for split in args.splits:
        merge_split(
            split,
            resolve_path(args.hybrid_dir),
            resolve_path(args.manifest_root),
            args.hybrid_manifest_prefix,
            resolve_path(args.aishell_dir),
            args.aishell_manifest_prefix,
            resolve_path(args.output_dir),
            resolve_path(args.manifest_root),
            args.output_manifest_prefix,
            args.force,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
