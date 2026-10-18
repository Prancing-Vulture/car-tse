#!/usr/bin/env python
"""Merge an existing target-KWS cache with an additional negative cache.

The merge is append-only and writes a new cache/manifests, so the promoted
cache and checkpoints remain untouched.  Both caches must use the same
waveform/embedding dimensions and split names.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import REPO_ROOT
from car_tse.io.manifest import read_jsonl, write_jsonl


def resolve(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def copy_rows(source: np.ndarray, target: np.memmap, offset: int, chunk_size: int = 4096) -> None:
    for start in range(0, len(source), max(1, int(chunk_size))):
        end = min(len(source), start + max(1, int(chunk_size)))
        target[offset + start : offset + end] = source[start:end]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache-dir", type=Path, required=True)
    parser.add_argument("--base-manifest-root", type=Path, default=REPO_ROOT / "manifests")
    parser.add_argument("--base-manifest-template", default="stage3_target_kws_gsc_marvin_hardneg_{split}.jsonl")
    parser.add_argument("--extra-cache-dir", type=Path, required=True)
    parser.add_argument("--extra-manifest-root", type=Path, default=REPO_ROOT / "manifests")
    parser.add_argument("--extra-manifest-template", required=True)
    parser.add_argument("--output-cache-dir", type=Path, required=True)
    parser.add_argument("--output-manifest-root", type=Path, default=REPO_ROOT / "manifests")
    parser.add_argument("--output-manifest-template", required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    args = parser.parse_args()

    base_cache = resolve(args.base_cache_dir)
    extra_cache = resolve(args.extra_cache_dir)
    output_cache = resolve(args.output_cache_dir)
    base_manifest_root = resolve(args.base_manifest_root)
    extra_manifest_root = resolve(args.extra_manifest_root)
    output_manifest_root = resolve(args.output_manifest_root)
    output_cache.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        base_rows = read_jsonl(base_manifest_root / split / str(args.base_manifest_template).format(split=split))
        extra_rows = read_jsonl(extra_manifest_root / split / str(args.extra_manifest_template).format(split=split))
        arrays: dict[str, tuple[str, np.ndarray]] = {}
        for suffix in ("wav_16000hz", "enrollment_embedding", "extracted_embedding"):
            base = np.load(base_cache / f"{split}_{suffix}.npy", mmap_mode="r", allow_pickle=False)
            extra = np.load(extra_cache / f"{split}_{suffix}.npy", mmap_mode="r", allow_pickle=False)
            if base.ndim != extra.ndim or base.shape[1:] != extra.shape[1:]:
                raise ValueError(f"shape mismatch for {split}/{suffix}: {base.shape} vs {extra.shape}")
            output = np.lib.format.open_memmap(
                output_cache / f"{split}_{suffix}.npy",
                mode="w+",
                dtype=np.float32,
                shape=(len(base) + len(extra), *base.shape[1:]),
            )
            copy_rows(base, output, 0)
            copy_rows(extra, output, len(base))
            output.flush()
            del output
            arrays[suffix] = (f"{split}_{suffix}.npy", base)
        output_manifest = output_manifest_root / split / str(args.output_manifest_template).format(split=split)
        write_jsonl(output_manifest, list(base_rows) + list(extra_rows))
        summaries.append(
            {
                "split": split,
                "base_rows": len(base_rows),
                "extra_rows": len(extra_rows),
                "merged_rows": len(base_rows) + len(extra_rows),
                "output_manifest": str(output_manifest),
            }
        )
    metadata_path = resolve(args.metadata) if args.metadata else output_cache / "merge_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "base_cache": str(base_cache),
                "extra_cache": str(extra_cache),
                "output_cache": str(output_cache),
                "summaries": summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
