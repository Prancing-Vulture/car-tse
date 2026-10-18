#!/usr/bin/env python
"""Merge two compact-verifier pair caches into one training set.

Each input cache contributes ``{split}_wav_16000hz.npy`` (extracted
waveforms), ``{split}_enrollment_wav.npy`` (enrollment waveforms), and a
per-split JSONL manifest with ``label`` fields.  The merged cache keeps the
row order base-then-extra so existing diagnostics remain interpretable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from car_tse.config import REPO_ROOT


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-extracted-dir", type=Path, required=True)
    parser.add_argument("--base-enrollment-dir", type=Path, required=True)
    parser.add_argument("--base-manifest-template", required=True)
    parser.add_argument("--extra-extracted-dir", type=Path, required=True)
    parser.add_argument("--extra-enrollment-dir", type=Path, required=True)
    parser.add_argument("--extra-manifest-template", required=True)
    parser.add_argument("--manifest-root", type=Path, default=REPO_ROOT / "manifests")
    parser.add_argument("--output-extracted-dir", type=Path, required=True)
    parser.add_argument("--output-enrollment-dir", type=Path, required=True)
    parser.add_argument("--output-manifest-template", required=True)
    args = parser.parse_args()

    manifest_root = resolve(args.manifest_root)
    base_extracted = resolve(args.base_extracted_dir)
    base_enrollment = resolve(args.base_enrollment_dir)
    extra_extracted = resolve(args.extra_extracted_dir)
    extra_enrollment = resolve(args.extra_enrollment_dir)
    output_extracted = resolve(args.output_extracted_dir)
    output_enrollment = resolve(args.output_enrollment_dir)
    output_extracted.mkdir(parents=True, exist_ok=True)
    output_enrollment.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        base_rows = load_rows(
            manifest_root / split / str(args.base_manifest_template).format(split=split)
        )
        extra_rows = load_rows(
            manifest_root / split / str(args.extra_manifest_template).format(split=split)
        )
        base_extracted_split = np.load(
            base_extracted / f"{split}_wav_16000hz.npy", mmap_mode="r", allow_pickle=False
        )
        extra_extracted_split = np.load(
            extra_extracted / f"{split}_wav_16000hz.npy", mmap_mode="r", allow_pickle=False
        )
        base_enrollment_split = np.load(
            base_enrollment / f"{split}_enrollment_wav.npy", mmap_mode="r", allow_pickle=False
        )
        extra_enrollment_split = np.load(
            extra_enrollment / f"{split}_enrollment_wav.npy", mmap_mode="r", allow_pickle=False
        )
        if not (
            len(base_rows) == len(base_extracted_split) == len(base_enrollment_split)
            and len(extra_rows) == len(extra_extracted_split) == len(extra_enrollment_split)
        ):
            raise ValueError(f"{split}: manifest/cache length mismatch")
        if base_extracted_split.shape[1] != extra_extracted_split.shape[1]:
            raise ValueError(f"{split}: extracted-wav length mismatch")
        if base_enrollment_split.shape[1] != extra_enrollment_split.shape[1]:
            raise ValueError(f"{split}: enrollment-wav length mismatch")
        np.save(
            output_extracted / f"{split}_wav_16000hz.npy",
            np.concatenate(
                [np.asarray(base_extracted_split), np.asarray(extra_extracted_split)], axis=0
            ).astype(np.float32),
        )
        np.save(
            output_enrollment / f"{split}_enrollment_wav.npy",
            np.concatenate(
                [np.asarray(base_enrollment_split), np.asarray(extra_enrollment_split)], axis=0
            ).astype(np.float32),
        )
        rows = base_rows + extra_rows
        manifest_dir = manifest_root / split
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / str(args.output_manifest_template).format(split=split)
        with manifest_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        summary[split] = {
            "base": len(base_rows),
            "extra": len(extra_rows),
            "total": len(rows),
            "positives": int(sum(1 for row in rows if int(row.get("label", 0)) == 1)),
            "manifest": str(manifest_path),
        }
        print(f"{split}: merged {len(rows)} rows -> {manifest_path}", flush=True)

    summary_path = output_extracted / "merge_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
