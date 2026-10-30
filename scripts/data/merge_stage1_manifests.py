#!/usr/bin/env python
"""Append explicit absent-target hard negatives to Stage-1 manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import REPO_ROOT
from car_tse.io.manifest import read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, default=REPO_ROOT / "manifests")
    parser.add_argument("--base-name", default="stage1_gsc_marvin_cabin.jsonl")
    parser.add_argument("--hardneg-name", default="stage1_target_kws_nonwake_hardneg.jsonl")
    parser.add_argument("--output-name", default="stage1_gsc_marvin_nonwake_hardneg_cabin.jsonl")
    args = parser.parse_args()
    root = args.manifest_root if args.manifest_root.is_absolute() else REPO_ROOT / args.manifest_root
    for split in ("train", "val", "test"):
        base_path = root / split / args.base_name
        hardneg_path = root / split / args.hardneg_name
        output_path = root / split / args.output_name
        rows = read_jsonl(base_path) + read_jsonl(hardneg_path)
        write_jsonl(output_path, rows)
        print(f"{split}: {len(rows)} rows -> {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
