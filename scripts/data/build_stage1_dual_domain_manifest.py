#!/usr/bin/env python
"""Build a small balanced Stage 1 AISHELL/CAVEMOVE manifest.

The generated JSONL is intentionally ignored with the rest of ``manifests/``;
this script records how to recreate it without versioning local audio paths.
Only target-present AISHELL rows are used here because this is a bounded
target-extraction frontend adaptation, not a presence-head experiment.
"""

from __future__ import annotations

import argparse
import random
import sys
from itertools import zip_longest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import REPO_ROOT
from car_tse.io.manifest import read_jsonl, write_jsonl


def resolve(value: Path) -> Path:
    return value if value.is_absolute() else REPO_ROOT / value


def select_rows(path: Path, count: int, seed: int, aishell: bool) -> list[dict]:
    rows = read_jsonl(path)
    if aishell:
        rows = [row for row in rows if int(row.get("target_present", 0)) == 1]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[: min(max(0, int(count)), len(rows))]


def interleave(aishell_rows: list[dict], cavemove_rows: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for aishell, cavemove in zip_longest(aishell_rows, cavemove_rows):
        if aishell is not None:
            row = dict(aishell)
            row["domain"] = "aishell5"
            merged.append(row)
        if cavemove is not None:
            row = dict(cavemove)
            row["domain"] = "cavemove"
            merged.append(row)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, default=REPO_ROOT / "manifests")
    parser.add_argument("--output-name", default="stage1_dual_domain_rms010.jsonl")
    parser.add_argument("--aishell-train-count", type=int, default=256)
    parser.add_argument("--cavemove-train-count", type=int, default=256)
    parser.add_argument("--aishell-val-count", type=int, default=144)
    parser.add_argument("--cavemove-val-count", type=int, default=144)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()

    root = resolve(args.manifest_root)
    sources = {
        "train": (
            root / "presence_aishell_target" / "train" / "stage2_presence_aishell_target.jsonl",
            root / "train" / "stage1_cavemove_cabin.jsonl",
            args.aishell_train_count,
            args.cavemove_train_count,
        ),
        "val": (
            root / "presence_aishell_target" / "val" / "stage2_presence_aishell_target.jsonl",
            root / "val" / "stage1_cavemove_cabin.jsonl",
            args.aishell_val_count,
            args.cavemove_val_count,
        ),
    }
    for split, (aishell_path, cavemove_path, aishell_count, cavemove_count) in sources.items():
        aishell_rows = select_rows(aishell_path, aishell_count, args.seed + len(split), aishell=True)
        cavemove_rows = select_rows(cavemove_path, cavemove_count, args.seed + 100 + len(split), aishell=False)
        output = root / split / args.output_name
        rows = interleave(aishell_rows, cavemove_rows)
        write_jsonl(output, rows)
        print(
            f"{split}: {len(aishell_rows)} AISHELL + {len(cavemove_rows)} CAVEMOVE "
            f"= {len(rows)} rows -> {output}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
