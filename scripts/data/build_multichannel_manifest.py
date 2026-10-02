#!/usr/bin/env python
"""Add multichannel mixture paths from AISHELL-5 4-ch recordings."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import resolve_paths
from car_tse.data.prep_common import manifest_dir, prep_argparser
from car_tse.io.manifest import read_jsonl, write_jsonl


def main() -> int:
    parser = prep_argparser("Build multichannel manifest")
    parser.add_argument("--input-manifest", default="stage1_triplets_aishell.jsonl")
    parser.add_argument("--fallback-manifest", default="stage1_triplets.jsonl")
    args = parser.parse_args()
    paths = resolve_paths(args.paths)
    in_path = manifest_dir(paths, args.split) / args.input_manifest
    rows = read_jsonl(in_path)
    if not rows:
        in_path = manifest_dir(paths, args.split) / args.fallback_manifest
        rows = read_jsonl(in_path)
    aishell = paths["corpus_paths"]["aishell5"]
    mc_files = [p for p in aishell.rglob("*.wav") if p.exists()] if aishell.exists() else []
    rng = random.Random(7)
    for row in rows:
        if mc_files:
            row["mixture_multich_wav"] = str(rng.choice(mc_files))
        row.setdefault("meta", {})["multichannel"] = True
    out = manifest_dir(paths, args.split) / "stage1_mc_triplets.jsonl"
    write_jsonl(out, rows)
    print(f"Wrote {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
