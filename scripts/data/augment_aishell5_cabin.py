#!/usr/bin/env python
"""Attach AISHELL-5 car noise SNR metadata to stage1 manifests (paths only)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import resolve_paths
from car_tse.data.prep_common import manifest_dir, prep_argparser
from car_tse.io.manifest import read_jsonl, write_jsonl

SNR_DB = [-5, 0, 5, 10, 15]


def main() -> int:
    parser = prep_argparser("AISHELL-5 cabin noise augmentation metadata")
    parser.add_argument("--input-manifest", default="stage1_triplets.jsonl")
    args = parser.parse_args()
    paths = resolve_paths(args.paths)
    aishell = paths["corpus_paths"]["aishell5"]
    noise_files = list(aishell.rglob("*.wav")) if aishell.exists() else []
    in_path = manifest_dir(paths, args.split) / args.input_manifest
    rows = read_jsonl(in_path)
    if not rows:
        print(f"No rows in {in_path}")
        return 1
    rng = random.Random(99)
    for row in rows:
        meta = row.get("meta") or {}
        if noise_files:
            meta["aishell5_noise"] = str(rng.choice(noise_files))
            meta["snr_db"] = rng.choice(SNR_DB)
        row["meta"] = meta
        row["condition"] = "cabin_noise"
    out = manifest_dir(paths, args.split) / "stage1_triplets_aishell.jsonl"
    write_jsonl(out, rows)
    print(f"Wrote {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
