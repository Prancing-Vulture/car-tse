#!/usr/bin/env python
"""Attach playback reference paths for EC training (assistant reply stubs)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import resolve_paths
from car_tse.data.prep_common import manifest_dir, prep_argparser
from car_tse.io.manifest import read_jsonl, write_jsonl


def main() -> int:
    parser = prep_argparser("EC playback reference pairs")
    parser.add_argument("--input-manifest", default="stage1_mc_triplets.jsonl")
    parser.add_argument("--reply-dir", default="custom_incar/replies")
    args = parser.parse_args()
    paths = resolve_paths(args.paths)
    in_path = manifest_dir(paths, args.split) / args.input_manifest
    rows = read_jsonl(in_path)
    reply_root = paths["corpus_paths"].get("custom_incar", Path(paths["data_root"]) / "custom_incar")
    reply_dir = Path(reply_root) / "replies" if args.reply_dir == "custom_incar/replies" else Path(args.reply_dir)
    replies = list(reply_dir.glob("*.wav")) if reply_dir.exists() else []
    for i, row in enumerate(rows):
        if replies:
            row["playback_ref_wav"] = str(replies[i % len(replies)])
        else:
            row["playback_ref_wav"] = row.get("mixture_wav", "")
    out = manifest_dir(paths, args.split) / "stage1_mc_ec.jsonl"
    write_jsonl(out, rows)
    print(f"Wrote {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
