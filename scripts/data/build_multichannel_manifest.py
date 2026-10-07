#!/usr/bin/env python
"""Add multichannel mixture paths from AISHELL-5 4-ch recordings."""

from __future__ import annotations

import random
import json
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
    parser.add_argument("--multich-subdir", default="mix_multich")
    parser.add_argument("--output-manifest", default="stage1_mc_triplets.jsonl")
    args = parser.parse_args()
    paths = resolve_paths(args.paths)
    in_path = manifest_dir(paths, args.split) / args.input_manifest
    rows = read_jsonl(in_path)
    if not rows:
        in_path = manifest_dir(paths, args.split) / args.fallback_manifest
        rows = read_jsonl(in_path)
    aishell = paths["corpus_paths"]["aishell5"]
    mc_groups = []
    if aishell.exists():
        # AISHELL-5 stores the four far-field microphones as aligned files
        # DX01C01.wav ... DX04C01.wav inside each session directory.  Keep
        # them as a path list so the dataset can load aligned channels lazily.
        session_dirs = sorted({p.parent for p in aishell.rglob("DX01*.wav")})
        for session in session_dirs:
            channel_paths = [session / f"DX0{i}C01.wav" for i in range(1, 5)]
            if all(p.exists() for p in channel_paths):
                mc_groups.append(channel_paths)
        if not mc_groups:
            mc_groups = [[p] for p in aishell.rglob("*.wav") if p.exists()]
    rng = random.Random(7)
    metadata_by_file = {}
    for row in rows:
        mono_path = Path(row.get("mixture_wav", ""))
        metadata_path = mono_path.parent.parent / f"{args.multich_subdir}_metadata.jsonl"
        if not metadata_by_file and metadata_path.exists():
            with metadata_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        item = json.loads(line)
                        metadata_by_file[item["file"]] = item
        paired_path = mono_path.parent.parent / args.multich_subdir / mono_path.name
        if paired_path.exists():
            row["mixture_multich_wav"] = str(paired_path)
            row.pop("mixture_multich_wavs", None)
        elif mc_groups:
            group = rng.choice(mc_groups)
            row["mixture_multich_wavs"] = [str(p) for p in group]
            row["mixture_multich_wav"] = str(group[0])
        if mono_path.name in metadata_by_file:
            row.setdefault("meta", {})["multich_augmentation"] = metadata_by_file[mono_path.name]
        row.setdefault("meta", {})["multichannel"] = True
    out = manifest_dir(paths, args.split) / args.output_manifest
    write_jsonl(out, rows)
    print(f"Wrote {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
