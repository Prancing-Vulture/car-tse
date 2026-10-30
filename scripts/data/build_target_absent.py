#!/usr/bin/env python
"""Passenger-only and noise-only examples for presence training."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import resolve_paths
from car_tse.data.prep_common import manifest_dir, prep_argparser
from car_tse.io.manifest import read_jsonl, write_jsonl


def main() -> int:
    parser = prep_argparser("Build target-absent manifest")
    parser.add_argument("--input-manifest", default="stage1_mc_triplets.jsonl")
    args = parser.parse_args()
    paths = resolve_paths(args.paths)
    in_path = manifest_dir(paths, args.split) / args.input_manifest
    rows = read_jsonl(in_path)
    mix_dir = paths["corpus_paths"]["librimix"]
    s2_samples = list(mix_dir.rglob("s2/*.wav"))[: max(1, len(rows) // 10)] if mix_dir.exists() else []
    out_rows = []
    for row in rows:
        out_rows.append(row)
    for i, s2 in enumerate(s2_samples):
        ref = rows[i % len(rows)] if rows else {}
        out_rows.append(
            {
                "id": f"absent_{args.split}_{i}",
                "mixture_wav": str(s2),
                "enroll_wav": ref.get("enroll_wav", str(s2)),
                "target_wav": str(s2),
                "sample_rate": paths.get("sample_rate", 8000),
                "speaker_id": ref.get("speaker_id", ""),
                "seat": "passenger",
                "condition": "target_absent",
                "target_present": 0,
                "speaker_label": ref.get("speaker_label", 0),
                "transcript": "",
                "intent": "",
                "playback_ref_wav": "",
                # Keep optional batch fields present for DataLoader collation.
                # The bootstrap presence model uses the mono stage-1 path;
                # MC presence training can replace this with a paired s2
                # spatial mixture when that augmentation is enabled.
                "mixture_multich_wav": ref.get("mixture_multich_wav", ""),
                "meta": {"source": "passenger_only"},
            }
        )
    out = manifest_dir(paths, args.split) / "stage2_presence.jsonl"
    write_jsonl(out, out_rows)
    print(f"Wrote {len(out_rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
