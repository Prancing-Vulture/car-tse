#!/usr/bin/env python
"""Add balanced post-TSE hard-negative absent examples to presence training."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import REPO_ROOT
from car_tse.io.manifest import read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, default=REPO_ROOT / "manifests" / "train" / "stage2_cavemove_presence.jsonl")
    parser.add_argument("--hardneg-manifest", type=Path, default=REPO_ROOT / "manifests" / "train" / "stage3_kws_tse_hardneg_train.jsonl")
    parser.add_argument("--enrollment-wav", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "manifests" / "train" / "stage2_presence_tse_hardneg_train.jsonl")
    parser.add_argument("--absent-count", type=int, default=4000)
    parser.add_argument("--present-count", type=int, default=0, help="Optional balanced subset size; 0 keeps all present rows")
    parser.add_argument("--no-base", action="store_true", help="Write only converted absent hard-negative rows")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    base = [] if args.no_base else read_jsonl(args.base_manifest)
    hardneg = [row for row in read_jsonl(args.hardneg_manifest) if int(row.get("label", 0)) == 0]
    rng = random.Random(args.seed)
    if args.present_count and len(base) > args.present_count:
        base = rng.sample(base, args.present_count)
    if len(hardneg) > args.absent_count:
        hardneg = rng.sample(hardneg, args.absent_count)
    enrollment = str(args.enrollment_wav if args.enrollment_wav.is_absolute() else REPO_ROOT / args.enrollment_wav)
    absent = []
    for index, row in enumerate(hardneg):
        absent.append(
            {
                "id": f"tse_hardneg_absent_{index}",
                "mixture_wav": row["wav"],
                "enroll_wav": enrollment,
                "target_wav": enrollment,
                "sample_rate": 16000,
                "speaker_id": "tse_hardneg_absent",
                "speaker_label": -1,
                "condition": "post_tse_hard_negative",
                "target_present": 0,
                "transcript": row.get("meta", {}).get("intent", ""),
                "intent": "",
                "playback_ref_wav": "",
                "cabin_noise_wav": row.get("cabin_noise_wav") or (row.get("wav", "") if row.get("noise_only", False) else ""),
                "cabin_noise_channels": row.get("cabin_noise_channels", [0, 2, 4, 6]),
                "cabin_noise_snr_db": row.get("cabin_noise_snr_db", 0.0),
                "meta": {"source": "post_tse_hard_negative_presence", "source_row": row.get("id", "")},
            }
        )
    output_rows = list(base) + absent
    write_jsonl(args.output, output_rows)
    print(f"base_present={len(base)} added_absent={len(absent)} total={len(output_rows)} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
