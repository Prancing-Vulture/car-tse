#!/usr/bin/env python
"""Libri2Mix mix_both -> (mixture, enroll, target) manifests."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import resolve_paths
from car_tse.data.prep_common import manifest_dir, prep_argparser
from car_tse.io.manifest import write_jsonl

SPLIT_MAP = {
    # The full LibriMix recipe uses train-360.  The workspace bootstrap uses
    # the locally available train-clean-100 subset when train-360 is absent.
    "train": ("train-360", "train-100", "local-train"),
    "val": ("dev", "dev-clean", "local-val"),
    "test": ("test", "test-clean", "local-test"),
}


def find_librimix_subset(root: Path, split: str) -> Path | None:
    for subset in SPLIT_MAP[split]:
        for sr in ("wav8k", "wav16k"):
            for kind in ("min", "max"):
                # Support both layouts published by LibriMix: one with the
                # subset first (``train-360/wav8k/min/mix_both``), and the
                # canonical archive layout with sample rate/kind first
                # (``wav8k/min/train-360/mix_both``).
                for mix_dir in (
                    root / subset / sr / kind / "mix_both",
                    root / sr / kind / subset / "mix_both",
                ):
                    # A partially restored corpus can leave the preferred
                    # train-360 directory in place but empty.  Continue to
                    # train-100 rather than creating a valid-looking empty
                    # manifest from that directory.
                    if mix_dir.is_dir() and next(mix_dir.glob("*.wav"), None) is not None:
                        return mix_dir
    return None


def main() -> int:
    parser = prep_argparser("Build Libri2Mix triplet manifest")
    parser.add_argument("--max-rows", type=int, default=0, help="0 = all")
    args = parser.parse_args()
    paths = resolve_paths(args.paths)
    root = paths["corpus_paths"]["librimix"]
    mix_dir = find_librimix_subset(root, args.split)
    if mix_dir is None:
        print(f"No Libri2Mix mix_both under {root}")
        return 1
    s1_dir = mix_dir.parent / "s1"
    meta_path = Path(paths["manifest_root"]) / "meta" / "librispeech_speakers.json"
    spk_split = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    spk_to_label = {s: i for i, s in enumerate(spk_split.get("train", []))}
    rows = []
    mixes = sorted(mix_dir.glob("*.wav"))
    if args.max_rows:
        mixes = mixes[: args.max_rows]
    enroll_pool = spk_split.get("utterances", {})
    rng = random.Random(42)
    for i, mix_path in enumerate(mixes):
        stem = mix_path.stem
        s1_path = s1_dir / f"{stem}.wav"
        if not s1_path.exists():
            continue
        spk_id = stem.split("_")[0] if "_" in stem else "unknown"
        enroll_utts = enroll_pool.get(spk_id, [])
        enroll_wav = rng.choice(enroll_utts) if enroll_utts else str(s1_path)
        rows.append(
            {
                "id": f"librimix_{args.split}_{i}",
                "mixture_wav": str(mix_path),
                "enroll_wav": enroll_wav,
                "target_wav": str(s1_path),
                "sample_rate": paths.get("sample_rate", 8000),
                "speaker_id": spk_id,
                "seat": "",
                "condition": "librimix",
                "target_present": 1,
                "transcript": "",
                "intent": "",
                "playback_ref_wav": "",
                "speaker_label": spk_to_label.get(spk_id, 0),
                "meta": {"source": "librimix"},
            }
        )
    out = manifest_dir(paths, args.split) / "stage1_triplets.jsonl"
    write_jsonl(out, rows)
    print(f"Wrote {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
