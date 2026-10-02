#!/usr/bin/env python
"""Speaker-disjoint LibriSpeech train/val/test speaker lists."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import resolve_paths
from car_tse.data.prep_common import prep_argparser


def collect_speakers(librispeech_root: Path) -> dict[str, list[str]]:
    speakers: dict[str, list[str]] = {}
    for flac in librispeech_root.rglob("*.flac"):
        parts = flac.relative_to(librispeech_root).parts
        if len(parts) < 3:
            continue
        spk = parts[0]
        speakers.setdefault(spk, []).append(str(flac))
    return speakers


def main() -> int:
    args = prep_argparser("Build LibriSpeech speaker splits").parse_args()
    paths = resolve_paths(args.paths)
    root = paths["corpus_paths"]["librispeech"]
    if not root.exists():
        print(f"Missing {root}")
        return 1
    speakers = collect_speakers(root)
    spk_ids = sorted(speakers.keys())
    random.seed(42)
    random.shuffle(spk_ids)
    n = len(spk_ids)
    n_train = int(0.9 * n)
    n_val = int(0.05 * n)
    split = {
        "train": spk_ids[:n_train],
        "val": spk_ids[n_train : n_train + n_val],
        "test": spk_ids[n_train + n_val :],
        "utterances": speakers,
    }
    out = Path(paths["manifest_root"]) / "meta" / "librispeech_speakers.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(split, f)
    print(f"Wrote {out} ({n} speakers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
