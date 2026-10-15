#!/usr/bin/env python
"""Build speaker-disjoint manifests for real custom in-car wake recordings.

Expected layout under ``--root``::

    wake/<speaker>/*.wav
    negative/<speaker>/*.wav
    enrollment/<speaker>/*.wav       # optional; one clip is recorded in rows

The speaker directory can be replaced by any stable speaker identifier.  The
split is assigned at speaker level so a speaker never appears in more than one
split.  Negative recordings without a speaker directory are grouped by their
first relative path component, which keeps common ``negative/<source>/*.wav``
layouts disjoint as well.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def speaker_id(path: Path, root: Path, class_name: str) -> str:
    relative = path.relative_to(root / class_name)
    parts = relative.parts
    if len(parts) >= 2:
        return parts[0]
    return f"{class_name}_unassigned"


def collect(root: Path, class_name: str, label: int) -> list[dict]:
    class_root = root / class_name
    if not class_root.exists():
        return []
    rows: list[dict] = []
    for wav in sorted(class_root.rglob("*.wav")):
        rows.append(
            {
                "wav": str(wav.resolve()),
                "label": int(label),
                "speaker_id": speaker_id(wav, root, class_name),
                "source": "custom_incar",
                "class": class_name,
            }
        )
    return rows


def assign_splits(rows: list[dict], seed: int, val_fraction: float, test_fraction: float) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["speaker_id"])].append(row)
    speakers = sorted(grouped)
    random.Random(seed).shuffle(speakers)
    n = len(speakers)
    n_test = max(1, round(n * test_fraction)) if n >= 3 else 0
    n_val = max(1, round(n * val_fraction)) if n - n_test >= 2 else 0
    if n_test + n_val >= n and n >= 3:
        n_val = max(1, n - n_test - 1)
    split_speakers = {
        "test": set(speakers[:n_test]),
        "val": set(speakers[n_test : n_test + n_val]),
    }
    split_speakers["train"] = set(speakers[n_test + n_val :])
    output: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for split, speaker_set in split_speakers.items():
        for speaker in sorted(speaker_set):
            for row in grouped[speaker]:
                row = dict(row)
                row["split"] = split
                output[split].append(row)
    return output


def attach_enrollment(rows: list[dict], root: Path) -> None:
    enrollment_root = root / "enrollment"
    by_speaker: dict[str, list[Path]] = {}
    if enrollment_root.exists():
        for directory in sorted(path for path in enrollment_root.iterdir() if path.is_dir()):
            by_speaker[directory.name] = sorted(directory.rglob("*.wav"))
    for row in rows:
        clips = by_speaker.get(str(row["speaker_id"]), [])
        row["enroll_wav"] = str(clips[0].resolve()) if clips else ""


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build speaker-disjoint custom in-car KWS manifests")
    parser.add_argument("--root", type=Path, default=Path("data/raw/custom_incar"))
    parser.add_argument("--output-root", type=Path, default=Path("manifests"))
    parser.add_argument("--output-prefix", default="stage3_kws_custom_incar")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = collect(root, "wake", 1) + collect(root, "negative", 0)
    if not rows:
        raise SystemExit(
            f"No WAV files found under {root}. Add wake/<speaker>/*.wav and "
            "negative/<speaker>/*.wav before building the manifest."
        )
    splits = assign_splits(rows, args.seed, args.val_fraction, args.test_fraction)
    for split, split_rows in splits.items():
        attach_enrollment(split_rows, root)
        for index, row in enumerate(split_rows):
            row["id"] = f"custom_incar_{split}_{index:06d}"
        output = args.output_root.resolve() / split / f"{args.output_prefix}_{split}.jsonl"
        write_jsonl(output, split_rows)
        positives = sum(int(row["label"]) for row in split_rows)
        speakers = len({str(row["speaker_id"]) for row in split_rows})
        print(f"{split}: {len(split_rows)} rows ({positives} positive), {speakers} speakers -> {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
