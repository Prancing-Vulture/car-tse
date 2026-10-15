#!/usr/bin/env python
"""Build fusion-training rows from continuous evaluator JSON reports.

Pass development reports with explicit train/validation assignment. This script
rejects target speaker IDs shared across splits to reduce speaker leakage.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_split(paths: list[Path], split: str) -> tuple[list[dict], set[str]]:
    output = []
    speakers: set[str] = set()
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        for scene in report.get("scenes_detail", []):
            truth = scene.get("truth", {})
            wake_start = truth.get("target_wake_start_seconds")
            wake_end = truth.get("target_wake_end_seconds")
            source = truth.get("target_wake_source", {})
            speaker = str(source.get("speaker_id", ""))
            if not speaker or wake_start is None or wake_end is None:
                raise ValueError(f"{path}: scene lacks target speaker or wake timing")
            speakers.add(speaker)
            for window in scene.get("windows", []):
                presence = window.get("presence_probability")
                wake = window.get("wake_model_probability", window.get("wake_probability"))
                start = window.get("kws_crop_start_seconds")
                end = window.get("kws_crop_end_seconds")
                if None in (presence, wake, start, end) or end <= start:
                    continue
                overlap = max(0.0, min(float(end), float(wake_end)) - max(float(start), float(wake_start)))
                # A crop is positive only when at least half of its duration is
                # labeled wake speech. Partial boundary crops remain negatives.
                label = int(overlap / (float(end) - float(start)) >= 0.5)
                output.append({
                    "presence_probability": float(presence),
                    "wake_probability": float(wake),
                    "speaker_similarity": window.get("speaker_similarity"),
                    "label": label,
                    "split": split,
                    "speaker_id": speaker,
                    "source_report": path.name,
                })
    return output, speakers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-report", type=Path, action="append", required=True)
    parser.add_argument("--val-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train, train_speakers = load_split(args.train_report, "train")
    val, val_speakers = load_split(args.val_report, "val")
    overlap = train_speakers & val_speakers
    if overlap:
        raise ValueError(f"target speakers occur in both splits: {sorted(overlap)}")
    for split, rows in (("train", train), ("val", val)):
        if not any(row["label"] == 1 for row in rows) or not any(row["label"] == 0 for row in rows):
            raise ValueError(f"{split} split must contain positive and negative windows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["presence_probability", "wake_probability", "speaker_similarity", "label", "split", "speaker_id", "source_report"]
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(train + val)
    print(f"wrote={args.output} train_rows={len(train)} val_rows={len(val)} heldout_speakers={len(val_speakers)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
