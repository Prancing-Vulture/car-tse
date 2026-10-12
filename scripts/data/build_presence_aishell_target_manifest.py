#!/usr/bin/env python
"""Build real AISHELL-5 target-present/target-absent presence manifests.

AISHELL-5 provides four far-field cabin channels (DX01C01--DX04C01),
near-field reference channels (DA02/DA03 in the released sessions), and
TextGrid intervals whose tier names identify the active speaker.  This script
uses non-empty near-field intervals as target-present examples and pairs them
with target-absent far-field sessions.  It is a presence experiment only:
AISHELL-5 does not contain the project's wake phrase labels.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


INTERVAL_START = re.compile(r"^\s*xmin\s*=\s*([0-9.eE+-]+)")
INTERVAL_END = re.compile(r"^\s*xmax\s*=\s*([0-9.eE+-]+)")
TEXT_VALUE = re.compile(r'^\s*text\s*=\s*"(.*)"\s*$')
TIER_NAME = re.compile(r'^\s*name\s*=\s*"([^"]+)"\s*$')


def parse_textgrid(path: Path, min_duration: float, max_duration: float) -> tuple[str, list[tuple[float, float, str]]]:
    tier = path.stem
    intervals: list[tuple[float, float, str]] = []
    start: float | None = None
    end: float | None = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            tier_match = TIER_NAME.match(line)
            if tier_match and tier == path.stem:
                tier = tier_match.group(1)
            start_match = INTERVAL_START.match(line)
            if start_match:
                start = float(start_match.group(1))
                continue
            end_match = INTERVAL_END.match(line)
            if end_match:
                end = float(end_match.group(1))
                continue
            text_match = TEXT_VALUE.match(line)
            if text_match and start is not None and end is not None:
                text = text_match.group(1).replace('\\"', '"').strip()
                duration = end - start
                if text and min_duration <= duration <= max_duration:
                    intervals.append((start, end, text))
                start = None
                end = None
    return tier, intervals


def session_rows(session: Path, channels: list[str], max_intervals: int, min_duration: float, max_duration: float) -> list[dict]:
    channel_paths = [session / name for name in channels]
    if not all(path.exists() for path in channel_paths):
        return []
    rows: list[dict] = []
    for textgrid in sorted(session.glob("DA*.TextGrid")):
        nearfield = textgrid.with_suffix(".wav")
        if not nearfield.exists():
            continue
        speaker, intervals = parse_textgrid(textgrid, min_duration, max_duration)
        for interval_index, (start, end, text) in enumerate(intervals[:max_intervals]):
            # Use a one-second enrollment crop from the same near-field turn;
            # the Stage 1 cache expands/pads it to its configured enrollment size.
            enroll_start = start
            rows.append(
                {
                    "id": f"aishell5_present_{session.parent.name}_{session.name}_{speaker}_{interval_index:03d}",
                    "mixture_wav": str(channel_paths[0]),
                    "mixture_multich_wavs": [str(path) for path in channel_paths],
                    "enroll_wav": str(nearfield),
                    "target_wav": str(nearfield),
                    "sample_rate": 16000,
                    "speaker_id": speaker,
                    "target_present": 1,
                    "condition": "aishell5_farfield_target_present",
                    "segment_start_seconds": max(0.0, start - 0.25),
                    "segment_end_seconds": end,
                    "enroll_start_seconds": enroll_start,
                    "meta": {"source": "AISHELL-5", "text": text, "session": session.name},
                }
            )
    return rows


def absent_row(session: Path, channels: list[str], enrollment: Path, index: int, split: str) -> dict | None:
    channel_paths = [session / name for name in channels]
    if not all(path.exists() for path in channel_paths):
        return None
    return {
        "id": f"aishell5_absent_{split}_{session.name}_{index:05d}",
        "mixture_wav": str(channel_paths[0]),
        "mixture_multich_wavs": [str(path) for path in channel_paths],
        "enroll_wav": str(enrollment),
        "target_wav": str(channel_paths[0]),
        "sample_rate": 16000,
        "speaker_id": f"absent_{split}",
        "target_present": 0,
        "condition": "aishell5_farfield_target_absent",
        "meta": {"source": "AISHELL-5", "split": split},
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build AISHELL-5 target-present presence manifests")
    parser.add_argument("--aishell-root", type=Path, default=Path("data/raw/aishell5/extracted"))
    parser.add_argument("--output-root", type=Path, default=Path("manifests/presence_aishell_target"))
    parser.add_argument("--channels", nargs="+", default=["DX01C01.wav", "DX02C01.wav", "DX03C01.wav", "DX04C01.wav"])
    parser.add_argument("--max-intervals-per-speaker", type=int, default=4)
    parser.add_argument("--max-positive-rows", type=int, default=1200)
    parser.add_argument("--min-duration", type=float, default=0.4)
    parser.add_argument("--max-duration", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    root = args.aishell_root.resolve()
    output_root = args.output_root.resolve()
    randomizer = random.Random(args.seed)
    official_splits = {"train": "train", "val": "Dev", "test": "Eval1"}
    for split, source_split in official_splits.items():
        sessions = sorted(path for path in (root / source_split).iterdir() if path.is_dir())
        positives: list[dict] = []
        for session in sessions:
            positives.extend(
                session_rows(
                    session,
                    list(args.channels),
                    max(1, int(args.max_intervals_per_speaker)),
                    float(args.min_duration),
                    float(args.max_duration),
                )
            )
        randomizer.shuffle(positives)
        if not positives:
            raise SystemExit(f"No target-present intervals found in {root / source_split}")
        positives = positives[: max(1, int(args.max_positive_rows))]
        available_enrollments = [Path(row["enroll_wav"]) for row in positives]
        negatives: list[dict] = []
        for index in range(len(positives)):
            session = sessions[index % len(sessions)]
            row = absent_row(session, list(args.channels), available_enrollments[index % len(available_enrollments)], index, split)
            if row is not None:
                negatives.append(row)
        rows = positives + negatives[: len(positives)]
        randomizer.shuffle(rows)
        output = output_root / split / "stage2_presence_aishell_target.jsonl"
        write_jsonl(output, rows)
        print(
            f"{split}: {len(positives)} present + {len(positives)} absent = {len(rows)} rows "
            f"from {len(sessions)} sessions -> {output}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
