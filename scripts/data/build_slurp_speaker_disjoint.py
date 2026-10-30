#!/usr/bin/env python3
"""Build a deterministic speaker-disjoint SLURP stress split.

The official SLURP JSONL partitions are retained for the published-protocol
score.  This utility combines only the real recordings from those partitions,
keeps every speaker together, and writes a separate 80/10/10 split.  It is a
stress/generalization protocol, not a replacement for the official SLURP
benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


OFFICIAL_SPLITS = ("train", "devel", "test")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=Path("D:/datasets/slurp/manifests"),
        help="Directory produced by audit_slurp.py.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("D:/datasets/slurp/manifests_speaker_disjoint"),
    )
    parser.add_argument("--seed", type=int, default=20261018)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for split in OFFICIAL_SPLITS:
        path = args.manifest_root / f"slurp_{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.extend(read_jsonl(path))
    rows = [row for row in rows if row.get("speaker_id") and row.get("label", -1) >= 0]
    if not rows:
        raise RuntimeError("No real speaker-labelled rows found")

    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_speaker[str(row["speaker_id"])].append(row)

    # Stable pseudo-random order makes the assignment independent of JSONL
    # ordering while keeping it reproducible without a global random state.
    rng = random.Random(args.seed)
    speakers = list(by_speaker)
    rng.shuffle(speakers)
    speakers.sort(key=lambda speaker: hashlib.sha256(f"{args.seed}:{speaker}".encode()).hexdigest())

    total = len(rows)
    targets = {"train": int(round(total * 0.80)), "devel": int(round(total * 0.10))}
    targets["test"] = total - targets["train"] - targets["devel"]
    assigned: dict[str, list[dict[str, Any]]] = {split: [] for split in OFFICIAL_SPLITS}
    assigned_speakers: dict[str, list[str]] = {split: [] for split in OFFICIAL_SPLITS}
    counts = {split: 0 for split in OFFICIAL_SPLITS}

    # Greedy assignment by target deficit keeps the number of recordings close
    # to 80/10/10 while never splitting a speaker.
    for speaker in speakers:
        group = by_speaker[speaker]
        split = max(
            OFFICIAL_SPLITS,
            key=lambda candidate: targets[candidate] - counts[candidate],
        )
        assigned[split].extend(group)
        assigned_speakers[split].append(speaker)
        counts[split] += len(group)

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "seed": args.seed,
        "source_manifest_root": str(args.manifest_root.resolve()),
        "protocol": "speaker_disjoint_stress_only",
        "total_recordings": total,
        "total_speakers": len(speakers),
        "splits": {},
    }
    for split in OFFICIAL_SPLITS:
        path = args.output_root / f"slurp_{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in assigned[split]:
                output_row = dict(row)
                output_row["derived_split"] = split
                output_row["source_official_split"] = row.get("official_split")
                handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
        split_rows = assigned[split]
        summary["splits"][split] = {
            "recordings": len(split_rows),
            "speakers": len(assigned_speakers[split]),
            "speaker_ids": sorted(assigned_speakers[split]),
            "raw_intents": len({row.get("intent") for row in split_rows}),
            "actions": len({row.get("action") for row in split_rows}),
            "distant_recordings": sum(row.get("microphone") == "distant" for row in split_rows),
            "close_talk_recordings": sum(row.get("microphone") == "close_talk" for row in split_rows),
            "missing_audio": sum(not row.get("audio_exists", False) for row in split_rows),
        }

    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "total_recordings": summary["total_recordings"],
        "total_speakers": summary["total_speakers"],
        "splits": {
            split: {
                key: summary["splits"][split][key]
                for key in ("recordings", "speakers", "raw_intents", "actions", "distant_recordings", "close_talk_recordings", "missing_audio")
            }
            for split in OFFICIAL_SPLITS
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
