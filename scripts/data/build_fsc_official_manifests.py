#!/usr/bin/env python
"""Build official Fluent Speech Commands train/valid/test manifests.

The original local preparation script used a 90/10 slice of train_data.csv.
That is useful for a quick bootstrap run, but it is not the official FSC
validation/test protocol. This script keeps the existing label IDs and emits
separate manifests without changing any training data or checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def intent_name(row: dict[str, str]) -> str:
    parts = [row.get("action", ""), row.get("object", ""), row.get("location", "")]
    return "|".join(part for part in parts if part and part != "none") or "unknown"


def build_split(root: Path, split: str, label_map: dict[str, int]) -> list[dict]:
    csv_path = root / "data" / f"{split}_data.csv"
    rows: list[dict] = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            intent = intent_name(row)
            if intent not in label_map:
                raise KeyError(f"Intent {intent!r} is absent from the existing label map")
            wav = root / row["path"]
            if not wav.exists():
                raise FileNotFoundError(wav)
            rows.append(
                {
                    "wav": str(wav),
                    "label": label_map[intent],
                    "intent": intent,
                    "transcript": row.get("transcription", ""),
                    "speaker_id": row.get("speakerId", ""),
                    "official_split": split,
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/fluent_speech_commands/extracted/fluent_speech_commands_dataset"),
    )
    parser.add_argument("--label-map", type=Path, default=Path("manifests/meta/intent_label_map.json"))
    parser.add_argument("--output-root", type=Path, default=Path("manifests/official"))
    args = parser.parse_args()

    label_map = {str(k): int(v) for k, v in json.loads(args.label_map.read_text(encoding="utf-8")).items()}
    args.output_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        rows = build_split(args.dataset_root, split, label_map)
        out = args.output_root / f"stage4_intent_fsc_{split}.jsonl"
        with out.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
