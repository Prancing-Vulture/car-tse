#!/usr/bin/env python3
"""Audit the official SLURP release and optionally build flat manifests.

The SLURP repository keeps annotations in JSONL and links each annotation to
one or more recordings.  This utility does not alter the downloaded dataset.
It reports the official split sizes, microphone variant, label inventory,
speaker overlap, and whether the expected audio files are present.  Optional
flat manifests make the clean intent comparison reproducible without changing
the original annotation files.

The audio release is non-commercial (CC BY-NC 4.0); keep the dataset outside
the git repository and record its checksum/license separately.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SPLITS = ("train", "devel", "test", "train_synthetic")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
    return rows


def audio_root(dataset_root: Path, split: str) -> Path:
    return dataset_root / "audio" / ("slurp_synth" if split == "train_synthetic" else "slurp_real")


def speaker_id(
    row: dict[str, Any], recording_file: str, metadata: dict[str, Any]
) -> str | None:
    record = metadata.get(str(row.get("slurp_id")), {})
    recording = record.get("recordings", {}).get(recording_file, {})
    value = recording.get("usrid")
    return str(value) if value else None


def iter_recordings(
    rows: Iterable[dict[str, Any]], split: str, dataset_root: Path, metadata: dict[str, Any]
) -> Iterable[dict[str, Any]]:
    root = audio_root(dataset_root, split)
    for row in rows:
        for recording in row.get("recordings", []):
            filename = str(recording["file"])
            path = root / filename
            yield {
                "slurp_id": row.get("slurp_id"),
                "sentence": row.get("sentence", ""),
                "sentence_annotation": row.get("sentence_annotation", ""),
                "intent": row.get("intent", ""),
                "action": row.get("action", ""),
                "scenario": row.get("scenario", ""),
                "entities": row.get("entities", []),
                "recording_file": filename,
                "speaker_id": speaker_id(row, filename, metadata),
                "microphone": "close_talk" if filename.endswith("-headset.flac") else "distant",
                "audio_path": str(path),
                "wav": str(path),
                "audio_exists": path.is_file(),
                "recording_status": recording.get("status"),
                "wer": recording.get("wer"),
                "ent_wer": recording.get("ent_wer"),
                "official_split": split,
            }


def summarize_split(
    rows: list[dict[str, Any]], recordings: list[dict[str, Any]]
) -> dict[str, Any]:
    speakers = sorted({item["speaker_id"] for item in recordings if item["speaker_id"]})
    entity_types = sorted(
        {
            str(entity.get("type"))
            for row in rows
            for entity in row.get("entities", [])
            if entity.get("type")
        }
    )
    return {
        "annotation_rows": len(rows),
        "recordings": len(recordings),
        "audio_present": sum(bool(item["audio_exists"]) for item in recordings),
        "audio_missing": sum(not item["audio_exists"] for item in recordings),
        "close_talk_recordings": sum(item["microphone"] == "close_talk" for item in recordings),
        "distant_recordings": sum(item["microphone"] == "distant" for item in recordings),
        "speakers_with_metadata": len(speakers),
        "speaker_ids": speakers,
        "intents": len({row.get("intent") for row in rows}),
        "actions": len({row.get("action") for row in rows}),
        "scenarios": len({row.get("scenario") for row in rows}),
        "entity_types": entity_types,
        "intent_counts": dict(sorted(Counter(str(row.get("intent")) for row in rows).items())),
        "scenario_counts": dict(sorted(Counter(str(row.get("scenario")) for row in rows).items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("D:/datasets/slurp"),
        help="Official SLURP checkout containing dataset/ and audio/.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for the audit JSON summary.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=None,
        help="Optional directory for one flat JSONL manifest per official split.",
    )
    args = parser.parse_args()

    annotation_root = args.dataset_root / "dataset" / "slurp"
    metadata_path = annotation_root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing SLURP metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    # SLURP's raw ``intent`` strings are not a stable cross-split inventory
    # (for example, some partitions use ``music_likeness`` while another uses
    # ``likeness``).  The canonical 46-value ``action`` field is consistent
    # and matches the paper's primary semantic metric.  Synthetic-only actions
    # are left with label=-1 in the flat manifests.
    train_rows_for_labels = read_jsonl(annotation_root / "train.jsonl")
    train_actions = sorted({str(row.get("action")) for row in train_rows_for_labels})
    label_map = {action: index for index, action in enumerate(train_actions)}

    split_rows: dict[str, list[dict[str, Any]]] = {}
    split_recordings: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        path = annotation_root / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing SLURP annotation split: {path}")
        rows = read_jsonl(path)
        recordings = list(iter_recordings(rows, split, args.dataset_root, metadata))
        split_rows[split] = rows
        split_recordings[split] = recordings

    summary: dict[str, Any] = {
        "dataset_root": str(args.dataset_root.resolve()),
        "annotation_root": str(annotation_root.resolve()),
        "splits": {},
        "speaker_overlap": {},
        "real_train_action_label_map": label_map,
    }
    for split in SPLITS:
        summary["splits"][split] = summarize_split(split_rows[split], split_recordings[split])
        if args.manifest_dir is not None:
            args.manifest_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = args.manifest_dir / f"slurp_{split}.jsonl"
            with manifest_path.open("w", encoding="utf-8") as handle:
                for item in split_recordings[split]:
                    item = dict(item)
                    item["label"] = label_map.get(str(item["action"]), -1)
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    speaker_sets = {
        split: {item["speaker_id"] for item in split_recordings[split] if item["speaker_id"]}
        for split in SPLITS
    }
    for left, right in (("train", "devel"), ("train", "test"), ("devel", "test")):
        summary["speaker_overlap"][f"{left}_vs_{right}"] = sorted(speaker_sets[left] & speaker_sets[right])

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({
        "dataset_root": summary["dataset_root"],
        "splits": {
            split: {
                key: summary["splits"][split][key]
                for key in (
                    "annotation_rows",
                    "recordings",
                    "audio_present",
                    "audio_missing",
                    "close_talk_recordings",
                    "distant_recordings",
                    "speakers_with_metadata",
                    "intents",
                    "actions",
                    "scenarios",
                )
            }
            for split in SPLITS
        },
        "speaker_overlap_counts": {
            key: len(value) for key, value in summary["speaker_overlap"].items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
