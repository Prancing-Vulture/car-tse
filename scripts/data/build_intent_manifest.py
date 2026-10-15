#!/usr/bin/env python
"""Fluent Speech Commands (+ optional SLURP / custom) intent manifests."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import resolve_paths
from car_tse.data.prep_common import manifest_dir, prep_argparser
from car_tse.io.manifest import write_jsonl


def scan_fsc(root: Path) -> tuple[list[dict], dict[str, int]]:
    rows = []
    label_map: dict[str, int] = {}
    # The Zenodo archive extracts with ``data/`` directly below the dataset
    # root; also accept the nested layout used by older mirrors.
    tsv = root / "data" / "train_data.csv"
    if not tsv.exists():
        tsv = root / "data" / "fluent_speech_commands_dataset" / "data" / "train_data.csv"
    if not tsv.exists():
        for wav in root.rglob("*.wav"):
            intent = wav.parent.name
            if intent not in label_map:
                label_map[intent] = len(label_map)
            rows.append({"wav": str(wav), "label": label_map[intent], "intent": intent})
        return rows, label_map
    with open(tsv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # FSC's ``action`` column alone collapses the task to six labels.
            # Use the action/object/location tuple to preserve the actual
            # command intent taxonomy.
            intent = r.get("intent", "")
            if not intent:
                parts = [r.get("action", ""), r.get("object", ""), r.get("location", "")]
                intent = "|".join(p for p in parts if p and p != "none") or "unknown"
            if intent not in label_map:
                label_map[intent] = len(label_map)
            wav_path = root / r.get("wav_path", r.get("path", ""))
            if wav_path.exists():
                rows.append({"wav": str(wav_path), "label": label_map[intent], "intent": intent})
    return rows, label_map


def main() -> int:
    parser = prep_argparser("Build intent manifests")
    args = parser.parse_args()
    paths = resolve_paths(args.paths)
    fsc = paths["corpus_paths"]["fluent_speech_commands"]
    rows, label_map = scan_fsc(fsc)
    custom = paths["corpus_paths"]["custom_incar"]
    if custom.exists():
        for wav in custom.rglob("*.wav"):
            intent = wav.parent.name
            if intent not in label_map:
                label_map[intent] = len(label_map)
            rows.append({"wav": str(wav), "label": label_map[intent], "intent": intent})
    mid = len(rows)
    train_rows = rows[: int(0.9 * mid)] if mid else []
    eval_rows = rows[int(0.9 * mid) :] if mid else []
    meta = Path(paths["manifest_root"]) / "meta" / "intent_label_map.json"
    meta.parent.mkdir(parents=True, exist_ok=True)
    with open(meta, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)
    write_jsonl(manifest_dir(paths, "train") / "stage4_intent_train.jsonl", train_rows)
    write_jsonl(manifest_dir(paths, "val") / "stage4_intent_eval.jsonl", eval_rows)
    print(f"Intent train={len(train_rows)} val={len(eval_rows)} labels={len(label_map)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
