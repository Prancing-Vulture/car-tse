#!/usr/bin/env python
"""Add real in-car command speech as hard negatives for binary KWS."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import REPO_ROOT
from car_tse.io.manifest import read_jsonl, write_jsonl


def hard_negative(row: dict, split: str) -> dict:
    return {
        "wav": row["wav"],
        "label": 0,
        "split": split,
        "wake_word": "marvin",
        "noise_only": False,
        "cabin_noise_wav": row.get("cabin_noise_wav"),
        "cabin_noise_snr_db": row.get("cabin_noise_snr_db", 10.0),
        "meta": {
            "source": "fluent_speech_commands_hard_negative",
            "intent": row.get("intent", ""),
            "cabin_noise_dataset": "CAVEMOVE",
            "cabin_noise_type": (row.get("meta") or {}).get("cabin_noise_type", "noise"),
            "cabin_vehicle": (row.get("meta") or {}).get("cabin_vehicle", "unknown"),
            "kws_hard_negative": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, default=REPO_ROOT / "manifests")
    parser.add_argument("--max-hard-train", type=int, default=5130)
    parser.add_argument("--max-hard-val", type=int, default=585)
    parser.add_argument("--max-hard-test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    root = args.manifest_root
    base_names = {
        "train": "stage3_kws_cabin_train.jsonl",
        "val": "stage3_kws_cabin_val.jsonl",
        "test": "stage3_kws_cabin_test.jsonl",
    }
    source_train = read_jsonl(root / "train" / "stage4_cavemove_intent.jsonl")
    source_val = read_jsonl(root / "val" / "stage4_cavemove_intent.jsonl")
    if not source_train or not source_val:
        print("Missing stage4 CAVEMOVE intent manifests")
        return 1
    rng = random.Random(args.seed)
    train_pool = list(source_train)
    rng.shuffle(train_pool)
    train_rows = train_pool[: min(args.max_hard_train, len(train_pool))]
    test_rows = train_pool[min(args.max_hard_train, len(train_pool)) :]
    rng_val = random.Random(args.seed + 1)
    rng_val.shuffle(source_val)
    val_rows = source_val[: min(args.max_hard_val, len(source_val))]
    selected = {"train": train_rows, "val": val_rows, "test": test_rows[: args.max_hard_test]}

    for split, filename in base_names.items():
        base = read_jsonl(root / split / filename)
        if not base:
            print(f"Missing base KWS manifest: {root / split / filename}")
            return 1
        hard = [hard_negative(row, split) for row in selected[split]]
        combined = base + hard
        random.Random(args.seed + 10 + {"train": 0, "val": 1, "test": 2}[split]).shuffle(combined)
        output_name = filename.replace("stage3_kws_cabin", "stage3_kws_cabin_hardneg")
        output = root / split / output_name
        write_jsonl(output, combined)
        positives = sum(int(row.get("label") == 1) for row in combined)
        hard_count = sum(int((row.get("meta") or {}).get("kws_hard_negative", False)) for row in combined)
        print(f"{split}: rows={len(combined)} positives={positives} hard_negatives={hard_count} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
