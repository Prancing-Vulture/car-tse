#!/usr/bin/env python
"""Combine clean English identity and cabin-noise Stage 1 rows deterministically."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare(rows: list[dict], domain: str, limit: int, seed: int) -> list[dict]:
    rows = [dict(row) for row in rows]
    random.Random(seed).shuffle(rows)
    rows = rows[:limit] if limit > 0 else rows
    for row in rows:
        row.setdefault("domain", domain)
    return rows


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-train", type=Path, default=REPO_ROOT / "manifests/train/stage1_english_clean_identity.jsonl")
    parser.add_argument("--noisy-train", type=Path, default=REPO_ROOT / "manifests/train/stage1_dual_domain_rms010.jsonl")
    parser.add_argument("--clean-val", type=Path, default=REPO_ROOT / "manifests/val/stage1_english_clean_identity.jsonl")
    parser.add_argument("--noisy-val", type=Path, default=REPO_ROOT / "manifests/val/stage1_dual_domain_rms010.jsonl")
    parser.add_argument("--train-output", type=Path, default=REPO_ROOT / "manifests/train/stage1_english_clean_cabin_joint.jsonl")
    parser.add_argument("--val-output", type=Path, default=REPO_ROOT / "manifests/val/stage1_english_clean_cabin_joint.jsonl")
    parser.add_argument("--clean-train-limit", type=int, default=1024)
    parser.add_argument("--noisy-train-limit", type=int, default=1024)
    parser.add_argument("--clean-val-limit", type=int, default=512)
    parser.add_argument("--noisy-val-limit", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()

    train = prepare(read_rows(args.clean_train), "english_clean_identity", args.clean_train_limit, args.seed)
    train += prepare(read_rows(args.noisy_train), "cavemove_cabin_noise", args.noisy_train_limit, args.seed + 1)
    val = prepare(read_rows(args.clean_val), "english_clean_identity", args.clean_val_limit, args.seed + 2)
    val += prepare(read_rows(args.noisy_val), "cavemove_cabin_noise", args.noisy_val_limit, args.seed + 3)
    random.Random(args.seed + 4).shuffle(train)
    random.Random(args.seed + 5).shuffle(val)
    write(args.train_output, train)
    write(args.val_output, val)
    print(json.dumps({"train": len(train), "val": len(val), "train_domains": {"english_clean_identity": args.clean_train_limit, "cavemove_cabin_noise": args.noisy_train_limit}, "val_domains": {"english_clean_identity": args.clean_val_limit, "cavemove_cabin_noise": args.noisy_val_limit}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
