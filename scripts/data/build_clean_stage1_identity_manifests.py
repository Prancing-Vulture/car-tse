#!/usr/bin/env python
"""Build speaker-disjoint clean English identity manifests for Stage 1.

Each example uses one FSC utterance as both the clean microphone mixture and
the clean target, with a different utterance from the same speaker as the
enrollment.  Four repeated microphone paths preserve the multichannel input
contract while making this an explicit clean-domain identity/preservation
adaptation set.  The resulting checkpoint must still be checked against the
noisy extraction regression before promotion.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")


def build_rows(rows: list[dict], max_rows: int, seed: int) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        wav = resolve(str(row["wav"]))
        speaker = str(row.get("speaker_id", wav.parent.name))
        if wav.exists():
            grouped[speaker].append({"wav": relative(wav), "speaker_id": speaker})

    rng = random.Random(seed)
    speakers = sorted(grouped)
    rng.shuffle(speakers)
    candidates: list[dict] = []
    for speaker in speakers:
        utterances = sorted(grouped[speaker], key=lambda item: item["wav"])
        if len(utterances) < 2:
            continue
        for index, target in enumerate(utterances):
            enrollment = utterances[(index + 1) % len(utterances)]
            wav = target["wav"]
            enroll = enrollment["wav"]
            candidates.append(
                {
                    "mixture_wav": wav,
                    "target_wav": wav,
                    "enroll_wav": enroll,
                    "mixture_multich_wavs": [wav, wav, wav, wav],
                    "speaker_id": speaker,
                    "target_present": 1,
                }
            )
    rng.shuffle(candidates)
    return candidates[:max_rows] if max_rows > 0 else candidates


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", type=Path, default=REPO_ROOT / "manifests/official/stage4_intent_fsc_train.jsonl")
    parser.add_argument("--val-source", type=Path, default=REPO_ROOT / "manifests/official/stage4_intent_fsc_valid.jsonl")
    parser.add_argument("--train-output", type=Path, default=REPO_ROOT / "manifests/train/stage1_english_clean_identity.jsonl")
    parser.add_argument("--val-output", type=Path, default=REPO_ROOT / "manifests/val/stage1_english_clean_identity.jsonl")
    parser.add_argument("--max-train", type=int, default=4096)
    parser.add_argument("--max-val", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()

    train_rows = build_rows(read_rows(args.train_source), args.max_train, args.seed)
    val_rows = build_rows(read_rows(args.val_source), args.max_val, args.seed + 1)
    write_rows(args.train_output, train_rows)
    write_rows(args.val_output, val_rows)
    print(json.dumps({"train": len(train_rows), "val": len(val_rows), "train_output": str(args.train_output), "val_output": str(args.val_output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
