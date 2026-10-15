#!/usr/bin/env python
"""Build speaker-disjoint clean-English manifests for Stage-2 presence.

Positive rows pair two utterances from the same FSC speaker. Negative rows use
an utterance from another speaker with the same enrollment reference. The
manifest is a clean-domain diagnostic; it must be tested against the retained
noisy-domain presence head before promotion.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(resolved)


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_rows(source: Path, *, max_rows: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for row in read_rows(source):
        wav = resolve(str(row["wav"]))
        speaker = str(row.get("speaker_id") or wav.parent.name)
        if wav.exists():
            grouped[speaker].append(wav)
    for speaker in grouped:
        grouped[speaker] = sorted(set(grouped[speaker]))

    speakers = sorted(speaker for speaker, wavs in grouped.items() if len(wavs) >= 2)
    if len(speakers) < 2:
        raise RuntimeError("clean presence manifest needs at least two speakers with two utterances each")
    rng = random.Random(seed)
    positive: list[dict[str, Any]] = []
    for speaker in speakers:
        utterances = grouped[speaker]
        for index, target in enumerate(utterances):
            enrollment = utterances[(index + 1) % len(utterances)]
            target_ref = repo_relative(target)
            enrollment_ref = repo_relative(enrollment)
            positive.append(
                {
                    "mixture_wav": target_ref,
                    "mixture_multich_wavs": [target_ref] * 4,
                    "enroll_wav": enrollment_ref,
                    "target_wav": target_ref,
                    "speaker_id": speaker,
                    "source_speaker_id": speaker,
                    "group_id": speaker,
                    "target_present": 1,
                    "condition": "english_clean_same_speaker",
                    "meta": {"source": "FSC", "domain": "clean_english"},
                }
            )

    negative: list[dict[str, Any]] = []
    for index, enrollment_speaker in enumerate(speakers):
        other_speaker = speakers[(index + 1) % len(speakers)]
        enrollment = grouped[enrollment_speaker][0]
        for target in grouped[other_speaker]:
            target_ref = repo_relative(target)
            negative.append(
                {
                    "mixture_wav": target_ref,
                    "mixture_multich_wavs": [target_ref] * 4,
                    "enroll_wav": repo_relative(enrollment),
                    "target_wav": target_ref,
                    "speaker_id": enrollment_speaker,
                    "source_speaker_id": other_speaker,
                    "group_id": enrollment_speaker,
                    "target_present": 0,
                    "condition": "english_clean_non_target_speaker",
                    "meta": {"source": "FSC", "domain": "clean_english"},
                }
            )

    rng.shuffle(positive)
    rng.shuffle(negative)
    if max_rows > 0:
        per_class = max(1, int(max_rows) // 2)
        positive = positive[:per_class]
        negative = negative[:per_class]
    rows = positive + negative
    rng.shuffle(rows)
    for index, row in enumerate(rows):
        row["id"] = f"clean_presence_{index:06d}"
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=REPO_ROOT / "manifests/official/stage4_intent_fsc_train.jsonl")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "manifests/train/stage2_presence_english_clean_identity.jsonl")
    parser.add_argument("--max-rows", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    rows = build_rows(args.source, max_rows=int(args.max_rows), seed=int(args.seed))
    write_rows(args.output, rows)
    counts = {"positive": sum(int(row["target_present"]) for row in rows), "negative": sum(int(not row["target_present"]) for row in rows)}
    print(json.dumps({"rows": len(rows), "counts": counts, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
