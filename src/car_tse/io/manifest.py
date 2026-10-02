"""JSONL manifest read/write."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


MANIFEST_FIELDS = (
    "id",
    "mixture_wav",
    "enroll_wav",
    "target_wav",
    "sample_rate",
    "speaker_id",
    "seat",
    "condition",
    "target_present",
    "transcript",
    "intent",
    "playback_ref_wav",
    "meta",
)


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path | str, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
