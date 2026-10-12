#!/usr/bin/env python
"""Index an extracted ANIR background-noise archive without copying audio."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import soundfile as sf


NAME_RE = re.compile(r"^con_(?P<condition>[^_]+_[^_]+)_mic_(?P<microphone>\d+)\.wav$", re.IGNORECASE)


def inspect(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    unmatched: list[str] = []
    for path in sorted(root.rglob("*.wav")):
        match = NAME_RE.match(path.name)
        if match is None:
            unmatched.append(str(path))
            continue
        info = sf.info(str(path))
        rows.append(
            {
                "path": str(path),
                "condition": match.group("condition"),
                "microphone": int(match.group("microphone")),
                "sample_rate_hz": int(info.samplerate),
                "channels": int(info.channels),
                "frames": int(info.frames),
                "duration_seconds": float(info.duration),
                "subtype": str(info.subtype),
            }
        )
    return {
        "dataset": "ANIR Corpus - V class background noise",
        "root": str(root),
        "file_count": len(rows),
        "unmatched_file_count": len(unmatched),
        "files": rows,
        "unmatched_files": unmatched,
        "interpretation": "Noise-only automotive recordings for augmentation and target-absent diagnostics; not speech or wake-word supervision.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inspect(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("root", "file_count", "unmatched_file_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
