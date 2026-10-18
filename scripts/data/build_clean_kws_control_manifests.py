#!/usr/bin/env python
"""Build raw-English, speech-only KWS control manifests.

The source cabin manifests contain raw Speech Commands clips plus optional
CAVEMOVE metadata. This control intentionally drops ``noise_only`` rows and
retains only the clean WAV path and binary Marvin label; no cabin bed is added
by the clean-control config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    source_dir = (args.source_dir or repo_root / "manifests").resolve()
    output_dir = (args.output_dir or repo_root / "manifests" / "clean_control").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        source = source_dir / split / f"stage3_kws_cabin_{split}.jsonl"
        if not source.exists():
            raise FileNotFoundError(source)
        rows: list[dict[str, object]] = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if bool(row.get("noise_only", False)):
                continue
            rows.append(
                {
                    "wav": str(row["wav"]),
                    "label": int(row["label"]),
                    "split": split,
                    "wake_word": "marvin",
                    "clean_control": True,
                    "source_manifest": str(source),
                }
            )
        destination = output_dir / f"stage3_kws_clean_{split}.jsonl"
        destination.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        counts[split] = len(rows)
    print(json.dumps({"output_dir": str(output_dir), "counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
