#!/usr/bin/env python
"""Fail fast if expected corpus directories are missing."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import resolve_paths
from car_tse.data.prep_common import prep_argparser

REQUIRED = ["librispeech", "librimix"]
OPTIONAL = ["aishell5", "speech_commands_v2", "fluent_speech_commands"]


def main() -> int:
    args = prep_argparser("Verify dataset layout").parse_args()
    paths = resolve_paths(args.paths)
    missing = []
    for name in REQUIRED:
        p = paths["corpus_paths"][name]
        if not p.exists():
            missing.append(str(p))
    if missing:
        print("ERROR: missing required corpora:")
        for m in missing:
            print(f"  {m}")
        return 1
    for name in OPTIONAL:
        p = paths["corpus_paths"][name]
        status = "OK" if p.exists() else "skip (optional)"
        print(f"{name}: {status} -> {p}")
    print("Required corpora OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
