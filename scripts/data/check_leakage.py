#!/usr/bin/env python
"""Ensure speakers/sessions do not leak across train/val/test."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import resolve_paths
from car_tse.io.manifest import read_jsonl


def speakers_in_manifest(path: Path) -> set[str]:
    return {str(r.get("speaker_id", r.get("id", ""))) for r in read_jsonl(path)}


def main() -> int:
    paths = resolve_paths()
    root = Path(paths["manifest_root"])
    manifests = {
        "train": list((root / "train").glob("stage1*.jsonl")),
        "val": list((root / "val").glob("stage1*.jsonl")),
        "test": list((root / "test").glob("stage1*.jsonl")),
    }
    spk_sets = {}
    for split, files in manifests.items():
        s: set[str] = set()
        for f in files:
            s |= speakers_in_manifest(f)
        spk_sets[split] = s
    leaked = False
    for a in ("train", "val"):
        for b in ("val", "test"):
            if a == b:
                continue
            inter = spk_sets[a] & spk_sets[b]
            if inter and "" not in inter:
                print(f"LEAK {a}/{b}: {len(inter)} speaker ids overlap")
                leaked = True
    if leaked:
        return 1
    print("No speaker leakage detected across stage1 manifests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
