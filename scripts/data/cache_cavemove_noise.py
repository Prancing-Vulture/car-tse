#!/usr/bin/env python
"""Cache CAVEMOVE cabin recordings as mono 16 kHz NumPy arrays."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from car_tse.config import REPO_ROOT, resolve_paths
from car_tse.io.audio import read_mono


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "cache" / "cavemove_16k")
    args = parser.parse_args()
    paths = resolve_paths(args.paths)
    root = Path(paths["corpus_paths"]["cavemove"])
    files = sorted(p for p in root.rglob("*.wav") if p.parent.name in {"noise", "ventilation"})
    if not files:
        print(f"No CAVEMOVE noise/ventilation WAVs under {root}")
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(files, start=1):
        digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()
        target = args.output_dir / f"{digest}.npy"
        if not target.exists():
            wav, _ = read_mono(path, args.sample_rate)
            np.save(target, wav.astype(np.float32, copy=False), allow_pickle=False)
        if index % 5 == 0 or index == len(files):
            print(f"{index}/{len(files)} {path.parent.parent.name}/{path.parent.name}", flush=True)
    print(f"Cached {len(files)} CAVEMOVE recordings in {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
