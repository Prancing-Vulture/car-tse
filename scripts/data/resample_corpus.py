#!/usr/bin/env python
"""Resample manifest audio to target sample rate (cached paths in meta)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import soundfile as sf

from car_tse.config import resolve_paths
from car_tse.data.prep_common import prep_argparser
from car_tse.io.manifest import read_jsonl, write_jsonl


def resample_linear(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wav
    duration = wav.shape[0] / orig_sr
    n_out = int(duration * target_sr)
    x_old = np.linspace(0, 1, wav.shape[0], endpoint=False)
    x_new = np.linspace(0, 1, n_out, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


def main() -> int:
    parser = prep_argparser("Resample manifest wavs")
    parser.add_argument("--manifest", default="stage1_triplets.jsonl")
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()
    paths = resolve_paths(args.paths)
    target_sr = int(paths.get("sample_rate", 8000))
    cache = Path(paths["manifest_root"]) / "cache" / str(target_sr)
    cache.mkdir(parents=True, exist_ok=True)
    in_path = Path(paths["manifest_root"]) / args.split / args.manifest
    rows = read_jsonl(in_path)
    if args.max_rows:
        rows = rows[: args.max_rows]
    keys = ("mixture_wav", "enroll_wav", "target_wav", "playback_ref_wav")
    for row in rows:
        for key in keys:
            src = row.get(key)
            if not src or not Path(src).exists():
                continue
            data, sr = sf.read(src, always_2d=True)
            mono = data.mean(axis=1)
            if sr != target_sr:
                mono = resample_linear(mono, sr, target_sr)
            out = cache / f"{row['id']}_{key}.wav"
            sf.write(str(out), mono, target_sr)
            row[key] = str(out)
        row["sample_rate"] = target_sr
    write_jsonl(in_path, rows)
    print(f"Resampled {len(rows)} rows @ {target_sr} Hz -> {in_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
