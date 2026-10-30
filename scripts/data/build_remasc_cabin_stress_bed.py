#!/usr/bin/env python
"""Build a reproducible long stress bed from a ReMASC vehicle recording.

This is an evaluation stress asset only.  It tiles one real ReMASC vehicle
recording so the existing continuous proxy evaluator can exercise its normal
long-stream path.  The resulting stream must not be treated as a natural
continuous driver/passenger session or as production evidence.
"""

from __future__ import annotations

import argparse
import json
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if int(source_rate) == int(target_rate):
        return np.asarray(audio, dtype=np.float32)
    factor = gcd(int(source_rate), int(target_rate))
    up = int(target_rate) // factor
    down = int(source_rate) // factor
    return np.asarray(resample_poly(audio, up, down, axis=0), dtype=np.float32)


def build_bed(source: Path, output: Path, *, sample_rate: int, duration_seconds: float) -> dict[str, object]:
    audio, source_rate = sf.read(str(source), always_2d=True, dtype="float32")
    audio = resample_audio(audio, int(source_rate), int(sample_rate))
    samples = max(1, int(round(float(duration_seconds) * int(sample_rate))))
    repeats = int(np.ceil(samples / max(len(audio), 1)))
    tiled = np.tile(audio, (repeats, 1))[:samples]
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), tiled, int(sample_rate), subtype="FLOAT")
    metadata = {
        "source": str(source),
        "source_sample_rate_hz": int(source_rate),
        "source_channels": int(audio.shape[1]),
        "output": str(output),
        "output_sample_rate_hz": int(sample_rate),
        "output_channels": int(tiled.shape[1]),
        "output_duration_seconds": float(len(tiled) / int(sample_rate)),
        "construction": "resample if needed, then tile the short ReMASC recording",
        "use_decision": "synthetic stress diagnostic only; not real-driver evidence",
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    args = parser.parse_args()
    if not args.source.exists():
        raise FileNotFoundError(args.source)
    if args.duration_seconds <= 0:
        raise ValueError("--duration-seconds must be positive")
    print(json.dumps(build_bed(
        args.source,
        args.output,
        sample_rate=args.sample_rate,
        duration_seconds=args.duration_seconds,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
