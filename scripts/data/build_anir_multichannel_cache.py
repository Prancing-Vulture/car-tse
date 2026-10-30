#!/usr/bin/env python
"""Build a small 16-kHz multichannel ANIR noise cache for augmentation.

ANIR is distributed as one mono WAV per microphone at 48 kHz.  This utility
joins four explicitly selected microphones per condition and resamples them
to the 16-kHz contract used by the current car-TSE pipeline.  The cache is
written outside the repository by default because the source archive is large.
It contains noise only; it is not an evaluation or speech corpus.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


NAME_RE = re.compile(r"^con_(?P<condition>[^_]+_[^_]+)_mic_(?P<microphone>\d+)\.wav$", re.IGNORECASE)


def resample_linear(data: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(data, dtype=np.float32)
    data = np.asarray(data, dtype=np.float32)
    output_length = max(1, int(round(data.shape[0] * target_rate / source_rate)))
    old_x = np.linspace(0.0, 1.0, data.shape[0], endpoint=False)
    new_x = np.linspace(0.0, 1.0, output_length, endpoint=False)
    return np.stack(
        [np.interp(new_x, old_x, data[:, index]) for index in range(data.shape[1])],
        axis=1,
    ).astype(np.float32)


def discover(root: Path, microphones: list[int]) -> list[dict[str, Any]]:
    groups: dict[str, dict[int, Path]] = {}
    for path in sorted(root.rglob("*.wav")):
        match = NAME_RE.match(path.name)
        if match is None:
            continue
        groups.setdefault(match.group("condition"), {})[
            int(match.group("microphone"))
        ] = path
    rows: list[dict[str, Any]] = []
    for condition in sorted(groups):
        paths = groups[condition]
        if all(microphone in paths for microphone in microphones):
            rows.append(
                {
                    "condition": condition,
                    "microphones": list(microphones),
                    "source_paths": [str(paths[microphone]) for microphone in microphones],
                }
            )
    return rows


def build_row(row: dict[str, Any], output_root: Path, target_rate: int) -> dict[str, Any]:
    arrays: list[np.ndarray] = []
    source_rates: list[int] = []
    for source_path in row["source_paths"]:
        values, source_rate = sf.read(source_path, always_2d=True, dtype="float32")
        arrays.append(values.mean(axis=1).astype(np.float32))
        source_rates.append(int(source_rate))
    if len(set(source_rates)) != 1:
        raise ValueError(f"Mixed source rates in {row['condition']}: {source_rates}")
    length = min(len(values) for values in arrays)
    joined = np.stack([values[:length] for values in arrays], axis=1)
    joined = resample_linear(joined, source_rates[0], target_rate)
    output_path = output_root / f"anir_{row['condition']}_m{''.join(str(x) for x in row['microphones'])}.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), joined, target_rate, subtype="FLOAT")
    return {
        "id": f"anir_{row['condition']}_m{''.join(str(x) for x in row['microphones'])}",
        "condition": row["condition"],
        "microphones": row["microphones"],
        "path": str(output_path),
        "sample_rate_hz": int(target_rate),
        "channels": int(joined.shape[1]),
        "frames": int(joined.shape[0]),
        "duration_seconds": float(joined.shape[0] / target_rate),
        "source_sample_rate_hz": int(source_rates[0]),
        "source_paths": row["source_paths"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("D:/datasets/anir/v_class_noise_files"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("D:/datasets/anir/v_class_noise_16k_mc"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/external/anir_v_class_noise_16k_mc.json"),
    )
    parser.add_argument("--microphones", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    output_root = args.output_root.resolve()
    manifest_path = args.manifest.resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    if len(args.microphones) != 4:
        raise ValueError("The current spatial frontend contract requires exactly four microphones")
    rows = discover(root, list(args.microphones))
    if not rows:
        raise RuntimeError(f"No complete ANIR groups found under {root}")
    output_root.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        output_path = output_root / f"anir_{row['condition']}_m{''.join(str(x) for x in row['microphones'])}.wav"
        if output_path.exists() and not args.force:
            info = sf.info(str(output_path))
            output_rows.append(
                {
                    "id": f"anir_{row['condition']}_m{''.join(str(x) for x in row['microphones'])}",
                    "condition": row["condition"],
                    "microphones": row["microphones"],
                    "path": str(output_path),
                    "sample_rate_hz": int(info.samplerate),
                    "channels": int(info.channels),
                    "frames": int(info.frames),
                    "duration_seconds": float(info.duration),
                    "source_sample_rate_hz": 48000,
                    "source_paths": row["source_paths"],
                }
            )
        else:
            output_rows.append(build_row(row, output_root, int(args.sample_rate)))
        print(f"{index}/{len(rows)} {row['condition']}", flush=True)

    manifest = {
        "dataset": "ANIR Corpus - V class background noise",
        "root": str(root),
        "output_root": str(output_root),
        "sample_rate_hz": int(args.sample_rate),
        "microphones": list(args.microphones),
        "file_count": len(output_rows),
        "intended_use": "multichannel automotive-noise augmentation only",
        "not_evaluation_evidence": True,
        "source_archive_sha256": "8CC71CA50F89909D475798761669ABEA788C6C2CA5F1E022480EB63F5022A441",
        "files": output_rows,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"file_count": len(output_rows), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
