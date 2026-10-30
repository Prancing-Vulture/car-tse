#!/usr/bin/env python
"""Create hard-negative Stage-1 mixtures from non-wake Speech Commands speech."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import REPO_ROOT
from car_tse.io.manifest import read_jsonl, write_jsonl
from car_tse.io.audio import read_mono


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--speech-root",
        type=Path,
        default=REPO_ROOT / "data/raw/speech_commands_v2/extracted",
    )
    parser.add_argument("--manifest-root", type=Path, default=REPO_ROOT / "manifests")
    parser.add_argument(
        "--marvin-manifest-name", default="stage1_gsc_marvin_cabin.jsonl"
    )
    parser.add_argument("--output-manifest-name", default="stage1_target_kws_nonwake_hardneg.jsonl")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data/processed/target_kws_nonwake_stage1",
    )
    parser.add_argument("--train-limit", type=int, default=1200)
    parser.add_argument("--val-limit", type=int, default=300)
    parser.add_argument("--test-limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()

    speech_root = args.speech_root if args.speech_root.is_absolute() else REPO_ROOT / args.speech_root
    manifest_root = args.manifest_root if args.manifest_root.is_absolute() else REPO_ROOT / args.manifest_root
    output_root = args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root
    rng = random.Random(args.seed)
    nonwake = sorted(
        path
        for path in speech_root.glob("*/*.wav")
        if path.parent.name.lower() != "marvin"
    )
    if not nonwake:
        raise RuntimeError(f"No non-wake Speech Commands clips found under {speech_root}")
    rng.shuffle(nonwake)
    limits = {"train": args.train_limit, "val": args.val_limit, "test": args.test_limit}
    for split, limit in limits.items():
        marvin_manifest = manifest_root / split / args.marvin_manifest_name
        marvin_rows = read_jsonl(marvin_manifest)
        target_rows = [row for row in marvin_rows if int(float(row.get("target_present", 0))) == 1]
        if not target_rows:
            raise RuntimeError(f"No target rows in {marvin_manifest}")
        split_root = output_root / split
        split_root.mkdir(parents=True, exist_ok=True)
        silence_path = split_root / "silence_3s.wav"
        if not silence_path.exists():
            sf.write(str(silence_path), np.zeros(3 * 16000, dtype=np.float32), 16000, subtype="FLOAT")
        rows: list[dict] = []
        start_index = sum(limits[name] for name in limits if list(limits).index(name) < list(limits).index(split))
        selected = [nonwake[(start_index + index) % len(nonwake)] for index in range(max(0, int(limit)))]
        for index, source_path in enumerate(selected):
            source, _ = read_mono(source_path, 16000)
            mixture = np.zeros(3 * 16000, dtype=np.float32)
            start = int(round(0.75 * 16000))
            mixture[start : min(len(mixture), start + len(source))] = source[: max(0, len(mixture) - start)]
            digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:12]
            stem = f"{split}_{index:05d}_{digest}"
            mixture_path = split_root / f"{stem}_mix.wav"
            sf.write(str(mixture_path), mixture, 16000, subtype="FLOAT")
            target_row = target_rows[index % len(target_rows)]
            rows.append(
                {
                    "id": f"target_kws_nonwake_{split}_{stem}",
                    "mixture_wav": str(mixture_path),
                    "enroll_wav": target_row["enroll_wav"],
                    "target_wav": str(silence_path),
                    "sample_rate": 16000,
                    "speaker_id": target_row.get("speaker_id", ""),
                    "speaker_label": target_row.get("speaker_label", 0),
                    "condition": "nonwake_speech_hard_negative",
                    "target_present": 0,
                    "cabin_noise_wav": target_row.get("cabin_noise_wav", ""),
                    "cabin_noise_channels": target_row.get("cabin_noise_channels", [0, 2, 4, 6]),
                    # Include the hard passenger case used by the continuous
                    # evaluator in the deterministic cache pass.
                    "cabin_noise_snr_db": target_row.get("cabin_noise_snr_db", 0.0),
                    "meta": {
                        "source": str(source_path),
                        "source_word": source_path.parent.name,
                        "passenger_snr_db": -5.0,
                        "target_kws_hard_negative": True,
                    },
                }
            )
        write_jsonl(manifest_root / split / args.output_manifest_name, rows)
        print(f"{split}: {len(rows)} hard negatives -> {manifest_root / split / args.output_manifest_name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
