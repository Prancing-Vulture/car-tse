#!/usr/bin/env python
"""Build a speaker-disjoint English spatial cabin set for Stage-1 adaptation.

Clean LibriSpeech speech is rendered through CAVEMOVE driver/passenger
impulse responses and mixed with CAVEMOVE vehicle noise.  The generated files
are deliberately kept as a separate, uniquely named research artifact; no
existing manifest or checkpoint is overwritten.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "eval"))

from car_tse.io.audio import read_mono  # noqa: E402
from car_tse.io.manifest import write_jsonl  # noqa: E402
from eval_english_spatial_zone_ablation import (  # noqa: E402
    _load_noise,
    _render_clip,
    _select_noise,
    load_spatial_calibration,
    rms,
)


SAMPLE_RATE = 16_000
SEGMENT_SAMPLES = 3 * SAMPLE_RATE
ENROLLMENT_SAMPLES = int(1.5 * SAMPLE_RATE)


def _speaker_id(path: Path) -> str:
    return path.parent.parent.name


def _crop(wav: np.ndarray, samples: int, rng: np.random.Generator) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if len(wav) >= samples:
        start = int(rng.integers(0, len(wav) - samples + 1))
        return wav[start : start + samples].copy()
    out = np.zeros(samples, dtype=np.float32)
    out[: len(wav)] = wav
    return out


def _render_and_scale(
    clip: np.ndarray,
    impulse_response: np.ndarray,
    target_rms: float,
    snr_db: float,
) -> np.ndarray:
    rendered = _render_clip(clip, impulse_response)
    current = float(np.sqrt(np.mean(rendered * rendered) + 1.0e-8))
    desired = target_rms * (10.0 ** (-float(snr_db) / 20.0))
    if current > 1.0e-8:
        rendered *= desired / current
    return rendered.astype(np.float32, copy=False)


def build_rows(
    files_by_speaker: dict[str, list[Path]],
    speakers: list[str],
    noise_files: list[Path],
    calibration: dict[str, Any],
    output_root: Path,
    split: str,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    usable_speakers = [speaker for speaker in speakers if len(files_by_speaker[speaker]) >= 2]
    if len(usable_speakers) < 2:
        raise RuntimeError(f"Not enough speakers with two utterances for {split}")

    for index in range(int(count)):
        target_speaker = usable_speakers[index % len(usable_speakers)]
        passenger_candidates = [speaker for speaker in usable_speakers if speaker != target_speaker]
        passenger_speaker = passenger_candidates[int(rng.integers(0, len(passenger_candidates)))]
        target_path = files_by_speaker[target_speaker][int(rng.integers(0, len(files_by_speaker[target_speaker])))]
        enroll_candidates = [path for path in files_by_speaker[target_speaker] if path != target_path]
        enrollment_path = enroll_candidates[int(rng.integers(0, len(enroll_candidates)))]
        passenger_path = files_by_speaker[passenger_speaker][int(rng.integers(0, len(files_by_speaker[passenger_speaker])))]
        target, _ = read_mono(target_path, SAMPLE_RATE)
        enrollment, _ = read_mono(enrollment_path, SAMPLE_RATE)
        passenger, _ = read_mono(passenger_path, SAMPLE_RATE)
        target = _crop(target, SEGMENT_SAMPLES, rng)
        enrollment = _crop(enrollment, ENROLLMENT_SAMPLES, rng)
        passenger = _crop(passenger, SEGMENT_SAMPLES, rng)

        target_render = _render_clip(target, calibration["driver_ir"])
        target_level = float(np.sqrt(np.mean(target_render * target_render) + 1.0e-8))
        passenger_render = _render_and_scale(
            passenger,
            calibration["passenger_ir"],
            target_level,
            float(rng.choice([-10.0, -5.0, 0.0, 5.0])),
        )
        noise_path = noise_files[index % len(noise_files)]
        noise = _load_noise(noise_path, SEGMENT_SAMPLES)
        noise_level = float(np.sqrt(np.mean(noise * noise) + 1.0e-8))
        cabin_snr = float(rng.choice([-5.0, 0.0, 5.0, 10.0]))
        noise *= target_level * (10.0 ** (-cabin_snr / 20.0)) / max(noise_level, 1.0e-8)
        multichannel = target_render[:SEGMENT_SAMPLES] + passenger_render[:SEGMENT_SAMPLES] + noise
        mixture = multichannel.mean(axis=1).astype(np.float32)

        stem = f"{split}_{index:05d}"
        mixture_wav = output_root / f"{stem}_mixture.wav"
        multichannel_wav = output_root / f"{stem}_multichannel.wav"
        target_wav = output_root / f"{stem}_target.wav"
        enrollment_wav = output_root / f"{stem}_enrollment.wav"
        sf.write(str(mixture_wav), mixture, SAMPLE_RATE, subtype="FLOAT")
        sf.write(str(multichannel_wav), multichannel, SAMPLE_RATE, subtype="FLOAT")
        sf.write(str(target_wav), target, SAMPLE_RATE, subtype="FLOAT")
        sf.write(str(enrollment_wav), enrollment, SAMPLE_RATE, subtype="FLOAT")
        rows.append(
            {
                "id": stem,
                "mixture_wav": str(mixture_wav),
                "mixture_multich_wav": str(multichannel_wav),
                "enroll_wav": str(enrollment_wav),
                "target_wav": str(target_wav),
                "sample_rate": SAMPLE_RATE,
                "speaker_id": target_speaker,
                "target_present": 1,
                "target_activity_class": 1,
                "seat": "driver",
                "condition": "english_cavemove_spatial",
                "meta": {
                    "passenger_speaker_id": passenger_speaker,
                    "target_source": str(target_path),
                    "enrollment_source": str(enrollment_path),
                    "passenger_source": str(passenger_path),
                    "noise_source": str(noise_path),
                    "cabin_snr_db": cabin_snr,
                    "driver_ir": calibration["driver_ir_path"],
                    "passenger_ir": calibration["passenger_ir_path"],
                    "driver_delays_samples": calibration["driver_delays_samples"],
                    "english_only": True,
                },
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--librispeech-root",
        type=Path,
        default=REPO_ROOT / "data/raw/librispeech/extracted/LibriSpeech/train-clean-100",
    )
    parser.add_argument(
        "--cavemove-root",
        type=Path,
        default=REPO_ROOT / "data/raw/cavemove/extracted/cavemove_dataset",
    )
    parser.add_argument("--car", default="AlfaRomeo_146")
    parser.add_argument("--train-count", type=int, default=384)
    parser.add_argument("--val-count", type=int, default=96)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data/processed/english_spatial_stage1_2026-10-13",
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=REPO_ROOT / "manifests/train/stage1_english_spatial_cavemove_2026-10-13.jsonl",
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=REPO_ROOT / "manifests/val/stage1_english_spatial_cavemove_2026-10-13.jsonl",
    )
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    random.seed(int(args.seed))

    files_by_speaker: dict[str, list[Path]] = defaultdict(list)
    # LibriSpeech's directory layout already provides the speaker grouping.
    # Avoid probing tens of thousands of FLAC headers just to build a small
    # deterministic sample; short clips are safely zero-padded by _crop.
    for path in sorted(args.librispeech_root.rglob("*.flac")):
        files_by_speaker[_speaker_id(path)].append(path)
    speakers = sorted(files_by_speaker)
    if len(speakers) < 4:
        raise RuntimeError("LibriSpeech root does not contain enough English speakers")
    split_at = max(2, int(round(len(speakers) * 0.8)))
    train_speakers = speakers[:split_at]
    val_speakers = speakers[split_at:]
    if len(val_speakers) < 2:
        val_speakers = speakers[-max(2, len(speakers) // 5) :]
    calibration = load_spatial_calibration(args.cavemove_root, args.car)
    noise_files = _select_noise(args.cavemove_root, args.car)
    train_rows = build_rows(
        files_by_speaker,
        train_speakers,
        noise_files,
        calibration,
        args.output_root / "train",
        "train",
        args.train_count,
        int(args.seed),
    )
    val_rows = build_rows(
        files_by_speaker,
        val_speakers,
        noise_files,
        calibration,
        args.output_root / "val",
        "val",
        args.val_count,
        int(args.seed) + 1,
    )
    write_jsonl(args.train_manifest, train_rows)
    write_jsonl(args.val_manifest, val_rows)
    metadata = {
        "dataset": "english_spatial_stage1_cavemove",
        "english_only": True,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_speakers": train_speakers,
        "val_speakers": val_speakers,
        "speaker_disjoint": not bool(set(train_speakers) & set(val_speakers)),
        "driver_ir": calibration["driver_ir_path"],
        "passenger_ir": calibration["passenger_ir_path"],
        "driver_delays_samples": calibration["driver_delays_samples"],
        "noise_files": [str(path) for path in noise_files],
        "output_root": str(args.output_root),
        "train_manifest": str(args.train_manifest),
        "val_manifest": str(args.val_manifest),
    }
    metadata_path = args.output_root / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
