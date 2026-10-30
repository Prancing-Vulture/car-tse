#!/usr/bin/env python
"""Build same-domain target/passenger mixtures for Stage-1 adaptation.

The public corpora do not contain a natural driver wake-word plus command
recording.  This manifest nevertheless closes the most specific current gap:
SpEx+ sees a Marvin target, a different-speaker Marvin passenger, and a
CAVEMOVE cabin-noise reference while retaining a clean target waveform.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import REPO_ROOT
from car_tse.io.audio import read_mono


def speaker_key(path: Path) -> str:
    return path.stem.split("_nohash_")[0]


def vehicle_key(path: Path) -> str | None:
    return next(
        (name for name in ("AlfaRomeo_146", "Smart_forfour", "Volkswagen_Golf") if name in path.parts),
        None,
    )


def target_onset_for_copy(copy_index: int, sample_rate: int = 16000) -> int:
    """Jitter target onset within the first one-second runtime KWS crop."""

    return int(round((0.20 * (int(copy_index) % 2)) * sample_rate))


def partition_noise_paths(noise_paths: list[Path]) -> dict[str, list[Path]]:
    """Split recordings by source ID and reserve one vehicle for final testing."""

    by_vehicle: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(noise_paths):
        vehicle = vehicle_key(path)
        if vehicle is not None:
            by_vehicle[vehicle].append(path)
    if len(by_vehicle) < 3:
        raise RuntimeError(
            "Need at least three CAVEMOVE vehicle groups to keep train/validation noise "
            "recordings disjoint and reserve an unseen test vehicle"
        )
    test_vehicle = "Volkswagen_Golf" if "Volkswagen_Golf" in by_vehicle else sorted(by_vehicle)[-1]
    development_vehicles = sorted(set(by_vehicle) - {test_vehicle})
    pools = {"train": [], "val": [], "test": list(by_vehicle[test_vehicle])}
    for vehicle in development_vehicles:
        paths = by_vehicle[vehicle]
        if len(paths) < 2:
            raise RuntimeError(f"Need at least two noise recordings for train/validation vehicle {vehicle}")
        for index, path in enumerate(paths):
            # One of every five recordings per development vehicle is reserved
            # for validation. File identities never cross these two splits.
            pools["val" if index % 5 == 0 else "train"].append(path)
    if not pools["train"] or not pools["val"] or not pools["test"]:
        raise RuntimeError("Noise partition produced an empty split")
    return pools


def fit_clip(wav: np.ndarray, samples: int, start: int) -> np.ndarray:
    out = np.zeros(samples, dtype=np.float32)
    clip = np.asarray(wav, dtype=np.float32).reshape(-1)
    end = min(samples, start + len(clip))
    if end > start:
        out[start:end] = clip[: end - start]
    return out


def rms(wav: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(wav, dtype=np.float32) ** 2) + 1.0e-8))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--marvin-root", type=Path, default=REPO_ROOT / "data/raw/speech_commands_v2/extracted/marvin")
    parser.add_argument("--gsc-root", type=Path, default=None, help="Speech Commands root; defaults to the parent of --marvin-root")
    parser.add_argument("--noise-root", type=Path, default=REPO_ROOT / "data/raw/cavemove/extracted/cavemove_dataset")
    parser.add_argument("--manifest-root", type=Path, default=REPO_ROOT / "manifests")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "data/processed/gsc_marvin_stage1")
    parser.add_argument("--copies-per-target", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()

    marvin_root = args.marvin_root if args.marvin_root.is_absolute() else REPO_ROOT / args.marvin_root
    gsc_root = args.gsc_root if args.gsc_root is not None else marvin_root.parent
    gsc_root = gsc_root if gsc_root.is_absolute() else REPO_ROOT / gsc_root
    noise_root = args.noise_root if args.noise_root.is_absolute() else REPO_ROOT / args.noise_root
    manifest_root = args.manifest_root if args.manifest_root.is_absolute() else REPO_ROOT / args.manifest_root
    output_root = args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(marvin_root.glob("*.wav")):
        groups[speaker_key(path)].append(path)
    eligible = [(speaker, paths) for speaker, paths in sorted(groups.items()) if len(paths) >= 2]
    if len(eligible) < 30:
        raise RuntimeError(f"Need at least 30 Marvin speakers with two utterances; found {len(eligible)}")
    eligible_speakers = {speaker for speaker, _ in eligible}
    nonwake_by_speaker: dict[str, list[Path]] = defaultdict(list)
    for class_dir in sorted(gsc_root.iterdir()):
        if not class_dir.is_dir() or class_dir.name in {"marvin", "_background_noise_"}:
            continue
        for path in sorted(class_dir.glob("*_nohash_*.wav")):
            speaker = speaker_key(path)
            if speaker in eligible_speakers:
                nonwake_by_speaker[speaker].append(path)
    eligible = [
        (speaker, paths) for speaker, paths in eligible
        if nonwake_by_speaker.get(speaker)
    ]
    if len(eligible) < 30:
        raise RuntimeError(
            "Need at least 30 eligible Marvin speakers with a separate non-wake utterance; "
            f"found {len(eligible)}"
        )
    noise_paths = sorted(
        path for path in noise_root.rglob("*.wav") if "noise" in path.parts or "ventilation" in path.parts
    )
    if not noise_paths:
        raise RuntimeError(f"No CAVEMOVE noise files found under {noise_root}")
    noise_pools = partition_noise_paths(noise_paths)

    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    n_train = max(1, int(round(0.70 * len(eligible))))
    n_val = max(1, int(round(0.15 * len(eligible))))
    splits = {
        "train": eligible[:n_train],
        "val": eligible[n_train : n_train + n_val],
        "test": eligible[n_train + n_val :],
    }
    snrs = (-5.0, 0.0, 5.0, 10.0)
    duration_samples = 3 * 16000
    for split, target_groups in splits.items():
        if len(target_groups) < 2:
            continue
        split_root = output_root / split
        split_root.mkdir(parents=True, exist_ok=True)
        silence_path = split_root / "silence_3s.wav"
        if not silence_path.exists():
            sf.write(str(silence_path), np.zeros(duration_samples, dtype=np.float32), 16000, subtype="FLOAT")
        rows: list[dict] = []
        for target_index, (target_speaker, target_files) in enumerate(target_groups):
            passenger_candidates = [item for item in target_groups if item[0] != target_speaker]
            for copy_index in range(max(1, int(args.copies_per_target))):
                target_start = target_onset_for_copy(copy_index)
                target_path = target_files[copy_index % len(target_files)]
                enroll_path = target_files[(copy_index + 1) % len(target_files)]
                nonwake_path = nonwake_by_speaker[target_speaker][
                    copy_index % len(nonwake_by_speaker[target_speaker])
                ]
                passenger_speaker, passenger_files = passenger_candidates[
                    (target_index + copy_index) % len(passenger_candidates)
                ]
                passenger_path = passenger_files[(copy_index + 1) % len(passenger_files)]
                target, _ = read_mono(target_path, 16000)
                nonwake_target, _ = read_mono(nonwake_path, 16000)
                passenger, _ = read_mono(passenger_path, 16000)
                target_clip = fit_clip(target, duration_samples, target_start)
                nonwake_target_clip = fit_clip(nonwake_target, duration_samples, target_start)
                passenger_clip = fit_clip(passenger, duration_samples, target_start)
                snr_db = float(snrs[(target_index + copy_index) % len(snrs)])
                passenger_scaled = passenger_clip * (
                    rms(target_clip) * (10.0 ** (-snr_db / 20.0)) / max(rms(passenger_clip), 1.0e-6)
                )
                mixture = target_clip + passenger_scaled
                out_stem = f"{target_speaker}_{target_index:04d}_{copy_index}"
                mixture_path = split_root / f"{out_stem}_mix.wav"
                target_out_path = split_root / f"{out_stem}_target.wav"
                absent_mixture_path = split_root / f"{out_stem}_absent_mix.wav"
                nonwake_mixture_path = split_root / f"{out_stem}_nonwake_mix.wav"
                nonwake_target_out_path = split_root / f"{out_stem}_nonwake_target.wav"
                sf.write(str(mixture_path), mixture.astype(np.float32), 16000, subtype="FLOAT")
                sf.write(str(target_out_path), target_clip.astype(np.float32), 16000, subtype="FLOAT")
                sf.write(str(absent_mixture_path), passenger_scaled.astype(np.float32), 16000, subtype="FLOAT")
                nonwake_passenger_scaled = passenger_clip * (
                    rms(nonwake_target_clip)
                    * (10.0 ** (-snr_db / 20.0))
                    / max(rms(passenger_clip), 1.0e-6)
                )
                nonwake_mixture = nonwake_target_clip + nonwake_passenger_scaled
                sf.write(str(nonwake_mixture_path), nonwake_mixture.astype(np.float32), 16000, subtype="FLOAT")
                sf.write(str(nonwake_target_out_path), nonwake_target_clip.astype(np.float32), 16000, subtype="FLOAT")
                split_noise_paths = noise_pools[split]
                noise_path = split_noise_paths[
                    (target_index * max(1, int(args.copies_per_target)) + copy_index)
                    % len(split_noise_paths)
                ]
                rows.append(
                    {
                        "id": f"gsc_marvin_stage1_{split}_{out_stem}",
                        "mixture_wav": str(mixture_path),
                        "enroll_wav": str(enroll_path),
                        "target_wav": str(target_out_path),
                        "sample_rate": 16000,
                        "speaker_id": target_speaker,
                        "speaker_label": target_index,
                        "condition": "gsc_marvin_target_plus_marvin_passenger_cavemove",
                        "target_present": 1,
                        "kws_label": 1,
                        "target_onset_in_runtime_kws_crop_seconds": target_start / 16000.0,
                        "noise_vehicle": vehicle_key(noise_path),
                        "noise_recording_split": split,
                        "cabin_noise_wav": str(noise_path),
                        "cabin_noise_channels": [0, 2, 4, 6],
                        "cabin_noise_snr_db": float(snrs[(target_index + copy_index + 1) % len(snrs)]),
                        "meta": {
                            "source": "speech_commands_v2_marvin_plus_cavemove",
                            "passenger_speaker_id": passenger_speaker,
                            "passenger_snr_db": snr_db,
                            "speaker_disjoint_split": True,
                        },
                    }
                )
                rows.append(
                    {
                        "id": f"gsc_marvin_stage1_nonwake_{split}_{out_stem}",
                        "mixture_wav": str(nonwake_mixture_path),
                        "enroll_wav": str(enroll_path),
                        "target_wav": str(nonwake_target_out_path),
                        "sample_rate": 16000,
                        "speaker_id": target_speaker,
                        "speaker_label": target_index,
                        "condition": "gsc_nonwake_target_plus_marvin_passenger_cavemove",
                        "target_present": 1,
                        "kws_label": 0,
                        "target_onset_in_runtime_kws_crop_seconds": target_start / 16000.0,
                        "noise_vehicle": vehicle_key(noise_path),
                        "noise_recording_split": split,
                        "cabin_noise_wav": str(noise_path),
                        "cabin_noise_channels": [0, 2, 4, 6],
                        "cabin_noise_snr_db": float(snrs[(target_index + copy_index + 1) % len(snrs)]),
                        "meta": {
                            "source": "speech_commands_v2_nonwake_target_plus_marvin_passenger_cavemove",
                            "target_nonwake_source": str(nonwake_path),
                            "passenger_speaker_id": passenger_speaker,
                            "passenger_snr_db": snr_db,
                            "speaker_disjoint_split": True,
                        },
                    }
                )
                rows.append(
                    {
                        "id": f"gsc_marvin_stage1_absent_{split}_{out_stem}",
                        "mixture_wav": str(absent_mixture_path),
                        "enroll_wav": str(enroll_path),
                        "target_wav": str(silence_path),
                        "sample_rate": 16000,
                        "speaker_id": target_speaker,
                        "speaker_label": target_index,
                        "condition": "gsc_marvin_passenger_only_cavemove",
                        "target_present": 0,
                        "kws_label": 0,
                        "target_onset_in_runtime_kws_crop_seconds": target_start / 16000.0,
                        "noise_vehicle": vehicle_key(noise_path),
                        "noise_recording_split": split,
                        "cabin_noise_wav": str(noise_path),
                        "cabin_noise_channels": [0, 2, 4, 6],
                        "cabin_noise_snr_db": float(snrs[(target_index + copy_index + 1) % len(snrs)]),
                        "meta": {
                            "source": "speech_commands_v2_marvin_passenger_plus_cavemove",
                            "passenger_speaker_id": passenger_speaker,
                            "passenger_snr_db": snr_db,
                            "speaker_disjoint_split": True,
                        },
                    }
                )
        write_jsonl(manifest_root / split / "stage1_gsc_marvin_cabin.jsonl", rows)
        print(f"{split}: {len(rows)} rows, target_speakers={len(target_groups)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
