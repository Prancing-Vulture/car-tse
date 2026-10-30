#!/usr/bin/env python
"""Build a reproducible Stage 2 presence calibration manifest.

Positive rows use same-speaker Google Speech Commands ``marvin`` utterances
with CAVEMOVE noise augmentation supplied by ``PresenceDataset``.  Negative
rows use target-absent four-channel AISHELL-5 sessions with a fixed wake-word
enrollment.  The result is a domain-adaptation experiment, not a production
presence corpus.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import REPO_ROOT


def resolve_path(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def speaker_key(path: Path) -> str:
    return path.stem.split("_nohash_")[0]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def gsc_groups(root: Path) -> list[tuple[str, list[Path]]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(root.glob("*.wav")):
        grouped[speaker_key(path)].append(path)
    return sorted(
        [(speaker, files) for speaker, files in grouped.items() if len(files) >= 2],
        key=lambda item: item[0],
    )


def aishell_sessions(root: Path, split: str, channel_names: list[str]) -> list[list[Path]]:
    split_root = root / split
    sessions: list[list[Path]] = []
    for session in sorted(path for path in split_root.iterdir() if path.is_dir()):
        paths = [session / name for name in channel_names]
        if all(path.exists() for path in paths):
            sessions.append(paths)
    return sessions


def positive_rows(
    groups: list[tuple[str, list[Path]]],
    noise_paths: list[Path],
    enroll_fallback: Path,
    copies_per_speaker: int,
) -> list[dict]:
    rows: list[dict] = []
    snrs = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
    for speaker_index, (speaker, files) in enumerate(groups):
        for copy_index in range(max(1, copies_per_speaker)):
            target = files[copy_index % len(files)]
            enroll = files[(copy_index + 1) % len(files)] or enroll_fallback
            noise = noise_paths[(speaker_index * copies_per_speaker + copy_index) % len(noise_paths)]
            rows.append(
                {
                    "id": f"gsc_present_{speaker}_{copy_index}",
                    "mixture_wav": str(target),
                    "enroll_wav": str(enroll),
                    "target_wav": str(target),
                    "sample_rate": 16000,
                    "speaker_id": speaker,
                    "target_present": 1,
                    "condition": "gsc_marvin_plus_cavemove_noise",
                    "cabin_noise_wav": str(noise),
                    "cabin_noise_channels": [0, 2, 4, 6],
                    "cabin_noise_snr_db": snrs[(speaker_index + copy_index) % len(snrs)],
                    "meta": {"source": "speech_commands_v2_plus_cavemove", "calibration_only": True},
                }
            )
    return rows


def negative_rows(
    sessions: list[list[Path]],
    enrollments: list[Path],
    count: int,
    split: str,
) -> list[dict]:
    rows: list[dict] = []
    if not sessions:
        raise RuntimeError(f"No AISHELL-5 sessions available for split {split}")
    for index in range(count):
        channels = sessions[index % len(sessions)]
        rows.append(
            {
                "id": f"aishell5_absent_{split}_{index:05d}",
                "mixture_wav": str(channels[0]),
                "mixture_multich_wavs": [str(path) for path in channels],
                "enroll_wav": str(enrollments[index % len(enrollments)]),
                "target_wav": str(channels[0]),
                "sample_rate": 16000,
                "speaker_id": f"aishell5_absent_{split}",
                "target_present": 0,
                "condition": "aishell5_farfield_target_absent",
                # Keep cache generation on bounded windows.  Without this
                # field PresenceDataset reads every full AISHELL channel for
                # each row, which is needlessly slow for long recordings.
                "segment_start_seconds": 0.0,
                "meta": {"source": "AISHELL-5", "calibration_only": True},
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--marvin-root", type=Path, default=REPO_ROOT / "data/raw/speech_commands_v2/extracted/marvin")
    parser.add_argument("--aishell-root", type=Path, default=REPO_ROOT / "data/raw/aishell5/extracted")
    parser.add_argument("--noise-root", type=Path, default=REPO_ROOT / "data/raw/cavemove/extracted/cavemove_dataset")
    parser.add_argument("--manifest-root", type=Path, default=REPO_ROOT / "manifests")
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--copies-per-speaker", type=int, default=2)
    parser.add_argument("--channels", nargs="+", default=["DX01C01.wav", "DX02C01.wav", "DX03C01.wav", "DX04C01.wav"])
    parser.add_argument("--output-name", default="stage2_presence_gsc_aishell.jsonl")
    args = parser.parse_args()

    marvin_root = resolve_path(args.marvin_root)
    aishell_root = resolve_path(args.aishell_root)
    noise_root = resolve_path(args.noise_root)
    manifest_root = resolve_path(args.manifest_root)
    groups = gsc_groups(marvin_root)
    if len(groups) < 20:
        raise RuntimeError(f"Only {len(groups)} eligible marvin speakers found")
    noise_paths = sorted(
        path
        for path in noise_root.rglob("*.wav")
        if "noise" in path.parts or "ventilation" in path.parts
    )
    if not noise_paths:
        raise RuntimeError(f"No CAVEMOVE noise files found under {noise_root}")

    rng = random.Random(args.seed)
    rng.shuffle(groups)
    n_train = max(1, int(round(len(groups) * 0.70)))
    n_val = max(1, int(round(len(groups) * 0.15)))
    group_splits = {
        "train": groups[:n_train],
        "val": groups[n_train : n_train + n_val],
        "test": groups[n_train + n_val :],
    }
    fallback = groups[0][1][0]
    enrollments = [files[0] for _, files in groups]

    for split, split_groups in group_splits.items():
        if not split_groups:
            continue
        positive = positive_rows(split_groups, noise_paths, fallback, int(args.copies_per_speaker))
        aishell_split = {"train": "train", "val": "Dev", "test": "Eval1"}[split]
        sessions = aishell_sessions(aishell_root, aishell_split, list(args.channels))
        negative = negative_rows(sessions, enrollments, len(positive), split)
        rows = positive + negative
        rng.shuffle(rows)
        output = manifest_root / split / args.output_name
        write_jsonl(output, rows)
        print(
            f"{split}: {len(positive)} present + {len(negative)} absent = {len(rows)} rows; "
            f"speakers={len(split_groups)} sessions={len(sessions)} -> {output}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
