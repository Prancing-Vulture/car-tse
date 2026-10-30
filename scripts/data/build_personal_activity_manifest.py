#!/usr/bin/env python
"""Build a three-way personal-activity training manifest.

The existing presence manifest has target-positive Marvin clips and target-
absent AISHELL speech, but it does not explicitly mark non-target speech.
This utility preserves that data and adds deterministic Speech Commands clips
from other speakers/keywords as passenger-like non-target speech.  It is a
training/architecture diagnostic only; it is not a real-driver evaluation set.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def speaker_id(path: Path) -> str:
    return path.name.split("_nohash", 1)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--speech-commands-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-nontarget", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()

    base_path = args.base_manifest.resolve()
    gsc_root = args.speech_commands_root.resolve()
    output_path = args.output.resolve()
    rows = read_jsonl(base_path)
    marvin_dir = gsc_root / "marvin"
    if not rows or not marvin_dir.exists():
        raise RuntimeError("Base manifest and Speech Commands Marvin directory are required")

    marvin_by_speaker: dict[str, list[Path]] = {}
    for path in sorted(marvin_dir.glob("*.wav")):
        marvin_by_speaker.setdefault(speaker_id(path), []).append(path)
    enrollment_speakers = sorted(marvin_by_speaker)
    if len(enrollment_speakers) < 2:
        raise RuntimeError("Need at least two distinct Marvin enrollment speakers")

    non_target = [
        path
        for path in sorted(gsc_root.glob("*/*.wav"))
        if path.parent.name not in {"marvin", "_background_noise_"}
    ]
    if not non_target:
        raise RuntimeError("No non-target Speech Commands clips found")
    rng = random.Random(int(args.seed))
    rng.shuffle(non_target)
    non_target = non_target[: max(1, int(args.max_nontarget))]

    noise_paths = [
        str(row["cabin_noise_wav"])
        for row in rows
        if row.get("cabin_noise_wav") and Path(str(row["cabin_noise_wav"])).exists()
    ]
    if not noise_paths:
        raise RuntimeError("Base manifest has no available cabin-noise paths")

    output_rows: list[dict] = []
    for row in rows:
        clone = dict(row)
        # GSC Marvin rows are target speech; AISHELL rows are non-target speech
        # for this three-way diagnostic rather than an undifferentiated zero.
        clone["target_activity_class"] = 1 if int(row.get("target_present", 0)) else 2
        # Use the enrollment speaker as the held-out grouping key.  AISHELL
        # absent rows intentionally reuse a Marvin enrollment, so they must be
        # distributed with that enrollment group rather than forming one
        # giant validation-only group.
        clone["group_id"] = (
            str(row.get("speaker_id"))
            if int(row.get("target_present", 0))
            else speaker_id(Path(str(row["enroll_wav"])))
        )
        output_rows.append(clone)

    added: list[dict] = []
    for index, waveform in enumerate(non_target):
        source_speaker = speaker_id(waveform)
        candidates = [item for item in enrollment_speakers if item != source_speaker]
        enroll_speaker = candidates[index % len(candidates)]
        enroll = marvin_by_speaker[enroll_speaker][0]
        added.append(
            {
                "id": f"gsc_non_target_{index:05d}",
                "mixture_wav": str(waveform),
                "enroll_wav": str(enroll),
                "target_wav": str(waveform),
                "sample_rate": 16000,
                "speaker_id": enroll_speaker,
                "group_id": enroll_speaker,
                "source_speaker_id": source_speaker,
                "target_present": 0,
                "target_activity_class": 2,
                "condition": "gsc_non_target_speech_with_cabin_noise",
                "cabin_noise_wav": noise_paths[index % len(noise_paths)],
                "cabin_noise_channels": [0, 2, 4, 6],
                "cabin_noise_snr_db": [0.0, 10.0, 20.0][index % 3],
                "meta": {
                    "source": "speech_commands_v2_non_marvin_plus_cavemove_noise",
                    "personal_activity_class": "non_target_speech",
                    "training_diagnostic_only": True,
                },
            }
        )
    output_rows.extend(added)
    rng.shuffle(output_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "base_rows": len(rows),
                "added_non_target_rows": len(added),
                "output_rows": len(output_rows),
                "target_class_definition": {
                    "0": "non_speech",
                    "1": "target_speech",
                    "2": "non_target_speech",
                },
                "output": str(output_path),
                "not_evaluation_evidence": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
