#!/usr/bin/env python3
"""Record one four-microphone Xavier condition trial and save metadata.

The script records the selected hardware channels as a 16 kHz WAV. It does not
create engine, passenger, HVAC, or music conditions; arrange the real test
condition before recording. Audio is written under ignored data/raw/ by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("engine_noise", "passenger_speech", "air_conditioning", "music_playback")


def _device_arg(value: Optional[str]) -> Any:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _channels(value: str) -> List[int]:
    try:
        result = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("channel indices must be comma-separated integers") from exc
    if len(result) != 4 or len(set(result)) != 4 or min(result) < 0:
        raise argparse.ArgumentTypeError("select exactly four distinct, non-negative channel indices")
    return result


def _condition_prompt(condition: str) -> str:
    prompts = {
        "engine_noise": "Set the vehicle to the defined engine-noise condition and keep it constant.",
        "passenger_speech": "Include the planned passenger speech; mark whether it says the wake word in your annotations.",
        "air_conditioning": "Set and record the HVAC fan setting; keep it constant for this trial.",
        "music_playback": "Play the documented music track and volume; retain the playback file/reference if available.",
    }
    return prompts[condition]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, required=False)
    parser.add_argument("--duration", type=float, default=60.0, help="Recording duration in seconds")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--input-channels", type=int, default=4, help="Hardware channels to open")
    parser.add_argument(
        "--channel-indices",
        type=_channels,
        default=_channels("0,1,2,3"),
        help="Four simultaneous hardware channel indices to save, e.g. 0,2,4,6",
    )
    parser.add_argument("--device", default=None, help="Sounddevice input index or name")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data/raw/jetson_conditions",
    )
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args(argv)

    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        raise SystemExit(
            "Cannot load sounddevice. Install it in the Xavier venv and ensure "
            "PortAudio is installed (for example: sudo apt install libportaudio2)."
        ) from exc

    if args.list_devices:
        print(sd.query_devices())
        return 0
    if not args.condition:
        parser.error("--condition is required unless --list-devices is used")
    if args.duration <= 0 or args.sample_rate <= 0:
        parser.error("--duration and --sample-rate must be positive")
    if args.input_channels <= max(args.channel_indices):
        parser.error("--input-channels must exceed every selected channel index")

    device = _device_arg(args.device)
    device_info = sd.query_devices(device, "input")
    max_channels = int(device_info.get("max_input_channels", 0))
    if max_channels < args.input_channels:
        raise ValueError(
            "Input device supports {} channels, but --input-channels requested {}".format(
                max_channels, args.input_channels
            )
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_id = args.session_id or stamp
    destination_dir = (args.output_dir / args.condition).expanduser()
    if not destination_dir.is_absolute():
        destination_dir = (REPO_ROOT / destination_dir).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    wav_path = destination_dir / "{}_{}.wav".format(args.condition, session_id)
    metadata_path = wav_path.with_suffix(".json")

    print("Condition: {}".format(args.condition))
    print(_condition_prompt(args.condition))
    print("Device: {} | recording {} input channels, saving {} at {} Hz".format(
        device_info.get("name", "unknown"), args.input_channels, args.channel_indices, args.sample_rate
    ))
    input("Prepare the scene and speaker. Press Enter to start recording.")
    frames = int(round(args.duration * args.sample_rate))
    audio = sd.rec(
        frames,
        samplerate=args.sample_rate,
        channels=args.input_channels,
        dtype="float32",
        device=device,
    )
    sd.wait()
    selected = np.asarray(audio[:, args.channel_indices], dtype=np.float32)
    if selected.shape != (frames, 4) or not np.isfinite(selected).all():
        raise RuntimeError("Captured audio has an invalid shape or non-finite samples")
    sf.write(str(wav_path), selected, args.sample_rate, subtype="PCM_24")
    metadata = {
        "schema_version": 1,
        "recording_id": "{}_{}".format(args.condition, session_id),
        "condition": args.condition,
        "captured_at_utc": stamp,
        "wav_path": str(wav_path),
        "wav_sha256": _hash(wav_path),
        "sample_rate_hz": int(args.sample_rate),
        "duration_seconds": float(len(selected) / args.sample_rate),
        "saved_channels": 4,
        "input_device": str(device_info.get("name", "unknown")),
        "input_channel_count": int(args.input_channels),
        "selected_channel_indices": list(args.channel_indices),
        "audio_subtype": "PCM_24",
        "annotations_complete": False,
        "annotation_note": "Add expected wake/action labels and event end times to the condition JSONL manifest before scoring.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Saved WAV: {}".format(wav_path))
    print("Saved metadata: {}".format(metadata_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
