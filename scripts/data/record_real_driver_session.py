#!/usr/bin/env python
"""Record one consented multichannel real-driver pilot session.

The utility deliberately refuses to record unless the operator explicitly
confirms permission to use the recording. It writes the audio and metadata
but leaves event annotation to the operator, because timestamps and intent
labels must not be guessed by the recorder.

Example (after installing ``sounddevice``):

    python scripts/data/record_real_driver_session.py \
      --session-id D01_S01 --driver-id D01 --split tuning \
      --duration-seconds 180 --channel-count 4 \
      --microphone-setup "production cabin array" \
      --consent-record-id consent-D01-20261010 \
      --owner "project consent record" \
      --permitted-purpose "research evaluation" \
      --license-or-agreement "signed consent record" \
      --confirm-consent

Use ``--list-devices`` first to inspect available input devices.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "data" / "raw" / "custom_incar" / "sessions"


def parse_device(value: str | None) -> int | str | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def validate_capture_device(sd: Any, device: int | str | None, *, channel_count: int) -> dict[str, Any]:
    """Validate the requested input contract before creating a session folder.

    ``sounddevice.rec`` otherwise discovers an unsupported channel count only
    after the session directory has been created, which can leave an empty
    directory that looks like a pilot session to downstream audits.
    """

    info = sd.query_devices(device, "input")
    max_input_channels = int(info.get("max_input_channels", 0))
    if max_input_channels < int(channel_count):
        name = str(info.get("name", device if device is not None else "default input"))
        raise ValueError(
            f"input device {name!r} exposes {max_input_channels} channel(s), "
            f"but {int(channel_count)} requested"
        )
    sd.check_input_settings(
        device=device,
        channels=int(channel_count),
        samplerate=16000,
        dtype="float32",
    )
    return {
        "name": str(info.get("name", "")),
        "max_input_channels": max_input_channels,
        "requested_channels": int(channel_count),
        "sample_rate_hz": 16000,
    }


def write_metadata(
    session_dir: Path,
    *,
    session_id: str,
    driver_id: str,
    split: str,
    duration_seconds: float,
    channel_count: int,
    microphone_setup: str,
    consent_record_id: str,
    owner: str,
    permitted_purpose: str,
    license_or_agreement: str,
    vehicle: dict[str, Any],
) -> None:
    metadata = {
        "session_id": session_id,
        "split": split,
        "driver_id": driver_id,
        "audio_path": "audio.wav",
        "recorded_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "recording_sample_rate_hz": 16000,
        "duration_seconds_requested": float(duration_seconds),
        "channel_count": channel_count,
        "channel_map": [f"channel_{index}" for index in range(channel_count)],
        "microphone_setup": microphone_setup,
        "vehicle": vehicle,
        "consent": {
            "permission_to_use": True,
            "consent_record_id": consent_record_id,
            "owner": owner,
            "permitted_purpose": permitted_purpose,
            "license_or_agreement": license_or_agreement,
        },
        "notes": "Audio captured by record_real_driver_session.py; annotate before evaluation.",
    }
    (session_dir / "session_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (session_dir / "annotations.jsonl").touch()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--session-id")
    parser.add_argument("--driver-id")
    parser.add_argument("--split", choices=("tuning", "test"))
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--channel-count", type=int, default=4)
    parser.add_argument("--device", help="sounddevice input device index or name")
    parser.add_argument("--microphone-setup")
    parser.add_argument("--consent-record-id")
    parser.add_argument("--owner")
    parser.add_argument("--permitted-purpose")
    parser.add_argument("--license-or-agreement")
    parser.add_argument("--vehicle-make-model", default="redacted-or-description")
    parser.add_argument("--parked-or-moving", choices=("parked", "moving"), default="parked")
    parser.add_argument("--hvac", default="describe-hvac-state")
    parser.add_argument("--road-surface", default="describe-road-surface")
    parser.add_argument("--playback", default="off")
    parser.add_argument("--passenger-count", type=int, default=0)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument(
        "--confirm-consent",
        action="store_true",
        help="required: confirm that everyone recorded has granted permission to use the audio",
    )
    args = parser.parse_args()

    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError as exc:
        print(
            "sounddevice is required for recording. Install it in the active environment "
            "with: python -m pip install sounddevice",
            file=sys.stderr,
        )
        return 2

    if args.list_devices:
        print(sd.query_devices())
        return 0

    required = {
        "--session-id": args.session_id,
        "--driver-id": args.driver_id,
        "--split": args.split,
        "--duration-seconds": args.duration_seconds,
        "--microphone-setup": args.microphone_setup,
        "--consent-record-id": args.consent_record_id,
        "--owner": args.owner,
        "--permitted-purpose": args.permitted_purpose,
        "--license-or-agreement": args.license_or_agreement,
    }
    missing = [name for name, value in required.items() if value is None or value == ""]
    if missing:
        parser.error("missing required recording metadata: " + ", ".join(missing))
    if not args.confirm_consent:
        parser.error("refusing to record without --confirm-consent")
    if args.duration_seconds <= 0:
        parser.error("--duration-seconds must be positive")
    if args.channel_count < 1:
        parser.error("--channel-count must be positive")

    device = parse_device(args.device)
    try:
        device_info = validate_capture_device(
            sd,
            device,
            channel_count=int(args.channel_count),
        )
    except Exception as exc:
        print(f"Capture device does not satisfy the requested input contract: {exc}", file=sys.stderr)
        return 2

    session_dir = args.root / str(args.session_id)
    if session_dir.exists() and any(session_dir.iterdir()):
        parser.error(f"refusing to overwrite non-empty session directory: {session_dir}")
    session_dir.mkdir(parents=True, exist_ok=True)

    frames = int(round(float(args.duration_seconds) * 16000))
    print(
        f"Recording {args.duration_seconds:.1f}s, 16 kHz, {args.channel_count} channels "
        f"to {session_dir / 'audio.wav'}; press Ctrl+C to abort before saving.",
        flush=True,
    )
    try:
        recording = sd.rec(
            frames,
            samplerate=16000,
            channels=int(args.channel_count),
            dtype="float32",
            device=device,
        )
        sd.wait()
    except KeyboardInterrupt:
        print("Recording cancelled; no audio was saved.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Recording failed: {exc}", file=sys.stderr)
        return 1

    sf.write(str(session_dir / "audio.wav"), recording, 16000, subtype="PCM_16")
    write_metadata(
        session_dir,
        session_id=str(args.session_id),
        driver_id=str(args.driver_id),
        split=str(args.split),
        duration_seconds=float(args.duration_seconds),
        channel_count=int(args.channel_count),
        microphone_setup=str(args.microphone_setup),
        consent_record_id=str(args.consent_record_id),
        owner=str(args.owner),
        permitted_purpose=str(args.permitted_purpose),
        license_or_agreement=str(args.license_or_agreement),
        vehicle={
            "make_model": str(args.vehicle_make_model),
            "parked_or_moving": str(args.parked_or_moving),
            "hvac": str(args.hvac),
            "road_surface": str(args.road_surface),
            "playback": str(args.playback),
            "passenger_count": int(args.passenger_count),
        },
    )
    metadata_path = session_dir / "session_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["capture_device"] = device_info
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved audio and consent metadata under {session_dir}")
    print("Next: add timestamped labels to annotations.jsonl, then run validate_real_driver_sessions.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
