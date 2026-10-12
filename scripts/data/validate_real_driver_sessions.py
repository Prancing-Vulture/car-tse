"""Validate consented real-driver session folders before evaluation.

The validator is intentionally conservative: it checks the metadata and
annotation contract, but does not infer consent or repair labels. It is useful
for catching accidental speaker/split leakage before thresholds are tuned.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import soundfile as sf


ALLOWED_EVENTS = {
    "enrollment",
    "driver_wake",
    "driver_command",
    "passenger_wake",
    "passenger_speech",
    "no_command",
    "noise_only",
    "playback",
}
REQUIRED_METADATA = {
    "session_id",
    "split",
    "driver_id",
    "audio_path",
    "recording_sample_rate_hz",
    "channel_count",
    "consent",
}
REQUIRED_CONSENT_FIELDS = {
    "consent_record_id",
    "owner",
    "permitted_purpose",
    "license_or_agreement",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(value)
    return rows


def validate(root: Path, tuning_drivers: set[str], test_drivers: set[str]) -> dict[str, Any]:
    errors: list[str] = []
    sessions: list[dict[str, Any]] = []
    if tuning_drivers & test_drivers:
        errors.append("tuning and test driver lists overlap")

    session_dirs = sorted(path for path in root.iterdir() if path.is_dir()) if root.is_dir() else []
    if not session_dirs:
        errors.append(f"no session directories found under {root}")

    seen_ids: set[str] = set()
    for session_dir in session_dirs:
        metadata_path = session_dir / "session_metadata.json"
        annotations_path = session_dir / "annotations.jsonl"
        if not metadata_path.exists():
            errors.append(f"{session_dir}: missing session_metadata.json")
            continue
        if not annotations_path.exists():
            errors.append(f"{session_dir}: missing annotations.jsonl")
            continue
        try:
            metadata = _load_json(metadata_path)
            annotations = _load_jsonl(annotations_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{session_dir}: {exc}")
            continue

        missing = REQUIRED_METADATA - metadata.keys()
        if missing:
            errors.append(f"{session_dir}: missing metadata fields {sorted(missing)}")
        session_id = str(metadata.get("session_id", session_dir.name))
        driver_id = str(metadata.get("driver_id", ""))
        split = str(metadata.get("split", ""))
        if session_id != session_dir.name:
            errors.append(f"{session_id}: session_id must match directory name {session_dir.name!r}")
        if not driver_id.strip():
            errors.append(f"{session_id}: driver_id must be non-empty")
        if session_id in seen_ids:
            errors.append(f"duplicate session_id: {session_id}")
        seen_ids.add(session_id)
        if split not in {"tuning", "test"}:
            errors.append(f"{session_id}: split must be tuning or test")
        expected_split = "tuning" if driver_id in tuning_drivers else "test" if driver_id in test_drivers else None
        if expected_split and split != expected_split:
            errors.append(f"{session_id}: split={split} conflicts with driver list ({expected_split})")
        consent = metadata.get("consent")
        if not isinstance(consent, dict) or consent.get("permission_to_use") is not True:
            errors.append(f"{session_id}: consent.permission_to_use must be true")
        elif any(not str(consent.get(field, "")).strip() for field in REQUIRED_CONSENT_FIELDS):
            missing_consent = sorted(
                field for field in REQUIRED_CONSENT_FIELDS if not str(consent.get(field, "")).strip()
            )
            errors.append(f"{session_id}: consent fields must be non-empty: {missing_consent}")
        if metadata.get("recording_sample_rate_hz") != 16000:
            errors.append(f"{session_id}: recording_sample_rate_hz must be 16000")
        channels = 0
        try:
            channels = int(metadata.get("channel_count", 0))
            if channels < 1:
                errors.append(f"{session_id}: channel_count must be positive")
        except (TypeError, ValueError):
            errors.append(f"{session_id}: channel_count must be an integer")

        audio_path = Path(str(metadata.get("audio_path", "audio.wav")))
        if not audio_path.is_absolute():
            audio_path = session_dir / audio_path.name if audio_path.parent == Path(".") else root / audio_path
        if not audio_path.exists():
            errors.append(f"{session_id}: audio file not found: {audio_path}")
            audio_duration = None
        else:
            try:
                audio_info = sf.info(str(audio_path))
                audio_duration = float(audio_info.duration)
                if int(audio_info.samplerate) != 16000:
                    errors.append(f"{session_id}: audio sample rate must be 16000")
                if int(audio_info.channels) != channels:
                    errors.append(
                        f"{session_id}: audio channel count {audio_info.channels} does not match metadata {channels}"
                    )
            except (OSError, RuntimeError) as exc:
                audio_duration = None
                errors.append(f"{session_id}: unable to inspect audio: {exc}")

        event_counts: dict[str, int] = {}
        intervals_by_event: dict[str, list[tuple[float, float]]] = {}
        for row_number, row in enumerate(annotations, 1):
            event = row.get("event")
            event_key = event if isinstance(event, str) else str(event)
            event_counts[event_key] = event_counts.get(event_key, 0) + 1
            if event not in ALLOWED_EVENTS:
                errors.append(f"{session_id}:{row_number}: unsupported event {event!r}")
            if row.get("session_id") not in {None, session_id}:
                errors.append(f"{session_id}:{row_number}: session_id mismatch")
            try:
                start = float(row["start_s"])
                end = float(row["end_s"])
                if not (0 <= start < end):
                    errors.append(f"{session_id}:{row_number}: require 0 <= start_s < end_s")
                if audio_duration is not None and end > audio_duration + 1.0e-3:
                    errors.append(f"{session_id}:{row_number}: annotation ends after audio duration")
                if isinstance(event, str) and event in ALLOWED_EVENTS and 0 <= start < end:
                    intervals_by_event.setdefault(event, []).append((start, end))
            except (KeyError, TypeError, ValueError):
                errors.append(f"{session_id}:{row_number}: invalid start_s/end_s")
            if event == "driver_command" and not row.get("expected_intent"):
                errors.append(f"{session_id}:{row_number}: driver_command needs expected_intent")

        for required_event in (
            "enrollment",
            "driver_wake",
            "driver_command",
            "passenger_wake",
            "passenger_speech",
            "no_command",
            "noise_only",
        ):
            if not event_counts.get(required_event):
                errors.append(f"{session_id}: no {required_event} annotation")
        enrollment_intervals = intervals_by_event.get("enrollment", [])
        wake_intervals = intervals_by_event.get("driver_wake", [])
        command_intervals = intervals_by_event.get("driver_command", [])
        if enrollment_intervals and wake_intervals and not any(
            enrollment_end <= wake_start
            for _, enrollment_end in enrollment_intervals
            for wake_start, _ in wake_intervals
        ):
            errors.append(f"{session_id}: no enrollment interval precedes a driver_wake")
        if wake_intervals and command_intervals and not any(
            wake_end <= command_start
            for _, wake_end in wake_intervals
            for command_start, _ in command_intervals
        ):
            errors.append(f"{session_id}: no driver_command follows a driver_wake")
        sessions.append(
            {
                "session_id": session_id,
                "driver_id": driver_id,
                "split": split,
                "recording_sample_rate_hz": int(metadata.get("recording_sample_rate_hz", 0) or 0),
                "channel_count": channels,
                "event_counts": event_counts,
            }
        )

    if tuning_drivers or test_drivers:
        listed_drivers = set(tuning_drivers) | set(test_drivers)
        observed_drivers = {row["driver_id"] for row in sessions}
        for driver_id in sorted(observed_drivers - listed_drivers):
            errors.append(f"driver {driver_id}: missing from tuning/test driver assignment")

    return {
        "root": str(root),
        "session_count": len(sessions),
        "drivers": sorted({row["driver_id"] for row in sessions}),
        "sessions": sessions,
        "errors": errors,
        "valid": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/custom_incar/sessions"))
    parser.add_argument("--tuning-drivers", nargs="*", default=[])
    parser.add_argument("--test-drivers", nargs="*", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.root, set(args.tuning_drivers), set(args.test_drivers))
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
