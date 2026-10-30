from __future__ import annotations

import json

import numpy as np
import soundfile as sf

from scripts.data.validate_real_driver_sessions import validate


def _write_session(root, *, capture_contract=None):
    session = root / "D01_S01"
    session.mkdir(parents=True)
    sf.write(session / "audio.wav", np.zeros((7 * 16_000, 6), dtype=np.float32), 16_000)
    metadata = {
        "session_id": "D01_S01",
        "split": "tuning",
        "driver_id": "D01",
        "audio_path": "audio.wav",
        "recording_sample_rate_hz": 16_000,
        "channel_count": 6,
        "consent": {
            "permission_to_use": True,
            "consent_record_id": "consent-D01",
            "owner": "project",
            "permitted_purpose": "research",
            "license_or_agreement": "signed",
        },
    }
    if capture_contract is not None:
        metadata["capture_contract"] = capture_contract
    (session / "session_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    events = [
        {"event": "enrollment", "start_s": 0, "end_s": 1},
        {"event": "driver_wake", "start_s": 1, "end_s": 2},
        {"event": "driver_command", "start_s": 2, "end_s": 3, "expected_intent": "activate|lights"},
        {"event": "passenger_wake", "start_s": 3, "end_s": 4},
        {"event": "passenger_speech", "start_s": 4, "end_s": 5},
        {"event": "no_command", "start_s": 5, "end_s": 6},
        {"event": "noise_only", "start_s": 6, "end_s": 7},
    ]
    (session / "annotations.jsonl").write_text(
        "\n".join(json.dumps(row) for row in events) + "\n",
        encoding="utf-8",
    )


def test_six_channel_session_requires_explicit_playback_contract(tmp_path):
    _write_session(tmp_path)
    result = validate(tmp_path, {"D01"}, set())
    assert result["valid"] is False
    assert any("playback reference" in error for error in result["errors"])


def test_six_channel_session_with_playback_contract_passes(tmp_path):
    _write_session(
        tmp_path,
        capture_contract={
            "capture_profile": "production6",
            "minimum_input_channels": 6,
            "playback_reference_required": True,
            "playback_reference_declared": True,
            "microphone_channel_indices": [0, 1, 2, 3],
            "playback_reference_channel_indices": [4, 5],
        },
    )
    result = validate(tmp_path, {"D01"}, set())
    assert result["valid"] is True
    assert result["sessions"][0]["playback_reference_declared"] is True
