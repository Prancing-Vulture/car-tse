"""Contract tests for the real-driver pilot evidence validator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.data.validate_real_driver_sessions import validate
from scripts.eval import eval_real_driver_sessions as real_driver_eval
from scripts.eval.eval_real_driver_sessions import aggregate


def _write_session(root: Path, *, session_id: str = "D01_S01", valid: bool = True) -> None:
    session_dir = root / session_id
    session_dir.mkdir(parents=True)
    sf.write(session_dir / "audio.wav", np.zeros((16000 * 8, 4), dtype=np.float32), 16000, subtype="PCM_16")
    metadata = {
        "session_id": session_id,
        "split": "tuning",
        "driver_id": "D01",
        "audio_path": "audio.wav",
        "recording_sample_rate_hz": 16000,
        "channel_count": 4,
        "consent": {
            "permission_to_use": True,
            "consent_record_id": "consent-D01-S01",
            "owner": "pilot operator",
            "permitted_purpose": "research evaluation",
            "license_or_agreement": "signed consent record",
        },
    }
    if not valid:
        metadata["consent"]["license_or_agreement"] = ""
    (session_dir / "session_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    rows = [
        {"session_id": session_id, "event": "enrollment", "start_s": 0.5, "end_s": 2.0},
        {"session_id": session_id, "event": "driver_wake", "start_s": 2.5, "end_s": 3.2},
        {
            "session_id": session_id,
            "event": "driver_command",
            "start_s": 3.4,
            "end_s": 5.0,
            "expected_intent": "activate|lamp",
        },
        {"session_id": session_id, "event": "passenger_wake", "start_s": 5.2, "end_s": 5.8},
        {"session_id": session_id, "event": "passenger_speech", "start_s": 5.9, "end_s": 6.5},
        {"session_id": session_id, "event": "no_command", "start_s": 6.6, "end_s": 7.0},
        {"session_id": session_id, "event": "noise_only", "start_s": 7.0, "end_s": 7.8},
    ]
    (session_dir / "annotations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_validator_accepts_complete_consented_sequence(tmp_path: Path):
    _write_session(tmp_path)
    result = validate(tmp_path, {"D01"}, set())
    assert result["valid"] is True
    assert result["session_count"] == 1


def test_validator_rejects_incomplete_consent_record(tmp_path: Path):
    _write_session(tmp_path, valid=False)
    result = validate(tmp_path, {"D01"}, set())
    assert result["valid"] is False
    assert any("consent fields must be non-empty" in error for error in result["errors"])


def test_validator_rejects_missing_enrollment_to_wake_order(tmp_path: Path):
    _write_session(tmp_path)
    annotation_path = tmp_path / "D01_S01" / "annotations.jsonl"
    rows = [json.loads(line) for line in annotation_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["start_s"] = 4.0
    rows[0]["end_s"] = 5.0
    annotation_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = validate(tmp_path, {"D01"}, set())
    assert result["valid"] is False
    assert any("enrollment interval precedes" in error for error in result["errors"])


def test_validator_rejects_unassigned_driver_when_split_lists_are_provided(tmp_path: Path):
    _write_session(tmp_path)
    result = validate(tmp_path, {"D02"}, {"D03"})
    assert result["valid"] is False
    assert any("missing from tuning/test driver assignment" in error for error in result["errors"])


def test_aggregate_keeps_conservative_and_interval_false_accept_rates_separate():
    summary = aggregate(
        [
            {
                "duration_seconds": 3600.0,
                "driver_wake_events": 2,
                "driver_wakes_detected": 1,
                "driver_command_events": 1,
                "correct_commands": 1,
                "false_accept_events": 2,
                "negative_interval_false_accept_events": 1,
                "false_action_events": 1,
                "matched_wakes": [],
                "command_details": [],
            }
        ]
    )
    assert summary["false_accepts_per_hour"] == 2.0
    assert summary["negative_interval_false_accepts_per_hour"] == 1.0
    assert summary["false_actions_per_hour"] == 1.0


def test_safety_rates_use_negative_exposure_when_available():
    summary = aggregate(
        [
            {
                "duration_seconds": 3600.0,
                "negative_exposure_seconds": 600.0,
                "driver_wake_events": 1,
                "driver_wakes_detected": 1,
                "driver_command_events": 1,
                "correct_commands": 1,
                "false_accept_events": 2,
                "negative_interval_false_accept_events": 2,
                "false_action_events": 1,
                "matched_wakes": [],
                "command_details": [],
            }
        ]
    )
    # The historical total-recording rates remain available for comparison,
    # but safety rates are normalized by the labeled negative exposure.
    assert summary["false_accepts_per_hour"] == 2.0
    assert summary["false_actions_per_hour"] == 1.0
    assert summary["negative_exposure_hours"] == 600.0 / 3600.0
    assert summary["false_accepts_per_negative_hour"] == 12.0
    assert summary["false_actions_per_negative_hour"] == 6.0
    assert summary["confidence_intervals"]["driver_wake_recall_95"]["lower"] < 1.0
    assert summary["confidence_intervals"]["driver_wake_recall_95"]["upper"] == 1.0
    assert summary["confidence_intervals"]["false_accepts_per_negative_hour_95"]["rate_per_hour"] == 12.0
    assert summary["confidence_intervals"]["false_accepts_per_negative_hour_95"]["upper_per_hour"] > 12.0


def test_threshold_selection_requires_false_accept_and_false_action_limits(monkeypatch):
    def fake_evaluate_group(_pipeline, _sessions, *, kws_threshold, **_kwargs):
        # The lower KWS threshold has stronger recall but violates only the
        # false-accept limit.  A selector that filters on false actions alone
        # would incorrectly choose it.
        if float(kws_threshold) < 0.5:
            summary = {
                "driver_wake_recall": 1.0,
                "command_accuracy": 1.0,
                "false_accepts_per_hour": 2.0,
                "false_actions_per_hour": 0.0,
            }
        else:
            summary = {
                "driver_wake_recall": 0.8,
                "command_accuracy": 0.8,
                "false_accepts_per_hour": 0.0,
                "false_actions_per_hour": 0.0,
            }
        return summary, []

    monkeypatch.setattr(real_driver_eval, "evaluate_group", fake_evaluate_group)
    selected, _ = real_driver_eval.choose_thresholds(
        object(),
        [object()],
        presence_values=[0.15],
        kws_values=[0.3, 0.5],
        command_presence_values=[0.1],
        speaker_gate_values=[None],
        command_speaker_gate_threshold=None,
        intent_confidence_values=[None],
        hop_seconds=0.25,
        device=object(),
        max_false_accepts_per_hour=1.0,
        max_false_actions_per_hour=1.0,
    )
    assert selected["kws_threshold"] == 0.5
