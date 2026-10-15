from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from eval_continuous_full_pipeline_proxy import (  # noqa: E402
    find_disjoint_wake,
    find_independent_enrollment,
    is_target_command_event,
    is_target_wake_event,
)


def test_target_wake_needs_runtime_confirmation_and_positive_target_audio_support():
    windows = [
        {"kws_crop_start_seconds": 1.0, "kws_crop_end_seconds": 2.0},
        {"kws_crop_start_seconds": 1.5, "kws_crop_end_seconds": 2.5},
        {"kws_crop_start_seconds": 6.0, "kws_crop_end_seconds": 7.0},
    ]
    matching = {"wake_positive_support_window_ids": [0, 1]}
    passenger_only = {"wake_positive_support_window_ids": [2]}
    one_target_hit = {"wake_positive_support_window_ids": [0, 2]}
    insufficient_runtime_hits = {"wake_positive_support_window_ids": [0]}

    assert is_target_wake_event(matching, windows, 1.0, 2.0, required_hits=2)
    assert not is_target_wake_event(passenger_only, windows, 1.0, 2.0, required_hits=1)
    # One target-overlapping positive crop is enough for attribution because
    # the runtime itself has confirmed the event using the other support hit.
    assert is_target_wake_event(one_target_hit, windows, 1.0, 2.0, required_hits=2)
    assert not is_target_wake_event(
        insufficient_runtime_hits, windows, 1.0, 2.0, required_hits=2
    )


def test_target_command_must_follow_the_target_wake_and_match_its_time_range():
    candidate = {
        "associated_wake_event_id": "wake_0000",
        "decision_available_seconds": 4.5,
    }
    assert is_target_command_event(candidate, "wake_0000", 2.1, 4.1)
    assert not is_target_command_event(candidate, None, 2.1, 4.1)
    assert not is_target_command_event(candidate, "wake_0001", 2.1, 4.1)
    assert not is_target_command_event(
        {**candidate, "decision_available_seconds": 7.2}, "wake_0000", 2.1, 4.1
    )


def test_scene_sources_are_independent_and_speaker_disjoint(tmp_path: Path):
    target = tmp_path / "marvin" / "speaker_a_nohash_0.wav"
    enrollment = tmp_path / "marvin" / "speaker_a_nohash_1.wav"
    passenger = tmp_path / "marvin" / "speaker_b_nohash_0.wav"
    target.parent.mkdir(parents=True)
    target.touch()
    enrollment.touch()
    passenger.touch()

    enrollment_row = find_independent_enrollment(
        [
            {"wav": str(target), "label": 1},
            {"wav": str(enrollment), "label": 1},
        ],
        "speaker_a",
        target,
    )
    passenger_row = find_disjoint_wake(
        [{"wav": str(target)}, {"wav": str(passenger)}], {"speaker_a"}
    )

    assert Path(enrollment_row["wav"]).resolve() != target.resolve()
    assert passenger_row["wav"] == str(passenger)
