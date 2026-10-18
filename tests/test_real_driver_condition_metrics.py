from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.eval.eval_real_driver_sessions import Session, aggregate, summarize_conditions


def _session() -> Session:
    return Session(
        directory=Path("."),
        metadata={"session_id": "S01", "driver_id": "D01", "split": "tuning"},
        annotations=[],
        audio=np.zeros((8 * 16000, 1), dtype=np.float32),
        sample_rate=16000,
    )


def _event(event: str, start: float, end: float, condition: str, **extra):
    return {
        "event": event,
        "start_s": start,
        "end_s": end,
        "condition": condition,
        **extra,
    }


def test_condition_metrics_use_negative_exposure_union():
    session = _session()
    driver_wake = _event("driver_wake", 1.0, 1.5, "road_5db")
    driver_command = _event("driver_command", 1.6, 2.5, "road_5db", expected_intent="activate|lights")
    passenger_wake = _event("passenger_wake", 4.0, 5.0, "passenger")
    noise = _event("noise_only", 6.0, 8.0, "road_5db")
    negative_events = [passenger_wake, noise]
    matched_wakes = [{"truth": driver_wake, "prediction": {"window_center_seconds": 1.25}}]
    command_details = [{
        "truth": driver_command,
        "prediction": {"window_center_seconds": 2.0},
        "correct": True,
    }]
    unmatched = [{"window_center_seconds": 4.5}]
    false_actions = [{"window_center_seconds": 4.5}]

    by_condition = summarize_conditions(
        session,
        driver_wakes=[driver_wake],
        driver_commands=[driver_command],
        matched_wakes=matched_wakes,
        command_details=command_details,
        unmatched_wakes=unmatched,
        false_action_events=false_actions,
        eligible_negative_events=negative_events,
    )

    assert by_condition["road_5db"]["driver_wake_recall"] == 1.0
    assert by_condition["road_5db"]["command_accuracy"] == 1.0
    assert by_condition["road_5db"]["exposure_seconds"] == 2.0
    assert by_condition["passenger"]["false_accept_events"] == 1
    assert by_condition["passenger"]["false_actions_per_hour"] == 3600.0

    overall = aggregate([{
        "duration_seconds": 8.0,
        "driver_wake_events": 1,
        "driver_wakes_detected": 1,
        "driver_command_events": 1,
        "correct_commands": 1,
        "false_accept_events": 1,
        "negative_interval_false_accept_events": 1,
        "false_action_events": 1,
        "matched_wakes": matched_wakes,
        "command_details": command_details,
        "condition_metrics": by_condition,
    }])
    assert overall["by_condition"]["passenger"]["false_accepts_per_hour"] == 3600.0
