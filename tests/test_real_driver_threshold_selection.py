from __future__ import annotations

import torch

from scripts.eval import eval_real_driver_sessions as evaluator


def _choose(monkeypatch, *, safe: bool):
    def fake_evaluate_group(*args, **kwargs):
        presence = float(kwargs["presence_threshold"])
        summary = {
            "driver_wake_recall": 0.8 + presence * 0.1,
            "command_accuracy": 0.7,
            "false_accepts_per_hour": 0.5 if safe else 2.0,
            "false_actions_per_hour": 0.5 if safe else 2.0,
        }
        return summary, []

    monkeypatch.setattr(evaluator, "evaluate_group", fake_evaluate_group)
    return evaluator.choose_thresholds(
        pipeline=None,
        sessions=[object()],
        presence_values=[0.1, 0.2],
        kws_values=[0.5],
        command_presence_values=[0.1],
        speaker_gate_values=[None],
        command_speaker_gate_threshold=None,
        intent_confidence_values=[None],
        hop_seconds=0.25,
        device=torch.device("cpu"),
        max_false_accepts_per_hour=1.0,
        max_false_actions_per_hour=1.0,
    )


def test_threshold_selection_marks_safe_tuning_point(monkeypatch):
    selected, _ = _choose(monkeypatch, safe=True)

    assert selected["selection_status"] == "safe_tuning_operating_point"
    assert selected["safe_candidate_count"] == 2


def test_threshold_selection_marks_unsafe_diagnostic_fallback(monkeypatch):
    selected, _ = _choose(monkeypatch, safe=False)

    assert selected["selection_status"] == "no_safe_tuning_point_diagnostic_fallback"
    assert selected["safe_candidate_count"] == 0
