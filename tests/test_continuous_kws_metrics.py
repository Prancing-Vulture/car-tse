import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "eval"))

from eval_aishell5_continuous_kws import summarize_kws_threshold


def test_smoothed_wake_windows_are_reported_as_merged_events():
    # Three positive windows followed by a long gap and two more positive
    # windows should be two operating events, not five false accepts.
    scores = np.asarray([0.9, 0.9, 0.9] + [0.0] * 10 + [0.9, 0.9], dtype=np.float32)
    presence = np.ones_like(scores)
    result = summarize_kws_threshold(
        presence,
        scores,
        duration=float(len(scores)) * 0.25,
        presence_threshold=0.5,
        kws_threshold=0.5,
        smooth_window=1,
        required_hits=1,
        refractory_calls=0,
        hop_seconds=0.25,
    )

    assert result["full_path_smoothed_positive_windows"] == 5
    assert result["full_path_smoothed_events"] == 2
    assert result["kws_only_smoothed_positive_windows"] == 5
    assert result["kws_only_smoothed_events"] == 2


def test_late_speaker_fusion_can_veto_keyword_score_without_changing_kws_only():
    scores = np.asarray([0.9, 0.9], dtype=np.float32)
    presence = np.ones_like(scores)
    speaker = np.asarray([0.30, 0.10], dtype=np.float32)
    result = summarize_kws_threshold(
        presence,
        scores,
        duration=2.0,
        presence_threshold=0.5,
        kws_threshold=0.5,
        smooth_window=1,
        required_hits=1,
        refractory_calls=0,
        hop_seconds=1.0,
        speaker_array=speaker,
        speaker_fusion_mode="min",
        speaker_fusion_center=0.20,
        speaker_fusion_scale=0.05,
    )

    assert result["full_path_smoothed_positive_windows"] == 1
    assert result["kws_only_smoothed_positive_windows"] == 2
    assert result["speaker_fusion_mode"] == "min"


def test_activity_margin_can_veto_target_absent_windows_without_changing_kws_only():
    scores = np.asarray([0.9, 0.9], dtype=np.float32)
    presence = np.ones_like(scores)
    margins = np.asarray([-0.2, 0.1], dtype=np.float32)
    result = summarize_kws_threshold(
        presence,
        scores,
        duration=2.0,
        presence_threshold=0.5,
        kws_threshold=0.5,
        smooth_window=1,
        required_hits=1,
        refractory_calls=0,
        hop_seconds=1.0,
        activity_margin_array=margins,
        activity_margin_threshold=0.0,
    )

    assert result["activity_margin_pass_rate"] == 0.5
    assert result["full_path_smoothed_positive_windows"] == 1
    assert result["kws_only_smoothed_positive_windows"] == 2
