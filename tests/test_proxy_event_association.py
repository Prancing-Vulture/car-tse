import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "eval"))

from eval_continuous_full_pipeline_proxy import event_overlaps_label, intervals_overlap, row_speaker_id


def test_interval_overlap_accepts_boundary_crossing_window():
    assert intervals_overlap(0.0, 3.0, 2.9, 3.1)
    assert not intervals_overlap(0.0, 3.0, 3.0, 3.1)


def test_proxy_event_matching_uses_window_support_not_center_only():
    row = {
        "window_start_seconds": 0.0,
        "window_end_seconds": 3.0,
        "window_center_seconds": 1.5,
    }
    # The labelled event begins after the window centre, but is still inside
    # the model window and must therefore support the detection.
    assert event_overlaps_label(
        row,
        label_start=2.0,
        label_end=2.2,
        stage1_seconds=3.0,
        tolerance_before=0.0,
        tolerance_after=0.0,
    )


def test_row_speaker_id_prefers_manifest_identity_and_falls_back_to_parent():
    path = Path("/data/fsc/speaker-17/command.wav")
    assert row_speaker_id({"speaker_id": "manifest-42"}, path) == "manifest-42"
    assert row_speaker_id({}, path) == "speaker-17"
