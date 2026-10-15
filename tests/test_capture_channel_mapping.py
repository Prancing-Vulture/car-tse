from __future__ import annotations

import numpy as np
import pytest

from car_tse.io.capture import resolve_capture_channels, split_capture_audio


def _production_metadata() -> dict:
    return {
        "capture_contract": {
            "capture_profile": "production6",
            "playback_reference_required": True,
            "playback_reference_declared": True,
            "microphone_channel_indices": [0, 1, 2, 3],
            "playback_reference_channel_indices": [4, 5],
        }
    }


def test_six_channel_capture_routes_mics_and_averages_playback_reference():
    audio = np.stack([np.full(8, value, dtype=np.float32) for value in range(6)], axis=1)
    microphones, playback = split_capture_audio(audio, _production_metadata(), model_mics=4)
    assert microphones.shape == (8, 4)
    assert np.allclose(microphones[0], [0.0, 1.0, 2.0, 3.0])
    assert playback is not None
    assert np.allclose(playback, 4.5)


def test_four_channel_capture_keeps_all_channels_without_contract():
    audio = np.zeros((5, 4), dtype=np.float32)
    microphones, playback = split_capture_audio(audio, {}, model_mics=4)
    assert microphones.shape == (5, 4)
    assert playback is None


def test_wide_capture_without_mapping_is_rejected():
    with pytest.raises(ValueError, match="channel mapping"):
        resolve_capture_channels({}, channel_count=6, model_mics=4)


def test_overlapping_capture_mapping_is_rejected():
    metadata = _production_metadata()
    metadata["capture_contract"]["playback_reference_channel_indices"] = [3, 5]
    with pytest.raises(ValueError, match="overlap"):
        split_capture_audio(np.zeros((4, 6), dtype=np.float32), metadata)
