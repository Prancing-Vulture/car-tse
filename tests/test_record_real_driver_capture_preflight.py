from __future__ import annotations

import pytest

from scripts.data.record_real_driver_session import validate_capture_device


class FakeSoundDevice:
    def __init__(self, max_input_channels: int):
        self.max_input_channels = max_input_channels
        self.checked = None

    def query_devices(self, device, direction):
        assert direction == "input"
        return {
            "name": "fake input",
            "max_input_channels": self.max_input_channels,
        }

    def check_input_settings(self, **kwargs):
        self.checked = kwargs


def test_capture_preflight_records_requested_contract():
    sounddevice = FakeSoundDevice(4)
    info = validate_capture_device(sounddevice, 3, channel_count=4)

    assert info["max_input_channels"] == 4
    assert info["requested_channels"] == 4
    assert sounddevice.checked == {
        "device": 3,
        "channels": 4,
        "samplerate": 16000,
        "dtype": "float32",
    }


def test_capture_preflight_rejects_insufficient_channels():
    sounddevice = FakeSoundDevice(2)
    with pytest.raises(ValueError, match=r"2 channel\(s\), but 4 requested"):
        validate_capture_device(sounddevice, 1, channel_count=4)
