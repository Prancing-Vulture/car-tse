from __future__ import annotations

import pytest

from scripts.data.preflight_capture_hardware import preflight, probe_device


class _FakeSoundDevice:
    def __init__(self):
        self.devices = [
            {"name": "mono", "max_input_channels": 1},
            {"name": "array", "max_input_channels": 6},
            {"name": "output", "max_input_channels": 0},
        ]

    def query_devices(self, device=None, kind=None):
        if device is None:
            return self.devices
        return self.devices[int(device)]

    def check_input_settings(self, *, device, channels, samplerate, dtype):
        if int(device) == 1 and int(samplerate) == 16000:
            return None
        raise RuntimeError("unsupported format")


def test_preflight_filters_outputs_and_accepts_matching_array():
    report = preflight(_FakeSoundDevice(), sample_rate=16000, channel_count=4)
    assert report["recorded"] is False
    assert report["device_count"] == 2
    assert [row["name"] for row in report["supported_devices"]] == ["array"]


def test_probe_reports_channel_mismatch_without_format_probe():
    fake = _FakeSoundDevice()
    result = probe_device(fake, 0, sample_rate=16000, channel_count=4)
    assert result["supported"] is False
    assert "channel" in result["reason"]


def test_preflight_requires_positive_contract():
    with pytest.raises(ValueError):
        # The CLI validates this before calling the helper; this test keeps
        # the helper contract explicit if it is reused programmatically.
        preflight(_FakeSoundDevice(), sample_rate=0, channel_count=4)
