from __future__ import annotations

import numpy as np

from car_tse.frontend.spatial import delay_and_sum, gcc_phat_delay, mvdr_beamform


def test_delay_and_sum_aligns_integer_delays_without_circular_wrap() -> None:
    audio = np.zeros((2, 32), dtype=np.float32)
    audio[0, 8] = 1.0
    audio[1, 10] = 1.0

    result = delay_and_sum(audio, delays_samples=[0, -2])

    assert int(np.argmax(result)) == 8
    assert np.isclose(result[8], 1.0)
    assert np.allclose(result[:2], 0.0)


def test_delay_and_sum_accepts_time_first_audio() -> None:
    audio = np.zeros((16, 2), dtype=np.float32)
    audio[5, 0] = 1.0
    audio[5, 1] = 1.0

    result = delay_and_sum(audio, delays_samples=[0, 0], channel_axis=1)

    assert result.shape == (16,)
    assert np.isclose(result[5], 1.0)


def test_gcc_phat_recovers_a_small_positive_delay() -> None:
    reference = np.zeros(128, dtype=np.float32)
    reference[24] = 1.0
    signal = np.zeros_like(reference)
    signal[28] = 1.0

    assert gcc_phat_delay(reference, signal, max_delay_samples=16) == 4


def test_mvdr_returns_finite_input_length_output() -> None:
    rng = np.random.default_rng(4)
    audio = rng.normal(size=(4, 2048)).astype(np.float32)
    output = mvdr_beamform(
        audio,
        delays_samples=[0, 1, -1, 2],
        sample_rate=16000,
        frame_length=128,
        hop_length=64,
    )

    assert output.shape == (2048,)
    assert output.dtype == np.float32
    assert np.isfinite(output).all()

