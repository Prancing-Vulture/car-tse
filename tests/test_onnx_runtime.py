from __future__ import annotations

import numpy as np
import pytest

from car_tse.deployment.onnx_runtime import (
    OnnxInCarPipeline,
    _aggregate_top_fraction,
    _fit_audio,
    _normalize_extracted,
)


def test_fit_audio_crops_and_pads_without_changing_batch() -> None:
    value = np.arange(6, dtype=np.float32)[None, :]

    assert np.array_equal(_fit_audio(value, 4, name="audio"), [[0.0, 1.0, 2.0, 3.0]])
    assert np.array_equal(
        _fit_audio(value, 8, name="audio"),
        [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0]],
    )


def test_normalize_extracted_only_reduces_over_limit_peaks() -> None:
    value = np.asarray([[0.5, -2.0, 1.0]], dtype=np.float32)

    actual = _normalize_extracted(value)

    assert np.allclose(actual, [[0.2375, -0.95, 0.475]], atol=1.0e-6)


def test_aggregate_top_fraction_uses_largest_frames() -> None:
    probability = np.asarray([[0.1, 0.2, 0.9, 0.8]], dtype=np.float32)

    assert _aggregate_top_fraction(probability, 0.5) == pytest.approx(0.85)
    assert _aggregate_top_fraction(probability, 0.0) == pytest.approx(0.9)


class _FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray, np.ndarray | None, bool]] = []

    def run_components(
        self,
        multichannel,
        enrollment_audio,
        playback_reference=None,
        *,
        include_intent=True,
    ):
        multichannel = np.asarray(multichannel)
        enrollment_audio = np.asarray(enrollment_audio)
        self.calls.append((multichannel, enrollment_audio, playback_reference, bool(include_intent)))
        return {
            "extracted_audio": np.zeros((1, 48_000), dtype=np.float32),
            "downstream_audio": np.zeros((1, 48_000), dtype=np.float32),
            "speaker_embeddings": np.ones((2, 256), dtype=np.float32),
            "presence_logits": np.full((1, 301), 4.0, dtype=np.float32),
            "kws_logits": np.asarray([[0.0, 4.0]], dtype=np.float32),
            "target_kws_logits": np.asarray([[0.0, 4.0]], dtype=np.float32),
            "compact_verification_logit": np.asarray([[4.0]], dtype=np.float32),
            "intent_logits": np.asarray([[0.0, 3.0, 1.0]], dtype=np.float32),
        }


def test_composed_pipeline_preserves_playback_contract_and_wake_to_command() -> None:
    runtime = _FakeRuntime()
    pipeline = OnnxInCarPipeline(
        runtime,
        presence_threshold=0.1,
        speaker_gate_threshold=0.1,
        kws_threshold=0.5,
        kws_smoothing_window=1,
        kws_required_hits=1,
        intent_confidence_threshold=0.1,
        label_map={"1": "test_intent"},
    )
    capture = np.zeros((4, 48_000), dtype=np.float32)
    enrollment = np.zeros(24_000, dtype=np.float32)
    playback = np.zeros(48_000, dtype=np.float32)

    wake_output, state = pipeline.forward(
        capture,
        enrollment,
        multichannel=capture,
        playback_ref=playback,
    )
    command_output, _ = pipeline.forward(
        capture,
        enrollment,
        state,
        multichannel=capture,
        playback_ref=playback,
    )

    assert wake_output.wake_detected is True
    assert wake_output.command_pending is True
    assert command_output.intent_ready is True
    assert command_output.action is not None
    assert command_output.action.action == "test_intent"
    assert len(runtime.calls) == 2
    assert runtime.calls[0][3] is False
    assert runtime.calls[1][3] is True
    assert runtime.calls[0][0].shape == (4, 48_000)
    assert runtime.calls[0][2] is playback


def test_composed_pipeline_rejects_unvalidated_endpoint_strategy() -> None:
    with pytest.raises(ValueError, match="only the validated fixed command window"):
        OnnxInCarPipeline(_FakeRuntime(), command_window_strategy="endpoint")


def test_composed_pipeline_rejects_pre_extracted_shortcut() -> None:
    pipeline = OnnxInCarPipeline(_FakeRuntime())
    with pytest.raises(ValueError, match="pre_extracted is not supported"):
        pipeline.forward(
            np.zeros((4, 48_000), dtype=np.float32),
            np.zeros(24_000, dtype=np.float32),
            pre_extracted=np.zeros(48_000, dtype=np.float32),
        )
