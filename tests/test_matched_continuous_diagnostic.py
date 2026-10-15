from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

from car_tse.pipeline.in_car_pipeline import InCarPipeline, PipelineState
from scripts.eval.replay_matched_continuous_diagnostic import (
    CallContext,
    MatchedCapturePipeline,
    ensure_output_dir_available,
    validate_trial,
)


class PassthroughStage1(nn.Module):
    def forward(self, mixture, enrollment, multichannel=None, playback_ref=None):
        return mixture, enrollment, None, None


class PositiveFramePresence(nn.Module):
    is_framewise = True

    def __init__(self):
        super().__init__()
        self.mel = SimpleNamespace(hop=2)

    def forward(self, wav):
        return torch.full((wav.shape[0], 1, 4), 5.0, device=wav.device)


class PositiveKWS(nn.Module):
    def forward(self, wav):
        return torch.tensor([[0.0, 8.0]], device=wav.device).expand(wav.shape[0], -1)


class FixedIntent(nn.Module):
    def forward(self, wav):
        return torch.stack((wav.mean(dim=-1), -wav.mean(dim=-1)), dim=-1)


def make_pipeline() -> InCarPipeline:
    return InCarPipeline(
        PassthroughStage1(),
        PositiveFramePresence(),
        PositiveKWS(),
        FixedIntent(),
        presence_threshold=0.5,
        command_presence_threshold=0.5,
        kws_threshold=0.5,
        kws_sample_rate=8,
        kws_input_seconds=0.25,
        kws_smoothing_window=1,
        kws_required_hits=1,
        intent_sample_rate=8,
        intent_input_seconds=1.0,
        command_min_speech_seconds=0.0,
        device=torch.device("cpu"),
    )


def run_two_calls(pipeline, observed=None):
    state = PipelineState()
    outputs = []
    for index in range(2):
        context = CallContext(index, index * 8, (index + 1) * 8, 8, 16)
        if observed is not None:
            observed.call_context = context
        output, state = pipeline.forward(
            torch.linspace(0.1, 0.8, 8),
            torch.linspace(0.2, 0.6, 4),
            state,
            multichannel=torch.ones(4, 8),
            audio_start_sample=index * 8,
        )
        outputs.append(output)
    return outputs, state


def test_manifest_requires_explicit_matching_and_independent_enrollment(tmp_path: Path):
    source = tmp_path / "stream.wav"
    enrollment = tmp_path / "enrollment.wav"
    sf.write(source, np.zeros((100, 4), dtype=np.float32), 16_000, subtype="FLOAT")
    sf.write(enrollment, np.zeros(100, dtype=np.float32), 16_000, subtype="FLOAT")
    row = {
        "trial_id": "driver_001",
        "session_id": "session_01",
        "source_wav": str(source),
        "enrollment_wav": str(enrollment),
        "speaker_id": "driver_a",
        "stream_speaker_id": "driver_a",
        "wake_speaker_id": "driver_a",
        "command_speaker_id": "driver_a",
        "enrollment_speaker_id": "driver_a",
        "speaker_match_verified": True,
        "speaker_match_evidence": "same recorded session; reviewed by operator",
        "speaker_match_reviewer": "operator_01",
        "event_annotation_method": "manual waveform review",
        "annotation_reviewer": "operator_01",
        "sample_rate_hz": 16_000,
        "wake_start_sample": 10,
        "wake_end_sample": 30,
        "command_start_sample": 31,
        "command_end_sample": 80,
        "expected_intent": "increase_volume",
        "command_transcript": "increase the volume",
        "command_transcript_reviewed": True,
    }
    assert validate_trial(row) == []
    row["speaker_match_verified"] = False
    row["stream_speaker_id"] = "another_speaker"
    errors = validate_trial(row)
    assert any("speaker_match_verified" in error for error in errors)
    assert any("stream_speaker_id" in error for error in errors)
    row["speaker_match_verified"] = True
    row["stream_speaker_id"] = "driver_a"
    row["wake_start_sample"] = 10.5
    assert any("integer sample counts" in error for error in validate_trial(row))
    row["wake_start_sample"] = 10
    duplicate = tmp_path / "duplicate_enrollment.wav"
    duplicate.write_bytes(source.read_bytes())
    row["enrollment_wav"] = str(duplicate)
    assert any("content duplicates" in error for error in validate_trial(row))
    row["enrollment_wav"] = str(source)
    assert any("independent recording" in error for error in validate_trial(row))
    row["enrollment_wav"] = str(enrollment)
    row["command_end_sample"] = 101
    assert any("exceeds source length" in error for error in validate_trial(row))
    row["command_end_sample"] = 80
    enrollment_8k = tmp_path / "enrollment_8k.wav"
    sf.write(enrollment_8k, np.zeros(100, dtype=np.float32), 8_000, subtype="FLOAT")
    row["enrollment_wav"] = str(enrollment_8k)
    assert any("enrollment WAV sample rate" in error for error in validate_trial(row))


def test_observer_preserves_pipeline_outputs_and_saves_actual_stage4_input(tmp_path: Path):
    baseline = make_pipeline()
    expected_input = []
    handle = baseline.intent.register_forward_pre_hook(
        lambda _module, args: expected_input.append(args[0].detach().to(device="cpu").clone())
    )
    expected, expected_state = run_two_calls(baseline)
    handle.remove()
    observed = MatchedCapturePipeline.observe_existing(baseline, tmp_path, sample_rate_hz=8)
    original_intent_hooks = len(observed.intent._forward_pre_hooks)
    original_presence_hooks = len(observed.presence._forward_hooks)
    with observed:
        actual, final_state = run_two_calls(observed, observed)
        assert len(observed.intent._forward_pre_hooks) == original_intent_hooks + 1
        assert len(observed.presence._forward_hooks) == original_presence_hooks + 1
    assert len(observed.intent._forward_pre_hooks) == original_intent_hooks
    assert len(observed.presence._forward_hooks) == original_presence_hooks
    assert len(observed.call_rows) == 2
    for expected_output, actual_output in zip(expected, actual):
        assert expected_output.accepted == actual_output.accepted
        assert expected_output.wake_detected == actual_output.wake_detected
        assert expected_output.intent_ready == actual_output.intent_ready
        assert expected_output.action == actual_output.action
        if expected_output.intent_logits is not None:
            torch.testing.assert_close(expected_output.intent_logits, actual_output.intent_logits)
    row = observed.call_rows[-1]
    exact = row["stage4_exact_input"]
    assert exact is not None
    saved = torch.load(tmp_path / exact["path"], map_location="cpu", weights_only=True)
    assert list(saved.shape) == [1, 8]
    assert row["stage4_sample_rate_hz"] == 8
    assert row["intent_ready"] is True
    assert row["command_source_segments"]
    assert all(segment["source_mapping_complete"] for segment in row["command_source_segments"])
    assert row["command_source_segments"][0]["kind"] == "wake_preroll"
    assert row["presence_overall_tail_fresh"] is not None
    assert row["presence_logits"] is not None
    assert row["kws_threshold"] == observed.kws_threshold
    assert row["kws_refractory_left_before"] == 0
    assert torch.load(tmp_path / row["stage1_raw"]["path"], map_location="cpu", weights_only=True).shape[-1] == 8
    assert torch.load(tmp_path / row["stage1_normalized"]["path"], map_location="cpu", weights_only=True).shape[-1] == 8
    assert final_state.command_audio is None
    assert final_state.awaiting_command == expected_state.awaiting_command
    assert final_state.last_audio_end_sample == expected_state.last_audio_end_sample
    assert final_state.command_elapsed_samples == expected_state.command_elapsed_samples
    assert expected_input
    torch.testing.assert_close(saved, expected_input[-1])


def test_padded_final_window_pre_roll_is_saved_but_not_given_a_false_source_span(tmp_path: Path):
    observed = MatchedCapturePipeline.observe_existing(make_pipeline(), tmp_path, sample_rate_hz=8)
    observed.call_context = CallContext(0, 8, 12, 8, 12)
    with observed:
        output, state = observed.forward(
            torch.tensor([0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0]),
            torch.ones(4),
            PipelineState(),
            multichannel=torch.ones(4, 8),
            audio_start_sample=8,
        )
    assert output.wake_detected
    append = observed.call_rows[0]["command_appends"][0]
    assert append["kind"] == "wake_preroll"
    assert append["source_mapping_complete"] is False
    assert append["source_start_sample"] is None
    assert append["source_end_sample"] is None
    assert state.command_audio is not None


def test_observer_removes_hooks_after_exception(tmp_path: Path):
    pipeline = make_pipeline()
    observed = MatchedCapturePipeline.observe_existing(pipeline, tmp_path, sample_rate_hz=8)
    intent_hooks = len(observed.intent._forward_pre_hooks)
    presence_hooks = len(observed.presence._forward_hooks)
    try:
        with observed:
            raise RuntimeError("synthetic observer failure")
    except RuntimeError:
        pass
    assert len(observed.intent._forward_pre_hooks) == intent_hooks
    assert len(observed.presence._forward_hooks) == presence_hooks


def test_output_directory_guard_refuses_to_overwrite(tmp_path: Path):
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "trace.json").write_text("keep me", encoding="utf-8")
    try:
        ensure_output_dir_available(output_dir)
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing diagnostic output should be protected")
    assert (output_dir / "trace.json").read_text(encoding="utf-8") == "keep me"
