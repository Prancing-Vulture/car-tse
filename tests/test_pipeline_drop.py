"""Absent target should not trigger wake."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "speaker_extraction_SpEx"))

from car_tse.models.intent_classifier import IntentClassifier
from car_tse.models.kws_model import KWSModel
from car_tse.models.presence_head import PresenceHead
from car_tse.models.stage1_system import Stage1System
from car_tse.models.stage23_fusion import Stage23FusionHead
from car_tse.pipeline.in_car_pipeline import InCarPipeline, PipelineState, normalize_extracted_waveform, update_kws_state


def test_pipeline_drop_on_low_presence():
    cfg = {
        "L1_ms": 2.5,
        "L2_ms": 10.0,
        "L3_ms": 20.0,
        "N": 256,
        "B": 8,
        "O": 256,
        "P": 512,
        "Q": 3,
        "num_spks": 101,
        "spk_embed_dim": 256,
        "causal": False,
    }
    stage1 = Stage1System(spex_cfg=cfg, pretrained_dir=str(ROOT / "nonexistent"))
    presence = PresenceHead()
    with torch.no_grad():
        presence.fc.weight.zero_()
        presence.fc.bias.fill_(-10.0)
    kws = KWSModel()
    intent = IntentClassifier(num_classes=5)
    pipe = InCarPipeline(stage1, presence, kws, intent, presence_threshold=0.5, kws_threshold=0.99)
    mix = torch.randn(32000)
    enroll = torch.randn(16000)
    out, _ = pipe.forward(mix, enroll)
    assert out.accepted is False
    assert out.wake_detected is False


def test_pipeline_kws_requires_temporal_hits():
    state = PipelineState()
    assert update_kws_state(state, 0.8, 0.5, 3, 2, 2) is False
    assert update_kws_state(state, 0.1, 0.5, 3, 2, 2) is False
    assert update_kws_state(state, 0.8, 0.5, 3, 2, 2) is True
    assert update_kws_state(state, 0.8, 0.5, 3, 2, 2) is False


def test_stage23_joint_score_can_recover_wake_when_presence_gate_is_low():
    presence = PresenceHead()
    with torch.no_grad():
        presence.fc.weight.zero_()
        presence.fc.bias.fill_(-10.0)
    fusion = Stage23FusionHead()
    with torch.no_grad():
        for parameter in fusion.parameters():
            parameter.zero_()
        fusion.net[-1].bias.fill_(10.0)
    pipe = InCarPipeline(
        _IdentityStage1(),
        presence,
        _AlwaysWake(),
        IntentClassifier(num_classes=5),
        presence_threshold=0.99,
        kws_threshold=0.5,
        kws_smoothing_window=1,
        kws_required_hits=1,
        stage23_fusion=fusion,
    )

    output, _ = pipe.forward(torch.randn(16000), torch.randn(16000))

    assert output.presence_probability < 0.01
    assert output.wake_model_probability > 0.99
    assert output.wake_probability > 0.99
    assert output.wake_detected is True
def test_downstream_extraction_conditioning_only_attenuates_large_gain():
    wav = torch.tensor([[2.0, -1.0, 0.25], [0.2, -0.1, 0.05]])
    normalized = normalize_extracted_waveform(wav)
    assert torch.allclose(normalized[0].abs().max(), torch.tensor(0.95))
    assert torch.equal(normalized[1], wav[1])


class _IdentityStage1(nn.Module):
    def forward(self, mixture, enrollment, multichannel=None, playback_ref=None):
        return mixture, None, None, None


class _AlwaysPresent(nn.Module):
    def forward(self, s1):
        return torch.full((s1.shape[0],), 10.0, device=s1.device)


class _AlwaysWake(nn.Module):
    def forward(self, wav):
        return torch.tensor([[0.0, 10.0]], dtype=wav.dtype, device=wav.device).expand(wav.shape[0], -1)


class _WakeSequence(nn.Module):
    def __init__(self, scores):
        super().__init__()
        self.scores = list(scores)
        self.calls = 0

    def forward(self, wav):
        score = self.scores[min(self.calls, len(self.scores) - 1)]
        self.calls += 1
        return torch.tensor([[0.0, score]], dtype=wav.dtype, device=wav.device).expand(wav.shape[0], -1)


class _FramewisePresent(nn.Module):
    is_framewise = True

    class _Mel:
        hop = 1

    def __init__(self):
        super().__init__()
        self.mel = self._Mel()

    def forward(self, s1):
        return torch.full((s1.shape[0], s1.shape[-1]), 10.0, device=s1.device)


class _PresentThenAbsent(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, s1):
        self.calls += 1
        value = 10.0 if self.calls == 1 else -10.0
        return torch.full((s1.shape[0],), value, device=s1.device)


class _PresenceSequence(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = list(values)
        self.calls = 0

    def forward(self, s1):
        value = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        return torch.full((s1.shape[0],), value, device=s1.device)


class _FramewisePresentThenAbsent(_FramewisePresent):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, s1):
        self.calls += 1
        value = 10.0 if self.calls == 1 else -10.0
        return torch.full((s1.shape[0], s1.shape[-1]), value, device=s1.device)


class _CountingIntent(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.last_input_length = None
        self.last_input = None

    def forward(self, wav):
        self.calls += 1
        self.last_input_length = int(wav.shape[-1])
        self.last_input = wav.detach().clone()
        return torch.tensor([[0.0, 2.0]], dtype=wav.dtype, device=wav.device).expand(wav.shape[0], -1)


def test_pipeline_runs_intent_on_post_wake_frame_only():
    intent = _CountingIntent()
    pipe = InCarPipeline(
        _IdentityStage1(),
        _AlwaysPresent(),
        _AlwaysWake(),
        intent,
        presence_threshold=0.5,
        kws_threshold=0.5,
        kws_smoothing_window=1,
        kws_required_hits=1,
        intent_sample_rate=16000,
        intent_input_seconds=2.0,
    )
    mixture = torch.randn(3 * 16000)
    enrollment = torch.randn(16000)

    wake_output, state = pipe.forward(mixture, enrollment)
    assert wake_output.wake_detected is True
    assert wake_output.command_pending is True
    assert wake_output.intent_ready is False
    assert wake_output.intent_logits is None
    assert intent.calls == 0
    assert state.awaiting_command is True

    command_output, state = pipe.forward(mixture, enrollment, state)
    assert command_output.wake_detected is False
    assert command_output.command_pending is False
    assert command_output.intent_ready is True
    assert command_output.intent_logits is not None
    assert command_output.action is not None
    assert intent.calls == 1
    assert intent.last_input_length == 2 * 16000
    assert state.awaiting_command is False


def test_pipeline_buffers_only_new_audio_from_overlapping_windows():
    intent = _CountingIntent()
    pipe = InCarPipeline(
        _IdentityStage1(),
        _AlwaysPresent(),
        _AlwaysWake(),
        intent,
        presence_threshold=0.5,
        kws_threshold=0.5,
        kws_smoothing_window=1,
        kws_required_hits=1,
        kws_input_seconds=3.0,
        intent_sample_rate=1,
        intent_input_seconds=2.0,
    )
    enrollment = torch.zeros(1)

    _, state = pipe.forward(torch.tensor([0.0, 0.1, 0.2]), enrollment, audio_start_sample=0)
    first_command, state = pipe.forward(
        torch.tensor([0.1, 0.2, 0.3]), enrollment, state, audio_start_sample=1
    )
    assert first_command.command_pending is True
    assert intent.calls == 0
    assert torch.equal(state.command_audio, torch.tensor([[0.3]]))

    second_command, state = pipe.forward(
        torch.tensor([0.2, 0.3, 0.4]), enrollment, state, audio_start_sample=2
    )
    assert second_command.intent_ready is True
    assert intent.calls == 1
    assert torch.equal(intent.last_input, torch.tensor([[0.3, 0.4]]))
    assert state.command_audio is None
    assert state.awaiting_command is False


def test_pipeline_keeps_target_speech_after_wake_crop_as_preroll():
    intent = _CountingIntent()
    pipe = InCarPipeline(
        _IdentityStage1(),
        _FramewisePresent(),
        _AlwaysWake(),
        intent,
        presence_threshold=0.5,
        kws_threshold=0.5,
        kws_smoothing_window=1,
        kws_required_hits=1,
        kws_sample_rate=1,
        kws_input_seconds=1.0,
        intent_sample_rate=1,
        intent_input_seconds=2.0,
    )
    enrollment = torch.zeros(1)

    wake_output, state = pipe.forward(torch.tensor([0.0, 0.1, 0.2, 0.3]), enrollment)
    assert wake_output.wake_detected is True
    assert wake_output.intent_ready is False
    assert torch.equal(state.command_audio, torch.tensor([[0.1, 0.2, 0.3]]))

    command_output, state = pipe.forward(torch.tensor([0.4]), enrollment, state)
    assert command_output.intent_ready is True
    assert torch.equal(intent.last_input, torch.tensor([[0.1, 0.2]]))
    assert state.command_audio is None


def test_command_buffer_recovers_chunk_skipped_by_presence_gate():
    """An arrived command chunk remains available if the gate temporarily skips it."""
    intent = _CountingIntent()
    pipe = InCarPipeline(
        _IdentityStage1(),
        _PresenceSequence([10.0, -10.0, 10.0]),
        _AlwaysWake(),
        intent,
        presence_threshold=0.5,
        command_presence_threshold=0.5,
        kws_threshold=0.5,
        kws_sample_rate=1,
        kws_input_seconds=1.0,
        intent_sample_rate=1,
        intent_input_seconds=2.0,
        command_timeout_seconds=4.0,
    )
    enrollment = torch.zeros(1)

    wake, state = pipe.forward(torch.tensor([0.0]), enrollment, audio_start_sample=0)
    skipped, state = pipe.forward(
        torch.tensor([0.3]), enrollment, state, audio_start_sample=1
    )
    assert wake.wake_detected is True
    assert skipped.command_pending is True
    assert intent.calls == 0
    assert torch.equal(state.command_audio, torch.tensor([[0.3]]))

    result, state = pipe.forward(
        torch.tensor([0.4]), enrollment, state, audio_start_sample=2
    )
    assert result.intent_ready is True
    assert intent.calls == 1
    assert torch.equal(intent.last_input, torch.tensor([[0.3, 0.4]]))
    assert state.awaiting_command is False


def test_delayed_wake_uses_current_sliding_window_timestamp_for_preroll():
    pipe = InCarPipeline(
        _IdentityStage1(),
        _AlwaysPresent(),
        _WakeSequence([-10.0, 10.0]),
        _CountingIntent(),
        presence_threshold=0.5,
        kws_threshold=0.5,
        kws_sample_rate=1,
        kws_input_seconds=1.0,
        intent_sample_rate=1,
        intent_input_seconds=2.0,
    )
    enrollment = torch.zeros(1)

    _, state = pipe.forward(torch.tensor([0.0, 0.1, 0.2]), enrollment, audio_start_sample=0)
    wake, state = pipe.forward(
        torch.tensor([0.1, 0.2, 0.3]), enrollment, state, audio_start_sample=1
    )

    assert wake.wake_detected is True
    assert state.command_capture_start_sample == 2
    assert torch.equal(state.command_audio, torch.tensor([[0.2, 0.3]]))


def test_multi_hit_wake_backdates_capture_to_earliest_positive_crop():
    pipe = InCarPipeline(
        _IdentityStage1(),
        _AlwaysPresent(),
        _WakeSequence([-10.0, 10.0, 10.0]),
        _CountingIntent(),
        presence_threshold=0.0,
        kws_threshold=0.5,
        kws_smoothing_window=3,
        kws_required_hits=2,
        kws_sample_rate=10,
        kws_input_seconds=1.0,
        intent_sample_rate=10,
        intent_input_seconds=2.0,
        command_capture_preroll_seconds=0.25,
    )
    enrollment = torch.zeros(1)
    audio = torch.arange(50, dtype=torch.float32) / 100.0
    state = PipelineState()
    for start in (0, 10):
        window = audio[start : start + 30]
        _, state = pipe.forward(window, enrollment, state, audio_start_sample=start)
    window = audio[20:50]
    wake, state = pipe.forward(window, enrollment, state, audio_start_sample=20)

    assert wake.wake_detected is True
    # Positive KWS crops begin at 10 and 20; capture starts at the end of the
    # first one (20 samples) minus the 2-sample pre-roll, not at sample 30.
    assert state.command_capture_start_sample == 18
    assert state.command_audio is not None
    assert torch.isclose(state.command_audio[0, 0], torch.tensor(0.18))


def test_low_confidence_intent_abstains_to_noop_class():
    pipe = InCarPipeline(
        _IdentityStage1(), _AlwaysPresent(), _AlwaysWake(), _CountingIntent(),
        label_map={"0": "go", "1": "noop"}, intent_action_threshold=0.95,
    )
    state = PipelineState(command_audio=torch.ones(1, 16000), awaiting_command=True)
    output = pipe._infer_command(state, 1.0, None, torch.ones(1, 16000))

    assert output.intent_ready is True
    assert output.intent_confidence is not None
    assert output.intent_confidence < 0.95
    assert output.action is not None
    assert output.action.action == "noop"


def test_command_timeout_tracks_elapsed_audio_for_different_chunk_sizes():
    intent = _CountingIntent()
    pipe = InCarPipeline(
        _IdentityStage1(),
        _PresentThenAbsent(),
        _AlwaysWake(),
        intent,
        presence_threshold=0.5,
        kws_threshold=0.5,
        kws_smoothing_window=1,
        kws_required_hits=1,
        kws_sample_rate=1,
        kws_input_seconds=1.0,
        intent_sample_rate=2,
        intent_input_seconds=2.0,
        command_timeout_calls=4,
        command_timeout_seconds=2.0,
    )
    enrollment = torch.zeros(1)

    _, state = pipe.forward(torch.tensor([0.0]), enrollment)
    first_wait, state = pipe.forward(torch.zeros(2), enrollment, state)
    second_wait, state = pipe.forward(torch.zeros(1), enrollment, state)
    assert first_wait.command_pending is True
    assert second_wait.command_pending is True
    assert state.command_elapsed_samples == 3
    assert state.awaiting_command is True

    timeout, state = pipe.forward(torch.zeros(1), enrollment, state)
    assert timeout.command_pending is False
    assert state.command_elapsed_samples == 0
    assert state.awaiting_command is False


def test_wake_preroll_alone_does_not_trigger_intent_without_new_target_speech():
    intent = _CountingIntent()
    pipe = InCarPipeline(
        _IdentityStage1(),
        _FramewisePresentThenAbsent(),
        _AlwaysWake(),
        intent,
        presence_threshold=0.5,
        kws_threshold=0.5,
        kws_smoothing_window=1,
        kws_required_hits=1,
        kws_sample_rate=1,
        kws_input_seconds=1.0,
        intent_sample_rate=1,
        intent_input_seconds=2.0,
        command_timeout_calls=2,
    )
    enrollment = torch.zeros(1)

    _, state = pipe.forward(torch.tensor([0.0, 0.1, 0.2, 0.3]), enrollment)
    assert state.command_audio is not None
    first_wait, state = pipe.forward(torch.zeros(1), enrollment, state)
    timeout, state = pipe.forward(torch.zeros(1), enrollment, state)

    assert first_wait.command_pending is True
    assert timeout.intent_ready is False
    assert intent.calls == 0
    assert state.awaiting_command is False
