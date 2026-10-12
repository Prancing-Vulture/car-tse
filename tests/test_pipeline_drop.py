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


def test_downstream_extraction_conditioning_only_attenuates_large_gain():
    wav = torch.tensor([[2.0, -1.0, 0.25], [0.2, -0.1, 0.05]])
    normalized = normalize_extracted_waveform(wav)
    assert torch.allclose(normalized[0].abs().max(), torch.tensor(0.95))
    assert torch.equal(normalized[1], wav[1])


class _IdentityStage1(nn.Module):
    def forward(self, mixture, enrollment, multichannel=None, playback_ref=None):
        return mixture, None, None, None


class _EmbeddingIdentityStage1(_IdentityStage1):
    def speaker_embedding(self, waveform):
        return torch.tensor([[1.0, 0.0]], dtype=waveform.dtype, device=waveform.device).expand(
            waveform.shape[0], -1
        )


class _AlwaysPresent(nn.Module):
    def forward(self, s1):
        return torch.full((s1.shape[0],), 10.0, device=s1.device)


class _PassengerActivity(nn.Module):
    is_multiclass = True
    requires_enrollment = True
    target_class_index = 1
    num_classes = 3

    def forward(self, s1, enrollment_embedding):
        # Target speech is present enough to pass the ordinary presence
        # threshold, but the non-target posterior is larger.
        logits = torch.tensor([0.0, 4.0, 5.0], dtype=s1.dtype, device=s1.device)
        return logits.view(1, 3, 1).expand(s1.shape[0], -1, 8)

    @classmethod
    def activity_probabilities(cls, logits):
        return torch.softmax(logits.float(), dim=1)

    @staticmethod
    def aggregate_probability(probability, top_fraction=0.20):
        top_k = max(1, int(round(probability.shape[-1] * float(top_fraction))))
        return probability.topk(top_k, dim=-1).values.mean(dim=-1)


class _AlwaysWake(nn.Module):
    def forward(self, wav):
        return torch.tensor([[0.0, 10.0]], dtype=wav.dtype, device=wav.device).expand(wav.shape[0], -1)


class _AlwaysTargetWake(nn.Module):
    def forward(self, wav, enrollment_embedding, extracted_embedding):
        return torch.tensor([[0.0, 4.0]], dtype=wav.dtype, device=wav.device).expand(wav.shape[0], -1)


class _CountingIntent(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.last_input_length = None

    def forward(self, wav):
        self.calls += 1
        self.last_input_length = int(wav.shape[-1])
        return torch.tensor([[0.0, 2.0]], dtype=wav.dtype, device=wav.device).expand(wav.shape[0], -1)


class _LowConfidenceIntent(nn.Module):
    def forward(self, wav):
        return torch.zeros((wav.shape[0], 2), dtype=wav.dtype, device=wav.device)


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


def test_personal_activity_margin_vetoes_non_target_speech():
    pipe = InCarPipeline(
        _EmbeddingIdentityStage1(),
        _PassengerActivity(),
        _AlwaysWake(),
        _LowConfidenceIntent(),
        presence_threshold=0.1,
        presence_activity_margin_threshold=0.0,
        kws_threshold=0.5,
        kws_smoothing_window=1,
        kws_required_hits=1,
    )
    mixture = torch.randn(3 * 16000)
    enrollment = torch.randn(16000)

    output, _ = pipe.forward(mixture, enrollment)
    assert output.accepted is False
    assert output.wake_detected is False
    assert output.presence_non_target_probability is not None
    assert output.presence_activity_margin is not None
    assert output.presence_activity_margin < 0.0


def test_pipeline_can_delay_command_window_after_wake():
    pipe = InCarPipeline(
        _IdentityStage1(),
        _AlwaysPresent(),
        _AlwaysWake(),
        _LowConfidenceIntent(),
        presence_threshold=0.5,
        kws_threshold=0.5,
        kws_smoothing_window=1,
        kws_required_hits=1,
        command_delay_calls=1,
    )
    mixture = torch.randn(3 * 16000)
    enrollment = torch.randn(16000)

    wake_output, state = pipe.forward(mixture, enrollment)
    assert wake_output.wake_detected is True
    delayed_output, state = pipe.forward(mixture, enrollment, state)
    assert delayed_output.command_pending is True
    assert delayed_output.intent_ready is False
    command_output, _ = pipe.forward(mixture, enrollment, state)
    assert command_output.intent_ready is True


def test_pipeline_suppresses_low_confidence_command():
    pipe = InCarPipeline(
        _IdentityStage1(),
        _AlwaysPresent(),
        _AlwaysWake(),
        _LowConfidenceIntent(),
        presence_threshold=0.5,
        kws_threshold=0.5,
        kws_smoothing_window=1,
        kws_required_hits=1,
        intent_confidence_threshold=0.6,
    )
    mixture = torch.randn(3 * 16000)
    enrollment = torch.randn(16000)

    _, state = pipe.forward(mixture, enrollment)
    output, state = pipe.forward(mixture, enrollment, state)
    assert output.command_suppressed is True
    assert output.intent_ready is False
    assert output.action is None
    assert state.awaiting_command is False
    assert state.awaiting_command is False


def test_pipeline_can_use_precomputed_enrollment_bank_embedding():
    pipe = InCarPipeline(
        _EmbeddingIdentityStage1(),
        _AlwaysPresent(),
        _AlwaysWake(),
        _LowConfidenceIntent(),
        presence_threshold=0.5,
        kws_threshold=0.5,
        kws_smoothing_window=1,
        kws_required_hits=1,
        speaker_gate_threshold=0.0,
        speaker_gate_backend="spex",
    )
    mixture = torch.randn(3 * 16000)
    enrollment = torch.randn(16000)
    # The extracted mock embedding is [1, 0], while the seeded bank is its
    # opposite.  The gate must use the seeded bank, not recompute a single
    # enrollment embedding from the raw enrollment waveform.
    state = PipelineState(reference_speaker_embedding=torch.tensor([[-1.0, 0.0]]))
    output, _ = pipe.forward(mixture, enrollment, state)
    assert output.accepted is False
    assert output.speaker_similarity is not None
    assert output.speaker_similarity < 0.0


def test_pipeline_opt_in_late_speaker_fusion_vetoes_mismatched_embedding():
    pipe = InCarPipeline(
        _EmbeddingIdentityStage1(),
        _AlwaysPresent(),
        _AlwaysWake(),
        _LowConfidenceIntent(),
        target_kws=_AlwaysTargetWake(),
        target_kws_secondary=_AlwaysTargetWake(),
        target_kws_fusion_mode="min",
        presence_threshold=0.5,
        kws_threshold=0.5,
        kws_smoothing_window=1,
        kws_required_hits=1,
        speaker_fusion_mode="min",
        speaker_fusion_center=0.0,
        speaker_fusion_scale=0.1,
    )
    mixture = torch.randn(3 * 16000)
    enrollment = torch.randn(16000)

    matching_output, _ = pipe.forward(mixture, enrollment)
    assert matching_output.wake_detected is True
    assert matching_output.wake_probability is not None
    assert matching_output.wake_probability > 0.5

    mismatched_state = PipelineState(reference_speaker_embedding=torch.tensor([[-1.0, 0.0]]))
    mismatched_output, _ = pipe.forward(mixture, enrollment, mismatched_state)
    assert mismatched_output.wake_detected is False
    assert mismatched_output.wake_probability is not None
    assert mismatched_output.wake_probability < 0.5
