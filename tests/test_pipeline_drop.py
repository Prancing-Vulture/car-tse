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


def test_pipeline_kws_refractory_expires_before_second_wake():
    state = PipelineState()
    scores = [0.8, 0.8, 0.8]
    assert [update_kws_state(state, score, 0.5, 3, 2, 2) for score in scores] == [False, False, True]
    # Two calls are suppressed after the first wake, even though the score
    # stream remains positive.  A later wake is possible once the refractory
    # period has elapsed.
    assert update_kws_state(state, 0.8, 0.5, 3, 2, 2) is False
    assert update_kws_state(state, 0.8, 0.5, 3, 2, 2) is False
    assert update_kws_state(state, 0.8, 0.5, 3, 2, 2) is True


def test_post_action_refractory_blocks_only_after_action_commit():
    pipe = InCarPipeline(
        _IdentityStage1(),
        _AlwaysPresent(),
        _AlwaysWake(),
        _CountingIntent(),
        presence_threshold=0.5,
        kws_threshold=0.5,
        kws_smoothing_window=1,
        kws_required_hits=1,
        post_action_refractory_calls=2,
    )
    mixture = torch.randn(3 * 16000)
    enrollment = torch.randn(16000)

    wake, state = pipe.forward(mixture, enrollment)
    assert wake.wake_detected is True
    committed, state = pipe.forward(mixture, enrollment, state)
    assert committed.intent_ready is True
    assert committed.action is not None
    assert state.post_action_refractory_left == 2

    blocked, state = pipe.forward(mixture, enrollment, state)
    assert blocked.wake_detected is False
    assert blocked.intent_ready is False
    assert blocked.action is None
    blocked, state = pipe.forward(mixture, enrollment, state)
    assert blocked.wake_detected is False
    assert state.post_action_refractory_left == 0

    next_wake, _ = pipe.forward(mixture, enrollment, state)
    assert next_wake.wake_detected is True


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


class _SequencedIntent(nn.Module):
    def __init__(self, logits: list[tuple[float, float]]):
        super().__init__()
        self.logits = [torch.tensor(row, dtype=torch.float32) for row in logits]
        self.calls = 0

    def forward(self, wav):
        index = min(self.calls, len(self.logits) - 1)
        self.calls += 1
        return self.logits[index].to(wav.device, wav.dtype).expand(wav.shape[0], -1)


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


def test_confidence_buffered_command_waits_for_a_better_candidate():
    intent = _SequencedIntent([(1.0, 2.0), (0.0, 5.0)])
    pipe = InCarPipeline(
        _IdentityStage1(),
        _AlwaysPresent(),
        _AlwaysWake(),
        intent,
        presence_threshold=0.5,
        kws_threshold=0.5,
        kws_smoothing_window=1,
        kws_required_hits=1,
        command_window_strategy="confidence_buffered",
        command_buffer_calls=2,
        command_confidence_commit_threshold=0.90,
        intent_confidence_threshold=0.53,
    )
    pipe._prepare_kws_trimmed_intent_input = lambda wav: wav
    mixture = torch.randn(3 * 16000)
    enrollment = torch.randn(16000)

    _, state = pipe.forward(mixture, enrollment)
    pending, state = pipe.forward(mixture, enrollment, state)
    assert pending.command_pending is True
    assert pending.intent_ready is False
    assert state.awaiting_command is True
    committed, state = pipe.forward(mixture, enrollment, state)
    assert committed.intent_ready is True
    assert committed.intent_logits is not None
    assert intent.calls == 2
    assert state.awaiting_command is False


def test_confidence_buffered_command_suppresses_after_buffer_exhaustion():
    intent = _SequencedIntent([(0.0, 0.0), (0.0, 0.0)])
    pipe = InCarPipeline(
        _IdentityStage1(),
        _AlwaysPresent(),
        _AlwaysWake(),
        intent,
        presence_threshold=0.5,
        kws_threshold=0.5,
        kws_smoothing_window=1,
        kws_required_hits=1,
        command_window_strategy="confidence_buffered",
        command_buffer_calls=2,
        command_confidence_commit_threshold=0.90,
        intent_confidence_threshold=0.60,
    )
    pipe._prepare_kws_trimmed_intent_input = lambda wav: wav
    mixture = torch.randn(3 * 16000)
    enrollment = torch.randn(16000)

    _, state = pipe.forward(mixture, enrollment)
    pending, state = pipe.forward(mixture, enrollment, state)
    assert pending.command_pending is True
    suppressed, state = pipe.forward(mixture, enrollment, state)
    assert suppressed.command_suppressed is True
    assert suppressed.intent_ready is False
    assert state.awaiting_command is False


def test_confidence_buffered_command_commits_best_candidate_not_last():
    intent = _SequencedIntent([(5.0, 0.0), (0.0, 0.0)])
    pipe = InCarPipeline(
        _IdentityStage1(),
        _AlwaysPresent(),
        _AlwaysWake(),
        intent,
        presence_threshold=0.5,
        kws_threshold=0.5,
        kws_smoothing_window=1,
        kws_required_hits=1,
        command_window_strategy="confidence_buffered",
        command_buffer_calls=2,
        command_confidence_commit_threshold=1.0,
        intent_confidence_threshold=0.53,
    )
    pipe._prepare_kws_trimmed_intent_input = lambda wav: wav
    mixture = torch.randn(3 * 16000)
    enrollment = torch.randn(16000)

    _, state = pipe.forward(mixture, enrollment)
    pending, state = pipe.forward(mixture, enrollment, state)
    assert pending.command_pending is True
    committed, state = pipe.forward(mixture, enrollment, state)
    assert committed.intent_ready is True
    assert int(torch.argmax(committed.intent_logits, dim=-1).item()) == 0
    assert state.awaiting_command is False


def test_confidence_buffered_command_uses_longest_consecutive_run():
    intent = _SequencedIntent([(5.0, 0.0), (0.0, 5.0), (0.0, 6.0), (5.0, 0.0)])
    pipe = InCarPipeline(
        _IdentityStage1(),
        _AlwaysPresent(),
        _AlwaysWake(),
        intent,
        presence_threshold=0.5,
        kws_threshold=0.5,
        kws_smoothing_window=1,
        kws_required_hits=1,
        command_window_strategy="confidence_buffered",
        command_buffer_calls=4,
        command_confidence_commit_threshold=1.0,
        intent_confidence_threshold=0.53,
    )
    pipe._prepare_kws_trimmed_intent_input = lambda wav: wav
    mixture = torch.randn(3 * 16000)
    enrollment = torch.randn(16000)

    _, state = pipe.forward(mixture, enrollment)
    for _ in range(3):
        pending, state = pipe.forward(mixture, enrollment, state)
        assert pending.command_pending is True
    committed, state = pipe.forward(mixture, enrollment, state)
    assert committed.intent_ready is True
    assert int(torch.argmax(committed.intent_logits, dim=-1).item()) == 1
    assert state.awaiting_command is False


def test_intent_window_offset_selects_post_wake_samples():
    pipe = InCarPipeline(
        _IdentityStage1(),
        _AlwaysPresent(),
        _AlwaysWake(),
        _LowConfidenceIntent(),
        intent_sample_rate=10,
        intent_input_seconds=1.0,
        intent_window_offset_seconds=0.1,
    )
    waveform = torch.arange(20, dtype=torch.float32)
    selected = pipe._prepare_intent_input(waveform)
    assert torch.equal(selected, torch.arange(1, 11, dtype=torch.float32))


def test_kws_trim_command_window_uses_last_local_wake_hit():
    pipe = InCarPipeline(
        _IdentityStage1(),
        _AlwaysPresent(),
        _AlwaysWake(),
        _LowConfidenceIntent(),
        kws_sample_rate=10,
        kws_input_seconds=1.0,
        intent_sample_rate=10,
        intent_input_seconds=1.0,
        command_window_strategy="kws_trim",
        command_endpoint_kws_hop_seconds=0.5,
        command_endpoint_margin_seconds=0.2,
        kws_threshold=0.5,
    )
    pipe._regular_kws_score = lambda wav: 0.9 if float(wav[..., 0].item()) in {0.0, 5.0} else 0.1
    waveform = torch.arange(30, dtype=torch.float32)
    selected = pipe._prepare_kws_trimmed_intent_input(waveform)
    assert selected is not None
    assert torch.equal(selected, torch.arange(13, 23, dtype=torch.float32))


def test_kws_trim_latest_strategy_is_accepted():
    pipe = InCarPipeline(
        _IdentityStage1(),
        _AlwaysPresent(),
        _AlwaysWake(),
        _LowConfidenceIntent(),
        command_window_strategy="kws_trim_latest",
    )
    assert pipe.command_window_strategy == "kws_trim_latest"


def test_kws_trim_latest_uses_post_wake_frame():
    pipe = InCarPipeline(
        _IdentityStage1(),
        _AlwaysPresent(),
        _AlwaysWake(),
        _CountingIntent(),
        command_window_strategy="kws_trim_latest",
        presence_threshold=0.5,
        kws_threshold=0.5,
        kws_smoothing_window=1,
        kws_required_hits=1,
    )
    seen = []

    def capture(wav):
        seen.append(wav.detach().clone())
        return wav[..., : 2 * 16000]

    pipe._prepare_kws_trimmed_intent_input = capture
    enrollment = torch.randn(16000)
    _, state = pipe.forward(torch.ones(3 * 16000), enrollment)
    output, _ = pipe.forward(torch.full((3 * 16000,), 2.0), enrollment, state)
    assert output.intent_ready is True
    assert len(seen) == 1
    assert torch.all(seen[0] == 0.95)


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
    assert matching_output.target_kws_probability is not None
    assert matching_output.target_kws_probability > 0.5

    mismatched_state = PipelineState(reference_speaker_embedding=torch.tensor([[-1.0, 0.0]]))
    mismatched_output, _ = pipe.forward(mixture, enrollment, mismatched_state)
    assert mismatched_output.wake_detected is False
    assert mismatched_output.wake_probability is not None
    assert mismatched_output.wake_probability < 0.5
    assert mismatched_output.target_kws_probability is not None
    assert mismatched_output.target_kws_probability > 0.5
