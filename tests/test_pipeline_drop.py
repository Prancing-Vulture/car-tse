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


class _AlwaysPresent(nn.Module):
    def forward(self, s1):
        return torch.full((s1.shape[0],), 10.0, device=s1.device)


class _AlwaysWake(nn.Module):
    def forward(self, wav):
        return torch.tensor([[0.0, 10.0]], dtype=wav.dtype, device=wav.device).expand(wav.shape[0], -1)


class _CountingIntent(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.last_input_length = None

    def forward(self, wav):
        self.calls += 1
        self.last_input_length = int(wav.shape[-1])
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
