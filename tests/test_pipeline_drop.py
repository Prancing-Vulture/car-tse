"""Absent target should not trigger wake."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "speaker_extraction_SpEx"))

from car_tse.models.intent_classifier import IntentClassifier
from car_tse.models.kws_model import KWSModel
from car_tse.models.presence_head import PresenceHead
from car_tse.models.stage1_system import Stage1System
from car_tse.pipeline.in_car_pipeline import InCarPipeline


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
