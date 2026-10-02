"""SpEx+ adapter smoke test."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "speaker_extraction_SpEx"))

from car_tse.vendor.spex_plus_adapter import SpExPlusAdapter, spex_nnet_conf
from nnet.spex_plus import SpEx_Plus


def test_spex_forward_random():
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
    nnet = SpEx_Plus(**spex_nnet_conf(cfg, 8000))
    adapter = SpExPlusAdapter(nnet)
    mix = torch.randn(2, 32000)
    enroll = torch.randn(2, 16000)
    s1, s2, s3, logits = adapter(mix, enroll)
    assert s1.shape == mix.shape
    assert logits.shape[0] == 2
    assert torch.isfinite(s1).all()


@pytest.mark.skipif(
    not (ROOT / "checkpoints" / "spex_pretrained" / "best.pt.tar").exists(),
    reason="no pretrained checkpoint",
)
def test_load_pretrained():
    from car_tse.vendor.spex_plus_adapter import load_pretrained_spex

    cfg = {
        "L1_ms": 2.5,
        "L2_ms": 10.0,
        "L3_ms": 20.0,
        "N": 256,
        "B": 8,
        "O": 256,
        "P": 512,
        "Q": 3,
        "num_spks": 921,
        "spk_embed_dim": 256,
    }
    load_pretrained_spex(ROOT / "checkpoints" / "spex_pretrained", cfg)
