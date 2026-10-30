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
from car_tse.vendor.spex_plus_16k import CausalDepthwiseConv1d, SpExPlus16k
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


def test_spex_plus_16k_causal_shape_and_prefix_invariance():
    """The opt-in causal research path preserves shapes and has no future leak."""

    torch.manual_seed(7)
    model_kwargs = {
        "L1": 20,
        "L2": 80,
        "L3": 160,
        "N": 8,
        "ResNetBlock_cnt": 1,
        "TCN_cnt": 2,
        "speakers_cnt": 3,
    }
    noncausal = SpExPlus16k(**model_kwargs, causal=False).eval()
    causal = SpExPlus16k(**model_kwargs, causal=True).eval()
    causal.load_state_dict(noncausal.state_dict(), strict=True)
    mixture = torch.randn(2, 3200)
    reference = torch.randn(2, 1600)
    lengths = torch.full((2,), 1600, dtype=torch.long)
    with torch.no_grad():
        output = causal(mixture, reference, lengths)
    assert all(output[key].shape == (2, 3200) for key in ("s1", "s2", "s3"))
    assert all(torch.isfinite(output[key]).all() for key in ("s1", "s2", "s3"))

    depthwise = CausalDepthwiseConv1d(
        4, 4, 3, dilation=2, groups=4, left_padding=4
    ).eval()
    prefix = torch.randn(1, 4, 40)
    altered = prefix.clone()
    altered[..., 20:] += 10.0
    with torch.no_grad():
        original_output = depthwise(prefix)
        altered_output = depthwise(altered)
    assert torch.allclose(
        original_output[..., :20], altered_output[..., :20], atol=1e-6, rtol=1e-6
    )


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
