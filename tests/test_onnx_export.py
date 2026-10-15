from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from car_tse.models.kws_model import LogMelFront  # noqa: E402
from car_tse.models.intent_classifier import IntentClassifier  # noqa: E402
from scripts.export.export_onnx_components import _checkpoint_target  # noqa: E402


def test_exportable_logmel_matches_complex_stft_frontend() -> None:
    torch.manual_seed(7)
    waveform = torch.randn(2, 16_000)
    reference = LogMelFront(16_000, 40, n_fft=256, hop=80)(waveform)
    exportable = LogMelFront(
        16_000,
        40,
        n_fft=256,
        hop=80,
        exportable_frontend=True,
    )(waveform)

    assert reference.shape == exportable.shape == (2, 40, 201)
    assert torch.allclose(reference, exportable, atol=3.0e-4, rtol=3.0e-4)


def test_compact_intent_accepts_exportable_frontend() -> None:
    model = IntentClassifier(
        sample_rate=16_000,
        n_mels=80,
        hidden=16,
        num_layers=1,
        num_classes=3,
        normalize_input=True,
        exportable_frontend=True,
    )
    with torch.inference_mode():
        output = model(torch.randn(1, 32_000))
    assert output.shape == (1, 3)


def test_causal_intent_head_has_no_backward_direction() -> None:
    model = IntentClassifier(
        sample_rate=16_000,
        n_mels=80,
        hidden=16,
        num_layers=1,
        num_classes=3,
        bidirectional=False,
    )
    assert model.bidirectional is False
    assert model.lstm.bidirectional is False
    assert model.fc.in_features == 16
    with torch.inference_mode():
        output = model(torch.randn(1, 16_000))
    assert output.shape == (1, 3)


def test_wrapped_exports_load_checkpoint_into_underlying_model() -> None:
    class Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Linear(2, 2)

        def forward(self, value):
            return self.model(value)

    wrapper = Wrapper()
    assert _checkpoint_target(wrapper, "stage1") is wrapper.model
    assert _checkpoint_target(wrapper, "speaker_embedding") is wrapper.model
    assert _checkpoint_target(wrapper, "compact_verifier") is wrapper.model
    assert _checkpoint_target(wrapper, "kws") is wrapper
