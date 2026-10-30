from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

from train_stage3_kws import evaluate  # noqa: E402


class ProbabilityModel(nn.Module):
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        probability = wav[:, 0].clamp(1e-5, 1.0 - 1e-5)
        wake_logit = torch.logit(probability)
        return torch.stack((torch.zeros_like(wake_logit), wake_logit), dim=-1)


def test_validation_f1_uses_the_configured_pipeline_threshold():
    batch = {
        "wav": torch.tensor([[0.2], [0.4], [0.7], [0.9]]),
        "label": torch.tensor([0, 0, 1, 1]),
    }

    at_half = evaluate(ProbabilityModel(), [batch], torch.device("cpu"), 0.5)
    at_point_three = evaluate(ProbabilityModel(), [batch], torch.device("cpu"), 0.3)

    assert at_half["f1"] == 1.0
    assert at_point_three["f1"] == 0.8
    assert at_point_three["decision_threshold"] == 0.3
