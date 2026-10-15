"""Small learned fusion of Stage 2 presence and Stage 3 wake scores."""

from __future__ import annotations

import torch
import torch.nn as nn


class Stage23FusionHead(nn.Module):
    """Combine per-window presence, wake, and optional speaker scores.

    Inputs are probabilities in [0, 1] except speaker similarity, which is
    cosine similarity in [-1, 1]. The final feature is a validity bit so a
    missing speaker score is not confused with a real zero similarity.
    """

    input_features = 4

    def __init__(self, hidden: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.input_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        presence_probability: torch.Tensor,
        wake_probability: torch.Tensor,
        speaker_similarity: torch.Tensor | None = None,
        speaker_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        p = presence_probability.float().clamp(1.0e-5, 1.0 - 1.0e-5)
        w = wake_probability.float().clamp(1.0e-5, 1.0 - 1.0e-5)
        if speaker_similarity is None:
            s = torch.zeros_like(p)
            valid = torch.zeros_like(p)
        else:
            s = speaker_similarity.float().clamp(-1.0, 1.0)
            valid = (
                torch.ones_like(p)
                if speaker_valid is None
                else speaker_valid.float().clamp(0.0, 1.0)
            )
        features = torch.stack((p, w, s, valid), dim=-1)
        return self.net(features).squeeze(-1)
