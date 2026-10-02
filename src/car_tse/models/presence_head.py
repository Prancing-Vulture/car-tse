"""Target-presence binary head on extracted waveform."""

from __future__ import annotations

import torch
import torch.nn as nn


class PresenceHead(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=15, stride=4, padding=7),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=15, stride=4, padding=7),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, s1: torch.Tensor) -> torch.Tensor:
        x = s1.unsqueeze(1)
        h = self.net(x).squeeze(-1)
        return self.fc(h).squeeze(-1)
