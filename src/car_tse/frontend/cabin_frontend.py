"""Echo cancellation + spatial front-end for multichannel cabin audio."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CabinFrontend(nn.Module):
    """
    Multichannel input [B, C, T] + optional playback ref [B, T] -> mono mixture [B, T].
    Learned EC uses ref; spatial conv collapses channels.
    """

    def __init__(self, num_mics: int = 4, ec_hidden: int = 128, spatial_channels: int = 64):
        super().__init__()
        self.num_mics = num_mics
        self.ec = nn.Sequential(
            nn.Conv1d(num_mics + 1, ec_hidden, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(ec_hidden, num_mics, kernel_size=7, padding=3),
        )
        self.spatial = nn.Sequential(
            nn.Conv1d(num_mics, spatial_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(spatial_channels, 1, kernel_size=5, padding=2),
        )
        # Start from a safe, usable baseline.  A randomly initialized front
        # end would destroy the signal seen by a pretrained SpEx+ backbone on
        # the first update.  The residual branches therefore start at zero,
        # while the direct path is the same mean-pool baseline used in the
        # noisy transfer evaluation.
        nn.init.zeros_(self.ec[-1].weight)
        nn.init.zeros_(self.ec[-1].bias)
        nn.init.zeros_(self.spatial[-1].weight)
        nn.init.zeros_(self.spatial[-1].bias)

    def forward(
        self,
        multichannel: torch.Tensor,
        playback_ref: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if multichannel.dim() == 2:
            multichannel = multichannel.unsqueeze(1)
        b, c, t = multichannel.shape
        if playback_ref is None:
            playback_ref = torch.zeros(b, t, device=multichannel.device, dtype=multichannel.dtype)
        ref = playback_ref.unsqueeze(1)
        ec_in = torch.cat([multichannel, ref], dim=1)
        residual = self.ec(ec_in)
        enhanced = multichannel - residual
        mono = multichannel.mean(dim=1) + self.spatial(enhanced).squeeze(1)
        return mono
