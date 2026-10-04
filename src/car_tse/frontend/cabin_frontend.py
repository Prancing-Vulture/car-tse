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

    def __init__(
        self,
        num_mics: int = 4,
        ec_hidden: int = 128,
        spatial_channels: int = 64,
        input_rms_target: float | None = None,
    ):
        super().__init__()
        self.num_mics = num_mics
        self.input_rms_target = (
            None if input_rms_target is None or float(input_rms_target) <= 0 else float(input_rms_target)
        )
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
        input_scale = None
        if self.input_rms_target is not None:
            input_rms = multichannel.float().square().mean(dim=(1, 2), keepdim=True).sqrt()
            input_scale = (self.input_rms_target / input_rms.clamp_min(1.0e-8)).clamp(0.05, 20.0)
            multichannel = multichannel * input_scale.to(multichannel.dtype)
            if playback_ref is not None:
                playback_ref = playback_ref * input_scale[:, 0, 0].to(playback_ref.dtype).unsqueeze(1)
        if playback_ref is None:
            playback_ref = torch.zeros(b, t, device=multichannel.device, dtype=multichannel.dtype)
        ref = playback_ref.unsqueeze(1)
        ec_in = torch.cat([multichannel, ref], dim=1)
        residual = self.ec(ec_in)
        enhanced = multichannel - residual
        mono = multichannel.mean(dim=1) + self.spatial(enhanced).squeeze(1)
        return mono
