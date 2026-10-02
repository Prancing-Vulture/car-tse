"""Stage 1: optional cabin frontend + SpEx+."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from car_tse.frontend.cabin_frontend import CabinFrontend
from car_tse.vendor.spex_plus_adapter import SpExPlusAdapter, load_pretrained_spex


class Stage1System(nn.Module):
    def __init__(
        self,
        spex_cfg: dict[str, Any],
        sample_rate: int = 8000,
        use_cabin_frontend: bool = False,
        num_mics: int = 4,
        frontend_cfg: dict[str, Any] | None = None,
        pretrained_dir: str | None = None,
    ):
        super().__init__()
        self.use_cabin_frontend = use_cabin_frontend
        frontend_cfg = frontend_cfg or {}
        if use_cabin_frontend:
            self.frontend = CabinFrontend(
                num_mics=num_mics,
                ec_hidden=frontend_cfg.get("ec_hidden", 128),
                spatial_channels=frontend_cfg.get("spatial_channels", 64),
            )
        else:
            self.frontend = None
        nnet = load_pretrained_spex(
            pretrained_dir or ".",
            spex_cfg,
            sample_rate=sample_rate,
            strict=False,
        )
        self.spex = SpExPlusAdapter(nnet)

    def forward(
        self,
        mixture: torch.Tensor,
        enrollment: torch.Tensor,
        multichannel: torch.Tensor | None = None,
        playback_ref: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.frontend is not None:
            if multichannel is None:
                raise ValueError("multichannel required when cabin frontend is enabled")
            mixture = self.frontend(multichannel, playback_ref)
        return self.spex(mixture, enrollment)

    def set_encoder_frozen(self, frozen: bool) -> None:
        for p in self.spex.encoder_params():
            for param in p.parameters():
                param.requires_grad = not frozen

    def set_spex_frozen(self, frozen: bool) -> None:
        for p in self.spex.parameters():
            p.requires_grad = not frozen
