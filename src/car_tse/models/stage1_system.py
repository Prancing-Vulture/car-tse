"""Stage 1: optional cabin frontend + SpEx+."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from car_tse.frontend.cabin_frontend import CabinFrontend
from car_tse.vendor.spex_plus_adapter import SpExPlusAdapter, load_pretrained_spex
from car_tse.vendor.spex_plus_transfer_adapter import SpExPlus16kTransferAdapter


class Stage1System(nn.Module):
    def __init__(
        self,
        spex_cfg: dict[str, Any],
        sample_rate: int = 8000,
        use_cabin_frontend: bool = False,
        num_mics: int = 4,
        frontend_cfg: dict[str, Any] | None = None,
        pretrained_dir: str | None = None,
        backend: str = "vendor_8k",
    ):
        super().__init__()
        self.backend = backend
        self.use_cabin_frontend = use_cabin_frontend
        frontend_cfg = frontend_cfg or {}
        if use_cabin_frontend:
            self.frontend = CabinFrontend(
                num_mics=num_mics,
                ec_hidden=frontend_cfg.get("ec_hidden", 128),
                spatial_channels=frontend_cfg.get("spatial_channels", 64),
                input_rms_target=frontend_cfg.get("input_rms_target"),
            )
        else:
            self.frontend = None
        if backend == "transfer_spexplus_16k":
            if not pretrained_dir:
                raise ValueError("transfer_spexplus_16k requires a checkpoint path")
            self.spex = SpExPlus16kTransferAdapter(pretrained_dir)
        elif backend == "vendor_8k":
            nnet = load_pretrained_spex(
                pretrained_dir or ".",
                spex_cfg,
                sample_rate=sample_rate,
                strict=False,
            )
            self.spex = SpExPlusAdapter(nnet)
        else:
            raise ValueError(f"Unsupported Stage 1 backend: {backend}")

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

    @torch.no_grad()
    def speaker_embedding(self, waveform: torch.Tensor) -> torch.Tensor | None:
        """Return the normalized transfer-model speaker embedding when available."""

        embedder = getattr(self.spex, "speaker_embedding", None)
        if not callable(embedder):
            return None
        return embedder(waveform)

    @torch.no_grad()
    def speaker_similarity(
        self,
        enrollment: torch.Tensor,
        extracted: torch.Tensor,
    ) -> torch.Tensor | None:
        """Cosine similarity between enrollment and extracted speaker embeddings."""

        enrollment_embedding = self.speaker_embedding(enrollment)
        extracted_embedding = self.speaker_embedding(extracted)
        if enrollment_embedding is None or extracted_embedding is None:
            return None
        return F.cosine_similarity(enrollment_embedding, extracted_embedding, dim=-1)
