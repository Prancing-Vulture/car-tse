"""Adapter for the existing 16 kHz SpEx+ transfer checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from car_tse.vendor.spex_plus_16k import SpExPlus16k, load_spex_plus_16k


class SpExPlus16kTransferAdapter(nn.Module):
    """Expose the transfer model through the project's Stage 1 interface."""

    def __init__(self, checkpoint: str | Path, *, causal: bool = False):
        super().__init__()
        self.model: SpExPlus16k = load_spex_plus_16k(
            checkpoint,
            model_config={"causal": bool(causal)},
        )

    def forward(
        self,
        mixture: torch.Tensor,
        enrollment: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        reference_lengths = torch.full(
            (enrollment.shape[0],), enrollment.shape[-1], dtype=torch.long, device=enrollment.device
        )
        output = self.model(mixture, enrollment, reference_lengths)
        return output["s1"], output["s2"], output["s3"], output["speaker_pred"]

    @torch.no_grad()
    def speaker_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return the frozen SpEx+ speaker embedding for a waveform batch."""

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        waveform = torch.nan_to_num(waveform.float(), nan=0.0, posinf=0.0, neginf=0.0)
        encoded = self.model.speech_encoder(waveform)
        lengths = torch.full(
            (waveform.shape[0],), waveform.shape[-1], dtype=torch.long, device=waveform.device
        )
        _, embedding = self.model.speaker_encoder(encoded, lengths)
        return F.normalize(embedding.float(), dim=-1)

    def encoder_params(self) -> list[nn.Module]:
        return [self.model.speech_encoder, self.model.speaker_encoder]
