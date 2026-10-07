"""Pretrained wav2vec 2.0 intent classifier used after Stage 1 extraction."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from car_tse.train.utils import load_checkpoint


class Wav2VecIntentHead(nn.Module):
    def __init__(self, input_dim: int = 768, hidden: int = 512, num_classes: int = 31, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class Wav2VecIntentClassifier(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        encoder_checkpoint: Path | str,
        head_checkpoint: Path | str,
        num_classes: int = 31,
        input_sample_rate: int = 16000,
        input_seconds: float = 2.0,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        encoder_checkpoint = Path(encoder_checkpoint)
        encoder_payload = torch.load(encoder_checkpoint, map_location="cpu", weights_only=False)
        self.encoder.load_state_dict(encoder_payload["model_state_dict"], strict=True)
        self.head = Wav2VecIntentHead(
            input_dim=int(self.encoder.config.hidden_size),
            hidden=512,
            num_classes=num_classes,
            dropout=0.2,
        )
        load_checkpoint(Path(head_checkpoint), self.head)
        self.input_samples = int(input_sample_rate * input_seconds)

    def _features(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.shape[-1] > self.input_samples:
            wav = wav[..., : self.input_samples]
        elif wav.shape[-1] < self.input_samples:
            wav = torch.nn.functional.pad(wav, (0, self.input_samples - wav.shape[-1]))
        wav = wav - wav.mean(dim=-1, keepdim=True)
        wav = wav / wav.std(dim=-1, keepdim=True).clamp_min(1.0e-5)
        states = self.encoder(wav).last_hidden_state
        return states.mean(dim=1)

    def embedding(self, wav: torch.Tensor) -> torch.Tensor:
        """Return a normalized frozen-encoder embedding for optional speaker gating."""

        return F.normalize(self._features(wav), dim=-1)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.head(self._features(wav))
