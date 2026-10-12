"""Compact speaker-verification candidates for the in-car pipeline."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from car_tse.models.kws_model import LogMelFront


class CompactSpeakerEncoder(nn.Module):
    """Small log-mel speaker encoder intended for later distillation."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 40,
        embedding_dim: int = 128,
        n_fft: int = 400,
        hop: int = 160,
        channels: tuple[int, ...] = (32, 64, 128),
    ) -> None:
        super().__init__()
        self.front = LogMelFront(sample_rate, n_mels, n_fft=n_fft, hop=hop)
        layers: list[nn.Module] = []
        input_channels = 1
        for output_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(output_channels),
                    nn.GELU(),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            input_channels = output_channels
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.cnn = nn.Sequential(*layers)
        self.projection = nn.Linear(input_channels, int(embedding_dim))
        self.embedding_dim = int(embedding_dim)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        waveform = torch.nan_to_num(waveform.float(), nan=0.0, posinf=0.0, neginf=0.0)
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            features = self.front(waveform)
            features = (features - features.mean(dim=(1, 2), keepdim=True)) / features.std(
                dim=(1, 2), keepdim=True
            ).clamp_min(1.0e-4)
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            hidden = self.cnn(features.unsqueeze(1)).flatten(1)
            embedding = self.projection(hidden)
        return F.normalize(embedding.float(), dim=-1)


class CompactSpeakerVerifier(nn.Module):
    """Shared compact encoder plus an enrollment/extracted relation head."""

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden: int = 128,
        **encoder_kwargs: object,
    ) -> None:
        super().__init__()
        self.encoder = CompactSpeakerEncoder(embedding_dim=embedding_dim, **encoder_kwargs)
        relation_dim = int(embedding_dim) * 4 + 1
        self.relation = nn.Sequential(
            nn.LayerNorm(relation_dim),
            nn.Linear(relation_dim, int(hidden)),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(int(hidden), 1),
        )

    @staticmethod
    def relation_features(enrollment: torch.Tensor, extracted: torch.Tensor) -> torch.Tensor:
        enrollment = F.normalize(enrollment.float(), dim=-1)
        extracted = F.normalize(extracted.float(), dim=-1)
        return torch.cat(
            [
                enrollment,
                extracted,
                torch.abs(enrollment - extracted),
                enrollment * extracted,
                F.cosine_similarity(enrollment, extracted, dim=-1).unsqueeze(-1),
            ],
            dim=-1,
        )

    def forward(
        self,
        enrollment_waveform: torch.Tensor,
        extracted_waveform: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enrollment = self.encoder(enrollment_waveform)
        extracted = self.encoder(extracted_waveform)
        logits = self.relation(self.relation_features(enrollment, extracted)).squeeze(-1)
        return logits, enrollment, extracted
