"""16 kHz SpEx+ implementation used for transfer learning.

This is the MIT-licensed SpEx+ implementation already used by the local
Libri2Mix experiments in ``E:/prawin/snn``.  It is kept separate from the
vendored 8 kHz model because their parameter names and tensor shapes are not
checkpoint-compatible.  The implementation intentionally preserves the
original module names so ``runs/spexplus_no_jepa_30/best.pt`` can be loaded
without key rewriting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class GlobalLayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.beta = nn.Parameter(torch.zeros(channels, 1))
        self.gamma = nn.Parameter(torch.ones(channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        stats = x.float()
        mean = stats.mean(dim=(1, 2), keepdim=True)
        var = (stats - mean).pow(2).mean(dim=(1, 2), keepdim=True)
        normalized = self.gamma.float() * (stats - mean) / torch.sqrt(var + self.eps)
        return (normalized + self.beta.float()).to(input_dtype)


def _match_length(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.shape[-1] < length:
        return F.pad(x, (0, length - x.shape[-1]))
    return x[..., :length]


class SpeechEncoder(nn.Module):
    def __init__(self, L1: int = 20, L2: int = 80, L3: int = 160, channels: int = 256):
        super().__init__()
        self.L1, self.L2, self.L3 = int(L1), int(L2), int(L3)
        self.channels = int(channels)
        self.stride = self.L1 // 2
        self.short = nn.Sequential(nn.Conv1d(1, channels, L1, self.stride), nn.ReLU())
        self.middle = nn.Sequential(nn.Conv1d(1, channels, L2, self.stride), nn.ReLU())
        self.long = nn.Sequential(nn.Conv1d(1, channels, L3, self.stride), nn.ReLU())

    def forward(self, x: torch.Tensor, return_branches: bool = False):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("SpeechEncoder expects [batch, time] or [batch, 1, time]")
        x1 = self.short(x)
        final_length = x1.shape[-1]
        required_l2 = (final_length - 1) * self.stride + self.L2
        required_l3 = (final_length - 1) * self.stride + self.L3
        x2 = self.middle(_match_length(x, required_l2))
        x3 = self.long(_match_length(x, required_l3))
        if return_branches:
            return torch.cat([x1, x2, x3], dim=1), [x1, x2, x3]
        return torch.cat([x1, x2, x3], dim=1)


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ResNetBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.part1 = nn.Sequential(
            nn.Conv1d(channels, 2 * channels, 1, bias=False),
            nn.BatchNorm1d(2 * channels),
            nn.PReLU(),
            nn.Conv1d(2 * channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.part2 = nn.Sequential(nn.PReLU(), nn.MaxPool1d(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.part2(self.part1(x) + x)


class SpeakerEncoder(nn.Module):
    def __init__(
        self,
        channels: int = 256,
        resnet_blocks: int = 3,
        L1: int = 20,
        stride: int = 10,
        speakers_cnt: int = 251,
    ):
        super().__init__()
        self.L1 = int(L1)
        self.stride = int(stride)
        self.resnet_blocks_count = int(resnet_blocks)
        self.norm = ChannelLayerNorm(3 * channels)
        self.conv1 = nn.Conv1d(3 * channels, channels, 1)
        self.resnet_blocks = nn.Sequential(*(ResNetBlock(channels) for _ in range(resnet_blocks)))
        self.conv2 = nn.Conv1d(channels, channels, 1)
        self.classification = nn.Linear(channels, speakers_cnt)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(self.norm(x))
        x = self.resnet_blocks(x)
        x = self.conv2(x)
        frame_lengths = torch.div(
            lengths.to(x.device) - self.L1, self.stride, rounding_mode="floor"
        ) + 1
        for _ in range(self.resnet_blocks_count):
            frame_lengths = torch.div(frame_lengths, 3, rounding_mode="floor")
        frame_lengths = frame_lengths.clamp_min(1).to(x.dtype)
        valid_frames = (
            torch.arange(x.shape[-1], device=x.device).unsqueeze(0)
            < frame_lengths.long().unsqueeze(1)
        )
        x = x * valid_frames.unsqueeze(1).to(x.dtype)
        embedding = x.sum(dim=-1) / frame_lengths.unsqueeze(1)
        return self.classification(embedding), embedding


class TCN(nn.Module):
    def __init__(self, channels: int, kernel_size: int, speaker_channels: int, dilation: int):
        super().__init__()
        hidden = 2 * channels
        padding = dilation * (kernel_size - 1) // 2
        self.seq = nn.Sequential(
            nn.Conv1d(channels + speaker_channels, hidden, 1),
            nn.PReLU(),
            GlobalLayerNorm(hidden),
            nn.Conv1d(hidden, hidden, kernel_size, padding=padding, dilation=dilation, groups=hidden),
            nn.PReLU(),
            GlobalLayerNorm(hidden),
            nn.Conv1d(hidden, channels, 1),
        )

    def forward(self, x: torch.Tensor, speaker_embedding: torch.Tensor | None) -> torch.Tensor:
        residual = x
        if speaker_embedding is not None:
            repeated = speaker_embedding.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            x = torch.cat([x, repeated], dim=1)
        return residual + self.seq(x)


class StackedTCNs(nn.Module):
    def __init__(self, channels: int, speaker_channels: int, count: int):
        super().__init__()
        self.tcns = nn.ModuleList(
            [TCN(channels, 3, speaker_channels, 1)]
            + [TCN(channels, 3, 0, 2**i) for i in range(1, count)]
        )

    def forward(self, x: torch.Tensor, speaker_embedding: torch.Tensor) -> torch.Tensor:
        for index, tcn in enumerate(self.tcns):
            x = tcn(x, speaker_embedding if index == 0 else None)
        return x


class SpeakerExtractor(nn.Module):
    def __init__(self, channels: int = 256, speaker_channels: int = 256, tcn_count: int = 8):
        super().__init__()
        self.norm = ChannelLayerNorm(3 * channels)
        self.conv1 = nn.Conv1d(3 * channels, channels, 1)
        self.stacks = nn.ModuleList(
            [StackedTCNs(channels, speaker_channels, tcn_count) for _ in range(4)]
        )
        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(channels, channels, 1), nn.ReLU()) for _ in range(3)]
        )

    def forward(self, x: torch.Tensor, speaker_embedding: torch.Tensor) -> list[torch.Tensor]:
        x = self.conv1(self.norm(x))
        for stack in self.stacks:
            x = stack(x, speaker_embedding)
        return [conv(x) for conv in self.convs]


class SpeechDecoder(nn.Module):
    def __init__(
        self,
        L1: int = 20,
        L2: int = 80,
        L3: int = 160,
        channels: int = 256,
        stride: int = 10,
    ):
        super().__init__()
        self.short = nn.ConvTranspose1d(channels, 1, L1, stride)
        self.middle = nn.ConvTranspose1d(channels, 1, L2, stride)
        self.long = nn.ConvTranspose1d(channels, 1, L3, stride)

    def forward(self, x_short: torch.Tensor, x_middle: torch.Tensor, x_long: torch.Tensor):
        return (
            self.short(x_short).squeeze(1),
            self.middle(x_middle).squeeze(1),
            self.long(x_long).squeeze(1),
        )


class SpExPlus16k(nn.Module):
    """Checkpoint-compatible no-JEPA SpEx+ model."""

    def __init__(
        self,
        L1: int = 20,
        L2: int = 80,
        L3: int = 160,
        N: int = 256,
        ResNetBlock_cnt: int = 3,
        TCN_cnt: int = 8,
        speakers_cnt: int = 251,
    ):
        super().__init__()
        self.speech_encoder = SpeechEncoder(L1, L2, L3, N)
        self.speaker_encoder = SpeakerEncoder(N, ResNetBlock_cnt, L1, self.speech_encoder.stride, speakers_cnt)
        self.speaker_extractor = SpeakerExtractor(N, N, TCN_cnt)
        self.speech_decoder = SpeechDecoder(L1, L2, L3, N, self.speech_encoder.stride)

    @staticmethod
    def _fix_length(wave: torch.Tensor, length: int) -> torch.Tensor:
        if wave.shape[-1] < length:
            return F.pad(wave, (0, length - wave.shape[-1]))
        return wave[..., :length]

    def forward(
        self,
        mixture: torch.Tensor,
        reference: torch.Tensor,
        reference_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoded_mix, branches = self.speech_encoder(mixture, return_branches=True)
        encoded_ref = self.speech_encoder(reference)
        speaker_pred, speaker_embedding = self.speaker_encoder(encoded_ref, reference_lengths)
        masks = self.speaker_extractor(encoded_mix, speaker_embedding)
        decoded = self.speech_decoder(
            branches[0] * masks[0], branches[1] * masks[1], branches[2] * masks[2]
        )
        length = mixture.shape[-1]
        return {
            "s1": self._fix_length(decoded[0], length),
            "s2": self._fix_length(decoded[1], length),
            "s3": self._fix_length(decoded[2], length),
            "speaker_pred": speaker_pred,
            "speaker_embedding": speaker_embedding,
        }


def load_spex_plus_16k(
    checkpoint: str | Path,
    model_config: dict[str, Any] | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> SpExPlus16k:
    """Load the existing 16 kHz Libri2Mix model into the local implementation."""

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Transfer checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    saved_cfg = payload.get("model_config", {}) if isinstance(payload, dict) else {}
    cfg = {**saved_cfg, **(model_config or {})}
    model = SpExPlus16k(
        L1=int(cfg.get("L1", 20)),
        L2=int(cfg.get("L2", 80)),
        L3=int(cfg.get("L3", 160)),
        N=int(cfg.get("N", 256)),
        ResNetBlock_cnt=int(cfg.get("ResNetBlock_cnt", 3)),
        TCN_cnt=int(cfg.get("TCN_cnt", 8)),
        speakers_cnt=int(cfg.get("speakers_cnt", 251)),
    )
    state = payload.get("model", payload.get("model_state_dict", payload))
    # The local no-JEPA run was produced by a model class that still carried
    # its unused EMA target encoder and JEPA predictor.  Those auxiliary keys
    # are intentionally absent from this inference/training adapter; all
    # extraction-path keys must still match exactly.
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(
            "Transfer checkpoint is missing extraction weights: "
            + ", ".join(incompatible.missing_keys)
        )
    return model
