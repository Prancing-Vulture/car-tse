"""Target-presence binary head on extracted waveform."""

from __future__ import annotations

import torch
import torch.nn as nn

from car_tse.models.kws_model import LogMelFront


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


class FramePresenceHead(nn.Module):
    """Frame-level target-activity head for streaming presence decisions.

    The head operates on log-mel features and emits one target-speech logit per
    10 ms frame.  It is intentionally small so it can run beside the existing
    Stage 1 model.  The runtime aggregates the highest-scoring frames into a
    chunk-level gate; this preserves the current pipeline contract while
    avoiding a single decision over an entire three-second waveform.
    """

    is_framewise = True

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 40,
        hidden: int = 64,
        n_fft: int = 512,
        hop: int = 160,
    ):
        super().__init__()
        self.mel = LogMelFront(sample_rate, n_mels, n_fft=n_fft, hop=hop)
        groups = 8 if hidden % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv1d(n_mels, hidden, kernel_size=5, padding=2),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, dilation=2, padding=4),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    def forward(self, s1: torch.Tensor) -> torch.Tensor:
        if s1.dim() == 1:
            s1 = s1.unsqueeze(0)
        s1 = torch.nan_to_num(s1.float(), nan=0.0, posinf=0.0, neginf=0.0)
        # Keep the STFT/log-mel path in float32.  Half-precision complex STFT
        # and very quiet Stage 1 outputs can otherwise produce NaNs before the
        # small head has a chance to learn a useful scale.
        with torch.autocast(device_type=s1.device.type, enabled=False):
            features = self.mel(s1)
            # Per-clip feature normalization reduces the large Stage 1 gain
            # and matches the scale variation seen in cabin recordings.
            features = (features - features.mean(dim=(1, 2), keepdim=True)) / features.std(
                dim=(1, 2), keepdim=True
            ).clamp_min(1.0e-4)
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            return self.net(features).squeeze(1)

    @staticmethod
    def aggregate_probability(frame_probability: torch.Tensor, top_fraction: float = 0.20) -> torch.Tensor:
        """Aggregate frame probabilities into a target-present score."""

        if frame_probability.dim() == 1:
            frame_probability = frame_probability.unsqueeze(0)
        fraction = min(max(float(top_fraction), 1.0e-3), 1.0)
        top_k = max(1, int(round(frame_probability.shape[-1] * fraction)))
        return frame_probability.topk(top_k, dim=-1).values.mean(dim=-1)
