"""Target-presence binary head on extracted waveform."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        exportable_frontend: bool = False,
    ):
        super().__init__()
        self.mel = LogMelFront(
            sample_rate,
            n_mels,
            n_fft=n_fft,
            hop=hop,
            exportable_frontend=exportable_frontend,
        )
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


class PersonalFramePresenceHead(nn.Module):
    """Speaker-conditioned framewise target-activity head.

    The existing frame head only sees the extracted waveform.  This variant
    also receives the enrolled SpEx+ speaker embedding and uses it to modulate
    the acoustic features before predicting target activity.  It is a small,
    causal-compatible research candidate inspired by personal-VAD systems;
    it is intentionally separate from :class:`FramePresenceHead` so the
    retained checkpoint and its runtime behavior remain unchanged.
    """

    is_framewise = True
    requires_enrollment = True

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 40,
        hidden: int = 64,
        n_fft: int = 512,
        hop: int = 160,
        speaker_embedding_dim: int = 256,
        exportable_frontend: bool = False,
    ):
        super().__init__()
        self.mel = LogMelFront(
            sample_rate,
            n_mels,
            n_fft=n_fft,
            hop=hop,
            exportable_frontend=exportable_frontend,
        )
        groups = 8 if hidden % 8 == 0 else 1
        self.acoustic = nn.Sequential(
            nn.Conv1d(n_mels, hidden, kernel_size=5, padding=2),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, dilation=2, padding=4),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
        )
        self.speaker_modulation = nn.Sequential(
            nn.Linear(int(speaker_embedding_dim), hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * hidden),
        )
        self.output = nn.Conv1d(hidden, 1, kernel_size=1)

    def forward(self, s1: torch.Tensor, enrollment_embedding: torch.Tensor) -> torch.Tensor:
        if s1.dim() == 1:
            s1 = s1.unsqueeze(0)
        if enrollment_embedding.dim() == 1:
            enrollment_embedding = enrollment_embedding.unsqueeze(0)
        s1 = torch.nan_to_num(s1.float(), nan=0.0, posinf=0.0, neginf=0.0)
        enrollment_embedding = F.normalize(
            torch.nan_to_num(enrollment_embedding.float(), nan=0.0, posinf=0.0, neginf=0.0), dim=-1
        )
        with torch.autocast(device_type=s1.device.type, enabled=False):
            features = self.mel(s1)
            features = (features - features.mean(dim=(1, 2), keepdim=True)) / features.std(
                dim=(1, 2), keepdim=True
            ).clamp_min(1.0e-4)
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            hidden = self.acoustic(features)
            modulation = self.speaker_modulation(enrollment_embedding)
            scale, bias = modulation.chunk(2, dim=-1)
            # Bound the multiplicative condition at initialization so this
            # candidate cannot arbitrarily rescale quiet Stage 1 outputs.
            hidden = hidden * (1.0 + 0.1 * torch.tanh(scale).unsqueeze(-1)) + bias.unsqueeze(-1)
            return self.output(hidden).squeeze(1)

    @staticmethod
    def aggregate_probability(frame_probability: torch.Tensor, top_fraction: float = 0.20) -> torch.Tensor:
        return FramePresenceHead.aggregate_probability(frame_probability, top_fraction=top_fraction)


class PersonalActivityHead(nn.Module):
    """Three-way speaker-conditioned target activity head.

    The binary presence head treats every target-absent recording as the same
    negative class.  That is unsafe for an in-car assistant: passenger speech
    is acoustically different from silence, but both were previously trained
    as ``0``.  This head predicts per-frame ``non-speech``, ``target speech``,
    and ``non-target speech`` logits.  Only the target-speech probability is
    exposed to the existing runtime gate, so the state machine remains
    unchanged while the model can explicitly learn passenger rejection.

    The head is an opt-in research candidate.  It should be trained and
    calibrated on speaker-consistent driver/passenger sessions before being
    selected as the runtime default.
    """

    is_framewise = True
    is_multiclass = True
    requires_enrollment = True
    target_class_index = 1
    num_classes = 3

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 40,
        hidden: int = 64,
        n_fft: int = 512,
        hop: int = 160,
        speaker_embedding_dim: int = 256,
        exportable_frontend: bool = False,
    ):
        super().__init__()
        self.mel = LogMelFront(
            sample_rate,
            n_mels,
            n_fft=n_fft,
            hop=hop,
            exportable_frontend=exportable_frontend,
        )
        groups = 8 if hidden % 8 == 0 else 1
        self.acoustic = nn.Sequential(
            nn.Conv1d(n_mels, hidden, kernel_size=5, padding=2),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, dilation=2, padding=4),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
        )
        self.speaker_modulation = nn.Sequential(
            nn.Linear(int(speaker_embedding_dim), hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * hidden),
        )
        self.output = nn.Conv1d(hidden, self.num_classes, kernel_size=1)

    def forward(self, s1: torch.Tensor, enrollment_embedding: torch.Tensor) -> torch.Tensor:
        if s1.dim() == 1:
            s1 = s1.unsqueeze(0)
        if enrollment_embedding.dim() == 1:
            enrollment_embedding = enrollment_embedding.unsqueeze(0)
        s1 = torch.nan_to_num(s1.float(), nan=0.0, posinf=0.0, neginf=0.0)
        enrollment_embedding = F.normalize(
            torch.nan_to_num(enrollment_embedding.float(), nan=0.0, posinf=0.0, neginf=0.0), dim=-1
        )
        with torch.autocast(device_type=s1.device.type, enabled=False):
            features = self.mel(s1)
            features = (features - features.mean(dim=(1, 2), keepdim=True)) / features.std(
                dim=(1, 2), keepdim=True
            ).clamp_min(1.0e-4)
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            hidden = self.acoustic(features)
            modulation = self.speaker_modulation(enrollment_embedding)
            scale, bias = modulation.chunk(2, dim=-1)
            hidden = hidden * (1.0 + 0.1 * torch.tanh(scale).unsqueeze(-1)) + bias.unsqueeze(-1)
            return self.output(hidden)

    @classmethod
    def target_probability(cls, logits: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 3:
            raise ValueError(f"Expected [batch, classes, frames] logits, got {tuple(logits.shape)}")
        return torch.softmax(logits.float(), dim=1)[:, cls.target_class_index, :]

    @classmethod
    def activity_probabilities(cls, logits: torch.Tensor) -> torch.Tensor:
        """Return all non-speech/target/non-target frame probabilities."""

        if logits.dim() != 3:
            raise ValueError(f"Expected [batch, classes, frames] logits, got {tuple(logits.shape)}")
        if logits.shape[1] != cls.num_classes:
            raise ValueError(
                f"Expected {cls.num_classes} activity classes, got {logits.shape[1]}"
            )
        return torch.softmax(logits.float(), dim=1)

    @staticmethod
    def aggregate_probability(frame_probability: torch.Tensor, top_fraction: float = 0.20) -> torch.Tensor:
        return FramePresenceHead.aggregate_probability(frame_probability, top_fraction=top_fraction)

    @classmethod
    def aggregate_margin(
        cls,
        target_probability: torch.Tensor,
        non_target_probability: torch.Tensor,
        top_fraction: float = 0.20,
        mode: str = "independent_top",
    ) -> torch.Tensor:
        """Aggregate a target-vs-non-target activity margin.

        ``independent_top`` is the historical research behavior: aggregate
        the two class probabilities separately and subtract them.  The
        frame-aligned modes avoid comparing evidence from different frames:

        - ``target_top`` selects the strongest target-evidence frames and
          measures target-minus-passenger evidence on those same frames.
        - ``framewise_top`` selects the strongest framewise margins.
        - ``mean`` averages the framewise margin over the whole window.

        The opt-in modes are useful for personal-VAD style experiments, while
        retaining the historical default keeps old checkpoints and reports
        exactly reproducible.
        """

        if target_probability.dim() == 1:
            target_probability = target_probability.unsqueeze(0)
        if non_target_probability.dim() == 1:
            non_target_probability = non_target_probability.unsqueeze(0)
        if target_probability.shape != non_target_probability.shape:
            raise ValueError(
                "target and non-target probabilities must have the same shape, "
                f"got {tuple(target_probability.shape)} and {tuple(non_target_probability.shape)}"
            )
        mode = str(mode).lower()
        if mode == "independent_top":
            return cls.aggregate_probability(target_probability, top_fraction) - cls.aggregate_probability(
                non_target_probability, top_fraction
            )
        framewise_margin = target_probability - non_target_probability
        if mode == "mean":
            return framewise_margin.mean(dim=-1)
        if mode == "framewise_top":
            return cls.aggregate_probability(framewise_margin, top_fraction)
        if mode == "target_top":
            fraction = min(max(float(top_fraction), 1.0e-3), 1.0)
            top_k = max(1, int(round(target_probability.shape[-1] * fraction)))
            indices = target_probability.topk(top_k, dim=-1).indices
            return framewise_margin.gather(-1, indices).mean(dim=-1)
        raise ValueError(
            "margin mode must be one of: independent_top, target_top, framewise_top, mean"
        )
