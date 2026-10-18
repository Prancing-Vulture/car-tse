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
        # hidden <= 0 is the compact linear probe used by small-vocabulary
        # experiments. Keeping the default MLP unchanged preserves old
        # 31-class checkpoints and inference behavior.
        self.net = (
            nn.Linear(input_dim, num_classes)
            if int(hidden) <= 0
            else nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )
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
        head_hidden: int = 512,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        encoder_checkpoint = Path(encoder_checkpoint)
        encoder_payload = torch.load(encoder_checkpoint, map_location="cpu", weights_only=False)
        self.encoder.load_state_dict(encoder_payload["model_state_dict"], strict=True)
        self.head = Wav2VecIntentHead(
            input_dim=int(self.encoder.config.hidden_size),
            hidden=head_hidden,
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


class Wav2VecIntentEnsemble(nn.Module):
    """Weighted logit ensemble for compatible wav2vec-style intent models.

    This is intentionally opt-in.  Each member keeps its own encoder and head,
    while the returned logits are a weighted average before the pipeline's
    confidence and action gates.  Averaging logits rather than class IDs lets
    the command buffer and confidence threshold continue to operate exactly as
    they do for a single intent model.
    """

    def __init__(
        self,
        members: list[Wav2VecIntentClassifier],
        weights: list[float] | None = None,
        selection: str = "weighted_mean",
        gate_threshold: float = 0.90,
    ):
        super().__init__()
        if not members:
            raise ValueError("Wav2VecIntentEnsemble requires at least one member")
        num_classes = None
        for member in members:
            output = int(member.head.net[-1].out_features)
            if num_classes is None:
                num_classes = output
            elif output != num_classes:
                raise ValueError("all intent ensemble members must have the same class count")
        self.members = nn.ModuleList(members)
        self.selection = str(selection).lower()
        if self.selection not in {"weighted_mean", "confidence_fallback"}:
            raise ValueError("intent ensemble selection must be weighted_mean or confidence_fallback")
        self.gate_threshold = min(max(float(gate_threshold), 0.0), 1.0)
        raw_weights = torch.ones(len(members), dtype=torch.float32) if weights is None else torch.tensor(weights, dtype=torch.float32)
        if raw_weights.numel() != len(members) or not torch.isfinite(raw_weights).all() or (raw_weights < 0).any():
            raise ValueError("intent ensemble weights must be finite and non-negative")
        if float(raw_weights.sum()) <= 0.0:
            raise ValueError("intent ensemble weights must have a positive sum")
        self.register_buffer("weights", raw_weights / raw_weights.sum(), persistent=False)

    def embedding(self, wav: torch.Tensor) -> torch.Tensor:
        """Use the first member's representation for optional speaker gating."""

        return self.members[0].embedding(wav)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if self.selection == "confidence_fallback":
            if len(self.members) != 2:
                raise ValueError("confidence_fallback requires exactly two intent members")
            primary_logits = self.members[0](wav)
            primary_confidence = torch.softmax(primary_logits, dim=-1).max(dim=-1, keepdim=True).values
            fallback_logits = self.members[1](wav)
            use_fallback = primary_confidence < self.gate_threshold
            return torch.where(use_fallback, fallback_logits, primary_logits)
        logits = torch.stack([member(wav) for member in self.members], dim=0)
        weights = self.weights.to(device=logits.device, dtype=logits.dtype)
        return (logits * weights.view(-1, 1, 1)).sum(dim=0)
