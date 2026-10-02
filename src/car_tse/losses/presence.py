"""Stage 2 presence + absent output penalty."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def presence_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    s1: torch.Tensor,
    absent_penalty: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    bce = F.binary_cross_entropy_with_logits(logits, labels.float())
    absent_mask = labels < 0.5
    if absent_mask.any():
        energy = (s1[absent_mask] ** 2).mean(dim=-1).mean()
        loss = bce + absent_penalty * energy
        stats = {"bce": float(bce.detach()), "absent_energy": float(energy.detach())}
    else:
        loss = bce
        stats = {"bce": float(bce.detach()), "absent_energy": 0.0}
    return loss, stats
