"""Stage 1 combined loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from car_tse.metrics.si_sdr import si_sdr


def stage1_loss(
    s1: torch.Tensor,
    s2: torch.Tensor,
    s3: torch.Tensor,
    target: torch.Tensor,
    spk_logits: torch.Tensor,
    spk_labels: torch.Tensor | None,
    alpha: float = 0.1,
    beta: float = 0.1,
    gamma: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    l1 = -si_sdr(s1, target)
    l2 = -si_sdr(s2, target)
    l3 = -si_sdr(s3, target)
    loss = l1 + alpha * l2 + beta * l3
    stats = {"si_sdr_s1": float(-l1.detach()), "si_sdr_s2": float(-l2.detach()), "si_sdr_s3": float(-l3.detach())}
    if spk_labels is not None and gamma > 0:
        ce = F.cross_entropy(spk_logits, spk_labels)
        loss = loss + gamma * ce
        stats["ce"] = float(ce.detach())
    return loss, stats
