"""Stage 1 combined loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from car_tse.metrics.si_sdr import si_sdr, si_sdr_per_example


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
    target_present: torch.Tensor | None = None,
    absent_output_penalty: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if target_present is None or absent_output_penalty <= 0.0:
        l1 = -si_sdr(s1, target)
        l2 = -si_sdr(s2, target)
        l3 = -si_sdr(s3, target)
        loss = l1 + alpha * l2 + beta * l3
        stats = {
            "si_sdr_s1": float(-l1.detach()),
            "si_sdr_s2": float(-l2.detach()),
            "si_sdr_s3": float(-l3.detach()),
        }
    else:
        present = target_present.reshape(-1).to(dtype=torch.bool, device=s1.device)
        absent = ~present
        if present.any():
            sdr1 = si_sdr_per_example(s1[present], target[present]).mean()
            sdr2 = si_sdr_per_example(s2[present], target[present]).mean()
            sdr3 = si_sdr_per_example(s3[present], target[present]).mean()
            loss = -sdr1 - alpha * sdr2 - beta * sdr3
            stats = {
                "si_sdr_s1": float(sdr1.detach()),
                "si_sdr_s2": float(sdr2.detach()),
                "si_sdr_s3": float(sdr3.detach()),
            }
        else:
            loss = s1.float().sum() * 0.0
            stats = {"si_sdr_s1": 0.0, "si_sdr_s2": 0.0, "si_sdr_s3": 0.0}
        if absent.any():
            absent_energy = (
                s1[absent].float().square().mean()
                + alpha * s2[absent].float().square().mean()
                + beta * s3[absent].float().square().mean()
            )
            loss = loss + float(absent_output_penalty) * absent_energy
            stats["absent_output_energy"] = float(absent_energy.detach())
    if spk_labels is not None and gamma > 0:
        ce = F.cross_entropy(spk_logits, spk_labels)
        loss = loss + gamma * ce
        stats["ce"] = float(ce.detach())
    return loss, stats
