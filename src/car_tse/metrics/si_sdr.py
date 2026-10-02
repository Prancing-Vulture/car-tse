"""Scale-invariant SDR."""

from __future__ import annotations

import torch


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-batch mean SI-SDR (higher is better). est/ref: [B, T]."""
    ref = ref - ref.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)
    dot = torch.sum(est * ref, dim=-1, keepdim=True)
    ref_energy = torch.sum(ref ** 2, dim=-1, keepdim=True) + eps
    s_target = dot * ref / ref_energy
    e_noise = est - s_target
    ratio = torch.sum(s_target ** 2, dim=-1) / (torch.sum(e_noise ** 2, dim=-1) + eps)
    return 10 * torch.log10(ratio + eps).mean()
