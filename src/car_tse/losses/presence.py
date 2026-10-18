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


def target_activity_labels(
    target: torch.Tensor,
    target_present: torch.Tensor,
    n_fft: int = 512,
    hop: int = 160,
    threshold_db_below_peak: float = 30.0,
    floor: float = 1.0e-4,
) -> torch.Tensor:
    """Create weak frame labels from the clean target waveform.

    Present clips use target energy to mark speech-active frames; absent clips
    are all-negative.  This is deliberately a weak binary target-activity
    label, not a fabricated passenger-versus-nonspeech three-way label.
    """

    if target.dim() == 1:
        target = target.unsqueeze(0)
    target = torch.nan_to_num(target.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if target_present.dim() == 0:
        target_present = target_present.unsqueeze(0)
    window = torch.hann_window(n_fft, device=target.device, dtype=target.dtype)
    spectrum = torch.stft(
        target,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    frame_rms = spectrum.abs().pow(2).mean(dim=1).sqrt()
    peak = frame_rms.amax(dim=-1, keepdim=True)
    relative_floor = peak * (10.0 ** (-float(threshold_db_below_peak) / 20.0))
    active = (frame_rms >= relative_floor.clamp_min(float(floor))) & (frame_rms >= float(floor))
    return active.to(dtype=target.dtype) * target_present.to(dtype=target.dtype).unsqueeze(-1)


def frame_presence_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    positive_weight: float = 2.0,
    smooth_weight: float = 0.02,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted frame BCE with a small temporal smoothness regularizer."""

    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=0.0, neginf=0.0)
    labels = torch.nan_to_num(labels.float(), nan=0.0, posinf=0.0, neginf=0.0)
    weight = torch.tensor(float(max(1.0, positive_weight)), device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight)
    if logits.shape[-1] > 1:
        smooth = (torch.sigmoid(logits[..., 1:]) - torch.sigmoid(logits[..., :-1])).abs().mean()
    else:
        smooth = logits.new_zeros(())
    loss = bce + float(max(0.0, smooth_weight)) * smooth
    return loss, {"bce": float(bce.detach()), "smooth": float(smooth.detach())}


def target_activity_class_labels(
    target: torch.Tensor,
    activity_class: torch.Tensor,
    n_fft: int = 512,
    hop: int = 160,
    threshold_db_below_peak: float = 30.0,
    floor: float = 1.0e-4,
) -> torch.Tensor:
    """Create weak three-way frame labels for personal activity training.

    ``activity_class`` uses 0 for non-speech, 1 for target speech, and 2 for
    non-target speech.  The waveform supplies the speech-active frame mask;
    inactive frames are always class 0.  Unlike ``target_activity_labels``,
    this helper does not collapse non-target speech into silence.
    """

    if target.dim() == 1:
        target = target.unsqueeze(0)
    if activity_class.dim() == 0:
        activity_class = activity_class.unsqueeze(0)
    target = torch.nan_to_num(target.float(), nan=0.0, posinf=0.0, neginf=0.0)
    window = torch.hann_window(n_fft, device=target.device, dtype=target.dtype)
    spectrum = torch.stft(
        target,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    frame_rms = spectrum.abs().pow(2).mean(dim=1).sqrt()
    peak = frame_rms.amax(dim=-1, keepdim=True)
    relative_floor = peak * (10.0 ** (-float(threshold_db_below_peak) / 20.0))
    active = (frame_rms >= relative_floor.clamp_min(float(floor))) & (frame_rms >= float(floor))
    classes = activity_class.to(device=target.device, dtype=torch.long).unsqueeze(-1)
    return torch.where(active, classes.expand(-1, active.shape[-1]), torch.zeros_like(classes.expand(-1, active.shape[-1])))


def frame_activity_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: tuple[float, float, float] = (1.0, 2.0, 1.5),
    smooth_weight: float = 0.02,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Cross-entropy plus a small temporal regularizer for 3-way activity."""

    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=0.0, neginf=0.0)
    labels = torch.nan_to_num(labels.long(), nan=0, posinf=0, neginf=0)
    weights = torch.tensor(class_weights, device=logits.device, dtype=logits.dtype)
    ce = F.cross_entropy(logits, labels, weight=weights)
    target_probability = torch.softmax(logits, dim=1)[:, 1, :]
    if target_probability.shape[-1] > 1:
        smooth = (target_probability[..., 1:] - target_probability[..., :-1]).abs().mean()
    else:
        smooth = target_probability.new_zeros(())
    loss = ce + float(max(0.0, smooth_weight)) * smooth
    return loss, {"cross_entropy": float(ce.detach()), "smooth": float(smooth.detach())}
