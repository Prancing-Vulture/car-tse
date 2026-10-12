import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car_tse.models.presence_head import PersonalActivityHead, PersonalFramePresenceHead


def test_personal_presence_head_is_embedding_conditioned_and_framewise():
    torch.manual_seed(7)
    head = PersonalFramePresenceHead(hidden=16, speaker_embedding_dim=8)
    waveform = torch.randn(2, 16000)
    first = torch.randn(2, 8)
    second = first.clone()
    second[0, 0] += 3.0

    first_logits = head(waveform, first)
    second_logits = head(waveform, second)

    assert first_logits.shape[0] == 2
    assert first_logits.shape[-1] > 1
    assert not torch.allclose(first_logits[0], second_logits[0])
    assert torch.allclose(first_logits[1], second_logits[1])


def test_personal_activity_head_emits_three_way_target_probability():
    torch.manual_seed(11)
    head = PersonalActivityHead(hidden=16, speaker_embedding_dim=8)
    waveform = torch.randn(2, 16000)
    enrollment = torch.randn(2, 8)
    logits = head(waveform, enrollment)
    target_probability = head.target_probability(logits)

    assert logits.shape[0] == 2
    assert logits.shape[1] == 3
    assert target_probability.shape == (logits.shape[0], logits.shape[-1])
    assert torch.all(target_probability >= 0.0)
    assert torch.all(target_probability <= 1.0)


def test_personal_activity_margin_can_align_target_and_passenger_frames():
    target = torch.tensor([[0.9, 0.1, 0.8, 0.2]])
    non_target = torch.tensor([[0.1, 0.8, 0.2, 0.7]])

    independent = PersonalActivityHead.aggregate_margin(
        target, non_target, top_fraction=0.5, mode="independent_top"
    )
    target_top = PersonalActivityHead.aggregate_margin(
        target, non_target, top_fraction=0.5, mode="target_top"
    )
    framewise_top = PersonalActivityHead.aggregate_margin(
        target, non_target, top_fraction=0.5, mode="framewise_top"
    )

    assert torch.allclose(independent, torch.tensor([0.1]))
    assert torch.allclose(target_top, torch.tensor([0.7]))
    assert torch.allclose(framewise_top, torch.tensor([0.7]))
