"""End-to-end inference: SpEx+ -> presence -> KWS -> intent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from car_tse.models.intent_classifier import IntentClassifier
from car_tse.models.kws_model import KWSModel
from car_tse.models.presence_head import PresenceHead
from car_tse.models.stage1_system import Stage1System
from car_tse.pipeline.actions import ActionResult, intent_to_action
from car_tse.train.utils import load_checkpoint


@dataclass
class PipelineState:
    playback_ref: torch.Tensor | None = None
    last_reply_path: Path | None = None


@dataclass
class PipelineOutput:
    accepted: bool
    wake_detected: bool
    intent_logits: torch.Tensor | None = None
    action: ActionResult | None = None
    s1: torch.Tensor | None = None


class InCarPipeline:
    def __init__(
        self,
        stage1: Stage1System,
        presence: PresenceHead,
        kws: KWSModel,
        intent: IntentClassifier,
        presence_threshold: float = 0.5,
        kws_threshold: float = 0.5,
        wake_class_id: int = 0,
        label_map: dict[str, str] | None = None,
        device: torch.device | None = None,
    ):
        self.stage1 = stage1
        self.presence = presence
        self.kws = kws
        self.intent = intent
        self.presence_threshold = presence_threshold
        self.kws_threshold = kws_threshold
        self.wake_class_id = wake_class_id
        self.label_map = label_map
        self.device = device or torch.device("cpu")
        self.stage1.to(self.device)
        self.presence.to(self.device)
        self.kws.to(self.device)
        self.intent.to(self.device)
        self.stage1.eval()
        self.presence.eval()
        self.kws.eval()
        self.intent.eval()

    @classmethod
    def from_checkpoints(
        cls,
        stage1: Stage1System,
        ckpt_presence: Path,
        ckpt_kws: Path,
        ckpt_intent: Path,
        **kwargs: Any,
    ) -> "InCarPipeline":
        presence = PresenceHead()
        kws = KWSModel()
        intent = IntentClassifier()
        load_checkpoint(ckpt_presence, presence)
        load_checkpoint(ckpt_kws, kws)
        load_checkpoint(ckpt_intent, intent)
        return cls(stage1, presence, kws, intent, **kwargs)

    def forward(
        self,
        mixture: torch.Tensor,
        enrollment: torch.Tensor,
        state: PipelineState | None = None,
        multichannel: torch.Tensor | None = None,
    ) -> tuple[PipelineOutput, PipelineState]:
        state = state or PipelineState()
        mixture = mixture.to(self.device)
        enrollment = enrollment.to(self.device)
        if mixture.dim() == 1:
            mixture = mixture.unsqueeze(0)
        if enrollment.dim() == 1:
            enrollment = enrollment.unsqueeze(0)
        pb = state.playback_ref
        if pb is not None:
            pb = pb.to(self.device)
        with torch.no_grad():
            s1, _, _, _ = self.stage1(
                mixture,
                enrollment,
                multichannel=multichannel.to(self.device) if multichannel is not None else None,
                playback_ref=pb,
            )
            pres_logit = self.presence(s1)
            pres_prob = torch.sigmoid(pres_logit)
            if pres_prob.mean().item() < self.presence_threshold:
                return PipelineOutput(accepted=False, wake_detected=False, s1=s1), state
            kws_logits = self.kws(s1)
            kws_prob = torch.softmax(kws_logits, dim=-1)
            wake = kws_prob[..., self.wake_class_id].mean().item() >= self.kws_threshold
            if not wake:
                return PipelineOutput(accepted=True, wake_detected=False, s1=s1), state
            intent_logits = self.intent(s1)
            intent_id = str(int(torch.argmax(intent_logits, dim=-1).item()))
            action = intent_to_action(intent_id, self.label_map)
            if action.reply_wav and action.reply_wav.exists():
                state.last_reply_path = action.reply_wav
            return PipelineOutput(
                accepted=True,
                wake_detected=True,
                intent_logits=intent_logits,
                action=action,
                s1=s1,
            ), state
