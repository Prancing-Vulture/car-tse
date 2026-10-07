"""End-to-end inference: SpEx+ -> presence -> KWS -> post-wake intent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from car_tse.models.intent_classifier import IntentClassifier
from car_tse.models.kws_model import KWSModel, TargetSpeakerKWSModel
from car_tse.models.presence_head import FramePresenceHead, PresenceHead
from car_tse.models.stage1_system import Stage1System
from car_tse.models.wav2vec_intent import Wav2VecIntentClassifier
from car_tse.pipeline.actions import ActionResult, intent_to_action
from car_tse.train.utils import load_checkpoint


@dataclass
class PipelineState:
    playback_ref: torch.Tensor | None = None
    last_reply_path: Path | None = None
    kws_scores: list[float] = field(default_factory=list)
    kws_refractory_left: int = 0
    # A wake word arms a separate command-capture phase.  Keeping this in the
    # streaming state prevents intent from being inferred from the wake-word
    # window itself.
    awaiting_command: bool = False
    command_wait_calls: int = 0
    speaker_embedding: torch.Tensor | None = None
    target_kws_enrollment_embedding: torch.Tensor | None = None


@dataclass
class PipelineOutput:
    accepted: bool
    wake_detected: bool
    intent_logits: torch.Tensor | None = None
    action: ActionResult | None = None
    s1: torch.Tensor | None = None
    command_pending: bool = False
    intent_ready: bool = False
    presence_probability: float | None = None
    speaker_similarity: float | None = None
    wake_probability: float | None = None


def normalize_extracted_waveform(s1: torch.Tensor, peak_limit: float = 0.95) -> torch.Tensor:
    """Match the peak conditioning used when building downstream caches.

    Stage 1 can have a large learned output gain.  The KWS and intent caches
    clip that gain after extraction, while the runtime path historically fed
    the raw waveform to those heads.  Do not amplify quiet outputs; only
    attenuate samples whose per-example peak exceeds ``peak_limit``.
    """

    peak = s1.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(peak > peak_limit, peak_limit / peak.clamp_min(1.0e-8), torch.ones_like(peak))
    return s1 * scale


def update_kws_state(
    state: PipelineState,
    score: float,
    threshold: float,
    window_size: int,
    required_hits: int,
    refractory_calls: int,
) -> bool:
    window_size = max(1, int(window_size))
    required_hits = max(1, min(int(required_hits), window_size))
    refractory_calls = max(0, int(refractory_calls))
    state.kws_scores.append(float(score))
    state.kws_scores = state.kws_scores[-window_size:]
    if state.kws_refractory_left > 0:
        state.kws_refractory_left -= 1
        return False
    hits = sum(value >= threshold for value in state.kws_scores)
    wake = len(state.kws_scores) >= window_size and hits >= required_hits
    if wake:
        state.kws_refractory_left = refractory_calls
    return wake


class InCarPipeline:
    def __init__(
        self,
        stage1: Stage1System,
        presence: PresenceHead,
        kws: KWSModel,
        intent: torch.nn.Module,
        kws_secondary: KWSModel | None = None,
        target_kws: TargetSpeakerKWSModel | None = None,
        presence_threshold: float = 0.5,
        kws_threshold: float = 0.5,
        kws_fusion_mode: str = "mean",
        kws_fusion_alpha: float = 0.5,
        wake_class_id: int = 1,
        kws_sample_rate: int = 16000,
        kws_input_seconds: float = 1.0,
        kws_smoothing_window: int = 1,
        kws_required_hits: int = 1,
        kws_refractory_calls: int = 0,
        intent_sample_rate: int = 16000,
        intent_input_seconds: float = 2.0,
        command_timeout_calls: int = 4,
        command_presence_threshold: float | None = None,
        speaker_gate_threshold: float | None = None,
        speaker_gate_backend: str = "spex",
        presence_top_fraction: float = 0.20,
        label_map: dict[str, str] | None = None,
        device: torch.device | None = None,
    ):
        self.stage1 = stage1
        self.presence = presence
        self.kws = kws
        self.kws_secondary = kws_secondary
        self.target_kws = target_kws
        self.intent = intent
        self.presence_threshold = presence_threshold
        self.kws_threshold = kws_threshold
        self.kws_fusion_mode = str(kws_fusion_mode).lower()
        self.kws_fusion_alpha = min(max(float(kws_fusion_alpha), 0.0), 1.0)
        self.wake_class_id = wake_class_id
        self.kws_sample_rate = int(kws_sample_rate)
        self.kws_input_seconds = float(kws_input_seconds)
        self.kws_smoothing_window = max(1, int(kws_smoothing_window))
        self.kws_required_hits = max(1, min(int(kws_required_hits), self.kws_smoothing_window))
        self.kws_refractory_calls = max(0, int(kws_refractory_calls))
        self.intent_sample_rate = int(intent_sample_rate)
        self.intent_input_seconds = max(0.01, float(intent_input_seconds))
        self.command_timeout_calls = max(1, int(command_timeout_calls))
        self.command_presence_threshold = (
            float(presence_threshold)
            if command_presence_threshold is None
            else float(command_presence_threshold)
        )
        self.speaker_gate_threshold = (
            None if speaker_gate_threshold is None else float(speaker_gate_threshold)
        )
        self.speaker_gate_backend = str(speaker_gate_backend).lower()
        self.presence_top_fraction = min(max(float(presence_top_fraction), 1.0e-3), 1.0)
        self.label_map = label_map
        self.device = device or torch.device("cpu")
        self.stage1.to(self.device)
        self.presence.to(self.device)
        self.kws.to(self.device)
        if self.kws_secondary is not None:
            self.kws_secondary.to(self.device)
        if self.target_kws is not None:
            self.target_kws.to(self.device)
        self.intent.to(self.device)
        self.stage1.eval()
        self.presence.eval()
        self.kws.eval()
        if self.kws_secondary is not None:
            self.kws_secondary.eval()
        if self.target_kws is not None:
            self.target_kws.eval()
        self.intent.eval()

    @classmethod
    def from_checkpoints(
        cls,
        stage1: Stage1System,
        ckpt_presence: Path,
        ckpt_kws: Path,
        ckpt_intent: Path,
        kws_sample_rate: int = 16000,
        kws_num_classes: int = 2,
        intent_sample_rate: int = 16000,
        intent_type: str = "crnn",
        intent_encoder: str = "facebook/wav2vec2-base-960h",
        intent_encoder_checkpoint: Path | str | None = None,
        intent_num_classes: int = 31,
        **kwargs: Any,
    ) -> "InCarPipeline":
        pipeline_kwargs = dict(kwargs)
        presence_type = str(pipeline_kwargs.pop("presence_type", "scalar")).lower()
        presence_sample_rate = int(pipeline_kwargs.pop("presence_sample_rate", kws_sample_rate))
        if presence_type == "frame":
            presence = FramePresenceHead(
                sample_rate=presence_sample_rate,
                n_mels=int(pipeline_kwargs.pop("presence_n_mels", 40)),
                hidden=int(pipeline_kwargs.pop("presence_hidden", 64)),
                n_fft=int(pipeline_kwargs.pop("presence_n_fft", 512)),
                hop=int(pipeline_kwargs.pop("presence_hop", 160)),
            )
        elif presence_type == "scalar":
            presence = PresenceHead()
        else:
            raise ValueError(f"Unsupported presence_type: {presence_type}")
        kws = KWSModel(sample_rate=kws_sample_rate, num_classes=kws_num_classes)
        secondary_checkpoint = pipeline_kwargs.pop("kws_secondary_checkpoint", None)
        kws_secondary = None
        if secondary_checkpoint:
            kws_secondary = KWSModel(sample_rate=kws_sample_rate, num_classes=kws_num_classes)
        target_kws_checkpoint = pipeline_kwargs.pop("target_kws_checkpoint", None)
        target_kws = None
        if target_kws_checkpoint:
            target_kws = TargetSpeakerKWSModel(
                sample_rate=kws_sample_rate,
                n_mels=int(pipeline_kwargs.pop("target_kws_n_mels", 40)),
                channels=pipeline_kwargs.pop("target_kws_channels", [32, 64, 128]),
                num_classes=kws_num_classes,
                speaker_embedding_dim=int(pipeline_kwargs.pop("target_kws_speaker_embedding_dim", 256)),
                hidden=int(pipeline_kwargs.pop("target_kws_hidden", 256)),
            )
        intent_input_seconds = float(pipeline_kwargs.get("intent_input_seconds", 2.0))
        if intent_type == "wav2vec":
            if not intent_encoder_checkpoint:
                raise ValueError("intent_encoder_checkpoint is required for wav2vec intent")
            intent = Wav2VecIntentClassifier(
                encoder_name=intent_encoder,
                encoder_checkpoint=intent_encoder_checkpoint,
                head_checkpoint=ckpt_intent,
                num_classes=intent_num_classes,
                input_sample_rate=intent_sample_rate,
                input_seconds=intent_input_seconds,
            )
        else:
            intent = IntentClassifier(sample_rate=intent_sample_rate)
        load_checkpoint(ckpt_presence, presence)
        load_checkpoint(ckpt_kws, kws)
        if kws_secondary is not None:
            load_checkpoint(secondary_checkpoint, kws_secondary)
        if target_kws is not None:
            load_checkpoint(target_kws_checkpoint, target_kws)
        if intent_type != "wav2vec":
            load_checkpoint(ckpt_intent, intent)
        pipeline_kwargs.setdefault("intent_sample_rate", intent_sample_rate)
        pipeline_kwargs.setdefault("intent_input_seconds", intent_input_seconds)
        return cls(
            stage1,
            presence,
            kws,
            intent,
            kws_secondary=kws_secondary,
            target_kws=target_kws,
            kws_sample_rate=kws_sample_rate,
            **pipeline_kwargs,
        )

    def _reset_command_capture(self, state: PipelineState) -> None:
        state.awaiting_command = False
        state.command_wait_calls = 0

    def _prepare_intent_input(self, wav: torch.Tensor) -> torch.Tensor:
        """Return the fixed-length post-wake command window expected by Stage 4."""

        samples = max(1, int(round(self.intent_sample_rate * self.intent_input_seconds)))
        if wav.shape[-1] >= samples:
            return wav[..., :samples]
        return torch.nn.functional.pad(wav, (0, samples - wav.shape[-1]))

    def presence_probability(self, s1: torch.Tensor) -> torch.Tensor:
        """Return one target-present probability per batch item."""

        logits = self.presence(s1)
        if getattr(self.presence, "is_framewise", False):
            return FramePresenceHead.aggregate_probability(
                torch.sigmoid(logits),
                top_fraction=self.presence_top_fraction,
            )
        return torch.sigmoid(logits)

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
        if multichannel is not None and multichannel.dim() == 2:
            multichannel = multichannel.unsqueeze(0)
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
            pres_prob = self.presence_probability(s1)
            presence_probability = float(pres_prob.mean().item())
            speaker_similarity = None
            active_presence_threshold = (
                self.command_presence_threshold
                if state.awaiting_command
                else self.presence_threshold
            )
            if (
                self.speaker_gate_threshold is not None
                and presence_probability >= active_presence_threshold
            ):
                similarity = None
                if self.speaker_gate_backend in {"spex", "auto"}:
                    similarity = self.stage1.speaker_similarity(enrollment, s1)
                if similarity is None and self.speaker_gate_backend in {"wav2vec", "auto"}:
                    embedder = getattr(self.intent, "embedding", None)
                    if callable(embedder):
                        if state.speaker_embedding is None:
                            state.speaker_embedding = embedder(enrollment).detach()
                        extracted_embedding = embedder(self._prepare_intent_input(s1))
                        similarity = torch.nn.functional.cosine_similarity(
                            state.speaker_embedding,
                            extracted_embedding,
                            dim=-1,
                        )
                if similarity is not None:
                    speaker_similarity = float(similarity.mean().item())
            speaker_rejected = (
                speaker_similarity is not None
                and speaker_similarity < self.speaker_gate_threshold
            )
            if presence_probability < active_presence_threshold or speaker_rejected:
                state.kws_scores.clear()
                if not state.awaiting_command:
                    state.kws_refractory_left = 0
                    return PipelineOutput(
                        accepted=False,
                        wake_detected=False,
                        s1=s1,
                        presence_probability=presence_probability,
                        speaker_similarity=speaker_similarity,
                    ), state

                # Allow a short pause between the wake word and command.  The
                # timeout is measured in incoming streaming calls so it is
                # independent of the duration of each audio tensor.
                state.command_wait_calls += 1
                if state.command_wait_calls >= self.command_timeout_calls:
                    self._reset_command_capture(state)
                    state.kws_refractory_left = 0
                    return PipelineOutput(
                        accepted=False,
                        wake_detected=False,
                        s1=s1,
                        presence_probability=presence_probability,
                        speaker_similarity=speaker_similarity,
                    ), state
                return PipelineOutput(
                    accepted=False,
                    wake_detected=False,
                    command_pending=True,
                    s1=s1,
                    presence_probability=presence_probability,
                    speaker_similarity=speaker_similarity,
                ), state
            downstream_s1 = normalize_extracted_waveform(s1)

            if state.awaiting_command:
                # The first accepted target-speech frame after a wake is the
                # command window.  This is deliberately separate from the
                # wake frame; a later PVAD can replace this presence decision
                # without changing the state-machine contract.
                state.command_wait_calls += 1
                intent_input = self._prepare_intent_input(downstream_s1)
                intent_logits = self.intent(intent_input)
                intent_id = str(int(torch.argmax(intent_logits, dim=-1).item()))
                action = intent_to_action(intent_id, self.label_map)
                if action.reply_wav and action.reply_wav.exists():
                    state.last_reply_path = action.reply_wav
                self._reset_command_capture(state)
                state.kws_scores.clear()
                return PipelineOutput(
                    accepted=True,
                    wake_detected=False,
                    intent_logits=intent_logits,
                    action=action,
                    s1=s1,
                    command_pending=False,
                    intent_ready=True,
                    presence_probability=presence_probability,
                    speaker_similarity=speaker_similarity,
                ), state

            kws_input = downstream_s1
            if self.kws_input_seconds > 0:
                kws_samples = max(1, int(round(self.kws_sample_rate * self.kws_input_seconds)))
                if kws_input.shape[-1] >= kws_samples:
                    kws_input = kws_input[..., :kws_samples]
                else:
                    kws_input = torch.nn.functional.pad(kws_input, (0, kws_samples - kws_input.shape[-1]))
            target_kws_score = None
            if self.target_kws is not None:
                embedder = getattr(self.stage1, "speaker_embedding", None)
                if callable(embedder):
                    if state.target_kws_enrollment_embedding is None:
                        state.target_kws_enrollment_embedding = embedder(enrollment).detach()
                    extracted_embedding = embedder(s1).detach()
                    target_kws_logits = self.target_kws(
                        kws_input,
                        state.target_kws_enrollment_embedding,
                        extracted_embedding,
                    )
                    target_kws_prob = torch.softmax(target_kws_logits, dim=-1)
                    target_kws_score = target_kws_prob[..., self.wake_class_id].mean().item()
            if target_kws_score is None:
                kws_logits = self.kws(kws_input)
                kws_prob = torch.softmax(kws_logits, dim=-1)
                score = kws_prob[..., self.wake_class_id].mean().item()
            else:
                score = target_kws_score
            if self.kws_secondary is not None:
                secondary_prob = torch.softmax(self.kws_secondary(kws_input), dim=-1)
                secondary_score = secondary_prob[..., self.wake_class_id].mean().item()
                if self.kws_fusion_mode == "min":
                    score = min(score, secondary_score)
                elif self.kws_fusion_mode == "max":
                    score = max(score, secondary_score)
                elif self.kws_fusion_mode == "geomean":
                    score = float(max(score, 0.0) * max(secondary_score, 0.0)) ** 0.5
                else:
                    score = self.kws_fusion_alpha * score + (1.0 - self.kws_fusion_alpha) * secondary_score
            wake = update_kws_state(
                state,
                score,
                self.kws_threshold,
                self.kws_smoothing_window,
                self.kws_required_hits,
                self.kws_refractory_calls,
            )
            if not wake:
                return PipelineOutput(
                    accepted=True,
                    wake_detected=False,
                    s1=s1,
                    presence_probability=presence_probability,
                    speaker_similarity=speaker_similarity,
                    wake_probability=score,
                ), state

            # Do not infer intent from the wake-word frame.  The next accepted
            # frame enters the explicit command-capture branch above.
            state.awaiting_command = True
            state.command_wait_calls = 0
            state.kws_scores.clear()
            return PipelineOutput(
                accepted=True,
                wake_detected=True,
                s1=s1,
                command_pending=True,
                presence_probability=presence_probability,
                speaker_similarity=speaker_similarity,
                wake_probability=score,
            ), state
