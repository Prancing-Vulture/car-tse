"""End-to-end inference: SpEx+ -> presence -> KWS -> post-wake intent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from car_tse.models.intent_classifier import IntentClassifier
from car_tse.models.kws_model import KWSModel, TargetSpeakerKWSModel
from car_tse.models.presence_head import (
    FramePresenceHead,
    PersonalActivityHead,
    PersonalFramePresenceHead,
    PresenceHead,
)
from car_tse.models.speaker_verifier import CompactSpeakerVerifier
from car_tse.models.stage1_system import Stage1System
from car_tse.models.wav2vec_intent import Wav2VecIntentClassifier, Wav2VecIntentEnsemble
from car_tse.pipeline.actions import ActionResult, intent_to_action
from car_tse.train.utils import load_checkpoint


@dataclass
class PipelineState:
    playback_ref: torch.Tensor | None = None
    last_reply_path: Path | None = None
    kws_scores: list[float] = field(default_factory=list)
    kws_refractory_left: int = 0
    # Shared safety refractory for the wake detector.  It is armed after a
    # committed action, and can also be armed after an opt-in failed command
    # cycle.  The latter prevents a passenger wake from re-arming immediately
    # after a target wake timed out or was suppressed.
    post_action_refractory_left: int = 0
    # A wake word arms a separate command-capture phase.  Keeping this in the
    # streaming state prevents intent from being inferred from the wake-word
    # window itself.
    awaiting_command: bool = False
    command_wait_calls: int = 0
    speaker_embedding: torch.Tensor | None = None
    # Optional precomputed SpEx reference embedding.  When supplied, this is
    # an enrollment-bank representation (for example, the normalized mean of
    # several short utterances) and is reused by all speaker-conditioned
    # gates.  None preserves the historical single-utterance behavior.
    reference_speaker_embedding: torch.Tensor | None = None
    presence_enrollment_embedding: torch.Tensor | None = None
    target_kws_enrollment_embedding: torch.Tensor | None = None
    # The normalized Stage-1 frame that produced the accepted wake.  It is
    # retained only for the opt-in endpoint-aware command window strategy;
    # the historical fixed-window path leaves this unset.
    wake_context: torch.Tensor | None = None
    # The immediately preceding normalized Stage-1 frame.  The opt-in
    # pre-wake buffered strategy uses it when the wake detector fires late in
    # a co-channel scene, so the first command candidate can retain audio that
    # preceded the wake event.
    last_extracted_s1: torch.Tensor | None = None
    # Number of post-wake command candidates evaluated by the opt-in
    # confidence-buffered endpoint.  Keeping this in streaming state avoids
    # committing a low-confidence partial command and then re-arming on the
    # overlapping tail of the same utterance.
    command_candidate_calls: int = 0
    # Best intent candidate observed while the opt-in command buffer is open.
    # This prevents a weak trailing hop from replacing a strong earlier one.
    command_best_confidence: float = -1.0
    command_best_logits: torch.Tensor | None = None
    # Per-hop class/confidence history used for a stable buffered decision.
    command_candidate_history: list[tuple[int, float, torch.Tensor]] = field(
        default_factory=list
    )


@dataclass
class PipelineOutput:
    accepted: bool
    wake_detected: bool
    intent_logits: torch.Tensor | None = None
    action: ActionResult | None = None
    s1: torch.Tensor | None = None
    command_pending: bool = False
    intent_ready: bool = False
    command_suppressed: bool = False
    presence_probability: float | None = None
    presence_non_target_probability: float | None = None
    presence_activity_margin: float | None = None
    speaker_similarity: float | None = None
    wake_probability: float | None = None
    # Raw enrollment-conditioned wake score before optional speaker-score
    # fusion. This enables target-KWS attribution diagnostics without
    # changing the wake state-machine behavior.
    target_kws_probability: float | None = None


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
    if state.post_action_refractory_left > 0:
        state.post_action_refractory_left -= 1
        # Do not allow positive wake scores accumulated during the safety
        # window to form an immediate wake when it expires.
        state.kws_scores.clear()
        return False
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
        target_kws_secondary: TargetSpeakerKWSModel | None = None,
        compact_speaker_verifier: CompactSpeakerVerifier | None = None,
        target_kws_fusion_mode: str = "min",
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
        post_action_refractory_calls: int = 0,
        post_command_refractory_calls: int = 0,
        intent_sample_rate: int = 16000,
        intent_input_seconds: float = 2.0,
        intent_window_offset_seconds: float = 0.0,
        command_timeout_calls: int = 4,
        command_delay_calls: int = 0,
        command_window_strategy: str = "fixed",
        command_buffer_calls: int = 2,
        command_confidence_commit_threshold: float = 0.90,
        command_buffer_selection: str = "longest_run",
        command_buffer_confidence_margin: float = 0.30,
        command_endpoint_kws_hop_seconds: float = 0.25,
        command_endpoint_margin_seconds: float = 0.50,
        command_presence_threshold: float | None = None,
        presence_activity_margin_threshold: float | None = None,
        presence_activity_margin_mode: str = "independent_top",
        intent_confidence_threshold: float | None = None,
        speaker_gate_threshold: float | None = None,
        command_speaker_gate_threshold: float | None = None,
        command_speaker_gate_rescue_confidence: float | None = None,
        command_speaker_gate_rescue_floor: float = 0.0,
        speaker_gate_backend: str = "spex",
        speaker_fusion_mode: str = "none",
        speaker_fusion_center: float = 0.0,
        speaker_fusion_scale: float = 0.10,
        speaker_fusion_alpha: float = 0.5,
        compact_speaker_verifier_input_seconds: float = 1.0,
        compact_speaker_verifier_sample_rate: int = 16000,
        presence_top_fraction: float = 0.20,
        label_map: dict[str, str] | None = None,
        device: torch.device | None = None,
    ):
        self.stage1 = stage1
        self.presence = presence
        self.kws = kws
        self.kws_secondary = kws_secondary
        self.target_kws = target_kws
        self.target_kws_secondary = target_kws_secondary
        self.compact_speaker_verifier = compact_speaker_verifier
        self.target_kws_fusion_mode = str(target_kws_fusion_mode).lower()
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
        self.post_action_refractory_calls = max(0, int(post_action_refractory_calls))
        self.post_command_refractory_calls = max(0, int(post_command_refractory_calls))
        self.intent_sample_rate = int(intent_sample_rate)
        self.intent_input_seconds = max(0.01, float(intent_input_seconds))
        self.intent_window_offset_seconds = max(0.0, float(intent_window_offset_seconds))
        self.command_timeout_calls = max(1, int(command_timeout_calls))
        self.command_delay_calls = max(0, int(command_delay_calls))
        self.command_window_strategy = str(command_window_strategy).lower()
        if self.command_window_strategy not in {
            "fixed",
            "kws_trim",
            "kws_trim_latest",
            "confidence_buffered",
            "confidence_buffered_hybrid",
            "prewake_buffered_hybrid",
        }:
            raise ValueError(
                "command_window_strategy must be 'fixed', 'kws_trim', 'kws_trim_latest', "
                "'confidence_buffered_hybrid', or 'prewake_buffered_hybrid'"
            )
        self.command_buffer_calls = max(1, int(command_buffer_calls))
        self.command_confidence_commit_threshold = min(
            max(float(command_confidence_commit_threshold), 0.0), 1.0
        )
        self.command_buffer_selection = str(command_buffer_selection).lower()
        if self.command_buffer_selection not in {
            "longest_run",
            "best_confidence",
            "confidence_margin",
            "mean_logits",
        }:
            raise ValueError(
                "command_buffer_selection must be 'longest_run', 'best_confidence', "
                "'confidence_margin', or 'mean_logits'"
            )
        self.command_buffer_confidence_margin = max(
            float(command_buffer_confidence_margin), 0.0
        )
        self.command_endpoint_kws_hop_seconds = max(
            0.01, float(command_endpoint_kws_hop_seconds)
        )
        self.command_endpoint_margin_seconds = max(
            0.0, float(command_endpoint_margin_seconds)
        )
        self.command_presence_threshold = (
            float(presence_threshold)
            if command_presence_threshold is None
            else float(command_presence_threshold)
        )
        self.presence_activity_margin_threshold = (
            None
            if presence_activity_margin_threshold is None
            else float(presence_activity_margin_threshold)
        )
        self.presence_activity_margin_mode = str(presence_activity_margin_mode).lower()
        if self.presence_activity_margin_mode not in {
            "independent_top",
            "target_top",
            "framewise_top",
            "mean",
        }:
            raise ValueError(
                "presence_activity_margin_mode must be one of: "
                "independent_top, target_top, framewise_top, mean"
            )
        self.intent_confidence_threshold = (
            None if intent_confidence_threshold is None else float(intent_confidence_threshold)
        )
        self.speaker_gate_threshold = (
            None if speaker_gate_threshold is None else float(speaker_gate_threshold)
        )
        self.command_speaker_gate_threshold = (
            self.speaker_gate_threshold
            if command_speaker_gate_threshold is None
            else float(command_speaker_gate_threshold)
        )
        self.command_speaker_gate_rescue_confidence = (
            None
            if command_speaker_gate_rescue_confidence is None
            else min(max(float(command_speaker_gate_rescue_confidence), 0.0), 1.0)
        )
        self.command_speaker_gate_rescue_floor = min(
            max(float(command_speaker_gate_rescue_floor), 0.0), 1.0
        )
        self.speaker_gate_backend = str(speaker_gate_backend).lower()
        self.speaker_fusion_mode = str(speaker_fusion_mode).lower()
        if self.speaker_fusion_mode not in {"none", "min", "product", "mean"}:
            raise ValueError(
                "speaker_fusion_mode must be one of: none, min, product, mean"
            )
        self.speaker_fusion_center = float(speaker_fusion_center)
        self.speaker_fusion_scale = max(float(speaker_fusion_scale), 1.0e-4)
        self.speaker_fusion_alpha = min(max(float(speaker_fusion_alpha), 0.0), 1.0)
        self.compact_speaker_verifier_input_seconds = max(
            0.01, float(compact_speaker_verifier_input_seconds)
        )
        self.compact_speaker_verifier_sample_rate = int(compact_speaker_verifier_sample_rate)
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
        if self.target_kws_secondary is not None:
            self.target_kws_secondary.to(self.device)
        if self.compact_speaker_verifier is not None:
            self.compact_speaker_verifier.to(self.device)
        self.intent.to(self.device)
        self.stage1.eval()
        self.presence.eval()
        self.kws.eval()
        if self.kws_secondary is not None:
            self.kws_secondary.eval()
        if self.target_kws is not None:
            self.target_kws.eval()
        if self.target_kws_secondary is not None:
            self.target_kws_secondary.eval()
        if self.compact_speaker_verifier is not None:
            self.compact_speaker_verifier.eval()
        self.intent.eval()

    @classmethod
    def from_checkpoints(
        cls,
        stage1: Stage1System,
        ckpt_presence: Path,
        ckpt_kws: Path,
        ckpt_intent: Path,
        kws_sample_rate: int = 16000,
        kws_n_mels: int = 40,
        kws_channels: list[int] | None = None,
        kws_num_classes: int = 2,
        intent_sample_rate: int = 16000,
        intent_type: str = "crnn",
        intent_normalize_input: bool = False,
        intent_bidirectional: bool = True,
        intent_n_mels: int = 80,
        intent_hidden: int = 256,
        intent_num_layers: int = 2,
        intent_encoder: str = "facebook/wav2vec2-base-960h",
        intent_encoder_checkpoint: Path | str | None = None,
        intent_num_classes: int = 31,
        **kwargs: Any,
    ) -> "InCarPipeline":
        pipeline_kwargs = dict(kwargs)
        presence_type = str(pipeline_kwargs.pop("presence_type", "scalar")).lower()
        presence_sample_rate = int(pipeline_kwargs.pop("presence_sample_rate", kws_sample_rate))
        # These parameters belong to the optional framewise presence head.  Pop
        # them for every presence type so a scalar head cannot accidentally
        # receive them through the generic pipeline constructor.
        presence_n_mels = int(pipeline_kwargs.pop("presence_n_mels", 40))
        presence_hidden = int(pipeline_kwargs.pop("presence_hidden", 64))
        presence_n_fft = int(pipeline_kwargs.pop("presence_n_fft", 512))
        presence_hop = int(pipeline_kwargs.pop("presence_hop", 160))
        presence_speaker_embedding_dim = int(pipeline_kwargs.pop("presence_speaker_embedding_dim", 256))
        if presence_type == "frame":
            presence = FramePresenceHead(
                sample_rate=presence_sample_rate,
                n_mels=presence_n_mels,
                hidden=presence_hidden,
                n_fft=presence_n_fft,
                hop=presence_hop,
            )
        elif presence_type == "personal_frame":
            presence = PersonalFramePresenceHead(
                sample_rate=presence_sample_rate,
                n_mels=presence_n_mels,
                hidden=presence_hidden,
                n_fft=presence_n_fft,
                hop=presence_hop,
                speaker_embedding_dim=presence_speaker_embedding_dim,
            )
        elif presence_type in {"personal_activity", "personal_activity_3way"}:
            presence = PersonalActivityHead(
                sample_rate=presence_sample_rate,
                n_mels=presence_n_mels,
                hidden=presence_hidden,
                n_fft=presence_n_fft,
                hop=presence_hop,
                speaker_embedding_dim=presence_speaker_embedding_dim,
            )
        elif presence_type == "scalar":
            presence = PresenceHead()
        else:
            raise ValueError(f"Unsupported presence_type: {presence_type}")
        kws = KWSModel(
            sample_rate=kws_sample_rate,
            n_mels=int(kws_n_mels),
            channels=kws_channels,
            num_classes=kws_num_classes,
        )
        secondary_checkpoint = pipeline_kwargs.pop("kws_secondary_checkpoint", None)
        kws_secondary = None
        if secondary_checkpoint:
            kws_secondary = KWSModel(
                sample_rate=kws_sample_rate,
                n_mels=int(kws_n_mels),
                channels=kws_channels,
                num_classes=kws_num_classes,
            )
        target_kws_checkpoint = pipeline_kwargs.pop("target_kws_checkpoint", None)
        target_kws_secondary_checkpoint = pipeline_kwargs.pop("target_kws_secondary_checkpoint", None)
        # Consume target-KWS architecture options even when the target-KWS
        # branch is intentionally disabled.  Otherwise a regular-KWS
        # ablation forwards these branch-specific keys into ``__init__`` and
        # fails before inference starts.
        target_kws_n_mels = int(pipeline_kwargs.pop("target_kws_n_mels", 40))
        target_kws_channels = pipeline_kwargs.pop("target_kws_channels", [32, 64, 128])
        target_kws_speaker_embedding_dim = int(
            pipeline_kwargs.pop("target_kws_speaker_embedding_dim", 256)
        )
        target_kws_hidden = int(pipeline_kwargs.pop("target_kws_hidden", 256))
        compact_speaker_verifier_checkpoint = pipeline_kwargs.pop(
            "compact_speaker_verifier_checkpoint", None
        )
        compact_speaker_verifier_embedding_dim = int(
            pipeline_kwargs.pop("compact_speaker_verifier_embedding_dim", 128)
        )
        compact_speaker_verifier_hidden = int(
            pipeline_kwargs.pop("compact_speaker_verifier_hidden", 128)
        )
        compact_speaker_verifier_input_seconds = float(
            pipeline_kwargs.pop("compact_speaker_verifier_input_seconds", 1.0)
        )
        compact_speaker_verifier_sample_rate = int(
            pipeline_kwargs.pop("compact_speaker_verifier_sample_rate", 16000)
        )
        target_kws = None
        target_kws_secondary = None
        compact_speaker_verifier = None
        if target_kws_checkpoint:
            target_kws = TargetSpeakerKWSModel(
                sample_rate=kws_sample_rate,
                n_mels=target_kws_n_mels,
                channels=target_kws_channels,
                num_classes=kws_num_classes,
                speaker_embedding_dim=target_kws_speaker_embedding_dim,
                hidden=target_kws_hidden,
            )
            if target_kws_secondary_checkpoint:
                target_kws_secondary = TargetSpeakerKWSModel(
                    sample_rate=kws_sample_rate,
                    n_mels=target_kws_n_mels,
                    channels=target_kws_channels,
                    num_classes=kws_num_classes,
                    speaker_embedding_dim=target_kws_speaker_embedding_dim,
                    hidden=target_kws_hidden,
                )
        if compact_speaker_verifier_checkpoint:
            compact_speaker_verifier = CompactSpeakerVerifier(
                embedding_dim=compact_speaker_verifier_embedding_dim,
                hidden=compact_speaker_verifier_hidden,
                sample_rate=compact_speaker_verifier_sample_rate,
            )
        intent_input_seconds = float(pipeline_kwargs.get("intent_input_seconds", 2.0))
        intent_ensemble = pipeline_kwargs.pop("intent_ensemble", None)
        intent_ensemble_selection = str(
            pipeline_kwargs.pop("intent_ensemble_selection", "weighted_mean")
        )
        intent_ensemble_gate_threshold = float(
            pipeline_kwargs.pop("intent_ensemble_gate_threshold", 0.90)
        )
        if intent_type == "wav2vec" and intent_ensemble:
            members: list[Wav2VecIntentClassifier] = []
            weights: list[float] = []
            for index, spec in enumerate(intent_ensemble):
                if not isinstance(spec, dict):
                    raise ValueError(f"intent_ensemble member {index} must be a mapping")
                member_encoder = str(spec.get("encoder", intent_encoder))
                member_encoder_checkpoint = spec.get("encoder_checkpoint")
                member_head_checkpoint = spec.get("head_checkpoint")
                if not member_encoder_checkpoint or not member_head_checkpoint:
                    raise ValueError(
                        "each intent_ensemble member requires encoder_checkpoint and head_checkpoint"
                    )
                members.append(
                    Wav2VecIntentClassifier(
                        encoder_name=member_encoder,
                        encoder_checkpoint=member_encoder_checkpoint,
                        head_checkpoint=member_head_checkpoint,
                        num_classes=int(spec.get("num_classes", intent_num_classes)),
                        input_sample_rate=int(spec.get("sample_rate", intent_sample_rate)),
                        input_seconds=float(spec.get("input_seconds", intent_input_seconds)),
                    )
                )
                weights.append(float(spec.get("weight", 1.0)))
            intent = Wav2VecIntentEnsemble(
                members,
                weights,
                selection=intent_ensemble_selection,
                gate_threshold=intent_ensemble_gate_threshold,
            )
        elif intent_type == "wav2vec":
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
            intent = IntentClassifier(
                sample_rate=intent_sample_rate,
                n_mels=int(intent_n_mels),
                hidden=int(intent_hidden),
                num_layers=int(intent_num_layers),
                normalize_input=bool(intent_normalize_input),
                bidirectional=bool(intent_bidirectional),
            )
        load_checkpoint(ckpt_presence, presence)
        load_checkpoint(ckpt_kws, kws)
        if kws_secondary is not None:
            load_checkpoint(secondary_checkpoint, kws_secondary)
        if target_kws is not None:
            load_checkpoint(target_kws_checkpoint, target_kws)
        if target_kws_secondary is not None:
            load_checkpoint(target_kws_secondary_checkpoint, target_kws_secondary)
        if compact_speaker_verifier is not None:
            load_checkpoint(compact_speaker_verifier_checkpoint, compact_speaker_verifier)
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
            target_kws_secondary=target_kws_secondary,
            compact_speaker_verifier=compact_speaker_verifier,
            target_kws_fusion_mode=str(pipeline_kwargs.pop("target_kws_fusion_mode", "min")),
            kws_sample_rate=kws_sample_rate,
            compact_speaker_verifier_input_seconds=compact_speaker_verifier_input_seconds,
            compact_speaker_verifier_sample_rate=compact_speaker_verifier_sample_rate,
            **pipeline_kwargs,
        )

    def _reset_command_capture(
        self,
        state: PipelineState,
        *,
        arm_command_refractory: bool = False,
    ) -> None:
        state.awaiting_command = False
        state.command_wait_calls = 0
        state.command_candidate_calls = 0
        state.command_best_confidence = -1.0
        state.command_best_logits = None
        state.command_candidate_history.clear()
        state.wake_context = None
        if arm_command_refractory:
            state.post_action_refractory_left = max(
                state.post_action_refractory_left,
                self.post_command_refractory_calls,
            )

    def _fit_intent_input(self, wav: torch.Tensor, offset_samples: int = 0) -> torch.Tensor:
        """Crop/pad a waveform to the fixed Stage-4 input length."""

        samples = max(1, int(round(self.intent_sample_rate * self.intent_input_seconds)))
        offset = max(0, int(offset_samples))
        if wav.shape[-1] >= samples + offset:
            return wav[..., offset : offset + samples]
        if wav.shape[-1] >= samples:
            return wav[..., :samples]
        return torch.nn.functional.pad(wav, (0, samples - wav.shape[-1]))

    def _prepare_intent_input(self, wav: torch.Tensor) -> torch.Tensor:
        """Return the fixed-length post-wake command window expected by Stage 4."""

        offset = int(round(self.intent_sample_rate * self.intent_window_offset_seconds))
        return self._fit_intent_input(wav, offset)

    def _prepare_kws_input(self, wav: torch.Tensor) -> torch.Tensor:
        if self.kws_input_seconds <= 0:
            return wav
        samples = max(1, int(round(self.kws_sample_rate * self.kws_input_seconds)))
        if wav.shape[-1] >= samples:
            return wav[..., :samples]
        return torch.nn.functional.pad(wav, (0, samples - wav.shape[-1]))

    def _regular_kws_score(self, wav: torch.Tensor) -> float:
        """Score one local KWS crop without changing temporal wake state."""

        logits = self.kws(self._prepare_kws_input(wav))
        score = float(torch.softmax(logits, dim=-1)[..., self.wake_class_id].mean().item())
        if self.kws_secondary is not None:
            secondary = torch.softmax(
                self.kws_secondary(self._prepare_kws_input(wav)), dim=-1
            )[..., self.wake_class_id].mean().item()
            if self.kws_fusion_mode == "min":
                score = min(score, float(secondary))
            elif self.kws_fusion_mode == "max":
                score = max(score, float(secondary))
            elif self.kws_fusion_mode == "geomean":
                score = float(max(score, 0.0) * max(float(secondary), 0.0)) ** 0.5
            else:
                score = self.kws_fusion_alpha * score + (1.0 - self.kws_fusion_alpha) * float(secondary)
        return score

    def _prepare_kws_trimmed_intent_input(self, wav: torch.Tensor) -> torch.Tensor | None:
        """Trim the accepted wake frame after its last local KWS hit.

        This is deliberately opt-in.  It uses the ordinary KWS model as a
        local endpoint detector and leaves the temporal wake state untouched.
        If no local hit is visible, callers should fall back to the current
        streaming frame, which covers commands that begin after the wake frame.
        """

        kws_samples = max(1, int(round(self.kws_sample_rate * self.kws_input_seconds)))
        if wav.shape[-1] < kws_samples:
            return None
        hop_samples = max(
            1, int(round(self.kws_sample_rate * self.command_endpoint_kws_hop_seconds))
        )
        last_start = max(0, int(wav.shape[-1] - kws_samples))
        starts = list(range(0, last_start + 1, hop_samples))
        if starts[-1] != last_start:
            starts.append(last_start)
        scores = [self._regular_kws_score(wav[..., start : start + kws_samples]) for start in starts]
        active = [start for start, score in zip(starts, scores) if score >= self.kws_threshold]
        if not active:
            return None
        margin = int(round(self.intent_sample_rate * self.command_endpoint_margin_seconds))
        command_start = max(0, active[-1] + kws_samples - margin)
        if command_start >= wav.shape[-1]:
            return None
        offset = int(round(self.intent_sample_rate * self.intent_window_offset_seconds))
        return self._fit_intent_input(wav[..., command_start:], offset)

    def _prepare_compact_speaker_input(self, wav: torch.Tensor) -> torch.Tensor:
        """Return the one-second waveform expected by the compact verifier."""

        samples = max(
            1,
            int(
                round(
                    self.compact_speaker_verifier_sample_rate
                    * self.compact_speaker_verifier_input_seconds
                )
            ),
        )
        if wav.shape[-1] >= samples:
            return wav[..., :samples]
        return torch.nn.functional.pad(wav, (0, samples - wav.shape[-1]))

    def _speaker_fusion_probability(self, similarity: torch.Tensor) -> torch.Tensor:
        """Map cosine similarity to an opt-in calibrated speaker score.

        The center and scale are operating-point parameters, not learned
        defaults.  They must be selected on tuning sessions because cosine
        similarity is domain and enrollment dependent.  Keeping this mapping
        explicit makes the keyword and speaker branches independently
        inspectable and preserves the historical hard-gate behavior when the
        fusion mode is ``none``.
        """

        return torch.sigmoid(
            (similarity - self.speaker_fusion_center) / self.speaker_fusion_scale
        )

    def _fuse_speaker_score(
        self,
        keyword_score: float,
        similarity: torch.Tensor,
    ) -> float:
        """Fuse keyword evidence with an independent speaker-verification score."""

        speaker_score = float(self._speaker_fusion_probability(similarity).mean().item())
        keyword_score = float(keyword_score)
        if self.speaker_fusion_mode == "min":
            return min(keyword_score, speaker_score)
        if self.speaker_fusion_mode == "product":
            return keyword_score * speaker_score
        if self.speaker_fusion_mode == "mean":
            return (
                self.speaker_fusion_alpha * keyword_score
                + (1.0 - self.speaker_fusion_alpha) * speaker_score
            )
        return keyword_score

    def presence_probability(
        self,
        s1: torch.Tensor,
        enrollment_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one target-present probability per batch item."""

        target_probability, _, _ = self.presence_activity_scores(s1, enrollment_embedding)
        return target_probability

    def presence_activity_scores(
        self,
        s1: torch.Tensor,
        enrollment_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Return target, non-target, and target-minus-non-target scores.

        The margin is populated only for the opt-in three-way personal
        activity head. Other presence heads retain their historical behavior.
        """

        if getattr(self.presence, "requires_enrollment", False):
            if enrollment_embedding is None:
                raise ValueError("Personal presence head requires an enrollment embedding")
            logits = self.presence(s1, enrollment_embedding)
        else:
            logits = self.presence(s1)
        if getattr(self.presence, "is_multiclass", False):
            probabilities = self.presence.activity_probabilities(logits)
            target_frames = probabilities[:, self.presence.target_class_index, :]
            non_target_frames = probabilities[:, 2, :]
            target_probability = self.presence.aggregate_probability(
                target_frames,
                top_fraction=self.presence_top_fraction,
            )
            non_target_probability = self.presence.aggregate_probability(
                non_target_frames,
                top_fraction=self.presence_top_fraction,
            )
            aggregate_margin = getattr(self.presence, "aggregate_margin", None)
            if callable(aggregate_margin):
                activity_margin = aggregate_margin(
                    target_frames,
                    non_target_frames,
                    top_fraction=self.presence_top_fraction,
                    mode=self.presence_activity_margin_mode,
                )
            else:
                activity_margin = target_probability - non_target_probability
            return target_probability, non_target_probability, activity_margin
        if getattr(self.presence, "is_framewise", False):
            aggregate = getattr(self.presence, "aggregate_probability", FramePresenceHead.aggregate_probability)
            return aggregate(
                torch.sigmoid(logits),
                top_fraction=self.presence_top_fraction,
            ), None, None
        return torch.sigmoid(logits), None, None

    def forward(
        self,
        mixture: torch.Tensor,
        enrollment: torch.Tensor,
        state: PipelineState | None = None,
        multichannel: torch.Tensor | None = None,
        pre_extracted: torch.Tensor | None = None,
        playback_ref: torch.Tensor | None = None,
    ) -> tuple[PipelineOutput, PipelineState]:
        """Run one streaming step.

        ``pre_extracted`` is an opt-in evaluation hook for front-end
        ablations.  When supplied, it is treated as the Stage-1 waveform and
        the learned TSE forward pass is bypassed.  This lets a spatial
        beamformer be compared with TSE under the same presence, speaker,
        wake-word, intent, and action state machine.  It is deliberately not
        enabled by any production configuration.
        """
        state = state or PipelineState()
        mixture = mixture.to(self.device)
        enrollment = enrollment.to(self.device)
        if mixture.dim() == 1:
            mixture = mixture.unsqueeze(0)
        if enrollment.dim() == 1:
            enrollment = enrollment.unsqueeze(0)
        if multichannel is not None and multichannel.dim() == 2:
            multichannel = multichannel.unsqueeze(0)
        # ``state.playback_ref`` remains a backwards-compatible static hook.
        # Streaming callers should pass the reference for the current window
        # explicitly so a changing playback signal stays sample-aligned.
        pb = playback_ref if playback_ref is not None else state.playback_ref
        if pb is not None:
            pb = pb.to(self.device)
        with torch.no_grad():
            if pre_extracted is None:
                s1, _, _, _ = self.stage1(
                    mixture,
                    enrollment,
                    multichannel=multichannel.to(self.device) if multichannel is not None else None,
                    playback_ref=pb,
                )
            else:
                s1 = pre_extracted.to(self.device)
                if s1.dim() == 1:
                    s1 = s1.unsqueeze(0)
                if s1.dim() != 2:
                    raise ValueError(
                        "pre_extracted must have shape [samples] or [batch, samples]"
                    )
            if state.reference_speaker_embedding is None:
                speaker_embedder = getattr(self.stage1, "speaker_embedding", None)
                if callable(speaker_embedder):
                    state.reference_speaker_embedding = speaker_embedder(enrollment)
                    if state.reference_speaker_embedding is not None:
                        state.reference_speaker_embedding = state.reference_speaker_embedding.detach()
            reference_speaker_embedding = state.reference_speaker_embedding
            if getattr(self.presence, "requires_enrollment", False):
                if state.presence_enrollment_embedding is None:
                    if reference_speaker_embedding is None:
                        raise ValueError("personal presence head requires a speaker embedding")
                    state.presence_enrollment_embedding = reference_speaker_embedding
                presence_enrollment_embedding = state.presence_enrollment_embedding
            else:
                presence_enrollment_embedding = None
            pres_prob, non_target_prob, activity_margin = self.presence_activity_scores(
                s1, presence_enrollment_embedding
            )
            presence_probability = float(pres_prob.mean().item())
            presence_non_target_probability = (
                None if non_target_prob is None else float(non_target_prob.mean().item())
            )
            presence_activity_margin = (
                None if activity_margin is None else float(activity_margin.mean().item())
            )
            speaker_similarity = None
            active_speaker_gate_threshold = (
                self.command_speaker_gate_threshold
                if state.awaiting_command
                else self.speaker_gate_threshold
            )
            active_presence_threshold = (
                self.command_presence_threshold
                if state.awaiting_command
                else self.presence_threshold
            )
            extracted_speaker_embedding = None
            speaker_score_required = (
                active_speaker_gate_threshold is not None
                or (
                    not state.awaiting_command
                    and self.speaker_fusion_mode != "none"
                    and self.target_kws is not None
                )
            )
            if (
                speaker_score_required
                and presence_probability >= active_presence_threshold
            ):
                similarity = None
                if self.speaker_gate_backend in {"spex", "auto"}:
                    extracted_speaker_embedding = self.stage1.speaker_embedding(s1)
                    if (
                        reference_speaker_embedding is not None
                        and extracted_speaker_embedding is not None
                    ):
                        similarity = torch.nn.functional.cosine_similarity(
                            reference_speaker_embedding,
                            extracted_speaker_embedding,
                            dim=-1,
                        )
                    else:
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
                if similarity is None and self.speaker_gate_backend == "compact":
                    if self.compact_speaker_verifier is None:
                        raise RuntimeError(
                            "speaker_gate_backend='compact' requires "
                            "compact_speaker_verifier_checkpoint"
                        )
                    compact_enrollment = self._prepare_compact_speaker_input(enrollment)
                    compact_extracted = self._prepare_compact_speaker_input(
                        normalize_extracted_waveform(s1)
                    )
                    compact_logits, _, _ = self.compact_speaker_verifier(
                        compact_enrollment,
                        compact_extracted,
                    )
                    # For this backend the exposed speaker_similarity field is
                    # a calibrated verifier probability rather than SpEx cosine
                    # similarity.  The explicit backend/threshold in the config
                    # keeps that change auditable.
                    similarity = torch.sigmoid(compact_logits)
                if similarity is not None:
                    speaker_similarity = float(similarity.mean().item())
            speaker_rejected = (
                speaker_similarity is not None
                and active_speaker_gate_threshold is not None
                and speaker_similarity < active_speaker_gate_threshold
            )
            # A hard command-time speaker threshold is useful for rejecting
            # passenger speech, but it can also reject genuine target
            # commands after extraction when the compact verifier score is
            # low.  The opt-in rescue path lets those borderline frames reach
            # the intent head and only accepts them when the buffered command
            # is sufficiently confident.  Scores below the rescue floor are
            # still rejected before intent inference.
            speaker_rescue_allowed = (
                state.awaiting_command
                and speaker_rejected
                and self.command_speaker_gate_rescue_confidence is not None
                and speaker_similarity is not None
                and speaker_similarity >= self.command_speaker_gate_rescue_floor
            )
            speaker_rejected_for_frontend = speaker_rejected and not speaker_rescue_allowed
            activity_margin_rejected = (
                self.presence_activity_margin_threshold is not None
                and presence_activity_margin is not None
                and presence_activity_margin < self.presence_activity_margin_threshold
            )
            if (
                presence_probability < active_presence_threshold
                or speaker_rejected_for_frontend
                or activity_margin_rejected
            ):
                state.kws_scores.clear()
                if not state.awaiting_command:
                    return PipelineOutput(
                        accepted=False,
                        wake_detected=False,
                        s1=s1,
                        presence_probability=presence_probability,
                        presence_non_target_probability=presence_non_target_probability,
                        presence_activity_margin=presence_activity_margin,
                        speaker_similarity=speaker_similarity,
                    ), state

                # Allow a short pause between the wake word and command.  The
                # timeout is measured in incoming streaming calls so it is
                # independent of the duration of each audio tensor.
                state.command_wait_calls += 1
                if state.command_wait_calls >= self.command_timeout_calls:
                    self._reset_command_capture(state, arm_command_refractory=True)
                    return PipelineOutput(
                        accepted=False,
                        wake_detected=False,
                        s1=s1,
                        presence_probability=presence_probability,
                        presence_non_target_probability=presence_non_target_probability,
                        presence_activity_margin=presence_activity_margin,
                        speaker_similarity=speaker_similarity,
                    ), state
                return PipelineOutput(
                    accepted=False,
                    wake_detected=False,
                    command_pending=True,
                    s1=s1,
                    presence_probability=presence_probability,
                    presence_non_target_probability=presence_non_target_probability,
                    presence_activity_margin=presence_activity_margin,
                    speaker_similarity=speaker_similarity,
                ), state
            downstream_s1 = normalize_extracted_waveform(s1)
            previous_extracted_s1 = state.last_extracted_s1
            # Keep one normalized frame for the next hop.  This is deliberately
            # a one-frame context only; it does not alter the frozen TSE or
            # historical state-machine paths unless the new strategy is
            # explicitly selected.
            state.last_extracted_s1 = downstream_s1.detach().clone()

            if state.awaiting_command:
                # The first accepted target-speech frame after a wake is the
                # command window.  This is deliberately separate from the
                # wake frame; a later PVAD can replace this presence decision
                # without changing the state-machine contract.
                state.command_wait_calls += 1
                if state.command_wait_calls <= self.command_delay_calls:
                    # With sub-second hops, the first post-wake window can
                    # still contain the wake word.  An opt-in delay lets a
                    # streaming caller wait for a clean command window
                    # without changing the historical zero-delay behavior.
                    return PipelineOutput(
                        accepted=True,
                        wake_detected=False,
                        s1=s1,
                        command_pending=True,
                        presence_probability=presence_probability,
                        presence_non_target_probability=presence_non_target_probability,
                        presence_activity_margin=presence_activity_margin,
                        speaker_similarity=speaker_similarity,
                    ), state
                intent_input = self._prepare_intent_input(downstream_s1)
                if self.command_window_strategy in {
                    "kws_trim_latest",
                    "confidence_buffered",
                    "confidence_buffered_hybrid",
                    "prewake_buffered_hybrid",
                }:
                    trimmed = self._prepare_kws_trimmed_intent_input(downstream_s1)
                    if trimmed is None and state.wake_context is not None:
                        trimmed = self._prepare_kws_trimmed_intent_input(state.wake_context)
                elif self.command_window_strategy == "kws_trim" and state.wake_context is not None:
                    trimmed = self._prepare_kws_trimmed_intent_input(state.wake_context)
                else:
                    trimmed = None
                if trimmed is not None:
                    intent_input = trimmed
                candidate_inputs = [intent_input]
                if (
                    self.command_window_strategy
                    in {"confidence_buffered_hybrid", "prewake_buffered_hybrid"}
                    and trimmed is not None
                ):
                    # The fixed crop and endpoint-trimmed crop make different
                    # errors on short, overlapping command windows.  Keep
                    # this dual-crop path explicit and opt-in; it costs an
                    # additional intent forward and is a diagnostic candidate
                    # until its latency and false-action trade-off are tested.
                    candidate_inputs.insert(0, self._prepare_intent_input(downstream_s1))
                if (
                    self.command_window_strategy == "prewake_buffered_hybrid"
                    and state.wake_context is not None
                ):
                    # In simultaneous scenes the wake can be detected after
                    # the target command has already started.  Score the
                    # preceding overlapping frame as an additional candidate
                    # instead of forcing the intent head to start at the
                    # later post-wake window.
                    candidate_inputs.insert(
                        0, self._prepare_intent_input(state.wake_context)
                    )
                candidate_outputs = []
                for candidate_input in candidate_inputs:
                    candidate_logits = self.intent(candidate_input)
                    candidate_probability = torch.softmax(candidate_logits, dim=-1)
                    candidate_confidence = float(
                        candidate_probability.max(dim=-1).values.mean().item()
                    )
                    candidate_outputs.append(
                        (candidate_confidence, candidate_logits, candidate_probability)
                    )
                intent_confidence, intent_logits, intent_probability = max(
                    candidate_outputs, key=lambda item: item[0]
                )
                if self.command_window_strategy in {
                    "confidence_buffered",
                    "confidence_buffered_hybrid",
                    "prewake_buffered_hybrid",
                }:
                    # A short confidence buffer prevents an overlapping,
                    # partially spoken command from being committed before a
                    # later frame contains the complete utterance.  A high
                    # confidence candidate can still commit immediately to
                    # preserve low latency; otherwise we wait for at most the
                    # configured number of candidate evaluations.  This is an
                    # opt-in strategy because its false-action trade-off must
                    # be measured separately on cabin negatives.
                    state.command_candidate_calls += 1
                    if intent_confidence > state.command_best_confidence:
                        state.command_best_confidence = intent_confidence
                        state.command_best_logits = intent_logits.detach().clone()
                    candidate_intent_id = int(torch.argmax(intent_logits, dim=-1).item())
                    state.command_candidate_history.append(
                        (candidate_intent_id, intent_confidence, intent_logits.detach().clone())
                    )
                    reached_limit = state.command_candidate_calls >= self.command_buffer_calls
                    high_confidence = (
                        intent_confidence >= self.command_confidence_commit_threshold
                    )
                    if not reached_limit and not high_confidence:
                        return PipelineOutput(
                            accepted=True,
                            wake_detected=False,
                            intent_logits=intent_logits,
                            action=None,
                            s1=s1,
                            command_pending=True,
                            intent_ready=False,
                            presence_probability=presence_probability,
                            presence_non_target_probability=presence_non_target_probability,
                            presence_activity_margin=presence_activity_margin,
                            speaker_similarity=speaker_similarity,
                        ), state
                    if state.command_best_logits is not None:
                        # At the buffer limit, use the strongest candidate seen
                        # across all post-wake hops, not necessarily the last
                        # (which can be a trailing or partial frame).
                        intent_confidence = state.command_best_confidence
                        intent_logits = state.command_best_logits
                    if self.command_buffer_selection == "mean_logits" and state.command_candidate_history:
                        # Average the logits over the buffered post-wake
                        # candidates.  This is less sensitive to a single
                        # high-confidence partial/overlap frame than selecting
                        # the maximum-confidence hop, while retaining the
                        # existing confidence threshold after aggregation.
                        intent_logits = torch.stack(
                            [logits for _, _, logits in state.command_candidate_history],
                            dim=0,
                        ).mean(dim=0)
                        intent_confidence = float(
                            torch.softmax(intent_logits, dim=-1).max(dim=-1).values.mean().item()
                        )
                    if (
                        self.command_buffer_selection
                        in {"longest_run", "confidence_margin"}
                        and reached_limit
                        and state.command_candidate_history
                    ):
                        # Endpointed command windows can straddle two nearby
                        # intents.  Prefer the class with the longest
                        # consecutive run across the buffered hops; break ties
                        # by the earliest run so a later trailing frame cannot
                        # replace a stable earlier command solely due to a tiny
                        # confidence advantage.
                        best_run: dict[int, tuple[int, int]] = {}
                        best_by_class: dict[int, tuple[float, torch.Tensor]] = {}
                        current_id: int | None = None
                        current_run_length = 0
                        current_run_start = 0
                        for index, (candidate_id, confidence, logits) in enumerate(
                            state.command_candidate_history
                        ):
                            if candidate_id == current_id:
                                current_run_length += 1
                            else:
                                current_id = candidate_id
                                current_run_length = 1
                                current_run_start = index
                            previous_run = best_run.get(candidate_id)
                            current_run = (current_run_length, current_run_start)
                            if previous_run is None or current_run[0] > previous_run[0] or (
                                current_run[0] == previous_run[0]
                                and current_run[1] < previous_run[1]
                            ):
                                best_run[candidate_id] = current_run
                            previous = best_by_class.get(candidate_id)
                            if previous is None or confidence > previous[0]:
                                best_by_class[candidate_id] = (confidence, logits)
                        selected_id = max(
                            best_run,
                            key=lambda candidate_id: (
                                best_run[candidate_id][0],
                                -best_run[candidate_id][1],
                            ),
                        )
                        run_confidence, run_logits = best_by_class[selected_id]
                        if (
                            self.command_buffer_selection == "confidence_margin"
                            and state.command_best_logits is not None
                            and state.command_best_confidence - run_confidence
                            >= self.command_buffer_confidence_margin
                        ):
                            intent_confidence = state.command_best_confidence
                            intent_logits = state.command_best_logits
                        else:
                            intent_confidence = run_confidence
                            intent_logits = run_logits
                    if (
                        self.intent_confidence_threshold is not None
                        and intent_confidence < self.intent_confidence_threshold
                    ):
                        self._reset_command_capture(state, arm_command_refractory=True)
                        state.kws_scores.clear()
                        return PipelineOutput(
                            accepted=True,
                            wake_detected=False,
                            intent_logits=intent_logits,
                            action=None,
                            s1=s1,
                            command_pending=False,
                            intent_ready=False,
                            command_suppressed=True,
                            presence_probability=presence_probability,
                            presence_non_target_probability=presence_non_target_probability,
                            presence_activity_margin=presence_activity_margin,
                            speaker_similarity=speaker_similarity,
                        ), state
                    if (
                        speaker_rescue_allowed
                        and self.command_speaker_gate_rescue_confidence is not None
                        and intent_confidence < self.command_speaker_gate_rescue_confidence
                    ):
                        self._reset_command_capture(state, arm_command_refractory=True)
                        state.kws_scores.clear()
                        return PipelineOutput(
                            accepted=True,
                            wake_detected=False,
                            intent_logits=intent_logits,
                            action=None,
                            s1=s1,
                            command_pending=False,
                            intent_ready=False,
                            command_suppressed=True,
                            presence_probability=presence_probability,
                            presence_non_target_probability=presence_non_target_probability,
                            presence_activity_margin=presence_activity_margin,
                            speaker_similarity=speaker_similarity,
                        ), state
                elif (
                    self.intent_confidence_threshold is not None
                    and intent_confidence < self.intent_confidence_threshold
                ):
                    self._reset_command_capture(state, arm_command_refractory=True)
                    state.kws_scores.clear()
                    return PipelineOutput(
                        accepted=True,
                        wake_detected=False,
                        intent_logits=intent_logits,
                        action=None,
                        s1=s1,
                        command_pending=False,
                        intent_ready=False,
                        command_suppressed=True,
                        presence_probability=presence_probability,
                        presence_non_target_probability=presence_non_target_probability,
                        presence_activity_margin=presence_activity_margin,
                        speaker_similarity=speaker_similarity,
                    ), state
                intent_id = str(int(torch.argmax(intent_logits, dim=-1).item()))
                action = intent_to_action(intent_id, self.label_map)
                if action.reply_wav and action.reply_wav.exists():
                    state.last_reply_path = action.reply_wav
                self._reset_command_capture(state)
                state.kws_scores.clear()
                state.post_action_refractory_left = self.post_action_refractory_calls
                return PipelineOutput(
                    accepted=True,
                    wake_detected=False,
                    intent_logits=intent_logits,
                    action=action,
                    s1=s1,
                    command_pending=False,
                    intent_ready=True,
                    presence_probability=presence_probability,
                    presence_non_target_probability=presence_non_target_probability,
                    presence_activity_margin=presence_activity_margin,
                    speaker_similarity=speaker_similarity,
                ), state

            kws_input = self._prepare_kws_input(downstream_s1)
            target_kws_score = None
            target_kws_probability = None
            if self.target_kws is not None:
                embedder = getattr(self.stage1, "speaker_embedding", None)
                if callable(embedder):
                    if state.target_kws_enrollment_embedding is None:
                        if reference_speaker_embedding is None:
                            raise ValueError("target KWS requires a speaker embedding")
                        state.target_kws_enrollment_embedding = reference_speaker_embedding
                    if extracted_speaker_embedding is None:
                        extracted_speaker_embedding = embedder(s1).detach()
                    else:
                        extracted_speaker_embedding = extracted_speaker_embedding.detach()
                    target_kws_logits = self.target_kws(
                        kws_input,
                        state.target_kws_enrollment_embedding,
                        extracted_speaker_embedding,
                    )
                    target_kws_prob = torch.softmax(target_kws_logits, dim=-1)
                    target_kws_score = target_kws_prob[..., self.wake_class_id].mean().item()
                    if self.target_kws_secondary is not None:
                        secondary_logits = self.target_kws_secondary(
                            kws_input,
                            state.target_kws_enrollment_embedding,
                            extracted_speaker_embedding,
                        )
                        secondary_score = torch.softmax(secondary_logits, dim=-1)[..., self.wake_class_id].mean().item()
                        if self.target_kws_fusion_mode == "mean":
                            target_kws_score = 0.5 * (target_kws_score + secondary_score)
                        elif self.target_kws_fusion_mode == "geomean":
                            target_kws_score = float(max(target_kws_score, 0.0) * max(secondary_score, 0.0)) ** 0.5
                        else:
                            target_kws_score = min(target_kws_score, secondary_score)
                    target_kws_probability = target_kws_score
                    if (
                        self.speaker_fusion_mode != "none"
                        and similarity is not None
                    ):
                        target_kws_score = self._fuse_speaker_score(
                            target_kws_score,
                            similarity,
                        )
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
                    presence_non_target_probability=presence_non_target_probability,
                    presence_activity_margin=presence_activity_margin,
                    speaker_similarity=speaker_similarity,
                    wake_probability=score,
                    target_kws_probability=target_kws_probability,
                ), state

            # Do not infer intent from the wake-word frame.  The next accepted
            # frame enters the explicit command-capture branch above.
            state.awaiting_command = True
            state.command_wait_calls = 0
            state.command_candidate_calls = 0
            if self.command_window_strategy == "kws_trim":
                state.wake_context = downstream_s1.detach().clone()
            elif self.command_window_strategy == "prewake_buffered_hybrid":
                state.wake_context = (
                    downstream_s1.detach().clone()
                    if previous_extracted_s1 is None
                    else previous_extracted_s1.detach().clone()
                )
            state.kws_scores.clear()
            return PipelineOutput(
                accepted=True,
                wake_detected=True,
                s1=s1,
                command_pending=True,
                presence_probability=presence_probability,
                presence_non_target_probability=presence_non_target_probability,
                presence_activity_margin=presence_activity_margin,
                speaker_similarity=speaker_similarity,
                wake_probability=score,
                target_kws_probability=target_kws_probability,
            ), state
