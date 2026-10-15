"""End-to-end inference: SpEx+ -> presence -> KWS -> post-wake intent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from car_tse.models.intent_classifier import IntentClassifier
from car_tse.models.kws_model import KWSModel, TargetSpeakerKWSModel
from car_tse.models.presence_head import FramePresenceHead, PresenceHead
from car_tse.models.stage23_fusion import Stage23FusionHead
from car_tse.models.stage1_system import Stage1System
from car_tse.models.wav2vec_intent import Wav2VecIntentClassifier
from car_tse.pipeline.actions import ActionResult, intent_to_action
from car_tse.train.utils import load_checkpoint


@dataclass
class PipelineState:
    playback_ref: torch.Tensor | None = None
    last_reply_path: Path | None = None
    kws_scores: list[float] = field(default_factory=list)
    # Source-time starts paired with ``kws_scores``. These let a delayed
    # multi-hit wake decision backdate command capture to the end of the first
    # supporting crop rather than dropping speech heard during later checks.
    kws_window_start_samples: list[int | None] = field(default_factory=list)
    kws_refractory_left: int = 0
    # A wake word arms a separate command-capture phase.  Keeping this in the
    # streaming state prevents intent from being inferred from the wake-word
    # window itself.
    awaiting_command: bool = False
    command_wait_calls: int = 0
    command_elapsed_samples: int = 0
    command_timeout_samples: int | None = None
    command_silence_calls: int = 0
    command_new_samples: int = 0
    command_audio: torch.Tensor | None = None
    # Timestamped extracted-audio history is populated before presence/speaker
    # gates run.  That lets a later wake decision recover command audio from a
    # chunk that an earlier gate declined to append to command_audio.
    audio_history: torch.Tensor | None = None
    audio_history_start_sample: int | None = None
    stream_sample_cursor: int = 0
    command_capture_start_sample: int | None = None
    last_audio_end_sample: int | None = None
    last_call_elapsed_samples: int = 0
    speaker_embedding: torch.Tensor | None = None
    target_kws_enrollment_embedding: torch.Tensor | None = None


@dataclass
class PipelineOutput:
    accepted: bool
    wake_detected: bool
    intent_logits: torch.Tensor | None = None
    intent_confidence: float | None = None
    action: ActionResult | None = None
    s1: torch.Tensor | None = None
    command_pending: bool = False
    intent_ready: bool = False
    presence_probability: float | None = None
    speaker_similarity: float | None = None
    wake_probability: float | None = None
    wake_model_probability: float | None = None


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
    window_start_sample: int | None = None,
) -> bool:
    window_size = max(1, int(window_size))
    required_hits = max(1, min(int(required_hits), window_size))
    refractory_calls = max(0, int(refractory_calls))
    state.kws_scores.append(float(score))
    state.kws_scores = state.kws_scores[-window_size:]
    state.kws_window_start_samples.append(
        None if window_start_sample is None else max(0, int(window_start_sample))
    )
    state.kws_window_start_samples = state.kws_window_start_samples[-window_size:]
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
        stage23_fusion: Stage23FusionHead | None = None,
        stage23_fusion_threshold: float = 0.5,
        stage23_joint_enabled: bool = False,
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
        command_timeout_seconds: float | None = None,
        command_end_silence_calls: int = 2,
        command_min_speech_seconds: float = 0.25,
        command_capture_preroll_seconds: float = 0.25,
        intent_action_threshold: float | None = None,
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
        self.stage23_fusion = stage23_fusion
        self.stage23_fusion_threshold = float(stage23_fusion_threshold)
        self.stage23_joint_enabled = bool(stage23_joint_enabled)
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
        # Existing configs use `command_timeout_calls: 4`; retain that value
        # as a four-second default while allowing callers to set an explicit
        # timeout. The timeout itself is always tracked in audio samples.
        self.command_timeout_seconds = max(
            0.01,
            float(command_timeout_calls if command_timeout_seconds is None else command_timeout_seconds),
        )
        self.command_end_silence_calls = max(1, int(command_end_silence_calls))
        self.command_min_speech_seconds = max(0.0, float(command_min_speech_seconds))
        self.command_capture_preroll_seconds = max(0.0, float(command_capture_preroll_seconds))
        self.intent_action_threshold = (
            None if intent_action_threshold is None else min(max(float(intent_action_threshold), 0.0), 1.0)
        )
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
        if self.stage23_fusion is not None:
            self.stage23_fusion.to(self.device)
        self.intent.to(self.device)
        self.stage1.eval()
        self.presence.eval()
        self.kws.eval()
        if self.kws_secondary is not None:
            self.kws_secondary.eval()
        if self.target_kws is not None:
            self.target_kws.eval()
        if self.stage23_fusion is not None:
            self.stage23_fusion.eval()
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
        stage23_fusion_checkpoint = pipeline_kwargs.pop("stage23_fusion_checkpoint", None)
        stage23_fusion = None
        if stage23_fusion_checkpoint:
            stage23_fusion = Stage23FusionHead(
                hidden=int(pipeline_kwargs.pop("stage23_fusion_hidden", 8))
            )
            load_checkpoint(stage23_fusion_checkpoint, stage23_fusion)
        intent_input_seconds = float(pipeline_kwargs.get("intent_input_seconds", 2.0))
        if intent_type == "wav2vec":
            if not intent_encoder_checkpoint:
                raise ValueError("intent_encoder_checkpoint is required for wav2vec intent")
            intent_head_hidden = int(pipeline_kwargs.pop("intent_head_hidden", 512))
            intent = Wav2VecIntentClassifier(
                encoder_name=intent_encoder,
                encoder_checkpoint=intent_encoder_checkpoint,
                head_checkpoint=ckpt_intent,
                num_classes=intent_num_classes,
                input_sample_rate=intent_sample_rate,
                input_seconds=intent_input_seconds,
                head_hidden=intent_head_hidden,
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
            stage23_fusion=stage23_fusion,
            kws_sample_rate=kws_sample_rate,
            **pipeline_kwargs,
        )

    def _reset_command_capture(self, state: PipelineState) -> None:
        state.awaiting_command = False
        state.command_wait_calls = 0
        state.command_elapsed_samples = 0
        state.command_timeout_samples = None
        state.command_silence_calls = 0
        state.command_new_samples = 0
        state.command_audio = None
        state.command_capture_start_sample = None

    def _remember_audio(
        self,
        state: PipelineState,
        audio: torch.Tensor,
        start_sample: int,
    ) -> None:
        """Keep recent extracted audio independently of gate decisions."""
        if audio.shape[-1] == 0:
            return
        start_sample = max(0, int(start_sample))
        if state.audio_history is None or state.audio_history_start_sample is None:
            state.audio_history = audio.detach().clone()
            state.audio_history_start_sample = start_sample
        else:
            history_end = state.audio_history_start_sample + state.audio_history.shape[-1]
            if start_sample > history_end:
                # Preserve source-time alignment across real gaps.  Streaming
                # callers normally provide contiguous chunks; zeros represent
                # unavailable intervals rather than shifting later speech.
                gap = start_sample - history_end
                state.audio_history = torch.cat(
                    (state.audio_history, audio.new_zeros((*audio.shape[:-1], gap)), audio), dim=-1
                )
            elif start_sample < history_end:
                overlap = history_end - start_sample
                if overlap < audio.shape[-1]:
                    state.audio_history = torch.cat(
                        (state.audio_history, audio[..., overlap:].detach()), dim=-1
                    )
            else:
                state.audio_history = torch.cat((state.audio_history, audio.detach()), dim=-1)

        # Keep enough history for the wake context, the full command, and a
        # small scheduling margin, without allowing an indefinitely growing buffer.
        max_history = max(
            1,
            int(round(self.intent_sample_rate * (self.command_timeout_seconds + self.intent_input_seconds + 1.0))),
        )
        if state.audio_history.shape[-1] > max_history:
            trim = state.audio_history.shape[-1] - max_history
            state.audio_history = state.audio_history[..., trim:]
            state.audio_history_start_sample += trim

    def _history_from(self, state: PipelineState, start_sample: int) -> torch.Tensor | None:
        if state.audio_history is None or state.audio_history_start_sample is None:
            return None
        offset = max(0, int(start_sample) - state.audio_history_start_sample)
        if offset >= state.audio_history.shape[-1]:
            return state.audio_history[..., :0]
        return state.audio_history[..., offset:]

    def _advance_command_clock(self, state: PipelineState) -> bool:
        elapsed = max(0, int(state.last_call_elapsed_samples))
        if state.command_timeout_samples is None:
            state.command_timeout_samples = max(
                1,
                int(round(self.intent_sample_rate * self.command_timeout_seconds)),
            )
        state.command_elapsed_samples += elapsed
        return (
            state.command_timeout_samples is not None
            and state.command_elapsed_samples >= state.command_timeout_samples
        )

    def _new_audio_segment(
        self,
        wav: torch.Tensor,
        state: PipelineState,
        audio_start_sample: int | None,
        input_num_samples: int,
    ) -> torch.Tensor:
        """Return only samples not seen in earlier timestamped sliding windows.

        Callers that pass rolling windows should provide each window's start
        sample. Callers that pass consecutive chunks may omit it; each chunk
        is then treated as new audio, preserving the original streaming API.
        """

        if audio_start_sample is None:
            state.last_call_elapsed_samples = max(0, int(input_num_samples))
            return wav
        start = max(0, int(audio_start_sample))
        end = start + max(0, int(input_num_samples))
        previous_end = state.last_audio_end_sample
        overlap = 0 if previous_end is None else max(0, previous_end - start)
        state.last_audio_end_sample = end if previous_end is None else max(previous_end, end)
        state.last_call_elapsed_samples = (
            max(0, int(input_num_samples))
            if previous_end is None
            else max(0, end - previous_end)
        )
        if overlap >= input_num_samples:
            return wav[..., :0]
        return wav[..., min(overlap, wav.shape[-1]):]

    def _infer_command(
        self,
        state: PipelineState,
        presence_probability: float,
        speaker_similarity: float | None,
        s1: torch.Tensor,
    ) -> PipelineOutput:
        assert state.command_audio is not None
        intent_input = self._prepare_intent_input(state.command_audio)
        intent_logits = self.intent(intent_input)
        probabilities = torch.softmax(intent_logits, dim=-1)
        predicted_id = int(torch.argmax(intent_logits, dim=-1).item())
        intent_confidence = float(probabilities.max(dim=-1).values.item())
        intent_id = str(predicted_id)
        if self.intent_action_threshold is not None and intent_confidence < self.intent_action_threshold:
            # Small-vocabulary heads can explicitly abstain on uncertain
            # speech. Map the abstention to a configured no-op class if one is
            # present; otherwise preserve legacy argmax behavior.
            no_action_ids = [
                key for key, value in (self.label_map or {}).items()
                if str(value).lower() in {"noop", "no_op", "no_command", "abstain"}
            ]
            if no_action_ids:
                intent_id = str(no_action_ids[0])
        action = intent_to_action(intent_id, self.label_map)
        if action.reply_wav and action.reply_wav.exists():
            state.last_reply_path = action.reply_wav
        self._reset_command_capture(state)
        state.kws_scores.clear()
        state.kws_window_start_samples.clear()
        return PipelineOutput(
            accepted=True,
            wake_detected=False,
            intent_logits=intent_logits,
            intent_confidence=intent_confidence,
            action=action,
            s1=s1,
            command_pending=False,
            intent_ready=True,
            presence_probability=presence_probability,
            speaker_similarity=speaker_similarity,
        )

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

    def _presence_statistics(
        self,
        s1: torch.Tensor,
        tail_start_samples: int,
        fresh_start_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Return whole-window, post-KWS, and newly arrived audio presence."""

        logits = self.presence(s1)
        if not getattr(self.presence, "is_framewise", False):
            return torch.sigmoid(logits), None, None
        frame_probability = torch.sigmoid(logits)
        overall = FramePresenceHead.aggregate_probability(
            frame_probability,
            top_fraction=self.presence_top_fraction,
        )
        hop = max(1, int(getattr(self.presence.mel, "hop", 1)))

        def region_probability(start_samples: int) -> torch.Tensor:
            first_frame = min(
                frame_probability.shape[-1],
                (max(0, int(start_samples)) + hop - 1) // hop,
            )
            region = frame_probability[..., first_frame:]
            if region.shape[-1] == 0:
                return torch.zeros_like(overall)
            return FramePresenceHead.aggregate_probability(
                region,
                top_fraction=self.presence_top_fraction,
            )

        return overall, region_probability(tail_start_samples), region_probability(fresh_start_samples)

    def forward(
        self,
        mixture: torch.Tensor,
        enrollment: torch.Tensor,
        state: PipelineState | None = None,
        multichannel: torch.Tensor | None = None,
        audio_start_sample: int | None = None,
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
            kws_samples = (
                max(1, int(round(self.kws_sample_rate * self.kws_input_seconds)))
                if self.kws_input_seconds > 0
                else 0
            )
            downstream_s1 = normalize_extracted_waveform(s1)
            if audio_start_sample is None:
                new_audio_start_sample = int(state.stream_sample_cursor)
            else:
                timestamp_start = max(0, int(audio_start_sample))
                previous_end = state.last_audio_end_sample
                new_audio_start_sample = (
                    timestamp_start
                    if previous_end is None
                    else max(timestamp_start, previous_end)
                )
            new_downstream_audio = self._new_audio_segment(
                downstream_s1,
                state,
                audio_start_sample,
                mixture.shape[-1],
            )
            self._remember_audio(state, new_downstream_audio, new_audio_start_sample)
            state.stream_sample_cursor = max(
                state.stream_sample_cursor,
                new_audio_start_sample + int(new_downstream_audio.shape[-1]),
            )
            if state.awaiting_command and state.command_capture_start_sample is not None:
                recovered = self._history_from(state, state.command_capture_start_sample)
                if recovered is not None:
                    state.command_audio = recovered
            fresh_start_samples = max(0, s1.shape[-1] - new_downstream_audio.shape[-1])
            pres_prob, tail_presence_prob, fresh_presence_prob = self._presence_statistics(
                s1,
                kws_samples,
                fresh_start_samples,
            )
            presence_probability = float(pres_prob.mean().item())
            speaker_similarity = None
            joint_fusion_enabled = self.stage23_fusion is not None or self.stage23_joint_enabled
            active_presence_threshold = (
                self.command_presence_threshold
                if state.awaiting_command
                else self.presence_threshold
            )
            if (
                (self.speaker_gate_threshold is not None or joint_fusion_enabled)
                and (presence_probability >= active_presence_threshold or joint_fusion_enabled)
            ):
                similarity = None
                if self.speaker_gate_backend in {"spex", "auto"}:
                    speaker_similarity_fn = getattr(self.stage1, "speaker_similarity", None)
                    if callable(speaker_similarity_fn):
                        similarity = speaker_similarity_fn(enrollment, s1)
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
                and self.speaker_gate_threshold is not None
                and speaker_similarity < self.speaker_gate_threshold
            )
            command_region_rejected = (
                state.awaiting_command
                and fresh_presence_prob is not None
                and (
                    new_downstream_audio.shape[-1] == 0
                    or float(fresh_presence_prob.mean().item()) < self.command_presence_threshold
                )
            )
            if (
                presence_probability < active_presence_threshold
                or speaker_rejected
                or command_region_rejected
            ) and not (joint_fusion_enabled and not state.awaiting_command):
                state.kws_scores.clear()
                state.kws_window_start_samples.clear()
                if not state.awaiting_command:
                    state.kws_refractory_left = 0
                    return PipelineOutput(
                        accepted=False,
                        wake_detected=False,
                        s1=s1,
                        presence_probability=presence_probability,
                        speaker_similarity=speaker_similarity,
                    ), state

                # Allow a short pause between the wake word and command. The
                # timeout uses samples, so chunk sizes do not change its length.
                state.command_wait_calls += 1
                timeout_reached = self._advance_command_clock(state)
                minimum_speech_samples = (
                    max(1, int(round(self.intent_sample_rate * self.command_min_speech_seconds)))
                    if self.command_min_speech_seconds > 0
                    else 0
                )
                if (
                    state.command_audio is not None
                    and state.command_audio.shape[-1] > 0
                    and state.command_new_samples >= minimum_speech_samples
                ):
                    state.command_silence_calls += 1
                    if state.command_silence_calls >= self.command_end_silence_calls:
                        return self._infer_command(
                            state,
                            presence_probability,
                            speaker_similarity,
                            s1,
                        ), state
                if timeout_reached:
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
            if state.awaiting_command:
                # Accumulate only new audio after wake. With timestamped
                # sliding windows, overlap is removed before samples are
                # appended. This aligns the fixed-length intent crop to the
                # post-wake command instead of the front of an arbitrary
                # context window.
                state.command_wait_calls += 1
                timeout_reached = self._advance_command_clock(state)
                state.command_silence_calls = 0
                if new_downstream_audio.shape[-1] > 0:
                    state.command_new_samples += int(new_downstream_audio.shape[-1])
                required_samples = max(1, int(round(self.intent_sample_rate * self.intent_input_seconds)))
                if state.command_audio is not None and state.command_audio.shape[-1] >= required_samples:
                    return self._infer_command(
                        state,
                        presence_probability,
                        speaker_similarity,
                        s1,
                    ), state
                if timeout_reached:
                    self._reset_command_capture(state)
                    state.kws_refractory_left = 0
                    return PipelineOutput(
                        accepted=False,
                        wake_detected=False,
                        s1=s1,
                        presence_probability=presence_probability,
                        speaker_similarity=speaker_similarity,
                    ), state
                state.kws_scores.clear()
                state.kws_window_start_samples.clear()
                return PipelineOutput(
                    accepted=True,
                    wake_detected=False,
                    s1=s1,
                    command_pending=True,
                    intent_ready=False,
                    presence_probability=presence_probability,
                    speaker_similarity=speaker_similarity,
                ), state

            kws_input = downstream_s1
            if self.kws_input_seconds > 0:
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
            wake_model_probability = float(score)
            if self.stage23_fusion is not None:
                p_tensor = torch.tensor(presence_probability, device=self.device)
                w_tensor = torch.tensor(wake_model_probability, device=self.device)
                s_tensor = (
                    None
                    if speaker_similarity is None
                    else torch.tensor(speaker_similarity, device=self.device)
                )
                fused_logit = self.stage23_fusion(p_tensor, w_tensor, s_tensor)
                score = float(torch.sigmoid(fused_logit).item())
            wake = update_kws_state(
                state,
                score,
                self.stage23_fusion_threshold if self.stage23_fusion is not None else self.kws_threshold,
                self.kws_smoothing_window,
                self.kws_required_hits,
                self.kws_refractory_calls,
                window_start_sample=(
                    max(0, int(audio_start_sample))
                    if audio_start_sample is not None
                    else int(new_audio_start_sample)
                ),
            )
            if not wake:
                return PipelineOutput(
                    accepted=True,
                    wake_detected=False,
                    s1=s1,
                    presence_probability=presence_probability,
                    speaker_similarity=speaker_similarity,
                    wake_probability=score,
                    wake_model_probability=wake_model_probability,
                ), state

            # Do not infer intent from the wake-word frame.  The next accepted
            # frame enters the explicit command-capture branch above.
            state.awaiting_command = True
            state.command_wait_calls = 0
            state.command_elapsed_samples = 0
            state.command_timeout_samples = None
            state.command_silence_calls = 0
            state.command_new_samples = 0
            # If the current input is shorter than the configured KWS window,
            # the remainder was padding rather than source audio. Start at the
            # end of received audio so the next real samples are retained.
            observed_kws_samples = min(kws_samples, int(downstream_s1.shape[-1]))
            wake_window_start_sample = (
                max(0, int(audio_start_sample))
                if audio_start_sample is not None
                else new_audio_start_sample
            )
            # The decision may arrive only after several overlapping KWS
            # windows have been scored. Starting after the *latest* window
            # can discard a command spoken while confirmation was pending.
            # Begin after the earliest positive supporting window, with a
            # small configurable pre-roll to preserve commands that start
            # just before that boundary.
            positive_starts = [
                start for score_value, start in zip(state.kws_scores, state.kws_window_start_samples)
                if score_value >= (self.stage23_fusion_threshold if self.stage23_fusion is not None else self.kws_threshold)
                and start is not None
            ]
            capture_anchor = min(positive_starts) if positive_starts else wake_window_start_sample
            preroll_samples = int(round(self.command_capture_preroll_seconds * self.intent_sample_rate))
            state.command_capture_start_sample = max(
                0, capture_anchor + observed_kws_samples - preroll_samples
            )
            # KWS scores the first KWS-sized region. Capture the following
            # source-time audio from the rolling history regardless of whether
            # the presence gate localized that tail as speech; the gate may be
            # exactly why the command was previously lost.
            state.command_audio = self._history_from(
                state,
                state.command_capture_start_sample,
            )
            state.kws_scores.clear()
            state.kws_window_start_samples.clear()
            return PipelineOutput(
                accepted=True,
                wake_detected=True,
                s1=s1,
                command_pending=True,
                presence_probability=presence_probability,
                speaker_similarity=speaker_similarity,
                wake_probability=score,
                wake_model_probability=wake_model_probability,
            ), state
