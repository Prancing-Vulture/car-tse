"""ONNX Runtime composition for the retained in-car assistant graphs.

The exporter and golden harness validate individual graphs.  This module
composes the same graphs in the same order as :class:`InCarPipeline` and keeps
the wake-to-command state machine explicit.  It is an embedded-runtime bridge,
not a claim that a vendor-specific target SoC or C++ integration is complete.

The retained contract is deliberately strict: four microphone channels and a
sample-aligned playback reference feed Stage 1; enrollment is a fixed 1.5 s
window; Stage 1 emits a 3 s extracted window; downstream models use the fixed
windows recorded in the ONNX manifest.  Shape errors fail early instead of
silently cropping a production capture incorrectly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from car_tse.pipeline.actions import intent_to_action
from car_tse.pipeline.in_car_pipeline import (
    PipelineOutput,
    PipelineState,
    update_kws_state,
)


CORE_COMPONENTS = ("stage1", "kws", "presence", "target_kws", "compact_verifier", "intent")
REQUIRED_COMPONENTS = CORE_COMPONENTS + ("speaker_embedding",)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-np.clip(value, -80.0, 80.0)))


def _softmax(value: np.ndarray, axis: int = -1) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    shifted = value - np.max(value, axis=axis, keepdims=True)
    exponent = np.exp(np.clip(shifted, -80.0, 80.0))
    return exponent / np.maximum(exponent.sum(axis=axis, keepdims=True), 1.0e-12)


def _as_audio_batch(value: np.ndarray | torch.Tensor, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [samples] or [batch, samples], got {array.shape}")
    return np.ascontiguousarray(array)


def _fit_audio(value: np.ndarray, samples: int, *, name: str) -> np.ndarray:
    value = _as_audio_batch(value, name=name)
    samples = int(samples)
    if samples <= 0:
        raise ValueError(f"{name} requires a positive sample count")
    if value.shape[-1] == samples:
        return value
    if value.shape[-1] > samples:
        return np.ascontiguousarray(value[..., :samples])
    return np.pad(value, ((0, 0), (0, samples - value.shape[-1])))


def _normalize_extracted(value: np.ndarray, peak_limit: float = 0.95) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    peak = np.max(np.abs(value), axis=-1, keepdims=True)
    scale = np.where(peak > peak_limit, peak_limit / np.maximum(peak, 1.0e-8), 1.0)
    return np.ascontiguousarray(value * scale.astype(np.float32))


def _aggregate_top_fraction(probability: np.ndarray, top_fraction: float) -> float:
    probability = np.asarray(probability, dtype=np.float32)
    if probability.ndim == 1:
        probability = probability[None, :]
    fraction = min(max(float(top_fraction), 1.0e-3), 1.0)
    top_k = max(1, int(round(probability.shape[-1] * fraction)))
    return float(np.sort(probability, axis=-1)[..., -top_k:].mean())


def _torch_batch(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))


class OnnxComponentRuntime:
    """Load and compose the retained ONNX component graphs."""

    def __init__(
        self,
        manifest_path: Path | str,
        *,
        providers: Sequence[str] | None = None,
        require_playback_reference: bool = True,
        intra_op_num_threads: int | None = None,
        inter_op_num_threads: int | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - optional deployment dependency
            raise RuntimeError(
                "ONNX Runtime is required for OnnxComponentRuntime; install the "
                "project's 'embedded' optional dependencies."
            ) from exc

        self.manifest_path = Path(manifest_path).resolve()
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        rows = payload.get("components", [])
        if not rows and payload.get("component"):
            rows = [payload]
        self.manifest = payload
        self.require_playback_reference = bool(require_playback_reference)
        self.intra_op_num_threads = (
            None if intra_op_num_threads is None else max(1, int(intra_op_num_threads))
        )
        self.inter_op_num_threads = (
            None if inter_op_num_threads is None else max(1, int(inter_op_num_threads))
        )
        provider_list = list(providers or ["CPUExecutionProvider"])
        # The retained graphs have intentionally different fixed windows
        # (1.0/1.5/2.0/3.0 s).  ORT's memory-pattern planner otherwise tries to
        # reuse allocations across those calls and emits shape-mismatch
        # warnings in a streaming loop.  Disabling that planner makes the
        # variable-window contract explicit; target-device sessions should
        # choose their own allocator after profiling.
        session_options = ort.SessionOptions()
        session_options.enable_mem_pattern = False
        if self.intra_op_num_threads is not None:
            session_options.intra_op_num_threads = self.intra_op_num_threads
        if self.inter_op_num_threads is not None:
            session_options.inter_op_num_threads = self.inter_op_num_threads
        self.sessions: dict[str, Any] = {}
        self.rows: dict[str, dict[str, Any]] = {}
        for row in rows:
            component = str(row.get("component", ""))
            if component not in REQUIRED_COMPONENTS:
                continue
            model_path = Path(str(row["output"]))
            if not model_path.is_absolute():
                model_path = self.manifest_path.parent / model_path
            model_path = model_path.resolve()
            if not model_path.is_file():
                raise FileNotFoundError(f"ONNX model for {component} is missing: {model_path}")
            self.sessions[component] = ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=provider_list,
            )
            self.rows[component] = dict(row)
        missing = [component for component in REQUIRED_COMPONENTS if component not in self.sessions]
        if missing:
            raise ValueError(f"ONNX manifest is missing required components: {missing}")
        self._validate_contract()

    def _validate_contract(self) -> None:
        stage1_inputs = [meta.name for meta in self.sessions["stage1"].get_inputs()]
        expected = {"multichannel_audio", "enrollment_audio", "playback_reference"}
        if set(stage1_inputs) != expected:
            raise ValueError(
                "Stage 1 ONNX input contract mismatch: "
                f"expected {sorted(expected)}, got {stage1_inputs}"
            )
        speaker_inputs = [meta.name for meta in self.sessions["speaker_embedding"].get_inputs()]
        if speaker_inputs != ["enrollment_audio", "extracted_audio"]:
            raise ValueError(
                "speaker embedding ONNX input contract mismatch: "
                f"got {speaker_inputs}"
            )
        stage1_shape = self.sessions["stage1"].get_inputs()[0].shape
        if len(stage1_shape) != 3 or isinstance(stage1_shape[1], str) or int(stage1_shape[1]) != 4:
            raise ValueError(f"Stage 1 must expose four microphone channels, got {stage1_shape}")

    @staticmethod
    def _time_shape(session: Any, input_name: str, fallback: int | None = None) -> int | None:
        for meta in session.get_inputs():
            if meta.name != input_name:
                continue
            shape = meta.shape
            if len(shape) >= 2 and isinstance(shape[-1], (int, np.integer)):
                return int(shape[-1])
            return fallback
        raise ValueError(f"ONNX session has no input named {input_name}")

    def _run(self, component: str, values: Mapping[str, np.ndarray]) -> np.ndarray:
        session = self.sessions[component]
        names = [meta.name for meta in session.get_inputs()]
        missing = [name for name in names if name not in values]
        if missing:
            raise ValueError(f"missing {component} ONNX inputs: {missing}")
        feed = {name: np.ascontiguousarray(values[name], dtype=np.float32) for name in names}
        return np.asarray(session.run(None, feed)[0], dtype=np.float32)

    def run_components(
        self,
        multichannel_audio: np.ndarray | torch.Tensor,
        enrollment_audio: np.ndarray | torch.Tensor,
        playback_reference: np.ndarray | torch.Tensor | None = None,
        *,
        include_intent: bool = True,
    ) -> dict[str, np.ndarray | None]:
        """Run all retained graphs once and return their handoff tensors."""

        if isinstance(multichannel_audio, torch.Tensor):
            multichannel_audio = multichannel_audio.detach().cpu().numpy()
        multichannel = np.asarray(multichannel_audio, dtype=np.float32)
        if multichannel.ndim == 2:
            if multichannel.shape[0] == 4:
                multichannel = multichannel[None, ...]
            elif multichannel.shape[1] == 4:
                multichannel = multichannel.T[None, ...]
        if multichannel.ndim != 3 or multichannel.shape[1] != 4:
            raise ValueError(
                "multichannel_audio must have shape [4, samples], [samples, 4], "
                f"or [batch, 4, samples], got {multichannel.shape}"
            )
        if multichannel.shape[0] != 1:
            raise ValueError(
                "the retained stateful ONNX contract currently supports batch size 1, "
                f"got {multichannel.shape[0]}"
            )
        multichannel = np.ascontiguousarray(multichannel)
        stage1_session = self.sessions["stage1"]
        stage1_time = self._time_shape(stage1_session, "multichannel_audio")
        if stage1_time is not None:
            if multichannel.shape[-1] > stage1_time:
                multichannel = multichannel[..., :stage1_time]
            elif multichannel.shape[-1] < stage1_time:
                multichannel = np.pad(multichannel, ((0, 0), (0, 0), (0, stage1_time - multichannel.shape[-1])))

        enrollment_time = self._time_shape(stage1_session, "enrollment_audio")
        enrollment = _as_audio_batch(enrollment_audio, name="enrollment_audio")
        if enrollment_time is not None:
            enrollment = _fit_audio(enrollment, enrollment_time, name="enrollment_audio")
        if enrollment.shape[0] != multichannel.shape[0]:
            if enrollment.shape[0] == 1:
                enrollment = np.repeat(enrollment, multichannel.shape[0], axis=0)
            else:
                raise ValueError("multichannel and enrollment batch sizes do not match")

        if playback_reference is None:
            if self.require_playback_reference:
                raise ValueError(
                    "playback_reference is required by the production six-channel capture contract"
                )
            playback = np.zeros((multichannel.shape[0], multichannel.shape[-1]), dtype=np.float32)
        else:
            playback = _as_audio_batch(playback_reference, name="playback_reference")
            if playback.shape[0] == 1 and multichannel.shape[0] > 1:
                playback = np.repeat(playback, multichannel.shape[0], axis=0)
            if playback.shape[0] != multichannel.shape[0]:
                raise ValueError("multichannel and playback_reference batch sizes do not match")
            playback = _fit_audio(playback, multichannel.shape[-1], name="playback_reference")

        extracted = self._run(
            "stage1",
            {
                "multichannel_audio": multichannel,
                "enrollment_audio": enrollment,
                "playback_reference": playback,
            },
        )
        speaker_pair = self._run(
            "speaker_embedding",
            {
                "enrollment_audio": enrollment,
                "extracted_audio": extracted,
            },
        )
        if speaker_pair.ndim != 2 or speaker_pair.shape[0] != 2:
            raise ValueError(f"speaker embedding output must be [2, embedding], got {speaker_pair.shape}")
        enrollment_embedding = speaker_pair[:1]
        extracted_embedding = speaker_pair[1:2]
        presence_logits = self._run("presence", {"extracted_audio": extracted})
        downstream = _normalize_extracted(extracted)

        kws_time = self._time_shape(self.sessions["kws"], "audio")
        if kws_time is None:
            raise ValueError("retained KWS graph must expose a fixed audio window")
        kws_input = _fit_audio(downstream, kws_time, name="kws_audio")
        kws_logits = self._run("kws", {"audio": kws_input})
        target_kws_logits = self._run(
            "target_kws",
            {
                "audio": kws_input,
                "enrollment_embedding": enrollment_embedding,
                "extracted_embedding": extracted_embedding,
            },
        )

        verifier_time = self._time_shape(self.sessions["compact_verifier"], "enrollment_audio")
        if verifier_time is None:
            raise ValueError("retained compact verifier must expose a fixed audio window")
        compact_enrollment = _fit_audio(enrollment, verifier_time, name="compact_enrollment_audio")
        compact_extracted = _fit_audio(downstream, verifier_time, name="compact_extracted_audio")
        compact_logit = self._run(
            "compact_verifier",
            {
                "enrollment_audio": compact_enrollment,
                "extracted_audio": compact_extracted,
            },
        )

        intent_logits: np.ndarray | None = None
        if include_intent:
            intent_time = self._time_shape(self.sessions["intent"], "audio")
            if intent_time is None:
                raise ValueError("retained intent graph must expose a fixed audio window")
            intent_input = _fit_audio(downstream, intent_time, name="intent_audio")
            intent_logits = self._run("intent", {"audio": intent_input})
        return {
            "extracted_audio": extracted,
            "downstream_audio": downstream,
            "speaker_embeddings": speaker_pair,
            "presence_logits": presence_logits,
            "kws_logits": kws_logits,
            "target_kws_logits": target_kws_logits,
            "compact_verification_logit": compact_logit,
            "intent_logits": intent_logits,
        }


class OnnxInCarPipeline:
    """Stateful retained pipeline backed entirely by the ONNX components.

    This class intentionally implements the retained fixed-window contract.
    Endpoint-trimmed and confidence-buffered command strategies are not
    silently approximated here; use the PyTorch reference until they have a
    separately validated ONNX implementation.
    """

    def __init__(
        self,
        runtime: OnnxComponentRuntime,
        *,
        presence_threshold: float = 0.15,
        command_presence_threshold: float | None = 0.10,
        presence_top_fraction: float = 0.20,
        speaker_gate_threshold: float | None = 0.30,
        command_speaker_gate_threshold: float | None = 0.30,
        kws_threshold: float = 0.50,
        kws_smoothing_window: int = 2,
        kws_required_hits: int = 2,
        kws_refractory_calls: int = 0,
        post_action_refractory_calls: int = 0,
        wake_class_id: int = 1,
        command_timeout_calls: int = 4,
        command_delay_calls: int = 0,
        intent_confidence_threshold: float | None = 0.53,
        command_window_strategy: str = "fixed",
        label_map: dict[str, str] | None = None,
    ) -> None:
        if str(command_window_strategy).lower() != "fixed":
            raise ValueError(
                "OnnxInCarPipeline currently supports only the validated fixed "
                "command window; endpoint strategies require their own ONNX parity study"
            )
        self.runtime = runtime
        self.presence_threshold = float(presence_threshold)
        self.command_presence_threshold = (
            self.presence_threshold
            if command_presence_threshold is None
            else float(command_presence_threshold)
        )
        self.presence_top_fraction = float(presence_top_fraction)
        self.speaker_gate_threshold = (
            None if speaker_gate_threshold is None else float(speaker_gate_threshold)
        )
        self.command_speaker_gate_threshold = (
            self.speaker_gate_threshold
            if command_speaker_gate_threshold is None
            else float(command_speaker_gate_threshold)
        )
        self.kws_threshold = float(kws_threshold)
        self.kws_smoothing_window = max(1, int(kws_smoothing_window))
        self.kws_required_hits = max(1, min(int(kws_required_hits), self.kws_smoothing_window))
        self.kws_refractory_calls = max(0, int(kws_refractory_calls))
        self.post_action_refractory_calls = max(0, int(post_action_refractory_calls))
        self.wake_class_id = int(wake_class_id)
        self.command_timeout_calls = max(1, int(command_timeout_calls))
        self.command_delay_calls = max(0, int(command_delay_calls))
        self.intent_confidence_threshold = (
            None if intent_confidence_threshold is None else float(intent_confidence_threshold)
        )
        self.label_map = dict(label_map or {})

    @staticmethod
    def _reset_command_capture(state: PipelineState) -> None:
        state.awaiting_command = False
        state.command_wait_calls = 0
        state.command_candidate_calls = 0
        state.command_best_confidence = -1.0
        state.command_best_logits = None
        state.command_candidate_history.clear()
        state.wake_context = None

    def _output(
        self,
        *,
        accepted: bool,
        wake_detected: bool,
        components: dict[str, np.ndarray],
        presence_probability: float,
        speaker_similarity: float | None,
        wake_probability: float | None = None,
        command_pending: bool = False,
        intent_logits: torch.Tensor | None = None,
        action: Any | None = None,
        intent_ready: bool = False,
        command_suppressed: bool = False,
    ) -> PipelineOutput:
        return PipelineOutput(
            accepted=accepted,
            wake_detected=wake_detected,
            intent_logits=intent_logits,
            action=action,
            s1=_torch_batch(components["extracted_audio"]),
            command_pending=command_pending,
            intent_ready=intent_ready,
            command_suppressed=command_suppressed,
            presence_probability=presence_probability,
            speaker_similarity=speaker_similarity,
            wake_probability=wake_probability,
        )

    def forward(
        self,
        mixture: np.ndarray | torch.Tensor,
        enrollment_audio: np.ndarray | torch.Tensor,
        state: PipelineState | None = None,
        *,
        multichannel: np.ndarray | torch.Tensor | None = None,
        pre_extracted: np.ndarray | torch.Tensor | None = None,
        playback_ref: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[PipelineOutput, PipelineState]:
        """Run one fixed-window ONNX step with the retained state machine."""

        if pre_extracted is not None:
            raise ValueError("pre_extracted is not supported by the ONNX retained runtime")
        capture = multichannel if multichannel is not None else mixture
        if multichannel is None:
            capture_array = np.asarray(
                mixture.detach().cpu().numpy() if isinstance(mixture, torch.Tensor) else mixture
            )
            if capture_array.ndim < 2:
                raise ValueError(
                    "the ONNX retained runtime requires four microphone channels via multichannel"
                )
        state = state or PipelineState()
        components = self.runtime.run_components(
            capture,
            enrollment_audio,
            playback_ref,
            include_intent=state.awaiting_command,
        )
        presence_probability = _aggregate_top_fraction(
            _sigmoid(components["presence_logits"]),
            self.presence_top_fraction,
        )
        active_presence_threshold = (
            self.command_presence_threshold
            if state.awaiting_command
            else self.presence_threshold
        )
        speaker_pair = components["speaker_embeddings"]
        if state.reference_speaker_embedding is None:
            state.reference_speaker_embedding = _torch_batch(speaker_pair[:1]).detach()
        active_speaker_threshold = (
            self.command_speaker_gate_threshold
            if state.awaiting_command
            else self.speaker_gate_threshold
        )
        speaker_similarity: float | None = None
        if active_speaker_threshold is not None and presence_probability >= active_presence_threshold:
            speaker_similarity = float(_sigmoid(components["compact_verification_logit"]).mean())
        speaker_rejected = (
            speaker_similarity is not None
            and active_speaker_threshold is not None
            and speaker_similarity < active_speaker_threshold
        )
        if presence_probability < active_presence_threshold or speaker_rejected:
            state.kws_scores.clear()
            if not state.awaiting_command:
                state.kws_refractory_left = 0
                return self._output(
                    accepted=False,
                    wake_detected=False,
                    components=components,
                    presence_probability=presence_probability,
                    speaker_similarity=speaker_similarity,
                ), state
            state.command_wait_calls += 1
            if state.command_wait_calls >= self.command_timeout_calls:
                self._reset_command_capture(state)
                state.kws_refractory_left = 0
                return self._output(
                    accepted=False,
                    wake_detected=False,
                    components=components,
                    presence_probability=presence_probability,
                    speaker_similarity=speaker_similarity,
                ), state
            return self._output(
                accepted=False,
                wake_detected=False,
                components=components,
                presence_probability=presence_probability,
                speaker_similarity=speaker_similarity,
                command_pending=True,
            ), state

        if state.awaiting_command:
            state.command_wait_calls += 1
            if state.command_wait_calls <= self.command_delay_calls:
                return self._output(
                    accepted=True,
                    wake_detected=False,
                    components=components,
                    presence_probability=presence_probability,
                    speaker_similarity=speaker_similarity,
                    command_pending=True,
                ), state
            intent_logits = _torch_batch(components["intent_logits"])
            intent_probability = torch.softmax(intent_logits, dim=-1)
            confidence = float(intent_probability.max(dim=-1).values.mean().item())
            if (
                self.intent_confidence_threshold is not None
                and confidence < self.intent_confidence_threshold
            ):
                self._reset_command_capture(state)
                state.kws_scores.clear()
                return self._output(
                    accepted=True,
                    wake_detected=False,
                    components=components,
                    presence_probability=presence_probability,
                    speaker_similarity=speaker_similarity,
                    intent_logits=intent_logits,
                    command_suppressed=True,
                ), state
            intent_id = str(int(torch.argmax(intent_logits, dim=-1).item()))
            action = intent_to_action(intent_id, self.label_map)
            self._reset_command_capture(state)
            state.kws_scores.clear()
            state.post_action_refractory_left = self.post_action_refractory_calls
            return self._output(
                accepted=True,
                wake_detected=False,
                components=components,
                presence_probability=presence_probability,
                speaker_similarity=speaker_similarity,
                intent_logits=intent_logits,
                action=action,
                intent_ready=True,
            ), state

        target_probability = _softmax(components["target_kws_logits"], axis=-1)
        score = float(target_probability[..., self.wake_class_id].mean())
        wake = update_kws_state(
            state,
            score,
            self.kws_threshold,
            self.kws_smoothing_window,
            self.kws_required_hits,
            self.kws_refractory_calls,
        )
        if not wake:
            return self._output(
                accepted=True,
                wake_detected=False,
                components=components,
                presence_probability=presence_probability,
                speaker_similarity=speaker_similarity,
                wake_probability=score,
            ), state
        state.awaiting_command = True
        state.command_wait_calls = 0
        state.command_candidate_calls = 0
        state.kws_scores.clear()
        return self._output(
            accepted=True,
            wake_detected=True,
            components=components,
            presence_probability=presence_probability,
            speaker_similarity=speaker_similarity,
            wake_probability=score,
            command_pending=True,
        ), state
