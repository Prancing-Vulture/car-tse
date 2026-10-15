#!/usr/bin/env python
"""Replay one verified same-speaker stream while saving the actual stage inputs.

This is an evaluation-only observer. It does not tune thresholds or modify
model weights. The manifest validator intentionally requires explicit
same-speaker verification; matching speaker strings by themselves is not enough.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from car_tse.config import load_stage_config  # noqa: E402
from car_tse.pipeline.in_car_pipeline import (  # noqa: E402
    InCarPipeline,
    PipelineState,
    normalize_extracted_waveform,
)
from car_tse.train.utils import get_device  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def ensure_output_dir_available(output_dir: Path) -> None:
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(
            f"Refusing to overwrite existing diagnostic outputs in {output_dir}; choose a new --output-dir"
        )


def validate_trial(row: dict[str, Any], *, check_files: bool = True) -> list[str]:
    """Return validation errors for one explicit matched-stream manifest row."""

    errors: list[str] = []
    required = (
        "trial_id",
        "session_id",
        "source_wav",
        "enrollment_wav",
        "speaker_id",
        "stream_speaker_id",
        "wake_speaker_id",
        "command_speaker_id",
        "enrollment_speaker_id",
        "speaker_match_verified",
        "speaker_match_evidence",
        "speaker_match_reviewer",
        "event_annotation_method",
        "annotation_reviewer",
        "sample_rate_hz",
        "wake_start_sample",
        "wake_end_sample",
        "command_start_sample",
        "command_end_sample",
        "expected_intent",
        "command_transcript",
        "command_transcript_reviewed",
    )
    for key in required:
        if key not in row or row[key] in (None, ""):
            errors.append(f"missing required field: {key}")
    if errors:
        return errors

    if row["speaker_match_verified"] is not True:
        errors.append("speaker_match_verified must be true after human/provenance verification")
    if row["command_transcript_reviewed"] is not True:
        errors.append("command_transcript_reviewed must be true after listening to the source recording")
    if not str(row["speaker_match_evidence"]).strip():
        errors.append("speaker_match_evidence must describe how speaker equivalence was verified")
    speaker = str(row["speaker_id"])
    if str(row["stream_speaker_id"]) != speaker:
        errors.append("stream_speaker_id must equal speaker_id")
    if str(row["wake_speaker_id"]) != speaker:
        errors.append("wake_speaker_id must equal speaker_id")
    if str(row["command_speaker_id"]) != speaker:
        errors.append("command_speaker_id must equal speaker_id")
    if str(row["enrollment_speaker_id"]) != speaker:
        errors.append("enrollment_speaker_id must equal speaker_id")
    integer_fields = (
        "sample_rate_hz",
        "wake_start_sample",
        "wake_end_sample",
        "command_start_sample",
        "command_end_sample",
    )
    if any(not isinstance(row[field], int) or isinstance(row[field], bool) for field in integer_fields):
        errors.append("sample rate and event boundaries must be integer sample counts")
        return errors
    sample_rate, wake_start, wake_end, command_start, command_end = (int(row[field]) for field in integer_fields)
    if sample_rate <= 0:
        errors.append("sample_rate_hz must be positive")
    if not (0 <= wake_start < wake_end <= command_start < command_end):
        errors.append("require 0 <= wake_start < wake_end <= command_start < command_end")

    source_path = Path(str(row["source_wav"]))
    enrollment_path = Path(str(row["enrollment_wav"]))
    if source_path.resolve() == enrollment_path.resolve():
        errors.append("enrollment_wav must be an independent recording from source_wav")
    if check_files:
        for field, path in (("source_wav", source_path), ("enrollment_wav", enrollment_path)):
            if not path.is_file():
                errors.append(f"{field} does not exist: {path}")
        if source_path.is_file():
            info = sf.info(str(source_path))
            if info.samplerate != sample_rate:
                errors.append(
                    f"source WAV sample rate {info.samplerate} does not match sample_rate_hz {sample_rate}"
                )
            if info.channels < 1:
                errors.append("source WAV must have at least one channel")
            if max(wake_end, command_end) > info.frames:
                errors.append(f"event boundary exceeds source length {info.frames}")
        if enrollment_path.is_file():
            enrollment_info = sf.info(str(enrollment_path))
            if enrollment_info.samplerate != sample_rate:
                errors.append(
                    f"enrollment WAV sample rate {enrollment_info.samplerate} does not match sample_rate_hz {sample_rate}"
                )
        if source_path.is_file() and enrollment_path.is_file() and sha256_file(source_path) == sha256_file(enrollment_path):
            errors.append("enrollment_wav content duplicates source_wav; provide an independent recording")
    return errors


def load_trial_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each row must be a JSON object")
            for path_key in ("source_wav", "enrollment_wav"):
                audio_path = Path(str(value.get(path_key, "")))
                if str(audio_path) and not audio_path.is_absolute():
                    value[path_key] = str((path.parent / audio_path).resolve())
            errors = validate_trial(value)
            if errors:
                raise ValueError(f"{path}:{line_number}: " + "; ".join(errors))
            trial_id = str(value["trial_id"])
            if trial_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate trial_id {trial_id!r}")
            seen_ids.add(trial_id)
            rows.append(value)
    if not rows:
        raise ValueError(f"No trials found in {path}")
    return rows


@dataclass
class CallContext:
    call_index: int
    source_start_sample: int
    source_valid_end_sample: int
    sample_rate_hz: int
    source_file_frames: int


class MatchedCapturePipeline(InCarPipeline):
    """A passive subclass that records decisions and exact model-bound audio."""

    @classmethod
    def observe_existing(
        cls,
        pipeline: InCarPipeline,
        artifact_dir: Path,
        *,
        sample_rate_hz: int,
        save_stage1: bool = True,
    ) -> "MatchedCapturePipeline":
        observed = cls.__new__(cls)
        observed.__dict__.update(pipeline.__dict__)
        observed.artifact_dir = artifact_dir
        observed.sample_rate_hz = int(sample_rate_hz)
        observed.save_stage1 = bool(save_stage1)
        observed.call_context = None
        observed.presence_capture = None
        observed.presence_logits = None
        observed.pending_source_span = None
        observed.pending_new_audio_samples = 0
        observed.command_source_segments = []
        observed.current_appends = []
        observed.intent_inputs = []
        observed.call_rows = []
        observed._intent_hook_handle = None
        observed._presence_hook_handle = None
        return observed

    def __enter__(self) -> "MatchedCapturePipeline":
        def capture_intent_input(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
            if not args or not isinstance(args[0], torch.Tensor):
                return
            if self.call_context is None:
                return
            tensor = args[0].detach().to(device="cpu").clone()
            self.intent_inputs.append((self.call_context.call_index, tensor))

        self._intent_hook_handle = self.intent.register_forward_pre_hook(capture_intent_input)
        try:
            def capture_presence_logits(
                _module: torch.nn.Module, _args: tuple[Any, ...], output: Any
            ) -> None:
                self.presence_logits = (
                    output.detach().to(device="cpu").clone()
                    if isinstance(output, torch.Tensor)
                    else None
                )

            self._presence_hook_handle = self.presence.register_forward_hook(capture_presence_logits)
        except Exception:
            self._intent_hook_handle.remove()
            self._intent_hook_handle = None
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self._intent_hook_handle is not None:
            self._intent_hook_handle.remove()
            self._intent_hook_handle = None
        if self._presence_hook_handle is not None:
            self._presence_hook_handle.remove()
            self._presence_hook_handle = None
        return False

    def _presence_statistics(
        self,
        s1: torch.Tensor,
        tail_start_samples: int,
        fresh_start_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        result = super()._presence_statistics(s1, tail_start_samples, fresh_start_samples)
        self.presence_capture = tuple(
            None if value is None else value.detach().to(device="cpu").tolist()
            for value in result
        )
        return result

    def _new_audio_segment(
        self,
        wav: torch.Tensor,
        state: PipelineState,
        audio_start_sample: int | None,
        input_num_samples: int,
    ) -> torch.Tensor:
        previous_end = state.last_audio_end_sample
        segment = super()._new_audio_segment(wav, state, audio_start_sample, input_num_samples)
        self.pending_source_span = None
        self.pending_new_audio_samples = int(segment.shape[-1])
        if audio_start_sample is None or segment.shape[-1] == 0:
            return segment
        start = max(int(audio_start_sample), previous_end if previous_end is not None else int(audio_start_sample))
        end = start + int(segment.shape[-1])
        if (
            self.call_context is not None
            and self.sample_rate_hz == int(self.call_context.sample_rate_hz)
            and segment.shape[-1] == max(0, int(input_num_samples) - max(0, (previous_end or int(audio_start_sample)) - int(audio_start_sample)))
            and end <= self.call_context.source_valid_end_sample
        ):
            self.pending_source_span = (start, end)
        return segment

    def _save_tensor(self, relative_path: str, tensor: torch.Tensor) -> dict[str, Any]:
        path = self.artifact_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        detached = tensor.detach().to(device="cpu").contiguous()
        torch.save(detached, path)
        return {
            "path": path.relative_to(self.artifact_dir).as_posix(),
            "shape": list(detached.shape),
            "samples": int(detached.shape[-1]) if detached.ndim else 1,
            "sha256": sha256_tensor(detached),
        }

    def _record_append(self, context: CallContext, tensor: torch.Tensor, span: tuple[int, int] | None, kind: str) -> None:
        saved = self._save_tensor(f"command_appends/call_{context.call_index:05d}_{kind}.pt", tensor)
        complete = span is not None and span[1] - span[0] == int(tensor.shape[-1])
        segment = {
            **saved,
            "kind": kind,
            "source_start_sample": span[0] if complete else None,
            "source_end_sample": span[1] if complete else None,
            "source_mapping_complete": bool(complete),
        }
        self.current_appends.append(segment)
        self.command_source_segments.append(segment)

    def _infer_command(
        self,
        state: PipelineState,
        presence_probability: float,
        speaker_similarity: float | None,
        s1: torch.Tensor,
    ):
        context = self.call_context
        if context is not None and state.command_audio is not None:
            before_samples = int(context_before := getattr(self, "_call_before_command_samples", 0))
            current_samples = int(state.command_audio.shape[-1])
            delta = current_samples - before_samples
            if delta > 0:
                span = self.pending_source_span
                appended = state.command_audio[..., before_samples:]
                if int(appended.shape[-1]) == delta:
                    self._record_append(context, appended, span, "accepted")
        return super()._infer_command(state, presence_probability, speaker_similarity, s1)

    def forward(
        self,
        mixture: torch.Tensor,
        enrollment: torch.Tensor,
        state: PipelineState | None = None,
        multichannel: torch.Tensor | None = None,
        audio_start_sample: int | None = None,
    ):
        context = self.call_context
        if context is None:
            raise RuntimeError("Set call_context before each observed forward call")
        state = state or PipelineState()
        before_awaiting = bool(state.awaiting_command)
        before_samples = int(state.command_audio.shape[-1]) if state.command_audio is not None else 0
        before_end = state.last_audio_end_sample
        self._call_before_command_samples = before_samples
        self._call_before_kws_scores = list(state.kws_scores)
        self._call_before_kws_refractory = int(state.kws_refractory_left)
        self.presence_capture = None
        self.presence_logits = None
        self.pending_source_span = None
        self.pending_new_audio_samples = 0
        self.current_appends = []
        prior_intent_count = len(self.intent_inputs)
        output, state = super().forward(
            mixture,
            enrollment,
            state,
            multichannel=multichannel,
            audio_start_sample=audio_start_sample,
        )

        if output.wake_detected:
            self.command_source_segments.clear()
            buffer = state.command_audio
            if buffer is not None and buffer.shape[-1] > 0:
                actual_window_end = context.source_start_sample + int(mixture.shape[-1])
                start = actual_window_end - int(buffer.shape[-1])
                span = (
                    (start, actual_window_end)
                    if start >= context.source_start_sample
                    and actual_window_end == context.source_valid_end_sample
                    and int(buffer.shape[-1]) <= int(mixture.shape[-1])
                    else None
                )
                self._record_append(context, buffer, span, "wake_preroll")
        elif before_awaiting and not output.intent_ready:
            after_samples = int(state.command_audio.shape[-1]) if state.command_audio is not None else 0
            delta = after_samples - before_samples
            if delta > 0 and state.command_audio is not None:
                self._record_append(
                    context,
                    state.command_audio[..., before_samples:],
                    self.pending_source_span,
                    "accepted",
                )

        raw_ref = output.s1
        stage1_raw = None
        stage1_normalized = None
        if raw_ref is not None:
            stage1_raw = self._save_tensor(f"stage1/raw/call_{context.call_index:05d}.pt", raw_ref)
            stage1_normalized = self._save_tensor(
                f"stage1/normalized/call_{context.call_index:05d}.pt",
                normalize_extracted_waveform(raw_ref),
            )

        if output.wake_detected:
            decision_reason = "wake_detected"
        elif output.accepted:
            decision_reason = "presence_gate_passed"
        else:
            overall, _tail, fresh = self.presence_capture or (None, None, None)
            score = output.presence_probability
            active_threshold = self.command_presence_threshold if before_awaiting else self.presence_threshold
            if score is not None and score < active_threshold:
                decision_reason = "overall_below_active_threshold"
            elif (
                output.speaker_similarity is not None
                and self.speaker_gate_threshold is not None
                and output.speaker_similarity < self.speaker_gate_threshold
            ):
                decision_reason = "speaker_gate_rejected"
            elif before_awaiting and self.pending_new_audio_samples == 0:
                decision_reason = "fresh_region_empty"
            elif before_awaiting and fresh is not None and float(np.asarray(fresh).mean()) < self.command_presence_threshold:
                decision_reason = "fresh_region_below_command_threshold"
            else:
                decision_reason = "runtime_rejected_other"

        if output.wake_probability is None:
            kws_status = "not_evaluated"
        else:
            kws_status = "evaluated"
        row: dict[str, Any] = {
            "call_index": context.call_index,
            "source_window_start_sample": context.source_start_sample,
            "source_window_valid_end_sample": context.source_valid_end_sample,
            "fresh_source_segment": (
                {"start_sample": self.pending_source_span[0], "end_sample": self.pending_source_span[1]}
                if self.pending_source_span is not None
                else None
            ),
            "fresh_source_mapping_complete": bool(
                self.pending_source_span is not None or self.pending_new_audio_samples == 0
            ),
            "presence_overall_tail_fresh": self.presence_capture,
            "presence_logits": (
                self._save_tensor(f"stage2/logits/call_{context.call_index:05d}.pt", self.presence_logits)
                if self.presence_logits is not None
                else None
            ),
            "presence_probability_output": output.presence_probability,
            "presence_active_threshold": self.command_presence_threshold if before_awaiting else self.presence_threshold,
            "decision_reason": decision_reason,
            "speaker_similarity": output.speaker_similarity,
            "accepted": bool(output.accepted),
            "awaiting_command_before": before_awaiting,
            "awaiting_command_after": bool(state.awaiting_command),
            "command_buffer_samples_before": before_samples,
            "command_buffer_samples_after": int(state.command_audio.shape[-1]) if state.command_audio is not None else 0,
            "command_appends": list(self.current_appends),
            "wake_probability": output.wake_probability,
            "kws_status": kws_status,
            "kws_threshold": float(self.kws_threshold),
            "kws_required_hits": int(self.kws_required_hits),
            "kws_smoothing_window": int(self.kws_smoothing_window),
            "kws_smoothing_support_before": list(self._call_before_kws_scores),
            "kws_smoothing_support_after": list(state.kws_scores),
            "kws_refractory_left_before": self._call_before_kws_refractory,
            "kws_refractory_left_after": int(state.kws_refractory_left),
            "wake_detected": bool(output.wake_detected),
            "command_pending": bool(output.command_pending),
            "intent_ready": bool(output.intent_ready),
            "stage1_raw": stage1_raw,
            "stage1_normalized": stage1_normalized,
        }
        if output.intent_logits is not None:
            probabilities = torch.softmax(output.intent_logits.detach(), dim=-1)
            row["intent_logits"] = output.intent_logits.detach().to(device="cpu").tolist()
            row["intent_probabilities"] = probabilities.to(device="cpu").tolist()
            row["predicted_class_id"] = int(probabilities.argmax(dim=-1).item())
            row["action"] = output.action.action if output.action is not None else None
            row["command_source_segments"] = list(self.command_source_segments)
            tensor_match = [item for item in self.intent_inputs[prior_intent_count:] if item[0] == context.call_index]
            if tensor_match:
                exact_tensor = tensor_match[-1][1]
                row["stage4_exact_input"] = self._save_tensor(
                    f"stage4/input/call_{context.call_index:05d}.pt", exact_tensor
                )
                row["stage4_sample_rate_hz"] = int(self.intent_sample_rate)
            else:
                row["stage4_exact_input"] = None
        self.call_rows.append(row)
        return output, state


def run_trial(
    row: dict[str, Any],
    *,
    config_path: Path,
    paths_path: Path,
    output_dir: Path,
    hop_seconds: float = 0.5,
) -> dict[str, Any]:
    ensure_output_dir_available(output_dir)
    errors = validate_trial(row)
    if errors:
        raise ValueError("Invalid matched trial: " + "; ".join(errors))
    from scripts.eval.eval_continuous_full_pipeline_proxy import file_sha256, load_pipeline, resolve_path

    config_path = resolve_path(config_path)
    paths_path = resolve_path(paths_path)
    cfg = load_stage_config(config_path, paths_path)
    stage1_cfg_path = resolve_path(cfg["stage1_config"])
    stage1_cfg = load_stage_config(stage1_cfg_path, paths_path)
    configured_sample_rate = int(cfg["sample_rate"])
    stage1_sample_rate = int(stage1_cfg["sample_rate"])
    if stage1_sample_rate != configured_sample_rate:
        raise ValueError(
            f"Stage 1 sample rate {stage1_sample_rate} != pipeline sample rate {configured_sample_rate}"
        )
    device = get_device()
    pipeline = load_pipeline(cfg, cfg["paths"], device)
    sample_rate = int(cfg["sample_rate"])
    if int(row["sample_rate_hz"]) != sample_rate:
        raise ValueError(f"trial sample rate {row['sample_rate_hz']} != pipeline sample rate {sample_rate}")
    source_path = Path(row["source_wav"])
    enrollment_path = Path(row["enrollment_wav"])
    multichannel, source_sr = sf.read(str(source_path), always_2d=True, dtype="float32")
    enrollment_data, enroll_sr = sf.read(str(enrollment_path), always_2d=True, dtype="float32")
    if source_sr != sample_rate or enroll_sr != sample_rate:
        raise ValueError("source and enrollment WAV sample rates must match pipeline sample_rate")
    if multichannel.shape[1] != int(cfg.get("num_mics", multichannel.shape[1])):
        raise ValueError(
            f"source channel count {multichannel.shape[1]} != configured num_mics {cfg.get('num_mics')}"
        )
    mono = multichannel.mean(axis=1).astype(np.float32)
    enrollment = enrollment_data.mean(axis=1).astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_dir / "source_multichannel.wav"), multichannel, sample_rate, subtype="FLOAT")
    sf.write(str(output_dir / "source_mono_mean.wav"), mono, sample_rate, subtype="FLOAT")
    sf.write(str(output_dir / "enrollment_mono_mean.wav"), enrollment, sample_rate, subtype="FLOAT")

    segment_samples = int(round(float(cfg.get("segment_seconds", 3.0)) * sample_rate))
    hop_samples = max(1, int(round(float(hop_seconds) * sample_rate)))
    observer = MatchedCapturePipeline.observe_existing(pipeline, output_dir, sample_rate_hz=sample_rate)
    state = PipelineState()
    with observer:
        for call_index, start in enumerate(range(0, max(1, len(mono)), hop_samples)):
            valid_end = min(len(mono), start + segment_samples)
            mix = np.zeros(segment_samples, dtype=np.float32)
            channels = np.zeros((multichannel.shape[1], segment_samples), dtype=np.float32)
            if valid_end > start:
                count = valid_end - start
                mix[:count] = mono[start:valid_end]
                channels[:, :count] = multichannel[start:valid_end].T
            context = CallContext(call_index, start, valid_end, sample_rate, len(mono))
            observer.call_context = context
            with torch.inference_mode():
                _, state = observer.forward(
                    torch.from_numpy(mix).to(device),
                    torch.from_numpy(enrollment).to(device),
                    state,
                    multichannel=torch.from_numpy(channels).to(device),
                    audio_start_sample=start,
                )

    config_hash = file_sha256(config_path)
    paths_hash = file_sha256(paths_path)
    checkpoint_root = resolve_path(Path(cfg["paths"]["checkpoint_root"]))
    checkpoints: dict[str, dict[str, str]] = {}
    for name, value in sorted(cfg.get("checkpoints", {}).items()):
        path = Path(value)
        if not path.is_absolute():
            path = checkpoint_root / path
        path = resolve_path(path)
        if path.is_file():
            checkpoints[name] = {"path": str(path), "sha256": file_sha256(path)}
    pretrained_path = stage1_cfg.get("pretrained_checkpoint")
    if pretrained_path:
        path = resolve_path(pretrained_path)
        if path.is_file():
            checkpoints["stage1_pretrained"] = {"path": str(path), "sha256": file_sha256(path)}
    report = {
        "benchmark": "matched_continuous_instrumented_diagnostic",
        "performance_claim": False,
        "trial": row,
        "source_audio": {
            "path": str(source_path.resolve()),
            "sha256": file_sha256(source_path),
            "sample_rate_hz": sample_rate,
            "channels": int(multichannel.shape[1]),
            "frames": int(len(mono)),
            "channel_names": row.get("channel_names"),
            "mono_derivation": "arithmetic mean across source channels",
        },
        "enrollment_audio": {
            "path": str(enrollment_path.resolve()),
            "sha256": file_sha256(enrollment_path),
            "sample_rate_hz": enroll_sr,
            "frames": int(enrollment_data.shape[0]),
        },
        "configuration": {"path": str(config_path), "sha256": config_hash},
        "manifest_record": {
            "path": str(row.get("_manifest_path", "not_attached")),
            "sha256": row.get("_manifest_sha256"),
        },
        "paths_configuration": {"path": str(paths_path), "sha256": paths_hash},
        "stage1_configuration": {"path": str(stage1_cfg_path), "sha256": file_sha256(stage1_cfg_path)},
        "checkpoints": checkpoints,
        "observer_script": {"path": str(Path(__file__).resolve()), "sha256": file_sha256(Path(__file__).resolve())},
        "hop_seconds": float(hop_seconds),
        "calls": observer.call_rows,
        "limits": [
            "A single replay is a diagnostic, not production evidence.",
            "Unmapped append spans are null and must not be reconstructed from buffer length.",
            "Enrollment/stream speaker equivalence is asserted only from the validated manifest and its cited review process.",
        ],
    }
    try:
        report["git_revision"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
        ).stdout
        report["git_worktree_dirty"] = bool(status.strip())
        report["git_status_sha256"] = hashlib.sha256(status.encode("utf-8")).hexdigest()
        tracked_diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True
        ).stdout
        report["tracked_diff_sha256"] = hashlib.sha256(tracked_diff).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        report["git_revision"] = None
        report["git_worktree_dirty"] = None
        report["git_status_sha256"] = None
        report["tracked_diff_sha256"] = None
    report_path = output_dir / "trace.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL manifest with one or more verified trials")
    parser.add_argument("--config", type=Path, required=True, help="Locked continuous pipeline evaluation config")
    parser.add_argument("--paths", type=Path, required=True, help="Local paths YAML used by the config")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trial-id", default=None, help="Run one trial_id; defaults to the first manifest row")
    parser.add_argument("--hop-seconds", type=float, default=0.5)
    parser.add_argument("--wandb", action="store_true", help="Explicitly log the diagnostic report to W&B")
    parser.add_argument("--wandb-project", default="car-tse")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default=None)
    args = parser.parse_args()
    trials = load_trial_manifest(args.manifest)
    selected = next((trial for trial in trials if trial["trial_id"] == args.trial_id), None) if args.trial_id else trials[0]
    if selected is None:
        raise SystemExit(f"trial_id {args.trial_id!r} not found in {args.manifest}")
    selected = dict(selected)
    selected["_manifest_path"] = str(args.manifest.resolve())
    selected["_manifest_sha256"] = sha256_file(args.manifest.resolve())
    trial_dir = args.output_dir / str(selected["trial_id"])
    report = run_trial(
        selected,
        config_path=args.config,
        paths_path=args.paths,
        output_dir=trial_dir,
        hop_seconds=args.hop_seconds,
    )
    print(f"Wrote diagnostic trace: {trial_dir / 'trace.json'}; calls={len(report['calls'])}")
    if args.wandb:
        from car_tse.train.wandb_logging import log_evaluation

        log_evaluation(
            project=args.wandb_project,
            run_name=f"matched-diagnostic-{selected['trial_id']}",
            job_type="instrumented-continuous-diagnostic",
            config={"config": str(args.config), "manifest": str(args.manifest), "trial_id": selected["trial_id"]},
            metrics={"calls": len(report["calls"]), "intent_events": sum(bool(call["intent_ready"]) for call in report["calls"])},
            output_path=trial_dir / "trace.json",
            mode=args.wandb_mode,
            directory=trial_dir / "wandb",
            window_rows=report["calls"],
            report_artifact=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
