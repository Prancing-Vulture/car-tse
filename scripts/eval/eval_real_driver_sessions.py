#!/usr/bin/env python
"""Evaluate permitted continuous real-driver pilot sessions.

The evaluator refuses to score an invalid session tree. It uses the same
pipeline checkpoints as the proxy evaluator, selects presence,
    command-presence, KWS, speaker-gate, and intent-confidence thresholds on
tuning drivers only,
freezes the selected operating point, and then scores the held-out test
drivers without changing settings.

Required session layout::

    sessions/<session_id>/audio.wav
    sessions/<session_id>/session_metadata.json
    sessions/<session_id>/annotations.jsonl

The event schema is documented in ``reports/real_driver_pilot_protocol_2026-10-09.md``.
This evaluator reports real-driver evidence only when the metadata grants
permission and the validator accepts the complete event contract.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.io.audio import read_multichannel
from car_tse.io.capture import split_capture_audio
from car_tse.pipeline.in_car_pipeline import PipelineState, normalize_extracted_waveform
from car_tse.pipeline.windowing import full_context_window_starts
from car_tse.train.utils import get_device, write_metrics
from eval_continuous_full_pipeline_proxy import load_pipeline, resolve_path
from validate_real_driver_sessions import validate


SAMPLE_RATE = 16000
REQUIRED_EVENTS = {
    "enrollment",
    "driver_wake",
    "driver_command",
    "passenger_wake",
    "passenger_speech",
    "no_command",
    "noise_only",
}


ABLATION_VARIANTS: dict[str, dict[str, bool]] = {
    "full": {
        "use_tse": True,
        "use_presence": True,
        "use_speaker_matching": True,
        "use_wake_smoothing": True,
    },
    "no_tse": {
        "use_tse": False,
        "use_presence": True,
        "use_speaker_matching": False,
        "use_wake_smoothing": True,
    },
    "no_presence": {
        "use_tse": True,
        "use_presence": False,
        "use_speaker_matching": True,
        "use_wake_smoothing": True,
    },
    "no_speaker_matching": {
        "use_tse": True,
        "use_presence": True,
        "use_speaker_matching": False,
        "use_wake_smoothing": True,
    },
    "no_wake_smoothing": {
        "use_tse": True,
        "use_presence": True,
        "use_speaker_matching": True,
        "use_wake_smoothing": False,
    },
}


@dataclass(frozen=True)
class Session:
    directory: Path
    metadata: dict[str, Any]
    annotations: list[dict[str, Any]]
    audio: np.ndarray
    sample_rate: int

    @property
    def session_id(self) -> str:
        return str(self.metadata["session_id"])

    @property
    def driver_id(self) -> str:
        return str(self.metadata["driver_id"])

    @property
    def split(self) -> str:
        return str(self.metadata["split"])

    @property
    def duration_seconds(self) -> float:
        return float(len(self.audio)) / max(self.sample_rate, 1)


def load_session(root: Path, session_id: str) -> Session:
    directory = root / session_id
    metadata = json.loads((directory / "session_metadata.json").read_text(encoding="utf-8"))
    annotations = [
        json.loads(line)
        for line in (directory / "annotations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    audio_path = Path(str(metadata.get("audio_path", "audio.wav")))
    if not audio_path.is_absolute():
        audio_path = directory / audio_path.name if audio_path.parent == Path(".") else root / audio_path
    audio, sample_rate = read_multichannel(audio_path, SAMPLE_RATE)
    if audio.ndim == 1:
        audio = audio[:, None]
    return Session(directory, metadata, annotations, audio.astype(np.float32), int(sample_rate))


def fit_channels(audio: np.ndarray, num_mics: int) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.shape[1] < num_mics:
        repeats = int(np.ceil(num_mics / max(audio.shape[1], 1)))
        audio = np.tile(audio, (1, repeats))[:, :num_mics]
    elif audio.shape[1] > num_mics:
        audio = audio[:, :num_mics]
    return np.asarray(audio, dtype=np.float32)


def clip(audio: np.ndarray, start_s: float, end_s: float, sample_rate: int) -> np.ndarray:
    start = max(0, int(round(float(start_s) * sample_rate)))
    end = min(len(audio), max(start + 1, int(round(float(end_s) * sample_rate))))
    result = np.asarray(audio[start:end], dtype=np.float32)
    if result.ndim == 2:
        result = result.mean(axis=1)
    return result.reshape(-1)


def pad_or_crop(wav: np.ndarray, samples: int) -> np.ndarray:
    result = np.zeros(max(1, int(samples)), dtype=np.float32)
    result[: min(len(wav), len(result))] = wav[: len(result)]
    return result


def event_time(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def event_condition(row: dict[str, Any], session: Session) -> str:
    return str(
        row.get("condition")
        or row.get("acoustic_condition")
        or session.metadata.get("condition")
        or session.metadata.get("vehicle", {}).get("road_surface")
        or "unspecified"
    )


def event_contains(center: float, row: dict[str, Any], tolerance: float = 0.0) -> bool:
    return event_time(row, "start_s") - tolerance <= center <= event_time(row, "end_s") + tolerance


def event_overlaps_window(
    event: dict[str, Any],
    truth: dict[str, Any],
    stage1_seconds: float,
    *,
    tolerance_before: float = 0.0,
    tolerance_after: float = 0.0,
) -> bool:
    """Associate a model window by interval support, not only its centre."""

    window_start = float(event["window_start_seconds"])
    window_end = float(
        event.get("window_end_seconds", window_start + float(stage1_seconds))
    )
    truth_start = event_time(truth, "start_s") - float(tolerance_before)
    truth_end = event_time(truth, "end_s") + float(tolerance_after)
    return min(window_end, truth_end) > max(window_start, truth_start)


def load_enrollment(session: Session, sample_rate: int, max_seconds: float = 3.0) -> np.ndarray:
    rows = [row for row in session.annotations if row.get("event") == "enrollment"]
    if not rows:
        raise ValueError(f"{session.session_id}: no enrollment event")
    row = rows[0]
    wav = clip(session.audio, event_time(row, "start_s"), event_time(row, "end_s"), sample_rate)
    if len(wav) > int(round(max_seconds * sample_rate)):
        wav = wav[: int(round(max_seconds * sample_rate))]
    return wav


def _condition_for_center(
    center_seconds: float,
    rows: list[dict[str, Any]],
    session: Session,
    *,
    default: str = "unmatched_or_unlabeled",
) -> str:
    """Return the annotation condition covering a predicted event center."""

    matches = [row for row in rows if event_contains(center_seconds, row)]
    if not matches:
        return default
    # Prefer the narrowest interval when annotations overlap, which keeps a
    # passenger wake or command-like event from being attributed to a broad
    # session-level condition interval.
    row = min(matches, key=lambda item: event_time(item, "end_s") - event_time(item, "start_s"))
    return event_condition(row, session)


def _union_seconds(intervals: list[tuple[float, float]]) -> float:
    """Return the length of the union of half-open intervals."""

    if not intervals:
        return 0.0
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([float(start), float(end)])
        else:
            merged[-1][1] = max(merged[-1][1], float(end))
    return float(sum(end - start for start, end in merged))


def _rate_per_hour(events: int | float, exposure_seconds: float) -> float | None:
    """Return an event rate over labeled exposure, or None if exposure is absent."""

    if float(exposure_seconds) <= 0.0:
        return None
    return float(events) / (float(exposure_seconds) / 3600.0)


def _wilson_interval(
    successes: int | float,
    trials: int | float,
    *,
    confidence: float = 0.95,
) -> dict[str, float] | None:
    """Return a Wilson score interval for a binomial proportion."""

    n = float(trials)
    if n <= 0.0:
        return None
    k = min(max(float(successes), 0.0), n)
    # Keep the evaluator dependency-light while using the standard 95% normal
    # quantile used by the Wilson score interval. The confidence argument is
    # intentionally restricted to the declared default for now.
    if abs(float(confidence) - 0.95) > 1.0e-9:
        raise ValueError("only 95% Wilson intervals are currently supported")
    z = 1.959963984540054
    p = k / n
    denominator = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denominator
    half = z * math.sqrt((p * (1.0 - p) / n) + (z * z) / (4.0 * n * n)) / denominator
    return {
        "successes": float(k),
        "trials": float(n),
        "lower": max(0.0, center - half),
        "upper": min(1.0, center + half),
    }


def _poisson_rate_interval(
    events: int | float,
    exposure_seconds: float,
    *,
    confidence: float = 0.95,
) -> dict[str, float] | None:
    """Return a 95% Poisson event-rate interval per hour.

    Garwood limits are used when SciPy is available. A conservative normal/
    zero-count fallback keeps the reporting script usable in a minimal
    installation without silently omitting uncertainty.
    """

    seconds = float(exposure_seconds)
    if seconds <= 0.0:
        return None
    if abs(float(confidence) - 0.95) > 1.0e-9:
        raise ValueError("only 95% Poisson intervals are currently supported")
    count = max(0, int(round(float(events))))
    alpha = 1.0 - float(confidence)
    try:
        from scipy.stats import chi2

        lower_count = 0.0 if count == 0 else 0.5 * float(chi2.ppf(alpha / 2.0, 2 * count))
        upper_count = 0.5 * float(chi2.ppf(1.0 - alpha / 2.0, 2 * (count + 1)))
    except Exception:
        z = 1.959963984540054
        if count == 0:
            lower_count = 0.0
            upper_count = -math.log(alpha / 2.0)
        else:
            lower_count = max(0.0, count - z * math.sqrt(count))
            upper_count = count + z * math.sqrt(count + 1.0)
    hours = seconds / 3600.0
    return {
        "events": float(count),
        "exposure_seconds": seconds,
        "rate_per_hour": count / hours,
        "lower_per_hour": lower_count / hours,
        "upper_per_hour": upper_count / hours,
    }


def _confidence_intervals(
    *,
    driver_wakes_detected: int | float,
    driver_wake_events: int | float,
    correct_commands: int | float,
    driver_command_events: int | float,
    false_accept_events: int | float,
    negative_interval_false_accept_events: int | float,
    false_action_events: int | float,
    negative_exposure_seconds: float,
) -> dict[str, dict[str, float] | None]:
    """Build the uncertainty fields shared by session and aggregate reports."""

    return {
        "driver_wake_recall_95": _wilson_interval(driver_wakes_detected, driver_wake_events),
        "command_accuracy_95": _wilson_interval(correct_commands, driver_command_events),
        "false_accepts_per_negative_hour_95": _poisson_rate_interval(
            false_accept_events, negative_exposure_seconds
        ),
        "negative_interval_false_accepts_per_negative_hour_95": _poisson_rate_interval(
            negative_interval_false_accept_events, negative_exposure_seconds
        ),
        "false_actions_per_negative_hour_95": _poisson_rate_interval(
            false_action_events, negative_exposure_seconds
        ),
    }


def _safety_rate(summary: dict[str, Any], negative_key: str, legacy_key: str) -> float:
    """Read a negative-exposure rate with a compatibility fallback.

    Older synthetic/test summaries only contain the historical total-duration
    fields. Real-driver runs now emit the negative-exposure fields and must use
    those for threshold selection and production gates.
    """

    value = summary.get(negative_key)
    if value is not None:
        return float(value)
    return float(summary[legacy_key])


def summarize_conditions(
    session: Session,
    *,
    driver_wakes: list[dict[str, Any]],
    driver_commands: list[dict[str, Any]],
    matched_wakes: list[dict[str, Any]],
    command_details: list[dict[str, Any]],
    unmatched_wakes: list[dict[str, Any]],
    false_action_events: list[dict[str, Any]],
    eligible_negative_events: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build an exposure-aware condition breakdown for one session."""

    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "exposure_seconds": 0.0,
            "driver_wake_events": 0,
            "driver_wakes_detected": 0,
            "driver_command_events": 0,
            "commands_with_action": 0,
            "correct_commands": 0,
            "false_accept_events": 0,
            "negative_interval_false_accept_events": 0,
            "false_action_events": 0,
        }
    )

    # Exposure is computed from the union of labeled negative intervals, so
    # overlapping passenger/no-command/noise annotations do not inflate the
    # denominator.
    intervals_by_condition: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in eligible_negative_events:
        condition = event_condition(row, session)
        intervals_by_condition[condition].append((event_time(row, "start_s"), event_time(row, "end_s")))
    for condition, intervals in intervals_by_condition.items():
        stats[condition]["exposure_seconds"] = _union_seconds(intervals)

    matched_truth_ids = {id(item["truth"]) for item in matched_wakes}
    for truth in driver_wakes:
        condition = event_condition(truth, session)
        stats[condition]["driver_wake_events"] += 1
        stats[condition]["driver_wakes_detected"] += int(id(truth) in matched_truth_ids)

    for detail in command_details:
        truth = detail["truth"]
        condition = event_condition(truth, session)
        stats[condition]["driver_command_events"] += 1
        prediction = detail.get("prediction")
        if prediction is not None:
            stats[condition]["commands_with_action"] += 1
            stats[condition]["correct_commands"] += int(bool(detail.get("correct")))

    for event in unmatched_wakes:
        condition = _condition_for_center(
            float(event["window_center_seconds"]),
            eligible_negative_events,
            session,
        )
        stats[condition]["false_accept_events"] += 1
        if condition != "unmatched_or_unlabeled":
            stats[condition]["negative_interval_false_accept_events"] += 1

    for event in false_action_events:
        condition = _condition_for_center(
            float(event["window_center_seconds"]),
            eligible_negative_events,
            session,
        )
        stats[condition]["false_action_events"] += 1

    for condition, row in stats.items():
        exposure_hours = float(row["exposure_seconds"]) / 3600.0
        row["driver_wake_recall"] = row["driver_wakes_detected"] / max(row["driver_wake_events"], 1)
        row["command_accuracy"] = row["correct_commands"] / max(row["driver_command_events"], 1)
        row["false_accepts_per_hour"] = row["false_accept_events"] / max(exposure_hours, 1.0e-8)
        row["negative_interval_false_accepts_per_hour"] = (
            row["negative_interval_false_accept_events"] / max(exposure_hours, 1.0e-8)
        )
        row["false_actions_per_hour"] = row["false_action_events"] / max(exposure_hours, 1.0e-8)
        # These explicit aliases make the denominator unambiguous. The older
        # ``*_per_hour`` fields are kept for report/backward compatibility;
        # condition-level metrics have always used negative exposure here.
        row["false_accepts_per_negative_hour"] = _rate_per_hour(
            row["false_accept_events"], row["exposure_seconds"]
        )
        row["negative_interval_false_accepts_per_negative_hour"] = _rate_per_hour(
            row["negative_interval_false_accept_events"], row["exposure_seconds"]
        )
        row["false_actions_per_negative_hour"] = _rate_per_hour(
            row["false_action_events"], row["exposure_seconds"]
        )
        row["confidence_intervals"] = _confidence_intervals(
            driver_wakes_detected=row["driver_wakes_detected"],
            driver_wake_events=row["driver_wake_events"],
            correct_commands=row["correct_commands"],
            driver_command_events=row["driver_command_events"],
            false_accept_events=row["false_accept_events"],
            negative_interval_false_accept_events=row["negative_interval_false_accept_events"],
            false_action_events=row["false_action_events"],
            negative_exposure_seconds=float(row["exposure_seconds"]),
        )
    return {condition: dict(row) for condition, row in sorted(stats.items())}


def run_session(
    pipeline: Any,
    session: Session,
    *,
    hop_seconds: float,
    presence_threshold: float,
    kws_threshold: float,
    command_presence_threshold: float,
    speaker_gate_threshold: float | None,
    command_speaker_gate_threshold: float | None,
    intent_confidence_threshold: float | None,
    device: torch.device,
    save_windows: bool,
    variant: str = "full",
) -> dict[str, Any]:
    """Run one session with a fresh streaming state."""

    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"unknown real-driver ablation variant: {variant}")
    variant_settings = ABLATION_VARIANTS[variant]

    num_mics = int(getattr(pipeline.stage1, "frontend", None).num_mics) if getattr(pipeline.stage1, "frontend", None) is not None else 1
    multichannel, playback = split_capture_audio(
        session.audio,
        session.metadata,
        model_mics=num_mics,
    )
    multichannel = fit_channels(multichannel, num_mics)
    mixture = multichannel.mean(axis=1).astype(np.float32)
    stage1_seconds = float(getattr(pipeline, "segment_seconds", 3.0))
    # ``InCarPipeline`` does not need this attribute at runtime, so derive the
    # context from the loaded config's intent/KWS contract when absent.
    if stage1_seconds <= 0:
        stage1_seconds = 3.0
    window_samples = int(round(stage1_seconds * session.sample_rate))
    hop_samples = max(1, int(round(float(hop_seconds) * session.sample_rate)))
    enrollment = load_enrollment(session, session.sample_rate)
    enrollment_tensor = torch.from_numpy(enrollment).to(device).unsqueeze(0)
    state = PipelineState()
    windows: list[dict[str, Any]] = []
    wake_events: list[dict[str, Any]] = []
    command_events: list[dict[str, Any]] = []

    # The evaluator applies the candidate thresholds to this run only. The
    # caller restores the pipeline values before the next operating point.
    original_values = {
        "presence_threshold": pipeline.presence_threshold,
        "kws_threshold": pipeline.kws_threshold,
        "command_presence_threshold": pipeline.command_presence_threshold,
        "speaker_gate_threshold": pipeline.speaker_gate_threshold,
        "command_speaker_gate_threshold": pipeline.command_speaker_gate_threshold,
        "presence_activity_margin_threshold": pipeline.presence_activity_margin_threshold,
        "kws_smoothing_window": pipeline.kws_smoothing_window,
        "kws_required_hits": pipeline.kws_required_hits,
        "kws_refractory_calls": pipeline.kws_refractory_calls,
        "speaker_fusion_mode": pipeline.speaker_fusion_mode,
        "target_kws": pipeline.target_kws,
        "target_kws_secondary": pipeline.target_kws_secondary,
        "intent_confidence_threshold": pipeline.intent_confidence_threshold,
    }
    pipeline.presence_threshold = (
        float(presence_threshold) if variant_settings["use_presence"] else float("-inf")
    )
    pipeline.command_presence_threshold = (
        float(command_presence_threshold)
        if variant_settings["use_presence"]
        else float("-inf")
    )
    pipeline.kws_threshold = float(kws_threshold)
    pipeline.speaker_gate_threshold = (
        None
        if not variant_settings["use_speaker_matching"]
        else None if speaker_gate_threshold is None else float(speaker_gate_threshold)
    )
    pipeline.command_speaker_gate_threshold = (
        None
        if not variant_settings["use_speaker_matching"]
        else (
            pipeline.speaker_gate_threshold
            if command_speaker_gate_threshold is None
            else float(command_speaker_gate_threshold)
        )
    )
    pipeline.presence_activity_margin_threshold = (
        original_values["presence_activity_margin_threshold"]
        if variant_settings["use_presence"]
        else None
    )
    if not variant_settings["use_speaker_matching"]:
        pipeline.target_kws = None
        pipeline.target_kws_secondary = None
        pipeline.speaker_fusion_mode = "none"
    if not variant_settings["use_wake_smoothing"]:
        pipeline.kws_smoothing_window = 1
        pipeline.kws_required_hits = 1
        pipeline.kws_refractory_calls = 0
    pipeline.intent_confidence_threshold = (
        None if intent_confidence_threshold is None else float(intent_confidence_threshold)
    )
    pipeline.eval()
    try:
        with torch.inference_mode():
            # Do not turn the final partial recording tail into a synthetic
            # zero-padded negative window. This matches StreamingRunner's
            # full-context cadence and prevents tail-only false accepts.
            for start in full_context_window_starts(len(mixture), window_samples, hop_samples):
                end = min(start + window_samples, len(mixture))
                mix_window = np.zeros(window_samples, dtype=np.float32)
                mc_window = np.zeros((multichannel.shape[1], window_samples), dtype=np.float32)
                playback_window = None
                if end > start:
                    mix_window[: end - start] = mixture[start:end]
                    mc_window[:, : end - start] = multichannel[start:end].T
                    if playback is not None:
                        playback_window = np.zeros(window_samples, dtype=np.float32)
                        playback_window[: end - start] = playback[start:end]
                elif playback is not None:
                    playback_window = np.zeros(window_samples, dtype=np.float32)
                mix_tensor = torch.from_numpy(mix_window).to(device).unsqueeze(0)
                mc_tensor = torch.from_numpy(mc_window).to(device).unsqueeze(0)
                playback_tensor = (
                    None
                    if playback_window is None
                    else torch.from_numpy(playback_window).to(device).unsqueeze(0)
                )
                output, state = pipeline.forward(
                    mix_tensor,
                    enrollment_tensor,
                    state,
                    multichannel=mc_tensor,
                    pre_extracted=(mix_tensor if not variant_settings["use_tse"] else None),
                    playback_ref=playback_tensor,
                )
                center = start / session.sample_rate + stage1_seconds / 2.0
                row = {
                    "window_index": len(windows),
                    "window_start_seconds": start / session.sample_rate,
                    "window_end_seconds": start / session.sample_rate + stage1_seconds,
                    "window_center_seconds": center,
                    "accepted": bool(output.accepted),
                    "presence_probability": output.presence_probability,
                    "speaker_similarity": output.speaker_similarity,
                    "wake_probability": output.wake_probability,
                    "wake_detected": bool(output.wake_detected),
                    "command_pending": bool(output.command_pending),
                    "intent_ready": bool(output.intent_ready),
                    "command_suppressed": bool(output.command_suppressed),
                    "intent": None,
                    "intent_confidence": None,
                    "action": output.action.action if output.action is not None else None,
                }
                if output.intent_logits is not None:
                    probabilities = torch.softmax(output.intent_logits, dim=-1)[0]
                    predicted_id = str(int(probabilities.argmax().item()))
                    row["intent_id"] = predicted_id
                    row["intent"] = (
                        pipeline.label_map.get(predicted_id, predicted_id)
                        if pipeline.label_map
                        else predicted_id
                    )
                    row["intent_confidence"] = float(probabilities.max().cpu())
                windows.append(row)
                if output.wake_detected:
                    wake_events.append(row)
                if output.intent_ready:
                    command_events.append(row)
    finally:
        for name, value in original_values.items():
            setattr(pipeline, name, value)

    driver_wakes = [row for row in session.annotations if row.get("event") == "driver_wake"]
    driver_commands = [row for row in session.annotations if row.get("event") == "driver_command"]
    eligible_negative_events = [
        row
        for row in session.annotations
        if row.get("event") in {"passenger_wake", "passenger_speech", "no_command", "noise_only", "playback"}
    ]
    matched_wakes: list[dict[str, Any]] = []
    used_wake_indices: set[int] = set()
    for truth in driver_wakes:
        candidates = [
            (index, event)
            for index, event in enumerate(wake_events)
            if index not in used_wake_indices
            and event_overlaps_window(
                event,
                truth,
                stage1_seconds,
                tolerance_before=max(0.5, hop_seconds),
                tolerance_after=max(0.5, hop_seconds),
            )
        ]
        if candidates:
            index, event = min(candidates, key=lambda item: item[1]["window_center_seconds"])
            used_wake_indices.add(index)
            matched_wakes.append({"truth": truth, "prediction": event})

    unmatched_wakes = [event for index, event in enumerate(wake_events) if index not in used_wake_indices]
    command_matches: list[dict[str, Any]] = []
    used_command_indices: set[int] = set()
    for wake in matched_wakes:
        wake_time = float(wake["prediction"]["window_center_seconds"])
        command_candidates = [
            (index, event)
            for index, event in enumerate(command_events)
            if index not in used_command_indices
            and float(event["window_center_seconds"]) >= wake_time
            and float(event["window_center_seconds"]) <= wake_time + float(pipeline.command_timeout_calls) * hop_seconds + stage1_seconds
        ]
        if command_candidates:
            index, event = min(command_candidates, key=lambda item: item[1]["window_center_seconds"])
            used_command_indices.add(index)
            command_matches.append({"wake": wake, "prediction": event})

    unmatched_commands = [
        event for index, event in enumerate(command_events) if index not in used_command_indices
    ]
    correct_commands = 0
    command_latencies: list[float] = []
    wake_to_action_latencies: list[float] = []
    matched_truth_commands: set[int] = set()
    command_details: list[dict[str, Any]] = []
    for truth_index, truth in enumerate(driver_commands):
        candidates = [
            match
            for match in command_matches
            if event_overlaps_window(
                match["prediction"],
                truth,
                stage1_seconds,
                tolerance_before=4.0,
                tolerance_after=4.0,
            )
        ]
        if not candidates:
            command_details.append(
                {
                    "truth": truth,
                    "prediction": None,
                    "correct": False,
                }
            )
            continue
        match = min(candidates, key=lambda item: item["prediction"]["window_center_seconds"])
        matched_truth_commands.add(truth_index)
        prediction = match["prediction"]
        correct = str(prediction.get("intent")) == str(truth.get("expected_intent"))
        correct_commands += int(correct)
        action_time = float(prediction["window_center_seconds"])
        command_latencies.append(action_time - event_time(truth, "start_s"))
        wake_to_action_latencies.append(action_time - float(match["wake"]["prediction"]["window_center_seconds"]))
        command_details.append({"truth": truth, "prediction": prediction, "correct": correct})

    false_action_events = [
        event
        for event in command_events
        if event not in [match["prediction"] for match in command_matches if any(
            event is detail.get("prediction") for detail in [*command_details]
        )]
    ]
    # Identity-based filtering above is intentionally conservative; any
    # command event that is not linked to a labeled driver command is an
    # ineligible action for safety reporting.
    true_command_predictions = {
        id(detail["prediction"])
        for detail in command_details
        if detail.get("prediction") is not None
    }
    false_action_events = [event for event in command_events if id(event) not in true_command_predictions]
    negative_wakes = [
        event
        for event in unmatched_wakes
        if any(
            event_overlaps_window(event, truth, stage1_seconds)
            for truth in eligible_negative_events
        )
    ]
    negative_exposure_seconds = _union_seconds(
        [
            (event_time(row, "start_s"), event_time(row, "end_s"))
            for row in eligible_negative_events
        ]
    )
    negative_exposure_hours = negative_exposure_seconds / 3600.0
    condition_metrics = summarize_conditions(
        session,
        driver_wakes=driver_wakes,
        driver_commands=driver_commands,
        matched_wakes=matched_wakes,
        command_details=command_details,
        unmatched_wakes=unmatched_wakes,
        false_action_events=false_action_events,
        eligible_negative_events=eligible_negative_events,
    )
    duration_hours = session.duration_seconds / 3600.0
    result = {
        "session_id": session.session_id,
        "variant": variant,
        "driver_id": session.driver_id,
        "split": session.split,
        "conditions": sorted({event_condition(row, session) for row in session.annotations}),
        "duration_seconds": session.duration_seconds,
        "capture": {
            "recorded_channel_count": int(session.audio.shape[1]),
            "model_microphone_count": int(multichannel.shape[1]),
            "playback_reference_used": playback is not None,
        },
        "driver_wake_events": len(driver_wakes),
        "driver_wakes_detected": len(matched_wakes),
        "driver_wake_recall": len(matched_wakes) / max(len(driver_wakes), 1),
        "driver_command_events": len(driver_commands),
        "commands_with_action": len(matched_truth_commands),
        "correct_commands": correct_commands,
        "command_accuracy": correct_commands / max(len(driver_commands), 1),
        "command_accuracy_given_detected_wake": correct_commands / max(len(matched_wakes), 1),
        # ``false_accept_events`` is deliberately conservative: it includes
        # every unmatched wake event, including one outside an annotated
        # negative interval. The interval-specific value is reported
        # separately for debugging label coverage.
        "false_accept_events": len(unmatched_wakes),
        "negative_interval_false_accept_events": len(negative_wakes),
        "false_accepts_per_hour": len(unmatched_wakes) / max(duration_hours, 1.0e-8),
        "negative_interval_false_accepts_per_hour": len(negative_wakes) / max(duration_hours, 1.0e-8),
        "false_action_events": len(false_action_events),
        "false_actions_per_hour": len(false_action_events) / max(duration_hours, 1.0e-8),
        "negative_exposure_seconds": negative_exposure_seconds,
        "negative_exposure_hours": negative_exposure_hours,
        "false_accepts_per_negative_hour": _rate_per_hour(len(unmatched_wakes), negative_exposure_seconds),
        "negative_interval_false_accepts_per_negative_hour": _rate_per_hour(
            len(negative_wakes), negative_exposure_seconds
        ),
        "false_actions_per_negative_hour": _rate_per_hour(
            len(false_action_events), negative_exposure_seconds
        ),
        "confidence_intervals": _confidence_intervals(
            driver_wakes_detected=len(matched_wakes),
            driver_wake_events=len(driver_wakes),
            correct_commands=correct_commands,
            driver_command_events=len(driver_commands),
            false_accept_events=len(unmatched_wakes),
            negative_interval_false_accept_events=len(negative_wakes),
            false_action_events=len(false_action_events),
            negative_exposure_seconds=negative_exposure_seconds,
        ),
        "wake_latency_mean_seconds": (
            float(np.mean([
                float(item["prediction"]["window_center_seconds"]) - event_time(item["truth"], "start_s")
                for item in matched_wakes
            ]))
            if matched_wakes
            else None
        ),
        "command_to_action_latency_p50_seconds": (
            float(np.percentile(command_latencies, 50)) if command_latencies else None
        ),
        "command_to_action_latency_p95_seconds": (
            float(np.percentile(command_latencies, 95)) if command_latencies else None
        ),
        "wake_to_action_latency_p50_seconds": (
            float(np.percentile(wake_to_action_latencies, 50)) if wake_to_action_latencies else None
        ),
        "wake_to_action_latency_p95_seconds": (
            float(np.percentile(wake_to_action_latencies, 95)) if wake_to_action_latencies else None
        ),
        "matched_wakes": matched_wakes,
        "command_details": command_details,
        "unmatched_wakes": unmatched_wakes,
        "unmatched_commands": unmatched_commands,
        "condition_metrics": condition_metrics,
    }
    if save_windows:
        result["windows"] = windows
    return result


def aggregate(session_results: list[dict[str, Any]]) -> dict[str, Any]:
    duration = sum(float(row["duration_seconds"]) for row in session_results)
    negative_exposure_seconds = sum(
        float(row.get("negative_exposure_seconds", 0.0)) for row in session_results
    )
    wake_total = sum(int(row["driver_wake_events"]) for row in session_results)
    wake_detected = sum(int(row["driver_wakes_detected"]) for row in session_results)
    command_total = sum(int(row["driver_command_events"]) for row in session_results)
    correct = sum(int(row.get("correct_commands", 0)) for row in session_results)
    false_accepts = sum(int(row["false_accept_events"]) for row in session_results)
    negative_false_accepts = sum(int(row.get("negative_interval_false_accept_events", 0)) for row in session_results)
    false_actions = sum(int(row["false_action_events"]) for row in session_results)
    duration_hours = duration / 3600.0
    wake_latencies = [
        float(item["prediction"]["window_center_seconds"]) - event_time(item["truth"], "start_s")
        for row in session_results
        for item in row.get("matched_wakes", [])
    ]
    command_latencies = [
        float(detail["prediction"]["window_center_seconds"]) - event_time(detail["truth"], "start_s")
        for row in session_results
        for detail in row.get("command_details", [])
        if detail.get("prediction") is not None
    ]
    by_condition: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "exposure_seconds": 0.0,
            "driver_wake_events": 0,
            "driver_wakes_detected": 0,
            "driver_command_events": 0,
            "commands_with_action": 0,
            "correct_commands": 0,
            "false_accept_events": 0,
            "negative_interval_false_accept_events": 0,
            "false_action_events": 0,
        }
    )
    for session_result in session_results:
        for condition, metrics in session_result.get("condition_metrics", {}).items():
            destination = by_condition[condition]
            for key in destination:
                destination[key] += float(metrics.get(key, 0.0))
    for condition, metrics in by_condition.items():
        exposure_hours = float(metrics["exposure_seconds"]) / 3600.0
        metrics["driver_wake_recall"] = metrics["driver_wakes_detected"] / max(metrics["driver_wake_events"], 1)
        metrics["command_accuracy"] = metrics["correct_commands"] / max(metrics["driver_command_events"], 1)
        metrics["false_accepts_per_hour"] = metrics["false_accept_events"] / max(exposure_hours, 1.0e-8)
        metrics["negative_interval_false_accepts_per_hour"] = (
            metrics["negative_interval_false_accept_events"] / max(exposure_hours, 1.0e-8)
        )
        metrics["false_actions_per_hour"] = metrics["false_action_events"] / max(exposure_hours, 1.0e-8)
        metrics["false_accepts_per_negative_hour"] = _rate_per_hour(
            metrics["false_accept_events"], metrics["exposure_seconds"]
        )
        metrics["negative_interval_false_accepts_per_negative_hour"] = _rate_per_hour(
            metrics["negative_interval_false_accept_events"], metrics["exposure_seconds"]
        )
        metrics["false_actions_per_negative_hour"] = _rate_per_hour(
            metrics["false_action_events"], metrics["exposure_seconds"]
        )
        metrics["confidence_intervals"] = _confidence_intervals(
            driver_wakes_detected=metrics["driver_wakes_detected"],
            driver_wake_events=metrics["driver_wake_events"],
            correct_commands=metrics["correct_commands"],
            driver_command_events=metrics["driver_command_events"],
            false_accept_events=metrics["false_accept_events"],
            negative_interval_false_accept_events=metrics["negative_interval_false_accept_events"],
            false_action_events=metrics["false_action_events"],
            negative_exposure_seconds=float(metrics["exposure_seconds"]),
        )
    return {
        "sessions": len(session_results),
        "duration_seconds": duration,
        "duration_hours": duration_hours,
        "driver_wake_events": wake_total,
        "driver_wakes_detected": wake_detected,
        "driver_wake_recall": wake_detected / max(wake_total, 1),
        "driver_command_events": command_total,
        "correct_commands": correct,
        "command_accuracy": correct / max(command_total, 1),
        "false_accept_events": false_accepts,
        "false_accepts_per_hour": false_accepts / max(duration_hours, 1.0e-8),
        "negative_interval_false_accept_events": negative_false_accepts,
        "negative_interval_false_accepts_per_hour": negative_false_accepts / max(duration_hours, 1.0e-8),
        "false_action_events": false_actions,
        "false_actions_per_hour": false_actions / max(duration_hours, 1.0e-8),
        "negative_exposure_seconds": negative_exposure_seconds,
        "negative_exposure_hours": negative_exposure_seconds / 3600.0,
        "false_accepts_per_negative_hour": _rate_per_hour(
            false_accepts, negative_exposure_seconds
        ),
        "negative_interval_false_accepts_per_negative_hour": _rate_per_hour(
            negative_false_accepts, negative_exposure_seconds
        ),
        "false_actions_per_negative_hour": _rate_per_hour(
            false_actions, negative_exposure_seconds
        ),
        "confidence_intervals": _confidence_intervals(
            driver_wakes_detected=wake_detected,
            driver_wake_events=wake_total,
            correct_commands=correct,
            driver_command_events=command_total,
            false_accept_events=false_accepts,
            negative_interval_false_accept_events=negative_false_accepts,
            false_action_events=false_actions,
            negative_exposure_seconds=negative_exposure_seconds,
        ),
        "wake_latency_p50_seconds": float(np.percentile(wake_latencies, 50)) if wake_latencies else None,
        "wake_latency_p95_seconds": float(np.percentile(wake_latencies, 95)) if wake_latencies else None,
        "command_to_action_latency_p50_seconds": (
            float(np.percentile(command_latencies, 50)) if command_latencies else None
        ),
        "command_to_action_latency_p95_seconds": (
            float(np.percentile(command_latencies, 95)) if command_latencies else None
        ),
        "by_condition": {condition: dict(metrics) for condition, metrics in sorted(by_condition.items())},
    }


def evaluate_group(
    pipeline: Any,
    sessions: list[Session],
    *,
    hop_seconds: float,
    presence_threshold: float,
    kws_threshold: float,
    command_presence_threshold: float,
    speaker_gate_threshold: float | None,
    command_speaker_gate_threshold: float | None,
    intent_confidence_threshold: float | None,
    device: torch.device,
    save_windows: bool,
    variant: str = "full",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details = [
        run_session(
            pipeline,
            session,
            hop_seconds=hop_seconds,
            presence_threshold=presence_threshold,
            kws_threshold=kws_threshold,
            command_presence_threshold=command_presence_threshold,
            speaker_gate_threshold=speaker_gate_threshold,
            command_speaker_gate_threshold=command_speaker_gate_threshold,
            intent_confidence_threshold=intent_confidence_threshold,
            device=device,
            save_windows=save_windows,
            variant=variant,
        )
        for session in sessions
    ]
    return aggregate(details), details


def evaluate_direct_command_oracle(
    pipeline: Any,
    sessions: list[Session],
    *,
    device: torch.device,
    use_tse: bool = True,
) -> dict[str, Any]:
    """Score labeled command windows without wake-word gating.

    This is intentionally an oracle segmentation baseline: the evaluator is
    given the annotated command interval and therefore does not measure wake
    recall, false accepts, endpointing, or action safety. It answers only how
    much command intent accuracy changes when wake-word decisions are removed.
    """

    num_mics = int(getattr(pipeline.stage1, "frontend", None).num_mics) if getattr(pipeline.stage1, "frontend", None) is not None else 1
    stage1_seconds = float(getattr(pipeline, "segment_seconds", 3.0))
    window_samples = max(1, int(round(stage1_seconds * SAMPLE_RATE)))
    total = 0
    correct = 0
    by_condition: dict[str, dict[str, int]] = defaultdict(lambda: {"commands": 0, "correct": 0})
    predictions: list[dict[str, Any]] = []
    pipeline.eval()
    with torch.inference_mode():
        for session in sessions:
            microphones, playback = split_capture_audio(
                session.audio,
                session.metadata,
                model_mics=num_mics,
            )
            microphones = fit_channels(microphones, num_mics)
            mixture = microphones.mean(axis=1).astype(np.float32)
            enrollment = load_enrollment(session, SAMPLE_RATE)
            enrollment_tensor = torch.from_numpy(enrollment).to(device).unsqueeze(0)
            for truth in [row for row in session.annotations if row.get("event") == "driver_command"]:
                center = 0.5 * (event_time(truth, "start_s") + event_time(truth, "end_s"))
                start = max(0, min(len(mixture), int(round(center * SAMPLE_RATE - window_samples / 2.0))))
                end = min(len(mixture), start + window_samples)
                mix_window = np.zeros(window_samples, dtype=np.float32)
                mc_window = np.zeros((microphones.shape[1], window_samples), dtype=np.float32)
                count = max(0, end - start)
                if count:
                    mix_window[:count] = mixture[start:end]
                    mc_window[:, :count] = microphones[start:end].T
                playback_tensor = None
                if playback is not None:
                    playback_window = np.zeros(window_samples, dtype=np.float32)
                    if count:
                        playback_window[:count] = playback[start:end]
                    playback_tensor = torch.from_numpy(playback_window).to(device).unsqueeze(0)
                mix_tensor = torch.from_numpy(mix_window).to(device).unsqueeze(0)
                mc_tensor = torch.from_numpy(mc_window).to(device).unsqueeze(0)
                extracted, _, _, _ = pipeline.stage1(
                    mix_tensor,
                    enrollment_tensor,
                    multichannel=mc_tensor,
                    playback_ref=playback_tensor,
                )
                intent_input = extracted if use_tse else mix_tensor
                intent_input = normalize_extracted_waveform(intent_input)
                probabilities = torch.softmax(
                    pipeline.intent(pipeline._prepare_intent_input(intent_input)),
                    dim=-1,
                )[0]
                predicted_id = str(int(probabilities.argmax().item()))
                predicted_intent = (
                    pipeline.label_map.get(predicted_id, predicted_id)
                    if pipeline.label_map
                    else predicted_id
                )
                expected_intent = str(truth.get("expected_intent"))
                condition = event_condition(truth, session)
                is_correct = predicted_intent == expected_intent
                total += 1
                correct += int(is_correct)
                by_condition[condition]["commands"] += 1
                by_condition[condition]["correct"] += int(is_correct)
                predictions.append(
                    {
                        "session_id": session.session_id,
                        "driver_id": session.driver_id,
                        "condition": condition,
                        "expected_intent": expected_intent,
                        "predicted_intent": predicted_intent,
                        "confidence": float(probabilities.max().cpu()),
                        "correct": bool(is_correct),
                    }
                )
    return {
        "oracle_segmentation": True,
        "wake_word_steps_enabled": False,
        "use_tse": bool(use_tse),
        "sessions": len(sessions),
        "driver_command_events": total,
        "correct_commands": correct,
        "command_accuracy": correct / max(total, 1),
        "by_condition": {
            condition: {
                **counts,
                "command_accuracy": counts["correct"] / max(counts["commands"], 1),
            }
            for condition, counts in sorted(by_condition.items())
        },
        "predictions": predictions,
    }


def choose_thresholds(
    pipeline: Any,
    sessions: list[Session],
    *,
    presence_values: list[float],
    kws_values: list[float],
    command_presence_values: list[float],
    speaker_gate_values: list[float | None],
    command_speaker_gate_threshold: float | None,
    intent_confidence_values: list[float | None],
    hop_seconds: float,
    device: torch.device,
    max_false_accepts_per_hour: float,
    max_false_actions_per_hour: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for presence in presence_values:
        for kws in kws_values:
            for command_presence in command_presence_values:
                for speaker_gate in speaker_gate_values:
                    for intent_confidence in intent_confidence_values:
                        summary, _ = evaluate_group(
                            pipeline,
                            sessions,
                            hop_seconds=hop_seconds,
                            presence_threshold=presence,
                            kws_threshold=kws,
                            command_presence_threshold=command_presence,
                            speaker_gate_threshold=speaker_gate,
                            command_speaker_gate_threshold=command_speaker_gate_threshold,
                            intent_confidence_threshold=intent_confidence,
                            device=device,
                            save_windows=False,
                        )
                        safe_under_accept_limit = (
                            _safety_rate(
                                summary,
                                "false_accepts_per_negative_hour",
                                "false_accepts_per_hour",
                            )
                            <= max_false_accepts_per_hour
                        )
                        safe_under_action_limit = (
                            _safety_rate(
                                summary,
                                "false_actions_per_negative_hour",
                                "false_actions_per_hour",
                            )
                            <= max_false_actions_per_hour
                        )
                        safe = safe_under_accept_limit and safe_under_action_limit
                        candidates.append(
                            {
                                "presence_threshold": presence,
                                "kws_threshold": kws,
                                "command_presence_threshold": command_presence,
                                "speaker_gate_threshold": speaker_gate,
                                "command_speaker_gate_threshold": command_speaker_gate_threshold,
                                "intent_confidence_threshold": intent_confidence,
                                "safe_under_false_action_limit": safe_under_action_limit,
                                "safe_under_false_accept_limit": safe_under_accept_limit,
                                "safe_under_safety_limits": safe,
                                "false_accepts_per_negative_hour": summary.get(
                                    "false_accepts_per_negative_hour"
                                ),
                                "false_actions_per_negative_hour": summary.get(
                                    "false_actions_per_negative_hour"
                                ),
                                "summary": summary,
                            }
                        )
    # Threshold selection must satisfy the complete safety contract.  Using
    # only the false-action filter here can select a point that never acts but
    # still accepts passenger/no-command wake words at an unacceptable rate.
    safe_candidates = [row for row in candidates if row["safe_under_safety_limits"]]
    if safe_candidates:
        selected = max(
            safe_candidates,
            key=lambda row: (
                float(row["summary"]["driver_wake_recall"]),
                float(row["summary"]["command_accuracy"]),
                -_safety_rate(
                    row["summary"],
                    "false_actions_per_negative_hour",
                    "false_actions_per_hour",
                ),
            ),
        )
        selected["selection_status"] = "safe_tuning_operating_point"
    else:
        # Continue with the least unsafe point so the report contains useful
        # diagnostics, but never present it as a frozen production setting.
        selected = min(
            candidates,
            key=lambda row: (
                _safety_rate(
                    row["summary"],
                    "false_accepts_per_negative_hour",
                    "false_accepts_per_hour",
                ),
                _safety_rate(
                    row["summary"],
                    "false_actions_per_negative_hour",
                    "false_actions_per_hour",
                ),
                -float(row["summary"]["driver_wake_recall"]),
                -float(row["summary"]["command_accuracy"]),
            ),
        )
        selected["selection_status"] = "no_safe_tuning_point_diagnostic_fallback"
    selected["safe_candidate_count"] = len(safe_candidates)
    return selected, candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "data/raw/custom_incar/sessions")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/eval_end_to_end_cavemove_target_kws_temporal2_candidate.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.yaml")
    parser.add_argument("--tuning-drivers", nargs="*", default=[])
    parser.add_argument("--test-drivers", nargs="*", default=[])
    parser.add_argument("--presence-thresholds", nargs="+", type=float, default=[0.5, 0.7, 0.9])
    parser.add_argument("--kws-thresholds", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    parser.add_argument(
        "--command-presence-thresholds",
        nargs="+",
        type=float,
        default=[0.1, 0.3, 0.5, 0.7],
    )
    parser.add_argument(
        "--speaker-gate-thresholds",
        nargs="+",
        type=float,
        default=[-1.0, 0.15, 0.20, 0.25],
        help="Use -1 to represent a disabled speaker gate.",
    )
    parser.add_argument(
        "--intent-confidence-thresholds",
        nargs="+",
        type=float,
        default=[-1.0, 0.53],
        help="Use -1 to represent a disabled intent-confidence safety gate.",
    )
    parser.add_argument("--hop-seconds", type=float, default=0.25)
    parser.add_argument(
        "--min-driver-wake-recall",
        type=float,
        default=0.90,
        help="held-out gate for driver wake recall used in production_valid",
    )
    parser.add_argument(
        "--min-command-accuracy",
        type=float,
        default=0.90,
        help="held-out gate for end-to-end command accuracy used in production_valid",
    )
    parser.add_argument(
        "--max-false-accepts-per-hour",
        type=float,
        default=1.0,
        help="safety limit used during tuning and the held-out production gate",
    )
    parser.add_argument("--max-false-actions-per-hour", type=float, default=1.0)
    parser.add_argument(
        "--run-ablations",
        action="store_true",
        help=(
            "after freezing full-system tuning thresholds, evaluate the real-driver "
            "test sessions with each component ablation and a command-only oracle"
        ),
    )
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "logs/real_driver_pipeline_evaluation.json")
    args = parser.parse_args()

    validation = validate(args.root, set(args.tuning_drivers), set(args.test_drivers))
    if not validation["valid"]:
        report = {
            "benchmark": "real_driver_continuous_pipeline",
            "valid_input": False,
            "validation": validation,
            "production_valid": False,
            "limitation": "No evaluation was run because the permitted session tree failed validation.",
        }
        write_metrics(resolve_path(args.output), report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2

    cfg = load_stage_config(args.config, args.paths)
    device = get_device()
    pipeline = load_pipeline(cfg, cfg["paths"], device)
    sessions = [load_session(args.root, row["session_id"]) for row in validation["sessions"]]
    tuning = [session for session in sessions if session.split == "tuning"]
    test = [session for session in sessions if session.split == "test"]
    if not tuning or not test:
        raise RuntimeError("Both non-empty tuning and test session groups are required")

    selected, threshold_grid = choose_thresholds(
        pipeline,
        tuning,
        presence_values=[float(value) for value in args.presence_thresholds],
        kws_values=[float(value) for value in args.kws_thresholds],
        command_presence_values=[float(value) for value in args.command_presence_thresholds],
        speaker_gate_values=[None if float(value) < -0.999 else float(value) for value in args.speaker_gate_thresholds],
        command_speaker_gate_threshold=(
            float(cfg["command_speaker_gate_threshold"])
            if cfg.get("command_speaker_gate_threshold") is not None
            else None
        ),
        intent_confidence_values=[
            None if float(value) < -0.999 else float(value)
            for value in args.intent_confidence_thresholds
        ],
        hop_seconds=float(args.hop_seconds),
        device=device,
        max_false_accepts_per_hour=float(args.max_false_accepts_per_hour),
        max_false_actions_per_hour=float(args.max_false_actions_per_hour),
    )
    selected_presence = float(selected["presence_threshold"])
    selected_kws = float(selected["kws_threshold"])
    tuning_summary, tuning_details = evaluate_group(
        pipeline,
        tuning,
        hop_seconds=float(args.hop_seconds),
        presence_threshold=selected_presence,
        kws_threshold=selected_kws,
        command_presence_threshold=float(selected["command_presence_threshold"]),
        speaker_gate_threshold=selected["speaker_gate_threshold"],
        command_speaker_gate_threshold=selected["command_speaker_gate_threshold"],
        intent_confidence_threshold=selected["intent_confidence_threshold"],
        device=device,
        save_windows=False,
    )
    test_summary, test_details = evaluate_group(
        pipeline,
        test,
        hop_seconds=float(args.hop_seconds),
        presence_threshold=selected_presence,
        kws_threshold=selected_kws,
        command_presence_threshold=float(selected["command_presence_threshold"]),
        speaker_gate_threshold=selected["speaker_gate_threshold"],
        command_speaker_gate_threshold=selected["command_speaker_gate_threshold"],
        intent_confidence_threshold=selected["intent_confidence_threshold"],
        device=device,
        save_windows=True,
    )
    ablation_study = None
    if args.run_ablations:
        ablation_variants: dict[str, Any] = {}
        for variant_name in ABLATION_VARIANTS:
            if variant_name == "full":
                continue
            variant_summary, variant_details = evaluate_group(
                pipeline,
                test,
                hop_seconds=float(args.hop_seconds),
                presence_threshold=selected_presence,
                kws_threshold=selected_kws,
                command_presence_threshold=float(selected["command_presence_threshold"]),
                speaker_gate_threshold=selected["speaker_gate_threshold"],
                command_speaker_gate_threshold=selected["command_speaker_gate_threshold"],
                intent_confidence_threshold=selected["intent_confidence_threshold"],
                device=device,
                save_windows=False,
                variant=variant_name,
            )
            ablation_variants[variant_name] = {
                "summary": variant_summary,
                "sessions": variant_details,
            }
        ablation_study = {
            "threshold_source": "full-system tuning split; held-out ablations use the frozen full-system operating point",
            "variants": ablation_variants,
            "direct_command_oracle": evaluate_direct_command_oracle(
                pipeline,
                test,
                device=device,
                use_tse=True,
            ),
        }
    study_size_ready = (
        validation["session_count"] >= 3
        and len(validation["drivers"]) >= 3
        and bool(args.tuning_drivers)
        and bool(args.test_drivers)
        and not bool(set(args.tuning_drivers) & set(args.test_drivers))
        and set(validation["drivers"]).issubset(set(args.tuning_drivers) | set(args.test_drivers))
        and bool(tuning)
        and bool(test)
    )
    production_valid = (
        study_size_ready
        and selected.get("selection_status") == "safe_tuning_operating_point"
        and float(test_summary["driver_wake_recall"]) >= float(args.min_driver_wake_recall)
        and float(test_summary["command_accuracy"]) >= float(args.min_command_accuracy)
        and _safety_rate(
            test_summary,
            "false_accepts_per_negative_hour",
            "false_accepts_per_hour",
        ) <= float(args.max_false_accepts_per_hour)
        and _safety_rate(
            test_summary,
            "false_actions_per_negative_hour",
            "false_actions_per_hour",
        ) <= float(args.max_false_actions_per_hour)
    )
    report = {
        "benchmark": "real_driver_continuous_pipeline",
        "valid_input": True,
        "production_valid": production_valid,
        "device": str(device),
        "config": str(resolve_path(args.config)),
        "checkpoints": cfg.get("checkpoints"),
        "sample_rate": SAMPLE_RATE,
        "hop_seconds": float(args.hop_seconds),
        "tuning_drivers": sorted(set(args.tuning_drivers)),
        "test_drivers": sorted(set(args.test_drivers)),
        "selected_thresholds": {
            "presence_threshold": selected_presence,
            "kws_threshold": selected_kws,
            "command_presence_threshold": float(selected["command_presence_threshold"]),
            "speaker_gate_threshold": selected["speaker_gate_threshold"],
            "command_speaker_gate_threshold": selected["command_speaker_gate_threshold"],
            "intent_confidence_threshold": selected["intent_confidence_threshold"],
            "max_false_actions_per_hour": float(args.max_false_actions_per_hour),
            "max_false_accepts_per_hour": float(args.max_false_accepts_per_hour),
            "selection_status": selected.get("selection_status"),
            "safe_tuning_candidate_count": int(selected.get("safe_candidate_count", 0)),
            "selection_rule": "maximize wake recall, then command accuracy, subject to false-accept and false-action limits",
        },
        "production_gates": {
            "minimum_sessions": 3,
            "minimum_drivers": 3,
            "minimum_driver_wake_recall": float(args.min_driver_wake_recall),
            "minimum_command_accuracy": float(args.min_command_accuracy),
            "maximum_false_accepts_per_hour": float(args.max_false_accepts_per_hour),
            "maximum_false_actions_per_hour": float(args.max_false_actions_per_hour),
            "study_size_pass": study_size_ready,
            "safe_tuning_operating_point_pass": selected.get("selection_status") == "safe_tuning_operating_point",
            "held_out_wake_recall_pass": float(test_summary["driver_wake_recall"])
            >= float(args.min_driver_wake_recall),
            "held_out_command_accuracy_pass": float(test_summary["command_accuracy"])
            >= float(args.min_command_accuracy),
            "held_out_false_accept_pass": _safety_rate(
                test_summary,
                "false_accepts_per_negative_hour",
                "false_accepts_per_hour",
            ) <= float(args.max_false_accepts_per_hour),
            "held_out_false_action_pass": _safety_rate(
                test_summary,
                "false_actions_per_negative_hour",
                "false_actions_per_hour",
            ) <= float(args.max_false_actions_per_hour),
            "rate_denominator": "union of labeled negative intervals when available; legacy total-session duration otherwise",
        },
        "tuning": {
            "summary": tuning_summary,
            "sessions": tuning_details,
            "threshold_grid": threshold_grid,
        },
        "held_out_test": {
            "summary": test_summary,
            "sessions": test_details,
        },
        "ablation_study": ablation_study,
        "validation": validation,
        "limitation": (
            "A pilot result is not broad real-world reliability evidence. Report driver/session counts and exposure "
            "hours, keep the held-out test frozen, and do not claim production readiness unless every declared gate passes."
        ),
    }
    output = resolve_path(args.output)
    write_metrics(output, report)
    print(json.dumps({"output": str(output), "selected_thresholds": report["selected_thresholds"], "test": test_summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
