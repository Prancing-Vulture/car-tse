#!/usr/bin/env python
"""Run a same-scene ablation matrix for the in-car assistant.

The matrix uses the existing continuous cross-corpus proxy scene builder. It
is intentionally not presented as a real-driver benchmark: the public wake
and command clips are not verified to belong to the same speaker.

Variants:

* ``full``: TSE + presence + target-speaker KWS + temporal smoothing;
* ``no_tse``: channel-mean raw mixture in place of TSE output;
* ``no_presence``: TSE + target-speaker KWS, presence gate disabled;
* ``no_speaker_matching``: TSE + generic KWS, target conditioning disabled;
* ``no_wake_smoothing``: TSE + presence + target-speaker KWS, one-hit wake;
* ``direct_command_oracle``: intent on the labeled command window without
  wake-word gating (an oracle segmentation baseline, not a deployable path).

The same generated scenes and checkpoints are used for every variant. Results
include settings, source paths, scene-level event outputs, and the code
version when available so the comparison can be repeated.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.io.audio import read_mono, read_multichannel
from car_tse.pipeline.in_car_pipeline import (
    PipelineState,
    normalize_extracted_waveform,
    update_kws_state,
)
from car_tse.train.utils import get_device, write_metrics
from eval_continuous_full_pipeline_proxy import (
    build_scene,
    fit_start,
    load_jsonl,
    load_pipeline,
    resolve_path,
    select_noise_files,
    speaker_key,
)
from eval_continuous_kws import prepare_channels


VARIANTS: dict[str, dict[str, Any]] = {
    "full": {
        "use_tse": True,
        "use_presence": True,
        "use_speaker_matching": True,
        "use_wake_smoothing": True,
        "use_wake_gate": True,
    },
    "no_tse": {
        "use_tse": False,
        "use_presence": True,
        "use_speaker_matching": False,
        "use_wake_smoothing": True,
        "use_wake_gate": True,
    },
    "no_presence": {
        "use_tse": True,
        "use_presence": False,
        "use_speaker_matching": True,
        "use_wake_smoothing": True,
        "use_wake_gate": True,
    },
    "no_speaker_matching": {
        "use_tse": True,
        "use_presence": True,
        "use_speaker_matching": False,
        "use_wake_smoothing": True,
        "use_wake_gate": True,
    },
    "no_wake_smoothing": {
        "use_tse": True,
        "use_presence": True,
        "use_speaker_matching": True,
        "use_wake_smoothing": False,
        "use_wake_gate": True,
    },
    "raw_intent_after_wake": {
        "use_tse": True,
        "use_presence": True,
        "use_speaker_matching": True,
        "use_wake_smoothing": True,
        "use_wake_gate": True,
        "intent_source": "raw",
    },
    "fused_intent_after_wake": {
        "use_tse": True,
        "use_presence": True,
        "use_speaker_matching": True,
        "use_wake_smoothing": True,
        "use_wake_gate": True,
        "intent_source": "mean_logits",
    },
    "direct_command_oracle": {
        "use_tse": True,
        "use_presence": False,
        "use_speaker_matching": False,
        "use_wake_smoothing": False,
        "use_wake_gate": False,
    },
}


def git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def pad_or_crop(wav: torch.Tensor, samples: int) -> torch.Tensor:
    if wav.shape[-1] >= samples:
        return wav[..., :samples]
    return torch.nn.functional.pad(wav, (0, samples - wav.shape[-1]))


def kws_score(model: torch.nn.Module, wav: torch.Tensor, sample_rate: int, seconds: float, wake_class_id: int) -> float:
    samples = max(1, int(round(sample_rate * seconds)))
    model_input = pad_or_crop(wav, samples)
    probabilities = torch.softmax(model(model_input), dim=-1)
    return float(probabilities[..., wake_class_id].mean().detach().cpu())


def target_kws_score(
    pipeline: Any,
    wav: torch.Tensor,
    enrollment_embedding: torch.Tensor | None,
    extracted_embedding: torch.Tensor | None,
) -> float | None:
    if pipeline.target_kws is None or enrollment_embedding is None or extracted_embedding is None:
        return None
    logits = pipeline.target_kws(wav, enrollment_embedding, extracted_embedding)
    probabilities = torch.softmax(logits, dim=-1)
    return float(probabilities[..., pipeline.wake_class_id].mean().detach().cpu())


def collect_window_scores(
    pipeline: Any,
    mixture: np.ndarray,
    multichannel: np.ndarray,
    enrollment: np.ndarray,
    truth: dict[str, Any],
    sample_rate: int,
    stage1_seconds: float,
    hop_seconds: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Run the expensive Stage 1 pass once and retain downstream scores."""

    window_samples = int(round(stage1_seconds * sample_rate))
    hop_samples = max(1, int(round(hop_seconds * sample_rate)))
    enrollment_tensor = torch.from_numpy(enrollment).to(device).unsqueeze(0)
    with torch.inference_mode():
        enrollment_embedding = pipeline.stage1.speaker_embedding(enrollment_tensor)
        rows: list[dict[str, Any]] = []
        for start in range(0, max(1, len(mixture)), hop_samples):
            end = min(start + window_samples, len(mixture))
            mix_window = np.zeros(window_samples, dtype=np.float32)
            mc_window = np.zeros((multichannel.shape[1], window_samples), dtype=np.float32)
            if end > start:
                mix_window[: end - start] = mixture[start:end]
                mc_window[:, : end - start] = multichannel[start:end].T
            mix_tensor = torch.from_numpy(mix_window).to(device).unsqueeze(0)
            mc_tensor = torch.from_numpy(mc_window).to(device).unsqueeze(0)
            extracted, _, _, _ = pipeline.stage1(
                mix_tensor,
                enrollment_tensor,
                multichannel=mc_tensor,
            )
            extracted_embedding = pipeline.stage1.speaker_embedding(extracted)
            speaker_similarity = torch.nn.functional.cosine_similarity(
                enrollment_embedding,
                extracted_embedding,
                dim=-1,
            ).mean()
            extracted_downstream = normalize_extracted_waveform(extracted)
            raw_downstream = normalize_extracted_waveform(mix_tensor)
            target_kws_input = pad_or_crop(
                extracted_downstream,
                max(1, int(round(pipeline.kws_sample_rate * pipeline.kws_input_seconds))),
            )
            raw_kws_input = pad_or_crop(
                raw_downstream,
                max(1, int(round(pipeline.kws_sample_rate * pipeline.kws_input_seconds))),
            )
            center = start / sample_rate + stage1_seconds / 2.0
            rows.append(
                {
                    "window_index": len(rows),
                    "window_start_seconds": start / sample_rate,
                    "window_center_seconds": center,
                    "target_active": (
                        float(truth["target_wake_start_seconds"]) - 0.5
                        <= center
                        <= float(truth["target_command_end_seconds"]) + 0.5
                    ),
                    "passenger_region": 5.0 <= center <= 11.5,
                    "presence_probability": float(pipeline.presence_probability(extracted).mean().cpu()),
                    "raw_presence_probability": float(pipeline.presence_probability(raw_downstream).mean().cpu()),
                    "speaker_similarity": float(speaker_similarity.detach().cpu()),
                    "generic_kws_probability": kws_score(
                        pipeline.kws,
                        target_kws_input,
                        pipeline.kws_sample_rate,
                        pipeline.kws_input_seconds,
                        pipeline.wake_class_id,
                    ),
                    "raw_generic_kws_probability": kws_score(
                        pipeline.kws,
                        raw_kws_input,
                        pipeline.kws_sample_rate,
                        pipeline.kws_input_seconds,
                        pipeline.wake_class_id,
                    ),
                    "target_kws_probability": target_kws_score(
                        pipeline,
                        target_kws_input,
                        enrollment_embedding,
                        extracted_embedding,
                    ),
                    "extracted_waveform": extracted_downstream.detach(),
                    "raw_waveform": raw_downstream.detach(),
                }
            )
    return rows


def is_target_wake(center: float, truth: dict[str, Any]) -> bool:
    return (
        float(truth["target_wake_start_seconds"]) - 0.25
        <= center
        <= float(truth["target_wake_end_seconds"]) + 3.0
    )


def summarize_variant(
    name: str,
    settings: dict[str, Any],
    pipeline: Any,
    rows: list[dict[str, Any]],
    truth: dict[str, Any],
    sample_rate: int,
    kws_threshold: float,
    presence_threshold: float,
    device: torch.device,
) -> dict[str, Any]:
    if not settings["use_wake_gate"]:
        command_start = float(truth["target_command_start_seconds"])
        row = min(rows, key=lambda item: abs(item["window_start_seconds"] - command_start))
        probabilities = torch.softmax(
            pipeline.intent(pad_or_crop(row["extracted_waveform"], int(round(pipeline.intent_sample_rate * pipeline.intent_input_seconds)))),
            dim=-1,
        )[0]
        predicted_id = str(int(probabilities.argmax().item()))
        predicted = pipeline.label_map.get(predicted_id, predicted_id) if pipeline.label_map else predicted_id
        return {
            "variant": name,
            "mode": "oracle_command_window_without_wake_gate",
            "target_wake_detected": None,
            "target_intent": predicted,
            "target_intent_correct": predicted == str(truth["expected_intent"]),
            "false_wake_events": None,
            "false_action_events": None,
            "false_action_rate_per_hour": None,
            "target_wake_latency_seconds": None,
            "intent_confidence": float(probabilities.max().detach().cpu()),
            "note": "Uses the labeled command window; wake recall and false-action rate are not comparable.",
        }

    state = PipelineState()
    wake_events: list[dict[str, Any]] = []
    command_events: list[dict[str, Any]] = []
    suppressed_command_events: list[dict[str, Any]] = []
    smoothing_window = int(settings.get("smoothing_window", pipeline.kws_smoothing_window)) if settings["use_wake_smoothing"] else 1
    required_hits = int(settings.get("required_hits", pipeline.kws_required_hits)) if settings["use_wake_smoothing"] else 1
    refractory_calls = int(settings.get("refractory_calls", pipeline.kws_refractory_calls)) if settings["use_wake_smoothing"] else 0
    command_timeout_calls = int(settings.get("command_timeout_calls", pipeline.command_timeout_calls))
    command_capture_calls = max(1, int(settings.get("command_capture_calls", 1)))
    command_aggregation = str(settings.get("command_aggregation", "mean_logits"))
    speaker_gate_threshold = settings.get("speaker_gate_threshold")
    pending_intents: list[tuple[torch.Tensor, dict[str, Any]]] = []

    with torch.inference_mode():
        for row in rows:
            accepted = True
            presence = row["presence_probability"] if settings["use_tse"] else row["raw_presence_probability"]
            active_presence_threshold = (
                float(settings.get("command_presence_threshold", getattr(pipeline, "command_presence_threshold", presence_threshold)))
                if state.awaiting_command
                else presence_threshold
            )
            speaker_rejected = bool(
                settings.get("use_speaker_gate", False)
                and speaker_gate_threshold is not None
                and float(row.get("speaker_similarity", -1.0)) < float(speaker_gate_threshold)
            )
            if (settings["use_presence"] and presence < active_presence_threshold) or speaker_rejected:
                accepted = False
                state.kws_scores.clear()
                if state.awaiting_command:
                    state.command_wait_calls += 1
                    if state.command_wait_calls >= command_timeout_calls:
                        state.awaiting_command = False
                        state.command_wait_calls = 0
                        state.kws_refractory_left = 0
                        pending_intents.clear()
                else:
                    state.kws_refractory_left = 0

            if not accepted:
                continue

            if state.awaiting_command:
                waveform = row["extracted_waveform"] if settings["use_tse"] else row["raw_waveform"]
                intent_samples = int(round(pipeline.intent_sample_rate * pipeline.intent_input_seconds))
                intent_input = pad_or_crop(waveform, intent_samples)
                logits = pipeline.intent(intent_input)[0]
                if settings.get("intent_source") == "raw" and settings["use_tse"]:
                    raw_input = pad_or_crop(row["raw_waveform"], intent_samples)
                    logits = pipeline.intent(raw_input)[0]
                elif settings.get("intent_source") == "mean_logits" and settings["use_tse"]:
                    raw_input = pad_or_crop(row["raw_waveform"], intent_samples)
                    raw_logits = pipeline.intent(raw_input)[0]
                    logits = (logits + raw_logits) / 2.0
                pending_intents.append((logits.detach(), row))
                if len(pending_intents) < command_capture_calls:
                    continue
                if command_aggregation == "max_confidence":
                    candidate_logits, candidate_row = max(
                        pending_intents,
                        key=lambda item: float(torch.softmax(item[0], dim=-1).max().detach().cpu()),
                    )
                    probabilities = torch.softmax(candidate_logits, dim=-1)
                    event_row = candidate_row
                else:
                    aggregate_logits = torch.stack([item[0] for item in pending_intents]).mean(dim=0)
                    probabilities = torch.softmax(aggregate_logits, dim=-1)
                    event_row = pending_intents[-1][1]
                confidence = float(probabilities.max().detach().cpu())
                min_intent_confidence = settings.get("min_intent_confidence")
                if (
                    min_intent_confidence is not None
                    and confidence < float(min_intent_confidence)
                ):
                    suppressed_command_events.append(
                        {
                            "window_index": int(event_row["window_index"]),
                            "window_center_seconds": event_row["window_center_seconds"],
                            "intent_confidence": confidence,
                            "min_intent_confidence": float(min_intent_confidence),
                        }
                    )
                    state.awaiting_command = False
                    state.command_wait_calls = 0
                    state.kws_scores.clear()
                    pending_intents.clear()
                    continue
                predicted_id = str(int(probabilities.argmax().item()))
                predicted = pipeline.label_map.get(predicted_id, predicted_id) if pipeline.label_map else predicted_id
                command_events.append(
                    {
                        "window_index": int(event_row["window_index"]),
                        "window_center_seconds": event_row["window_center_seconds"],
                        "intent": predicted,
                        "intent_confidence": confidence,
                        "capture_calls": command_capture_calls,
                        "aggregation": command_aggregation,
                    }
                )
                state.awaiting_command = False
                state.command_wait_calls = 0
                state.kws_scores.clear()
                pending_intents.clear()
                continue

            if settings["use_tse"] and settings["use_speaker_matching"]:
                score = row["target_kws_probability"]
            else:
                score = row["generic_kws_probability"] if settings["use_tse"] else row["raw_generic_kws_probability"]
            if score is None:
                score = 0.0
            wake = update_kws_state(
                state,
                float(score),
                float(kws_threshold),
                smoothing_window,
                required_hits,
                refractory_calls,
            )
            if wake:
                wake_events.append(
                    {
                        "window_index": int(row["window_index"]),
                        "window_center_seconds": row["window_center_seconds"],
                        "probability": float(score),
                    }
                )
                state.awaiting_command = True
                state.command_wait_calls = 0
                state.kws_scores.clear()

    target_wake_events = [event for event in wake_events if is_target_wake(event["window_center_seconds"], truth)]
    false_wake_events = [event for event in wake_events if event not in target_wake_events]
    target_command_events = [
        event
        for event in command_events
        if float(truth["target_command_start_seconds"]) - 0.5
        <= event["window_center_seconds"]
        <= float(truth["target_command_end_seconds"]) + 3.0
    ]
    target_command_event = target_command_events[0] if target_command_events else None
    false_action_events = [event for event in command_events if event not in target_command_events]
    target_wake_event = target_wake_events[0] if target_wake_events else None
    latency = None
    if target_wake_event:
        latency = float(target_wake_event["window_center_seconds"] - truth["target_wake_start_seconds"])
    duration = float(truth["duration_seconds"])
    return {
        "variant": name,
        "mode": "streaming_wake_then_command",
        "target_wake_detected": bool(target_wake_event),
        "target_intent": target_command_event.get("intent") if target_command_event else None,
        "target_intent_correct": bool(
            target_command_event and target_command_event.get("intent") == str(truth["expected_intent"])
        ),
        "false_wake_events": len(false_wake_events),
        "false_action_events": len(false_action_events),
        "false_action_rate_per_hour": len(false_action_events) * 3600.0 / max(duration, 1.0e-6),
        "suppressed_command_events": len(suppressed_command_events),
        "target_wake_latency_seconds": latency,
        "wake_events": wake_events,
        "command_events": command_events,
        "suppressed_command_events_detail": suppressed_command_events,
        "target_presence_pass_rate": float(
            np.mean([
                (row["presence_probability"] if settings["use_tse"] else row["raw_presence_probability"]) >= presence_threshold
                for row in rows
                if row["target_active"]
            ])
        ),
        "passenger_presence_pass_rate": float(
            np.mean([
                (row["presence_probability"] if settings["use_tse"] else row["raw_presence_probability"]) >= presence_threshold
                for row in rows
                if row["passenger_region"]
            ])
        ),
        "speaker_similarity_mean": float(np.mean([row["speaker_similarity"] for row in rows])),
        "speaker_gate_pass_rate": (
            float(np.mean([row["speaker_similarity"] >= float(speaker_gate_threshold) for row in rows]))
            if speaker_gate_threshold is not None
            else None
        ),
    }


def strip_tensors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in row.items()
            if not isinstance(value, torch.Tensor)
        }
        for row in rows
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/eval_end_to_end_cavemove_target_kws_temporal2_candidate.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.yaml")
    parser.add_argument("--wake-manifest", type=Path, default=REPO_ROOT / "manifests/test/stage3_kws_cabin_test.jsonl")
    parser.add_argument("--intent-manifest", type=Path, default=REPO_ROOT / "manifests/official/stage4_intent_fsc_test.jsonl")
    parser.add_argument("--noise-root", type=Path, default=REPO_ROOT / "data/raw/cavemove/extracted/cavemove_dataset")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "logs/continuous_ablation_matrix.json")
    parser.add_argument("--scenes", type=int, default=2)
    parser.add_argument("--cabin-snr-db", nargs="+", type=float, default=[-5.0, 0.0, 10.0, 20.0])
    parser.add_argument("--passenger-snr-db", type=float, default=0.0)
    parser.add_argument("--kws-threshold", type=float, default=None)
    parser.add_argument("--presence-threshold", type=float, default=None)
    parser.add_argument("--hop-seconds", type=float, default=1.0)
    parser.add_argument(
        "--speaker-gate-thresholds",
        nargs="+",
        type=float,
        default=None,
        help="Optional extra full-pipeline variants with speaker-similarity gates.",
    )
    parser.add_argument(
        "--command-presence-thresholds",
        nargs="+",
        type=float,
        default=None,
        help="Optional post-wake presence thresholds; combined with speaker-gate thresholds when both are supplied.",
    )
    parser.add_argument(
        "--intent-confidence-thresholds",
        nargs="+",
        type=float,
        default=None,
        help="Optional minimum post-wake intent confidence; combined with speaker-gate thresholds when supplied.",
    )
    parser.add_argument(
        "--enrollment-source",
        choices=("wake", "command", "same_speaker_other"),
        default="wake",
        help=(
            "Diagnostic enrollment source. 'wake' is the realistic cross-corpus proxy; "
            "'command' reuses the command clip and is a leakage-prone oracle; "
            "'same_speaker_other' uses a different FSC utterance from the command speaker."
        ),
    )
    parser.add_argument("--seed", type=int, default=20261009)
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    cfg = load_stage_config(args.config, args.paths)
    if args.kws_threshold is not None:
        cfg["kws_threshold"] = float(args.kws_threshold)
    if args.presence_threshold is not None:
        cfg["presence_threshold"] = float(args.presence_threshold)
    variant_defs = dict(VARIANTS)
    command_presence_thresholds = args.command_presence_thresholds or []
    intent_confidence_thresholds = args.intent_confidence_thresholds or []
    for threshold in args.speaker_gate_thresholds or []:
        label = f"{float(threshold):.3f}".replace("-", "m").replace(".", "p")
        settings = dict(VARIANTS["full"])
        settings.update({"use_speaker_gate": True, "speaker_gate_threshold": float(threshold)})
        variant_defs[f"full_speaker_gate_{label}"] = settings
        for command_threshold in command_presence_thresholds:
            command_label = f"{float(command_threshold):.3f}".replace("-", "m").replace(".", "p")
            joint_settings = dict(settings)
            joint_settings["command_presence_threshold"] = float(command_threshold)
            variant_defs[f"full_gate_{label}_command_presence_{command_label}"] = joint_settings
        for confidence_threshold in intent_confidence_thresholds:
            confidence_label = f"{float(confidence_threshold):.3f}".replace("-", "m").replace(".", "p")
            confidence_settings = dict(settings)
            confidence_settings["min_intent_confidence"] = float(confidence_threshold)
            variant_defs[f"full_gate_{label}_intent_confidence_{confidence_label}"] = confidence_settings
    if command_presence_thresholds and not args.speaker_gate_thresholds:
        for command_threshold in command_presence_thresholds:
            command_label = f"{float(command_threshold):.3f}".replace("-", "m").replace(".", "p")
            settings = dict(VARIANTS["full"])
            settings["command_presence_threshold"] = float(command_threshold)
            variant_defs[f"full_command_presence_{command_label}"] = settings
    if intent_confidence_thresholds and not args.speaker_gate_thresholds:
        for confidence_threshold in intent_confidence_thresholds:
            confidence_label = f"{float(confidence_threshold):.3f}".replace("-", "m").replace(".", "p")
            settings = dict(VARIANTS["full"])
            settings["min_intent_confidence"] = float(confidence_threshold)
            variant_defs[f"full_intent_confidence_{confidence_label}"] = settings
    device = get_device()
    pipeline = load_pipeline(cfg, cfg["paths"], device)
    sample_rate = int(cfg["sample_rate"])
    stage1_seconds = float(cfg.get("segment_seconds", 3.0))
    presence_threshold = float(cfg.get("presence_threshold", 0.5))
    kws_threshold = float(cfg.get("kws_threshold", 0.5))

    wake_rows = [
        row for row in load_jsonl(resolve_path(args.wake_manifest))
        if int(row.get("label", 0)) == 1 and not bool(row.get("noise_only", False))
    ]
    wake_rows.sort(key=lambda row: str(row.get("wav", "")))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in wake_rows:
        grouped[speaker_key(resolve_path(row["wav"]))].append(row)
    target_wake_rows = [rows[0] for _, rows in sorted(grouped.items()) if rows][: int(args.scenes)]
    if len(target_wake_rows) < int(args.scenes):
        raise RuntimeError("Not enough Marvin speakers for requested scenes")
    passenger_wake_row = next(
        row for row in wake_rows
        if speaker_key(resolve_path(row["wav"])) != speaker_key(resolve_path(target_wake_rows[0]["wav"]))
    )
    negative_rows = [
        row for row in load_jsonl(resolve_path(args.wake_manifest))
        if int(row.get("label", 0)) == 0 and not bool(row.get("noise_only", False))
    ]
    passenger_nonwake_row = negative_rows[0]
    all_intent_rows = load_jsonl(resolve_path(args.intent_manifest))
    intent_rows = all_intent_rows
    by_intent: dict[str, dict[str, Any]] = {}
    for row in intent_rows:
        by_intent.setdefault(str(row["intent"]), row)
    intent_rows = [by_intent[key] for key in sorted(by_intent)]
    noise_files = select_noise_files(resolve_path(args.noise_root), 20.0)
    if not noise_files:
        raise RuntimeError("No sufficiently long CAVEMOVE noise streams found")

    scene_results: list[dict[str, Any]] = []
    for scene_index in range(min(int(args.scenes), len(target_wake_rows))):
        wake_row = target_wake_rows[scene_index]
        intent_row = intent_rows[scene_index % len(intent_rows)]
        target_wake, _ = read_mono(resolve_path(wake_row["wav"]), sample_rate)
        target_command, _ = read_mono(resolve_path(intent_row["wav"]), sample_rate)
        passenger_wake, _ = read_mono(resolve_path(passenger_wake_row["wav"]), sample_rate)
        passenger_nonwake, _ = read_mono(resolve_path(passenger_nonwake_row["wav"]), sample_rate)
        target_wake = fit_start(target_wake, sample_rate)
        target_command = fit_start(target_command, 2 * sample_rate)
        passenger_wake = fit_start(passenger_wake, sample_rate)
        passenger_nonwake = fit_start(passenger_nonwake, int(1.5 * sample_rate))
        enrollment_source_path = str(resolve_path(wake_row["wav"]))
        if args.enrollment_source == "wake":
            enrollment_source = target_wake
        elif args.enrollment_source == "command":
            enrollment_source = target_command
            enrollment_source_path = str(resolve_path(intent_row["wav"]))
        else:
            command_speaker_id = str(intent_row.get("speaker_id", ""))
            candidates = [
                row for row in all_intent_rows
                if str(row.get("speaker_id", "")) == command_speaker_id
                and str(row.get("wav", "")) != str(intent_row.get("wav", ""))
            ]
            candidates.sort(key=lambda row: str(row.get("wav", "")))
            if not candidates:
                raise RuntimeError(
                    f"No different-utterance enrollment found for command speaker {command_speaker_id!r}"
                )
            enrollment_source_path = str(resolve_path(candidates[0]["wav"]))
            enrollment_source, _ = read_mono(resolve_path(candidates[0]["wav"]), sample_rate)
        enrollment = fit_start(enrollment_source, int(round(stage1_seconds * sample_rate / 2.0)))

        for snr_index, cabin_snr_db in enumerate(args.cabin_snr_db):
            noise_path = noise_files[(scene_index + snr_index) % len(noise_files)]
            raw_noise, _ = read_multichannel(noise_path, sample_rate)
            cabin_noise = prepare_channels(
                raw_noise,
                list(cfg.get("cabin_noise_channels", [0, 2, 4, 6])),
                int(cfg.get("num_mics", 4)),
            )
            mixture, multichannel, _, _, truth = build_scene(
                target_wake,
                target_command,
                passenger_wake,
                passenger_nonwake,
                cabin_noise,
                sample_rate,
                float(cabin_snr_db),
                float(args.passenger_snr_db),
            )
            truth.update(
                {
                    "expected_intent": str(intent_row["intent"]),
                    "target_wake_source": str(resolve_path(wake_row["wav"])),
                    "target_command_source": str(resolve_path(intent_row["wav"])),
                    "passenger_wake_source": str(resolve_path(passenger_wake_row["wav"])),
                    "passenger_nonwake_source": str(resolve_path(passenger_nonwake_row["wav"])),
                    "noise_source": str(noise_path),
                    "enrollment_source_wav": enrollment_source_path,
                    "speaker_consistent": args.enrollment_source in ("command", "same_speaker_other"),
                    "enrollment_source": args.enrollment_source,
                }
            )
            rows = collect_window_scores(
                pipeline,
                mixture,
                multichannel,
                enrollment,
                truth,
                sample_rate,
                stage1_seconds,
                float(args.hop_seconds),
                device,
            )
            scene_variant_results = {
                name: summarize_variant(
                    name,
                    settings,
                    pipeline,
                    rows,
                    truth,
                    sample_rate,
                    kws_threshold,
                    presence_threshold,
                    device,
                )
                for name, settings in variant_defs.items()
            }
            scene_results.append(
                {
                    "scene_index": scene_index,
                    "cabin_snr_db": float(cabin_snr_db),
                    "truth": truth,
                    "variants": scene_variant_results,
                    "window_scores": strip_tensors(rows),
                }
            )
            print(
                f"scene={scene_index} cabin_snr={cabin_snr_db:+.1f}: "
                + ", ".join(
                    f"{name}={int(bool(result['target_wake_detected'])) if result['target_wake_detected'] is not None else 'oracle:' + str(int(result['target_intent_correct']))}"
                    for name, result in scene_variant_results.items()
                ),
                flush=True,
            )

    variant_summary: dict[str, Any] = {}
    for name in variant_defs:
        results = [scene["variants"][name] for scene in scene_results]
        wake_values = [row["target_wake_detected"] for row in results if row["target_wake_detected"] is not None]
        intent_values = [bool(row["target_intent_correct"]) for row in results]
        false_wakes = [row["false_wake_events"] for row in results if row["false_wake_events"] is not None]
        false_actions = [row["false_action_events"] for row in results if row["false_action_events"] is not None]
        latencies = [row["target_wake_latency_seconds"] for row in results if row["target_wake_latency_seconds"] is not None]
        duration = sum(float(scene["truth"]["duration_seconds"]) for scene in scene_results)
        variant_summary[name] = {
            "scenes": len(results),
            "target_wake_recall": float(np.mean(wake_values)) if wake_values else None,
            "target_intent_accuracy": float(np.mean(intent_values)) if intent_values else None,
            "false_wake_events": int(sum(false_wakes)) if false_wakes else None,
            "false_action_events": int(sum(false_actions)) if false_actions else None,
            "false_action_rate_per_hour": (
                float(sum(false_actions) * 3600.0 / max(duration, 1.0e-6)) if false_actions else None
            ),
            "target_wake_latency_mean_seconds": float(np.mean(latencies)) if latencies else None,
            "target_wake_latency_p95_seconds": float(np.percentile(latencies, 95)) if latencies else None,
            "note": (
                "direct_command_oracle uses labeled command segmentation and is not a streaming result"
                if name == "direct_command_oracle"
                else None
            ),
        }

    report = {
        "benchmark": "continuous_pipeline_ablation_matrix_cross_corpus_proxy",
        "device": str(device),
        "config": str(resolve_path(args.config)),
        "checkpoints": cfg.get("checkpoints"),
        "code_revision": git_revision(),
        "sample_rate": sample_rate,
        "stage1_context_seconds": stage1_seconds,
        "hop_seconds": float(args.hop_seconds),
        "presence_threshold": presence_threshold,
        "kws_threshold": kws_threshold,
        "passenger_snr_db": float(args.passenger_snr_db),
        "enrollment_source": args.enrollment_source,
        "scenes": len(scene_results),
        "duration_seconds": sum(float(scene["truth"]["duration_seconds"]) for scene in scene_results),
        "speaker_consistent": args.enrollment_source in ("command", "same_speaker_other"),
        "production_valid": False,
        "variants": variant_defs,
        "summary": variant_summary,
        "scenes_detail": scene_results,
        "limitation": (
            (
                (
                    "Diagnostic same-speaker oracle only: the command clip is reused as enrollment, which leaks "
                    "command-speaker audio and is not a deployable evaluation."
                    if args.enrollment_source == "command"
                    else "Diagnostic same-speaker enrollment only: a different FSC utterance from the command speaker "
                    "is used, but the Marvin wake clip is not verified to be from that speaker; this is not a real-driver evaluation."
                )
                if args.enrollment_source in ("command", "same_speaker_other")
                else "Cross-corpus proxy only. Marvin wake clips and FSC command clips are not verified to be the same "
                "speaker, so these results diagnose component interactions and are not real-driver reliability evidence."
            )
        ),
    }
    output = resolve_path(args.output)
    write_metrics(output, report)
    print(json.dumps({"output": str(output), "summary": variant_summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
