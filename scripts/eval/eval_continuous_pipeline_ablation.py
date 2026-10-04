#!/usr/bin/env python
"""Ablate the continuous in-car cascade on identical proxy scenes.

The evaluator keeps Stage 1 fixed and compares:

1. TSE -> intent directly;
2. TSE -> presence;
3. TSE -> KWS without presence;
4. TSE -> presence -> KWS -> post-wake command capture -> intent.

It uses the same cross-corpus continuous scenes as
``eval_continuous_full_pipeline_proxy.py``.  The result is diagnostic only:
the public Marvin and FSC corpora do not provide a verified same-speaker
wake-word/command pair.
"""

from __future__ import annotations

import argparse
import json
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
from car_tse.pipeline.in_car_pipeline import PipelineState, normalize_extracted_waveform, update_kws_state
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


def event_indices(flags: list[bool], merge_gap_steps: int = 2) -> list[int]:
    indices: list[int] = []
    last = -10**9
    for index, flag in enumerate(flags):
        if flag and index - last > max(0, int(merge_gap_steps)):
            indices.append(index)
            last = index
    return indices


def is_target_wake(center: float, truth: dict[str, Any]) -> bool:
    return (
        float(truth["target_wake_start_seconds"]) - 0.25
        <= center
        <= float(truth["target_wake_end_seconds"]) + 3.0
    )


def is_target_active(center: float, truth: dict[str, Any]) -> bool:
    return (
        float(truth["target_wake_start_seconds"]) - 0.5
        <= center
        <= float(truth["target_command_end_seconds"]) + 0.5
    )


def is_passenger_region(center: float) -> bool:
    return 5.0 <= center <= 11.5


def evaluate_scene_ablation(
    pipeline: Any,
    mixture: np.ndarray,
    multichannel: np.ndarray,
    enrollment: np.ndarray,
    truth: dict[str, Any],
    sample_rate: int,
    stage1_seconds: float,
    hop_seconds: float,
    presence_thresholds: list[float],
    kws_threshold: float,
    device: torch.device,
) -> dict[str, Any]:
    window_samples = int(round(stage1_seconds * sample_rate))
    hop_samples = max(1, int(round(hop_seconds * sample_rate)))
    rows: list[dict[str, Any]] = []
    extracted_by_index: dict[int, torch.Tensor] = {}
    direct_command_row: dict[str, Any] | None = None
    command_start = float(truth["target_command_start_seconds"])

    with torch.inference_mode():
        for start in range(0, max(1, len(mixture)), hop_samples):
            end = min(start + window_samples, len(mixture))
            mix_window = np.zeros(window_samples, dtype=np.float32)
            mc_window = np.zeros((multichannel.shape[1], window_samples), dtype=np.float32)
            if end > start:
                mix_window[: end - start] = mixture[start:end]
                mc_window[:, : end - start] = multichannel[start:end].T

            mix_tensor = torch.from_numpy(mix_window).to(device).unsqueeze(0)
            enrollment_tensor = torch.from_numpy(enrollment).to(device).unsqueeze(0)
            mc_tensor = torch.from_numpy(mc_window).to(device).unsqueeze(0)
            extracted, _, _, _ = pipeline.stage1(
                mix_tensor,
                enrollment_tensor,
                multichannel=mc_tensor,
            )
            presence_probability = float(pipeline.presence_probability(extracted).mean().cpu())
            downstream = normalize_extracted_waveform(extracted)
            kws_input = downstream
            kws_samples = max(1, int(round(pipeline.kws_sample_rate * pipeline.kws_input_seconds)))
            if kws_input.shape[-1] >= kws_samples:
                kws_input = kws_input[..., :kws_samples]
            else:
                kws_input = torch.nn.functional.pad(kws_input, (0, kws_samples - kws_input.shape[-1]))
            wake_probability = float(torch.softmax(pipeline.kws(kws_input), dim=-1)[0, pipeline.wake_class_id].cpu())

            window_start = start / sample_rate
            center = window_start + stage1_seconds / 2.0
            row: dict[str, Any] = {
                "window_index": len(rows),
                "window_start_seconds": window_start,
                "window_center_seconds": center,
                "presence_probability": presence_probability,
                "wake_probability": wake_probability,
                "target_active": is_target_active(center, truth),
                "passenger_region": is_passenger_region(center),
                "intent": None,
                "intent_confidence": None,
            }
            extracted_by_index[int(row["window_index"])] = downstream

            if direct_command_row is None or abs(window_start - command_start) < abs(
                direct_command_row["window_start_seconds"] - command_start
            ):
                probabilities = torch.softmax(pipeline.intent(downstream), dim=-1)[0]
                predicted_id = str(int(probabilities.argmax().item()))
                row["direct_intent_id"] = predicted_id
                row["direct_intent"] = pipeline.label_map.get(predicted_id, predicted_id) if pipeline.label_map else predicted_id
                row["direct_intent_confidence"] = float(probabilities.max().detach().cpu())
                direct_command_row = row
            rows.append(row)

    result: dict[str, Any] = {
        "truth": truth,
        "windows": rows,
        "expected_intent": str(truth["expected_intent"]),
        "direct_intent": direct_command_row.get("direct_intent") if direct_command_row else None,
        "direct_intent_confidence": direct_command_row.get("direct_intent_confidence") if direct_command_row else None,
        "direct_intent_correct": bool(
            direct_command_row and direct_command_row.get("direct_intent") == str(truth["expected_intent"])
        ),
    }

    target_rows = [row for row in rows if row["target_active"]]
    passenger_rows = [row for row in rows if row["passenger_region"]]
    result["presence_target_mean"] = float(np.mean([row["presence_probability"] for row in target_rows]))
    result["presence_target_max"] = float(max(row["presence_probability"] for row in target_rows))
    result["presence_passenger_max"] = float(max(row["presence_probability"] for row in passenger_rows))

    # KWS with no presence gate.
    kws_state = PipelineState()
    kws_flags: list[bool] = []
    for row in rows:
        kws_flags.append(
            update_kws_state(
                kws_state,
                float(row["wake_probability"]),
                float(kws_threshold),
                int(pipeline.kws_smoothing_window),
                int(pipeline.kws_required_hits),
                int(pipeline.kws_refractory_calls),
            )
        )
    kws_event_indices = event_indices(kws_flags)
    kws_target_events = [index for index in kws_event_indices if is_target_wake(rows[index]["window_center_seconds"], truth)]
    kws_false_events = [index for index in kws_event_indices if index not in kws_target_events]
    result["kws_without_presence"] = {
        "target_wake_detected": bool(kws_target_events),
        "false_events": len(kws_false_events),
        "false_action_rate_per_hour": len(kws_false_events) * 3600.0 / float(truth["duration_seconds"]),
        "event_windows": [rows[index] for index in kws_event_indices],
    }

    # Full cascade for every requested presence operating point.  Stage 1,
    # presence and KWS scores are reused; the post-wake state machine is
    # replayed per threshold so intent is evaluated on the following command
    # window rather than on the wake-word window.
    full_by_threshold: dict[str, Any] = {}
    for threshold in presence_thresholds:
        state = PipelineState()
        wake_events: list[dict[str, Any]] = []
        command_events: list[dict[str, Any]] = []
        for row in rows:
            window_index = int(row["window_index"])
            if float(row["presence_probability"]) < float(threshold):
                state.kws_scores.clear()
                if state.awaiting_command:
                    state.command_wait_calls += 1
                    if state.command_wait_calls >= int(pipeline.command_timeout_calls):
                        state.awaiting_command = False
                        state.command_wait_calls = 0
                        state.kws_refractory_left = 0
                else:
                    state.kws_refractory_left = 0
                continue

            if state.awaiting_command:
                probabilities = torch.softmax(
                    pipeline.intent(pipeline._prepare_intent_input(extracted_by_index[window_index])), dim=-1
                )[0]
                predicted_id = str(int(probabilities.argmax().item()))
                command_events.append(
                    {
                        **row,
                        "intent_ready": True,
                        "intent": pipeline.label_map.get(predicted_id, predicted_id) if pipeline.label_map else predicted_id,
                        "intent_confidence": float(probabilities.max().detach().cpu()),
                    }
                )
                state.awaiting_command = False
                state.command_wait_calls = 0
                state.kws_scores.clear()
                continue

            wake = update_kws_state(
                state,
                float(row["wake_probability"]),
                float(kws_threshold),
                int(pipeline.kws_smoothing_window),
                int(pipeline.kws_required_hits),
                int(pipeline.kws_refractory_calls),
            )
            if wake:
                state.awaiting_command = True
                state.command_wait_calls = 0
                state.kws_scores.clear()
                wake_events.append({**row, "wake_detected": True, "command_pending": True})
        target_events = [event for event in wake_events if is_target_wake(event["window_center_seconds"], truth)]
        false_events = [event for event in wake_events if event not in target_events]
        target_wake_event = target_events[0] if target_events else None
        target_command_events = [
            event
            for event in command_events
            if float(truth["target_command_start_seconds"]) - 0.5
            <= event["window_center_seconds"]
            <= float(truth["target_command_end_seconds"]) + 3.0
        ]
        target_command_event = target_command_events[0] if target_command_events else None
        full_by_threshold[str(float(threshold))] = {
            "target_wake_detected": bool(target_wake_event),
            "target_intent": target_command_event.get("intent") if target_command_event else None,
            "target_intent_correct": bool(
                target_command_event and target_command_event.get("intent") == str(truth["expected_intent"])
            ),
            "false_events": len(false_events),
            "false_action_rate_per_hour": len(false_events) * 3600.0 / float(truth["duration_seconds"]),
            "wake_event_windows": wake_events,
            "command_event_windows": command_events,
            "target_presence_pass_rate": float(
                np.mean([row["presence_probability"] >= threshold for row in target_rows])
            ),
            "passenger_presence_false_pass_rate": float(
                np.mean([row["presence_probability"] >= threshold for row in passenger_rows])
            ),
        }
    result["full_by_presence_threshold"] = full_by_threshold
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/eval_end_to_end_cavemove_wav2vec_ft8_kws_hybrid_presence90_norm_smooth.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.yaml")
    parser.add_argument("--wake-manifest", type=Path, default=REPO_ROOT / "manifests/test/stage3_kws_cabin_test.jsonl")
    parser.add_argument("--intent-manifest", type=Path, default=REPO_ROOT / "manifests/official/stage4_intent_fsc_test.jsonl")
    parser.add_argument("--noise-root", type=Path, default=REPO_ROOT / "data/raw/cavemove/extracted/cavemove_dataset")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "logs/continuous_pipeline_ablation.json")
    parser.add_argument("--scenes", type=int, default=2)
    parser.add_argument("--cabin-snr-db", nargs="+", type=float, default=[0.0, -5.0])
    parser.add_argument("--passenger-snr-db", type=float, default=0.0)
    parser.add_argument("--presence-thresholds", nargs="+", type=float, default=[0.50, 0.90])
    parser.add_argument("--kws-threshold", type=float, default=0.29)
    parser.add_argument("--hop-seconds", type=float, default=1.0)
    args = parser.parse_args()

    cfg = load_stage_config(args.config, args.paths)
    cfg["kws_threshold"] = float(args.kws_threshold)
    device = get_device()
    pipeline = load_pipeline(cfg, cfg["paths"], device)
    sample_rate = int(cfg["sample_rate"])
    stage1_seconds = float(cfg.get("segment_seconds", 3.0))

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
    intent_rows = load_jsonl(resolve_path(args.intent_manifest))
    by_intent: dict[str, dict[str, Any]] = {}
    for row in intent_rows:
        by_intent.setdefault(str(row["intent"]), row)
    intent_rows = [by_intent[key] for key in sorted(by_intent)]
    noise_files = select_noise_files(resolve_path(args.noise_root), 20.0)
    if not noise_files:
        raise RuntimeError("No CAVEMOVE noise streams found")

    scene_results: list[dict[str, Any]] = []
    for scene_index, wake_row in enumerate(target_wake_rows):
        intent_row = intent_rows[scene_index % len(intent_rows)]
        target_wake, _ = read_mono(resolve_path(wake_row["wav"]), sample_rate)
        target_command, _ = read_mono(resolve_path(intent_row["wav"]), sample_rate)
        passenger_wake, _ = read_mono(resolve_path(passenger_wake_row["wav"]), sample_rate)
        passenger_nonwake, _ = read_mono(resolve_path(passenger_nonwake_row["wav"]), sample_rate)
        target_wake = fit_start(target_wake, sample_rate)
        target_command = fit_start(target_command, 2 * sample_rate)
        passenger_wake = fit_start(passenger_wake, sample_rate)
        passenger_nonwake = fit_start(passenger_nonwake, int(1.5 * sample_rate))
        enrollment = fit_start(target_wake, int(round(stage1_seconds * sample_rate / 2.0)))
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
            truth["expected_intent"] = str(intent_row["intent"])
            truth["speaker_consistent"] = False
            result = evaluate_scene_ablation(
                pipeline,
                mixture,
                multichannel,
                enrollment,
                truth,
                sample_rate,
                stage1_seconds,
                float(args.hop_seconds),
                [float(value) for value in args.presence_thresholds],
                float(args.kws_threshold),
                device,
            )
            result.update({"scene_index": scene_index, "cabin_snr_db": float(cabin_snr_db)})
            scene_results.append(result)
            print(
                f"scene={scene_index} snr={cabin_snr_db:+.1f}: "
                f"direct_intent={result['direct_intent_correct']} "
                f"kws_no_presence={result['kws_without_presence']['target_wake_detected']} "
                f"full=" + ",".join(
                    f"p{threshold:g}:{result['full_by_presence_threshold'][str(float(threshold))]['target_wake_detected']}"
                    for threshold in args.presence_thresholds
                ),
                flush=True,
            )

    def mean(key: str) -> float:
        return float(np.mean([float(row[key]) for row in scene_results]))

    threshold_summary: dict[str, Any] = {}
    for threshold in args.presence_thresholds:
        key = str(float(threshold))
        rows = [row["full_by_presence_threshold"][key] for row in scene_results]
        threshold_summary[key] = {
            "target_wake_recall": float(np.mean([row["target_wake_detected"] for row in rows])),
            "target_intent_accuracy": float(np.mean([row["target_intent_correct"] for row in rows])),
            "false_wake_events": int(sum(int(row["false_events"]) for row in rows)),
            "false_action_rate_per_hour": float(
                sum(int(row["false_events"]) for row in rows) * 3600.0 / (14.0 * len(rows))
            ),
            "target_presence_pass_rate_mean": float(np.mean([row["target_presence_pass_rate"] for row in rows])),
            "passenger_presence_false_pass_rate_mean": float(
                np.mean([row["passenger_presence_false_pass_rate"] for row in rows])
            ),
        }

    kws_rows = [row["kws_without_presence"] for row in scene_results]
    report = {
        "benchmark": "continuous_pipeline_ablation_cross_corpus_proxy",
        "device": str(device),
        "config": str(resolve_path(args.config)),
        "effective_kws_threshold": float(args.kws_threshold),
        "presence_thresholds": [float(value) for value in args.presence_thresholds],
        "scenes": len(scene_results),
        "duration_seconds": 14.0 * len(scene_results),
        "speaker_consistent": False,
        "production_valid": False,
        "direct_tse_to_intent": {
            "accuracy": mean("direct_intent_correct"),
            "predictions": [
                {
                    "scene_index": row["scene_index"],
                    "snr_db": row["cabin_snr_db"],
                    "expected": row["expected_intent"],
                    "predicted": row["direct_intent"],
                }
                for row in scene_results
            ],
        },
        "tse_to_kws_without_presence": {
            "target_wake_recall": float(np.mean([row["target_wake_detected"] for row in kws_rows])),
            "false_wake_events": int(sum(int(row["false_events"]) for row in kws_rows)),
            "false_action_rate_per_hour": float(
                sum(int(row["false_events"]) for row in kws_rows) * 3600.0 / (14.0 * len(kws_rows))
            ),
        },
        "full_pipeline_by_presence_threshold": threshold_summary,
        "scenes_detail": scene_results,
        "limitation": (
            "Continuous cross-corpus proxy only. The target Marvin and target command clips are not verified to be "
            "the same speaker, so the direct-intent and full-pipeline results diagnose the current cascade but are "
            "not a production accuracy estimate."
        ),
    }
    output = resolve_path(args.output)
    write_metrics(output, report)
    print(json.dumps({
        "output": str(output),
        "direct_tse_to_intent": report["direct_tse_to_intent"],
        "tse_to_kws_without_presence": report["tse_to_kws_without_presence"],
        "full_pipeline_by_presence_threshold": report["full_pipeline_by_presence_threshold"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
