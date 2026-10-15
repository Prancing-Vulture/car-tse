#!/usr/bin/env python
"""Replay a saved continuous-evaluation run over presence/KWS thresholds.

No audio models are rerun: the saved frame scores are replayed through the
same presence and temporal KWS state rules. This is suitable for development
diagnostics only. Select thresholds on development sessions, then freeze them
before evaluating a held-out test set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.pipeline.in_car_pipeline import PipelineState, update_kws_state


def is_target(center: float, truth: dict[str, Any]) -> bool:
    return (
        float(truth["target_wake_start_seconds"]) - 0.25
        <= center
        <= float(truth["target_wake_end_seconds"]) + 3.0
    )


def replay(
    scenes: list[dict[str, Any]],
    presence_threshold: float,
    kws_threshold: float,
    speaker_threshold: float | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    target_wakes = false_wakes = command_captures = correct_intents = 0
    total_duration = 0.0
    latencies: list[float] = []
    for scene in scenes:
        truth = scene["truth"]
        total_duration += float(truth["duration_seconds"])
        state = PipelineState()
        wakes: list[dict[str, Any]] = []
        commands: list[dict[str, Any]] = []
        for row in scene["windows"]:
            similarity = row.get("speaker_similarity")
            identity_rejected = (
                speaker_threshold is not None
                and similarity is not None
                and float(similarity) < speaker_threshold
            )
            if float(row["presence_probability"]) < presence_threshold or identity_rejected:
                state.kws_scores.clear()
                if state.awaiting_command:
                    state.command_wait_calls += 1
                    if state.command_wait_calls >= int(config.get("command_timeout_calls", 4)):
                        state.awaiting_command = False
                        state.command_wait_calls = 0
                        state.kws_refractory_left = 0
                else:
                    state.kws_refractory_left = 0
                continue
            if state.awaiting_command:
                probabilities = row.get("intent_probabilities") or []
                predicted_id = str(max(range(len(probabilities)), key=probabilities.__getitem__)) if probabilities else ""
                label_map = config.get("_id_to_intent", {})
                commands.append(
                    {
                        **row,
                        "intent": label_map.get(predicted_id, predicted_id),
                    }
                )
                state.awaiting_command = False
                state.command_wait_calls = 0
                state.kws_scores.clear()
                continue
            if update_kws_state(
                state,
                float(row["wake_probability"]),
                float(kws_threshold),
                int(config.get("kws_smoothing_window", 1)),
                int(config.get("kws_required_hits", 1)),
                int(config.get("kws_refractory_calls", 0)),
            ):
                state.awaiting_command = True
                state.command_wait_calls = 0
                state.kws_scores.clear()
                wakes.append(row)
        target_events = [event for event in wakes if is_target(float(event["window_center_seconds"]), truth)]
        false_wakes += sum(event not in target_events for event in wakes)
        if target_events:
            target_wakes += 1
            latencies.append(
                float(target_events[0]["window_center_seconds"])
                - float(truth["target_wake_start_seconds"])
            )
        command_start = float(truth["target_command_start_seconds"])
        command_end = float(truth["target_command_end_seconds"])
        target_commands = [
            event for event in commands
            if command_start - 0.5 <= float(event["window_center_seconds"]) <= command_end + 3.0
        ]
        if target_commands:
            command_captures += 1
            correct_intents += int(target_commands[0]["intent"] == scene["expected_intent"])
    n = max(1, len(scenes))
    return {
        "scenes": len(scenes),
        "duration_seconds": total_duration,
        "presence_threshold": float(presence_threshold),
        "kws_threshold": float(kws_threshold),
        "speaker_threshold": speaker_threshold,
        "target_wake_recall": target_wakes / n,
        "target_command_capture_rate": command_captures / n,
        "target_intent_accuracy": correct_intents / n,
        "false_wake_events": false_wakes,
        "false_wake_rate_per_hour": false_wakes * 3600.0 / max(total_duration, 1.0e-9),
        "mean_wake_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--presence-thresholds", nargs="+", type=float, default=[0.30, 0.50, 0.65, 0.90])
    parser.add_argument("--kws-thresholds", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.29, 0.50])
    parser.add_argument("--speaker-thresholds", nargs="+", type=float, default=[-1.0, 0.0, 0.10, 0.20])
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    # The label IDs in this saved system are the same IDs trained with the
    # generated FSC mapping, available alongside E: manifests.
    manifest_root = Path("E:/car-tse-data/manifests")
    label_map_path = manifest_root / "meta" / "intent_label_map.json"
    if label_map_path.exists():
        label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
        config["_id_to_intent"] = {str(value): key for key, value in label_map.items()}
    combos = [
        replay(source["scenes_detail"], p, k, s, config)
        for p in args.presence_thresholds
        for k in args.kws_thresholds
        for s in args.speaker_thresholds
    ]
    report = {
        "benchmark": source.get("benchmark"),
        "source_report": str(args.input),
        "speaker_consistent": source.get("speaker_consistent", False),
        "production_valid": False,
        "threshold_selection": "diagnostic only; do not select a publishable operating point from these test scenes",
        "results": combos,
        "warning": (
            "This is a small cross-corpus proxy with unmatched target wake/command speakers. "
            "A threshold combination with zero false wakes and zero target recall is rejection, not success."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"replayed {len(combos)} threshold combinations -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
