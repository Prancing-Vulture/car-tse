"""Re-score historical multi-speaker proxy logs with audit-safe metrics.

This script deliberately performs no model inference.  It reads the emitted
``scenes_detail`` records and recomputes the safety metrics using event
identity and interval overlap.  That makes corrections to an old benchmark
reproducible even when the original evaluator did not save the newer fields.

The legacy counters are retained for comparison.  In particular, the old
target-wake association allows a broad +3 second post-wake tolerance, so a
passenger wake in that interval can be counted as target support.  The report
therefore exposes both the legacy-compatible numbers and source-attributed
counts instead of silently replacing the historical result.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def intervals_overlap(first_start: float, first_end: float, second_start: float, second_end: float) -> bool:
    return min(float(first_end), float(second_end)) > max(float(first_start), float(second_start))


def event_interval(row: dict[str, Any], stage1_seconds: float) -> tuple[float, float]:
    start = float(row["window_start_seconds"])
    return start, float(row.get("window_end_seconds", start + stage1_seconds))


def event_overlaps_label(
    row: dict[str, Any],
    label_start: float,
    label_end: float,
    stage1_seconds: float,
    tolerance_before: float,
    tolerance_after: float,
) -> bool:
    start, end = event_interval(row, stage1_seconds)
    return intervals_overlap(
        start,
        end,
        float(label_start) - float(tolerance_before),
        float(label_end) + float(tolerance_after),
    )


def truth_interval(
    truth: dict[str, Any],
    start_key: str,
    end_key: str,
    fallback_duration: float,
) -> tuple[float, float]:
    start = float(truth.get(start_key, -1.0))
    end = float(truth.get(end_key, start + fallback_duration))
    return start, end


def source_labels(row: dict[str, Any], truth: dict[str, Any], stage1_seconds: float) -> list[str]:
    labels: list[str] = []
    intervals = (
        ("target_wake", truth_interval(truth, "target_wake_start_seconds", "target_wake_end_seconds", 1.0)),
        ("passenger_wake", truth_interval(truth, "passenger_wake_start_seconds", "passenger_wake_end_seconds", 1.0)),
        (
            "passenger_nonwake",
            truth_interval(truth, "passenger_nonwake_start_seconds", "passenger_nonwake_end_seconds", 1.5),
        ),
    )
    start, end = event_interval(row, stage1_seconds)
    for label, (label_start, label_end) in intervals:
        if label_start >= 0.0 and intervals_overlap(start, end, label_start, label_end):
            labels.append(label)
    return labels


def rescore_scene(scene: dict[str, Any], stage1_seconds: float) -> dict[str, Any]:
    truth = dict(scene.get("truth", {}))
    wake_events = list(scene.get("wake_events", []))
    command_events = list(scene.get("command_events", []))
    expected_intent = str(scene.get("expected_intent", truth.get("expected_intent", "")))
    target_wake_start, target_wake_end = truth_interval(
        truth, "target_wake_start_seconds", "target_wake_end_seconds", 1.0
    )
    target_command_start, target_command_end = truth_interval(
        truth, "target_command_start_seconds", "target_command_end_seconds", 2.0
    )
    target_wake_events = [
        row
        for row in wake_events
        if event_overlaps_label(row, target_wake_start, target_wake_end, stage1_seconds, 0.25, 3.0)
    ]
    strict_target_wake_events = [
        row
        for row in wake_events
        if event_overlaps_label(row, target_wake_start, target_wake_end, stage1_seconds, 0.0, 0.0)
    ]
    target_command_events = [
        row
        for row in command_events
        if event_overlaps_label(row, target_command_start, target_command_end, stage1_seconds, 0.5, 3.0)
    ]
    strict_target_command_events = [
        row
        for row in command_events
        if event_overlaps_label(row, target_command_start, target_command_end, stage1_seconds, 0.0, 0.0)
    ]
    target_command_event = sorted(
        target_command_events, key=lambda row: float(row["window_start_seconds"])
    )[0] if target_command_events else None
    executed = [row for row in command_events if row.get("action")]
    target_event_start = (
        None if target_command_event is None else float(target_command_event["window_start_seconds"])
    )
    non_target = [
        row
        for row in executed
        if target_event_start is None or float(row["window_start_seconds"]) != target_event_start
    ]
    wrong_intent = [row for row in executed if row.get("intent") != expected_intent]
    wrong_any = [row for row in executed if row in non_target or row in wrong_intent]
    intent_correct = bool(target_command_event and target_command_event.get("intent") == expected_intent)
    wake_source_counts: Counter[str] = Counter()
    for row in wake_events:
        labels = source_labels(row, truth, stage1_seconds)
        if row in target_wake_events:
            wake_source_counts["target_tolerance_support"] += 1
        elif len(labels) == 1:
            wake_source_counts[labels[0]] += 1
        else:
            wake_source_counts["+".join(labels) if labels else "other"] += 1
    return {
        "executed_action_events": len(executed),
        "executed_non_target_actions": len(non_target),
        "wrong_intent_action_events": len(wrong_intent),
        "wrong_executed_action_events": len(wrong_any),
        "scene_with_wrong_intent_action": bool(wrong_intent),
        "scene_with_wrong_executed_action": bool(wrong_any),
        "fully_correct_scene": bool(intent_correct and len(command_events) == 1),
        "strict_fully_correct_action_scene": bool(
            len(executed) == 1
            and not wrong_any
            and target_command_event is not None
            and executed[0] in strict_target_command_events
        ),
        "target_wake_events": len(target_wake_events),
        "strict_target_wake_events": len(strict_target_wake_events),
        "false_wake_events": len(wake_events) - len(target_wake_events),
        "strict_false_wake_events": len(wake_events) - len(strict_target_wake_events),
        "target_command_events": len(target_command_events),
        "strict_target_command_events": len(strict_target_command_events),
        "false_command_events": len(command_events) - len(target_command_events),
        "strict_false_command_events": len(command_events) - len(strict_target_command_events),
        "wake_trigger_source_counts": dict(wake_source_counts),
    }


def rescore(payload: dict[str, Any]) -> dict[str, Any]:
    stage1_seconds = float(payload.get("stage1_context_seconds", 3.0))
    scenes = list(payload.get("scenes_detail", []))
    scene_metrics = [rescore_scene(scene, stage1_seconds) for scene in scenes]
    duration_seconds = sum(float(scene.get("duration_seconds", 0.0)) for scene in scenes)
    source_counts: Counter[str] = Counter()
    for metric in scene_metrics:
        source_counts.update(metric["wake_trigger_source_counts"])

    def total(key: str) -> int:
        return sum(int(metric.get(key, 0)) for metric in scene_metrics)

    def scene_total(key: str) -> int:
        return sum(bool(metric.get(key, False)) for metric in scene_metrics)

    scenes_count = len(scene_metrics)
    hours = max(duration_seconds / 3600.0, 1.0e-12)
    report = {
        "source_log_metadata": {
            "benchmark": payload.get("benchmark"),
            "config": payload.get("config"),
            "effective_command_buffer_calls": payload.get("effective_command_buffer_calls"),
            "proxy_command_seconds": payload.get("proxy_command_seconds"),
            "scene_sampling": payload.get("scene_sampling"),
            "scene_pairing": payload.get("scene_pairing"),
        },
        "scenes": scenes_count,
        "duration_seconds": duration_seconds,
        "duration_hours": duration_seconds / 3600.0,
        "executed_action_events": total("executed_action_events"),
        "executed_non_target_actions": total("executed_non_target_actions"),
        "executed_non_target_actions_per_hour": total("executed_non_target_actions") / hours,
        "wrong_intent_action_events": total("wrong_intent_action_events"),
        "wrong_executed_action_events": total("wrong_executed_action_events"),
        "wrong_executed_action_rate_per_hour": total("wrong_executed_action_events") / hours,
        "scenes_with_wrong_intent_action": scene_total("scene_with_wrong_intent_action"),
        "scenes_with_wrong_intent_action_rate": scene_total("scene_with_wrong_intent_action") / max(scenes_count, 1),
        "scenes_with_wrong_executed_action": scene_total("scene_with_wrong_executed_action"),
        "scenes_with_wrong_executed_action_rate": scene_total("scene_with_wrong_executed_action") / max(scenes_count, 1),
        "fully_correct_scenes": scene_total("fully_correct_scene"),
        "fully_correct_scene_rate": scene_total("fully_correct_scene") / max(scenes_count, 1),
        "strict_fully_correct_action_scenes": scene_total("strict_fully_correct_action_scene"),
        "strict_fully_correct_action_rate": scene_total("strict_fully_correct_action_scene") / max(scenes_count, 1),
        "legacy_target_wake_events": total("target_wake_events"),
        "legacy_false_wake_events": total("false_wake_events"),
        "legacy_false_wake_rate_per_hour": total("false_wake_events") / hours,
        "strict_target_wake_events": total("strict_target_wake_events"),
        "strict_false_wake_events": total("strict_false_wake_events"),
        "strict_false_wake_rate_per_hour": total("strict_false_wake_events") / hours,
        "legacy_target_command_events": total("target_command_events"),
        "legacy_false_command_events": total("false_command_events"),
        "legacy_false_command_rate_per_hour": total("false_command_events") / hours,
        "strict_target_command_events": total("strict_target_command_events"),
        "strict_false_command_events": total("strict_false_command_events"),
        "strict_false_command_rate_per_hour": total("strict_false_command_events") / hours,
        "wake_trigger_source_counts": dict(source_counts),
        "scene_metrics": scene_metrics,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Historical evaluator JSON log")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()
    payload = json.loads(args.log.read_text(encoding="utf-8"))
    report = rescore(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {
        key: report[key]
        for key in (
            "scenes",
            "duration_hours",
            "scenes_with_wrong_intent_action",
            "executed_non_target_actions",
            "fully_correct_scenes",
            "legacy_false_wake_events",
            "wake_trigger_source_counts",
        )
    }
    if args.output:
        summary["output"] = str(args.output)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
