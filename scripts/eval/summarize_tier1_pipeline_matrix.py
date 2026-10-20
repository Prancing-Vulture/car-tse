#!/usr/bin/env python
"""Aggregate stored Tier-1 continuous pipeline overlap ablation runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = [
    REPO_ROOT / "reports/evaluation/tier1_ablation_2026-10-19/pipeline_matrix/overlap_target_passenger_sir_m10.json",
    REPO_ROOT / "reports/evaluation/tier1_ablation_2026-10-19/pipeline_matrix/overlap_target_passenger_sir_m5.json",
    REPO_ROOT / "reports/evaluation/tier1_ablation_2026-10-19/pipeline_matrix/overlap_target_passenger_sir_0.json",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    results = [scene["variants"][variant] for scene in rows]
    usable = [result for result in results if result.get("target_wake_detected") is not None]
    captured = [result for result in results if result.get("target_intent") is not None]
    durations = [float(scene["truth"]["duration_seconds"]) for scene in rows]
    wake_latencies = [
        float(result["target_wake_latency_seconds"])
        for result in usable
        if result.get("target_wake_latency_seconds") is not None
    ]
    actual_sirs = [
        float(scene["truth"]["overlap_target_passenger_sir_db"])
        for scene in rows
        if scene["truth"].get("overlap_target_passenger_sir_db") is not None
    ]
    total_false_wakes = sum(
        int(result.get("false_wake_events") or 0) for result in usable
    )
    total_false_actions = sum(
        int(result.get("false_action_events") or 0) for result in usable
    )
    correct = sum(bool(result.get("target_intent_correct")) for result in results)
    captured_correct = sum(
        bool(result.get("target_intent_correct")) for result in captured
    )
    return {
        "scenes": len(rows),
        "actual_overlap_sir_mean_db": sum(actual_sirs) / len(actual_sirs) if actual_sirs else None,
        "actual_overlap_sir_min_db": min(actual_sirs) if actual_sirs else None,
        "actual_overlap_sir_max_db": max(actual_sirs) if actual_sirs else None,
        "target_wake_recall": (
            sum(bool(result["target_wake_detected"]) for result in usable) / len(usable)
            if usable
            else None
        ),
        "target_command_capture_rate": len(captured) / len(rows) if rows else None,
        "joint_correct_intent_rate": correct / len(rows) if rows else None,
        "intent_accuracy_given_captured_command": (
            captured_correct / len(captured) if captured else None
        ),
        "false_wake_events": total_false_wakes if usable else None,
        "false_action_events": total_false_actions if usable else None,
        "false_action_rate_per_hour": (
            total_false_actions * 3600.0 / max(sum(durations), 1.0e-9)
            if usable
            else None
        ),
        "wake_latency_mean_seconds": (
            sum(wake_latencies) / len(wake_latencies) if wake_latencies else None
        ),
        "wake_latency_p95_seconds": (
            sorted(wake_latencies)[round(0.95 * (len(wake_latencies) - 1))]
            if wake_latencies
            else None
        ),
        "target_presence_pass_rate_mean": (
            sum(float(result["target_presence_pass_rate"]) for result in usable) / len(usable)
            if usable
            else None
        ),
        "passenger_presence_pass_rate_mean": (
            sum(float(result["passenger_presence_pass_rate"]) for result in usable) / len(usable)
            if usable
            else None
        ),
        "speaker_gate_pass_rate_mean": (
            sum(float(result["speaker_gate_pass_rate"]) for result in usable if result.get("speaker_gate_pass_rate") is not None)
            / max(sum(result.get("speaker_gate_pass_rate") is not None for result in usable), 1)
            if any(result.get("speaker_gate_pass_rate") is not None for result in usable)
            else None
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No summary rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reports/evaluation/tier1_ablation_2026-10-19/pipeline_matrix",
    )
    args = parser.parse_args()

    grouped: dict[tuple[float, float, str], list[dict[str, Any]]] = defaultdict(list)
    input_metadata = []
    for input_path in args.inputs:
        path = input_path if input_path.is_absolute() else (REPO_ROOT / input_path)
        path = path.resolve()
        report = json.loads(path.read_text(encoding="utf-8"))
        requested_sir = float(report["requested_target_passenger_sir_db"])
        input_metadata.append({"path": str(path), "sha256": sha256_file(path)})
        for scene in report["scenes_detail"]:
            cabin_snr = float(scene["cabin_snr_db"])
            for variant in scene["variants"]:
                grouped[(requested_sir, cabin_snr, variant)].append(scene)

    by_condition_rows: list[dict[str, Any]] = []
    for (requested_sir, cabin_snr, variant), scenes in sorted(grouped.items()):
        values = aggregate(scenes, variant)
        by_condition_rows.append(
            {
                "requested_target_passenger_sir_db": requested_sir,
                "cabin_snr_db": cabin_snr,
                "variant": variant,
                **values,
            }
        )

    by_sir_groups: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for (requested_sir, _cabin_snr, variant), scenes in grouped.items():
        by_sir_groups[(requested_sir, variant)].extend(scenes)
    by_sir_rows = []
    for (requested_sir, variant), scenes in sorted(by_sir_groups.items()):
        values = aggregate(scenes, variant)
        by_sir_rows.append(
            {
                "requested_target_passenger_sir_db": requested_sir,
                "variant": variant,
                **values,
            }
        )

    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir = output_dir.resolve()
    by_condition_path = output_dir / "summary_by_requested_sir_cabin_snr.csv"
    by_sir_path = output_dir / "summary_by_requested_sir.csv"
    write_csv(by_condition_path, by_condition_rows)
    write_csv(by_sir_path, by_sir_rows)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "aggregation": "scene-level rates; conditional intent accuracy uses scenes with a captured target command; actual SIR is measured only over co-active target/passenger speech samples",
        "inputs": input_metadata,
        "outputs": {
            by_condition_path.name: sha256_file(by_condition_path),
            by_sir_path.name: sha256_file(by_sir_path),
        },
    }
    manifest_path = output_dir / "summary_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"by_condition": str(by_condition_path), "by_sir": str(by_sir_path), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
