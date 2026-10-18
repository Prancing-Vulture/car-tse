#!/usr/bin/env python
"""Run targeted runtime ablations on the fixed continuous proxy scenes.

This study keeps the trained checkpoints fixed and varies only runtime
choices that are likely to matter in an in-car deployment:

* temporal hop and wake smoothing settings;
* enrollment duration;
* the microphone/front-end channel subset.

The scenes are the same cross-corpus proxy used by the existing continuous
ablation evaluator.  Marvin clips and FSC command clips are not verified to
be from the same speaker, so this is a diagnostic study rather than a
real-driver benchmark.
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
from car_tse.train.utils import get_device, write_metrics
from eval_continuous_ablation_matrix import (
    VARIANTS,
    collect_window_scores,
    resolve_path,
    speaker_key,
    summarize_variant,
)
from eval_continuous_full_pipeline_proxy import (
    build_scene,
    fit_start,
    load_jsonl,
    load_pipeline,
    select_noise_files,
)
from eval_continuous_kws import prepare_channels


TEMPORAL_SETTINGS: dict[str, dict[str, int]] = {
    "one_hit": {"smoothing_window": 1, "required_hits": 1, "refractory_calls": 0},
    "two_of_two": {"smoothing_window": 2, "required_hits": 2, "refractory_calls": 0},
    "two_of_three_refractory": {
        "smoothing_window": 3,
        "required_hits": 2,
        "refractory_calls": 2,
    },
    "three_of_five_refractory": {
        "smoothing_window": 5,
        "required_hits": 3,
        "refractory_calls": 4,
    },
}

FRONTEND_MODES = ("all4", "mono", "front_pair", "rear_pair")


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


def frontend_variant(multichannel: np.ndarray, mode: str) -> np.ndarray:
    """Return a four-channel proxy for one channel/front-end ablation."""

    data = np.asarray(multichannel, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected [samples, channels], got {data.shape}")
    if data.shape[1] == 0:
        raise ValueError("Cannot build a front-end variant from zero channels")
    if mode == "all4":
        selected = data
    elif mode == "mono":
        mono = data.mean(axis=1, keepdims=True)
        selected = np.repeat(mono, 4, axis=1)
    elif mode == "front_pair":
        indices = [0, 1] if data.shape[1] >= 2 else [0]
        selected = data[:, indices]
    elif mode == "rear_pair":
        indices = [2, 3] if data.shape[1] >= 4 else list(range(data.shape[1]))
        selected = data[:, indices]
    else:
        raise ValueError(f"Unknown front-end mode: {mode}")
    if selected.shape[1] < 4:
        repeats = int(np.ceil(4 / selected.shape[1]))
        selected = np.tile(selected, (1, repeats))[:, :4]
    return np.asarray(selected[:, :4], dtype=np.float32)


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    """Keep report details useful without serializing every wake event."""

    keys = (
        "variant",
        "target_wake_detected",
        "target_intent",
        "target_intent_correct",
        "false_wake_events",
        "false_action_events",
        "false_action_rate_per_hour",
        "target_wake_latency_seconds",
        "target_presence_pass_rate",
        "passenger_presence_pass_rate",
        "speaker_similarity_mean",
        "speaker_gate_pass_rate",
        "intent_confidence",
        "mode",
        "note",
    )
    compact = {key: result[key] for key in keys if key in result}
    compact["wake_event_count"] = len(result.get("wake_events", []))
    compact["command_event_count"] = len(result.get("command_events", []))
    return compact


def aggregate(results: list[dict[str, Any]], duration_seconds: float) -> dict[str, Any]:
    wake = [bool(row["target_wake_detected"]) for row in results if row.get("target_wake_detected") is not None]
    intent = [bool(row["target_intent_correct"]) for row in results]
    false_wakes = [int(row.get("false_wake_events", 0)) for row in results if row.get("false_wake_events") is not None]
    false_actions = [int(row.get("false_action_events", 0)) for row in results if row.get("false_action_events") is not None]
    latencies = [
        float(row["target_wake_latency_seconds"])
        for row in results
        if row.get("target_wake_latency_seconds") is not None
    ]
    target_presence = [
        float(row["target_presence_pass_rate"])
        for row in results
        if row.get("target_presence_pass_rate") is not None
    ]
    passenger_presence = [
        float(row["passenger_presence_pass_rate"])
        for row in results
        if row.get("passenger_presence_pass_rate") is not None
    ]
    speaker_similarity = [
        float(row["speaker_similarity_mean"])
        for row in results
        if row.get("speaker_similarity_mean") is not None
    ]
    speaker_gate = [
        float(row["speaker_gate_pass_rate"])
        for row in results
        if row.get("speaker_gate_pass_rate") is not None
    ]
    return {
        "scenes": len(results),
        "target_wake_recall": float(np.mean(wake)) if wake else None,
        "target_intent_accuracy": float(np.mean(intent)) if intent else None,
        "false_wake_events": int(sum(false_wakes)) if false_wakes else None,
        "false_wake_rate_per_hour": (
            float(sum(false_wakes) * 3600.0 / max(duration_seconds, 1.0e-6))
            if false_wakes
            else None
        ),
        "false_action_events": int(sum(false_actions)) if false_actions else None,
        "false_action_rate_per_hour": (
            float(sum(false_actions) * 3600.0 / max(duration_seconds, 1.0e-6))
            if false_actions
            else None
        ),
        "target_wake_latency_mean_seconds": float(np.mean(latencies)) if latencies else None,
        "target_wake_latency_p95_seconds": float(np.percentile(latencies, 95)) if latencies else None,
        "target_presence_pass_rate_mean": float(np.mean(target_presence)) if target_presence else None,
        "passenger_presence_pass_rate_mean": (
            float(np.mean(passenger_presence)) if passenger_presence else None
        ),
        "speaker_similarity_mean": float(np.mean(speaker_similarity)) if speaker_similarity else None,
        "speaker_gate_pass_rate_mean": float(np.mean(speaker_gate)) if speaker_gate else None,
    }


def scene_descriptor(scene: dict[str, Any]) -> dict[str, Any]:
    truth = scene["truth"]
    return {
        "scene_index": int(scene["scene_index"]),
        "cabin_snr_db": float(truth["cabin_snr_db"]),
        "passenger_snr_db": float(truth["passenger_snr_db"]),
        "expected_intent": str(truth["expected_intent"]),
        "target_wake_source": str(truth["target_wake_source"]),
        "target_command_source": str(truth["target_command_source"]),
        "passenger_wake_source": str(truth["passenger_wake_source"]),
        "noise_source": str(truth["noise_source"]),
    }


def add_scene_result(
    collection: dict[str, list[dict[str, Any]]],
    scene: dict[str, Any],
    result: dict[str, Any],
) -> None:
    collection.setdefault(str(scene["scene_index"]), []).append(result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/eval_end_to_end_cavemove_target_kws_temporal2_candidate.yaml",
    )
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.yaml")
    parser.add_argument(
        "--wake-manifest",
        type=Path,
        default=REPO_ROOT / "manifests/test/stage3_kws_cabin_test.jsonl",
    )
    parser.add_argument(
        "--intent-manifest",
        type=Path,
        default=REPO_ROOT / "manifests/official/stage4_intent_fsc_test.jsonl",
    )
    parser.add_argument(
        "--noise-root",
        type=Path,
        default=REPO_ROOT / "data/raw/cavemove/extracted/cavemove_dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "logs/runtime_ablation_sweep_2026-10-10.json",
    )
    parser.add_argument("--scenes", type=int, default=2)
    parser.add_argument("--cabin-snr-db", nargs="+", type=float, default=[-5.0, 0.0, 10.0, 20.0])
    parser.add_argument("--passenger-snr-db", type=float, default=0.0)
    parser.add_argument("--kws-threshold", type=float, default=None)
    parser.add_argument("--presence-threshold", type=float, default=None)
    parser.add_argument("--hops", nargs="+", type=float, default=[0.5, 1.0])
    parser.add_argument("--enrollment-seconds", nargs="+", type=float, default=[0.5, 1.0, 1.5])
    parser.add_argument("--base-enrollment-seconds", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    cfg = load_stage_config(args.config, args.paths)
    if args.kws_threshold is not None:
        cfg["kws_threshold"] = float(args.kws_threshold)
    if args.presence_threshold is not None:
        cfg["presence_threshold"] = float(args.presence_threshold)
    device = get_device()
    pipeline = load_pipeline(cfg, cfg["paths"], device)
    sample_rate = int(cfg["sample_rate"])
    stage1_seconds = float(cfg.get("segment_seconds", 3.0))
    presence_threshold = float(cfg.get("presence_threshold", 0.5))
    kws_threshold = float(cfg.get("kws_threshold", 0.5))
    hops = sorted({float(value) for value in args.hops if float(value) > 0.0})
    enrollment_seconds = sorted(
        {float(value) for value in [*args.enrollment_seconds, args.base_enrollment_seconds] if float(value) > 0.0}
    )
    base_enrollment_seconds = float(args.base_enrollment_seconds)
    if not hops or not enrollment_seconds:
        raise ValueError("At least one positive hop and enrollment duration are required")

    wake_rows = [
        row
        for row in load_jsonl(resolve_path(args.wake_manifest))
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
        row
        for row in wake_rows
        if speaker_key(resolve_path(row["wav"])) != speaker_key(resolve_path(target_wake_rows[0]["wav"]))
    )
    negative_rows = [
        row
        for row in load_jsonl(resolve_path(args.wake_manifest))
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
        raise RuntimeError("No sufficiently long CAVEMOVE noise streams found")

    scenes: list[dict[str, Any]] = []
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
                    "speaker_consistent": False,
                }
            )
            scenes.append(
                {
                    "scene_index": len(scenes),
                    "truth": truth,
                    "mixture": mixture,
                    "multichannel": multichannel,
                }
            )

    total_duration = sum(float(scene["truth"]["duration_seconds"]) for scene in scenes)
    cache: dict[tuple[int, float, float, str], list[dict[str, Any]]] = {}

    def get_rows(scene: dict[str, Any], hop: float, enrollment: float, frontend: str) -> list[dict[str, Any]]:
        key = (int(scene["scene_index"]), float(hop), float(enrollment), frontend)
        if key not in cache:
            mc = frontend_variant(scene["multichannel"], frontend)
            mixture = mc.mean(axis=1).astype(np.float32)
            enrollment_wav = fit_start(
                scene["target_wake_audio"],
                int(round(enrollment * sample_rate)),
            )
            print(
                f"collect scene={scene['scene_index'] + 1}/{len(scenes)} "
                f"hop={hop:g}s enrollment={enrollment:g}s frontend={frontend}",
                flush=True,
            )
            cache[key] = collect_window_scores(
                pipeline,
                mixture,
                mc,
                enrollment_wav,
                scene["truth"],
                sample_rate,
                stage1_seconds,
                hop,
                device,
            )
        return cache[key]

    # The target wake audio is needed to form each enrollment variant. Keep it
    # out of the serialized report because it is already identified by source.
    # Reconstruct it from the source recorded in the truth dictionary.
    for scene in scenes:
        target_wake, _ = read_mono(Path(scene["truth"]["target_wake_source"]), sample_rate)
        scene["target_wake_audio"] = target_wake

    temporal_results: dict[str, Any] = {}
    for hop in hops:
        hop_key = f"{hop:g}s"
        temporal_results[hop_key] = {}
        for temporal_name, temporal_override in TEMPORAL_SETTINGS.items():
            by_variant: dict[str, list[dict[str, Any]]] = {"full": [], "no_speaker_matching": []}
            per_scene: list[dict[str, Any]] = []
            for scene in scenes:
                rows = get_rows(scene, hop, base_enrollment_seconds, "all4")
                scene_variants: dict[str, Any] = {}
                for variant_name in by_variant:
                    settings = dict(VARIANTS[variant_name])
                    settings.update(temporal_override)
                    result = summarize_variant(
                        variant_name,
                        settings,
                        pipeline,
                        rows,
                        scene["truth"],
                        sample_rate,
                        kws_threshold,
                        presence_threshold,
                        device,
                    )
                    by_variant[variant_name].append(result)
                    scene_variants[variant_name] = compact_result(result)
                per_scene.append({"scene": scene_descriptor(scene), "variants": scene_variants})
            temporal_results[hop_key][temporal_name] = {
                "settings": temporal_override,
                "summary": {
                    variant: aggregate(results, total_duration)
                    for variant, results in by_variant.items()
                },
                "per_scene": per_scene,
            }

    enrollment_results: dict[str, Any] = {}
    for enrollment in enrollment_seconds:
        key = f"{enrollment:g}s"
        results: list[dict[str, Any]] = []
        per_scene: list[dict[str, Any]] = []
        for scene in scenes:
            rows = get_rows(scene, hops[-1], enrollment, "all4")
            result = summarize_variant(
                "full",
                dict(VARIANTS["full"]),
                pipeline,
                rows,
                scene["truth"],
                sample_rate,
                kws_threshold,
                presence_threshold,
                device,
            )
            results.append(result)
            per_scene.append({"scene": scene_descriptor(scene), "result": compact_result(result)})
        enrollment_results[key] = {
            "enrollment_seconds": enrollment,
            "hop_seconds": hops[-1],
            "summary": aggregate(results, total_duration),
            "per_scene": per_scene,
        }

    frontend_results: dict[str, Any] = {}
    for frontend in FRONTEND_MODES:
        results = []
        per_scene = []
        for scene in scenes:
            rows = get_rows(scene, hops[-1], base_enrollment_seconds, frontend)
            result = summarize_variant(
                "full",
                dict(VARIANTS["full"]),
                pipeline,
                rows,
                scene["truth"],
                sample_rate,
                kws_threshold,
                presence_threshold,
                device,
            )
            results.append(result)
            per_scene.append({"scene": scene_descriptor(scene), "result": compact_result(result)})
        frontend_results[frontend] = {
            "frontend": frontend,
            "enrollment_seconds": base_enrollment_seconds,
            "hop_seconds": hops[-1],
            "summary": aggregate(results, total_duration),
            "per_scene": per_scene,
        }

    # These two sweeps reuse the base feature cache.  They exercise the
    # runtime state machine only, so no additional Stage-1 passes are needed.
    speaker_gate_thresholds = (-0.2, 0.0, 0.2, 0.4, 0.6, 0.8)
    command_presence_thresholds = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
    speaker_gate_results: dict[str, Any] = {}
    timeout_results: dict[str, Any] = {}
    command_presence_results: dict[str, Any] = {}
    command_capture_results: dict[str, Any] = {}
    base_hop = hops[-1]
    base_rows_by_scene = {
        int(scene["scene_index"]): get_rows(scene, base_hop, base_enrollment_seconds, "all4")
        for scene in scenes
    }

    disabled_results = []
    disabled_per_scene = []
    for scene in scenes:
        result = summarize_variant(
            "full",
            dict(VARIANTS["full"]),
            pipeline,
            base_rows_by_scene[int(scene["scene_index"])],
            scene["truth"],
            sample_rate,
            kws_threshold,
            presence_threshold,
            device,
        )
        disabled_results.append(result)
        disabled_per_scene.append({"scene": scene_descriptor(scene), "result": compact_result(result)})
    speaker_gate_results["disabled"] = {
        "threshold": None,
        "summary": aggregate(disabled_results, total_duration),
        "per_scene": disabled_per_scene,
    }

    for threshold in speaker_gate_thresholds:
        key = f"{threshold:g}"
        results = []
        per_scene = []
        for scene in scenes:
            settings = dict(VARIANTS["full"])
            settings.update({"use_speaker_gate": True, "speaker_gate_threshold": threshold})
            result = summarize_variant(
                "full_with_speaker_gate",
                settings,
                pipeline,
                base_rows_by_scene[int(scene["scene_index"])],
                scene["truth"],
                sample_rate,
                kws_threshold,
                presence_threshold,
                device,
            )
            results.append(result)
            per_scene.append({"scene": scene_descriptor(scene), "result": compact_result(result)})
        speaker_gate_results[key] = {
            "threshold": threshold,
            "summary": aggregate(results, total_duration),
            "per_scene": per_scene,
        }

    for timeout in (1, 2, 4, 6):
        results = []
        per_scene = []
        for scene in scenes:
            settings = dict(VARIANTS["full"])
            settings["command_timeout_calls"] = timeout
            result = summarize_variant(
                "full_with_timeout",
                settings,
                pipeline,
                base_rows_by_scene[int(scene["scene_index"])],
                scene["truth"],
                sample_rate,
                kws_threshold,
                presence_threshold,
                device,
            )
            results.append(result)
            per_scene.append({"scene": scene_descriptor(scene), "result": compact_result(result)})
        timeout_results[str(timeout)] = {
            "command_timeout_calls": timeout,
            "summary": aggregate(results, total_duration),
            "per_scene": per_scene,
        }

    for threshold in command_presence_thresholds:
        key = f"{threshold:g}"
        results = []
        per_scene = []
        for scene in scenes:
            settings = dict(VARIANTS["full"])
            settings["command_presence_threshold"] = threshold
            result = summarize_variant(
                "full_with_command_presence_threshold",
                settings,
                pipeline,
                base_rows_by_scene[int(scene["scene_index"])],
                scene["truth"],
                sample_rate,
                kws_threshold,
                presence_threshold,
                device,
            )
            results.append(result)
            per_scene.append({"scene": scene_descriptor(scene), "result": compact_result(result)})
        command_presence_results[key] = {
            "command_presence_threshold": threshold,
            "summary": aggregate(results, total_duration),
            "per_scene": per_scene,
        }

    # Test delayed command decisions after a wake. This is a runtime-only
    # proxy for buffering a short post-wake command segment instead of acting
    # on the first accepted frame. The low command-presence threshold is
    # intentional here: it isolates capture timing from the wake gate.
    for capture_calls in (1, 2, 3):
        for aggregation in ("mean_logits", "max_confidence"):
            key = f"calls={capture_calls}|{aggregation}"
            results = []
            per_scene = []
            for scene in scenes:
                settings = dict(VARIANTS["full"])
                settings.update(
                    {
                        "command_presence_threshold": 0.05,
                        "command_capture_calls": capture_calls,
                        "command_aggregation": aggregation,
                    }
                )
                result = summarize_variant(
                    "full_with_command_capture",
                    settings,
                    pipeline,
                    base_rows_by_scene[int(scene["scene_index"])],
                    scene["truth"],
                    sample_rate,
                    kws_threshold,
                    presence_threshold,
                    device,
                )
                results.append(result)
                per_scene.append({"scene": scene_descriptor(scene), "result": compact_result(result)})
            command_capture_results[key] = {
                "command_capture_calls": capture_calls,
                "command_aggregation": aggregation,
                "command_presence_threshold": 0.05,
                "summary": aggregate(results, total_duration),
                "per_scene": per_scene,
            }

    report = {
        "benchmark": "runtime_ablation_sweep_cross_corpus_proxy",
        "device": str(device),
        "config": str(resolve_path(args.config)),
        "checkpoints": cfg.get("checkpoints"),
        "code_revision": git_revision(),
        "sample_rate": sample_rate,
        "stage1_context_seconds": stage1_seconds,
        "presence_threshold": presence_threshold,
        "kws_threshold": kws_threshold,
        "scenes": len(scenes),
        "duration_seconds": total_duration,
        "speaker_consistent": False,
        "production_valid": False,
        "scene_conditions": [scene_descriptor(scene) for scene in scenes],
        "study_plan": {
            "temporal_hops_seconds": hops,
            "temporal_settings": TEMPORAL_SETTINGS,
            "base_enrollment_seconds": base_enrollment_seconds,
            "enrollment_seconds": enrollment_seconds,
            "frontends": list(FRONTEND_MODES),
            "speaker_gate_thresholds": [float(value) for value in speaker_gate_thresholds],
            "command_timeout_calls": [1, 2, 4, 6],
            "command_presence_thresholds": [float(value) for value in command_presence_thresholds],
            "command_capture_calls": [1, 2, 3],
            "command_aggregation": ["mean_logits", "max_confidence"],
        },
        "temporal": temporal_results,
        "enrollment": enrollment_results,
        "frontend": frontend_results,
        "speaker_gate": speaker_gate_results,
        "command_timeout": timeout_results,
        "command_presence": command_presence_results,
        "command_capture": command_capture_results,
        "limitation": (
            "Cross-corpus proxy only. Marvin wake clips and FSC command clips are not verified to be the same "
            "speaker, and the front-end modes are channel simulations rather than recordings from alternate "
            "vehicle microphone placements. These results diagnose runtime interactions and do not establish "
            "real-driver reliability."
        ),
    }
    output = resolve_path(args.output)
    write_metrics(output, report)
    compact_summary = {
        "output": str(output),
        "scenes": len(scenes),
        "temporal_hops": hops,
        "enrollment_seconds": enrollment_seconds,
        "frontends": list(FRONTEND_MODES),
        "cache_collections": len(cache),
        "device": str(device),
    }
    print(json.dumps(compact_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
