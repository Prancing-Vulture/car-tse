#!/usr/bin/env python
"""Benchmark the retained ONNX pipeline through the live-capture ring path.

This benchmark uses one deterministic CAVEMOVE proxy row, repeats it to form a
short stream, and feeds arbitrary capture-sized frames through
"StreamingRunner". It measures the deployment plumbing and host reference
throughput; it is not a real-driver accuracy or target-SoC result.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from car_tse.config import load_stage_config  # noqa: E402
from car_tse.deployment.onnx_runtime import OnnxComponentRuntime  # noqa: E402
from car_tse.pipeline.streaming_runner import StreamingRunner  # noqa: E402
from eval_onnx_composed_pipeline_parity import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_MANIFEST,
    _build_onnx_pipeline,
    _build_dataset,
    _dataset_input,
)


def _repeat_to_length(value: np.ndarray, samples: int) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 1:
        repeats = int(np.ceil(samples / max(1, value.shape[0])))
        return np.tile(value, repeats)[:samples]
    if value.ndim == 2:
        repeats = int(np.ceil(samples / max(1, value.shape[0])))
        return np.tile(value, (repeats, 1))[:samples]
    raise ValueError(f"expected mono or time-first multichannel audio, got {value.shape}")


def run(
    *,
    config_path: Path,
    paths_path: Path,
    onnx_manifest: Path,
    row_index: int,
    seconds: float,
    frame_ms: float,
    intra_op_threads: int | None = None,
    inter_op_threads: int | None = None,
    hop_ms: float | None = None,
) -> dict[str, Any]:
    cfg = load_stage_config(config_path, paths_path)
    dataset, input_manifest = _build_dataset(cfg)
    inputs, metadata = _dataset_input(cfg, dataset, row_index)
    sample_rate = int(cfg.get("sample_rate", 16_000))
    total_samples = max(1, int(round(float(seconds) * sample_rate)))
    frame_samples = max(1, int(round(float(frame_ms) * sample_rate / 1000.0)))

    mixture = _repeat_to_length(inputs["mixture"].squeeze(0).numpy(), total_samples)
    multichannel = _repeat_to_length(
        inputs["multichannel"].squeeze(0).numpy().T,
        total_samples,
    )
    playback = _repeat_to_length(
        inputs["playback_ref"].squeeze(0).numpy(),
        total_samples,
    )
    enrollment = inputs["enrollment"].squeeze(0).numpy()
    hop_seconds = (
        float(hop_ms) / 1000.0
        if hop_ms is not None
        else float(cfg.get("hop_seconds", 0.25))
    )

    runtime = OnnxComponentRuntime(
        onnx_manifest,
        require_playback_reference=True,
        intra_op_num_threads=intra_op_threads,
        inter_op_num_threads=inter_op_threads,
    )
    pipeline = _build_onnx_pipeline(runtime, cfg)
    runner = StreamingRunner(
        forward=pipeline.forward,
        enrollment=enrollment,
        sample_rate=sample_rate,
        window_seconds=float(cfg.get("segment_seconds", 3.0)),
        hop_seconds=hop_seconds,
        num_mics=int(cfg.get("num_mics", 4)),
        max_lag_seconds=2.0,
        overload_policy="drop_oldest",
        max_windows_per_call=8,
        use_playback_ref=True,
    )

    started = time.perf_counter()
    pushed = 0
    process_calls = 0
    while pushed < total_samples:
        end = min(total_samples, pushed + frame_samples)
        runner.push(
            mixture[pushed:end],
            multichannel=multichannel[pushed:end],
            playback_ref=playback[pushed:end],
        )
        process_calls += 1
        runner.process_available()
        pushed = end
    elapsed = time.perf_counter() - started
    stats = runner.stats()
    return {
        "benchmark": "onnx_streaming_runner",
        "status": "ok",
        "config": str(config_path.resolve()),
        "onnx_manifest": str(onnx_manifest.resolve()),
        "input_manifest": str(input_manifest.resolve()),
        "device": "CPUExecutionProvider",
        "runtime_options": {
            "intra_op_threads": runtime.intra_op_num_threads,
            "inter_op_threads": runtime.inter_op_num_threads,
            "mem_pattern": False,
            "intent_only_while_awaiting_command": True,
        },
        "input": {
            **metadata,
            "sample_rate": sample_rate,
            "stream_seconds": total_samples / sample_rate,
            "frame_ms": float(frame_ms),
            "frame_samples": frame_samples,
            "hop_ms": hop_seconds * 1000.0,
            "microphone_channels": int(multichannel.shape[1]),
            "playback_reference": True,
        },
        "runner": {
            **stats,
            "process_calls": process_calls,
            "wall_seconds_including_feeding": elapsed,
            "inference_real_time_factor": stats["real_time_factor"],
            "capture_frame_real_time_factor": elapsed / max(total_samples / sample_rate, 1.0e-9),
        },
        "events": {
            "wake_events": int(runner.wake_events),
            "intent_events": int(runner.intent_events),
        },
        "limitations": (
            "Deterministic repeated CAVEMOVE proxy audio through the host CPU ONNX "
            "runtime. This validates ring-buffer continuity and playback-reference "
            "routing; it is not real-driver accuracy, a target-SoC benchmark, or "
            "production sign-off."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.yaml")
    parser.add_argument("--onnx-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--frame-ms", type=float, default=20.0)
    parser.add_argument(
        "--hop-ms",
        type=float,
        default=None,
        help="Optional StreamingRunner hop in milliseconds; omit for the config value.",
    )
    parser.add_argument(
        "--intra-op-threads",
        type=int,
        default=None,
        help="Optional ONNX Runtime intra-op thread count; omit for the default.",
    )
    parser.add_argument(
        "--inter-op-threads",
        type=int,
        default=None,
        help="Optional ONNX Runtime inter-op thread count; omit for the default.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "logs/onnx_streaming_runner_cpu_10s_2026-10-14.json",
    )
    args = parser.parse_args()
    result = run(
        config_path=args.config.resolve(),
        paths_path=args.paths.resolve(),
        onnx_manifest=args.onnx_manifest.resolve(),
        row_index=int(args.row_index),
        seconds=float(args.seconds),
        frame_ms=float(args.frame_ms),
        intra_op_threads=args.intra_op_threads,
        inter_op_threads=args.inter_op_threads,
        hop_ms=args.hop_ms,
    )
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"status": result["status"], "runner": result["runner"], "events": result["events"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
