#!/usr/bin/env python
"""Benchmark the loaded end-to-end in-car pipeline.

This complements the older component-only latency script.  It loads the same
checkpoint/configuration used by the continuous evaluator and measures the
actual ``InCarPipeline.forward`` path for a wake-stage frame and a post-wake
command frame.  The command path is forced through the intent branch so the
measurement remains a compute benchmark even when a synthetic random input
does not satisfy the safety gates.

The result is an engineering resource measurement, not a real-driver quality
claim.  It reports the input context, p50/p95 wall-clock latency, real-time
factor, parameter count, checkpoint bytes, and peak CUDA allocation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.pipeline.in_car_pipeline import PipelineState
from car_tse.train.utils import get_device, write_metrics
from eval_continuous_full_pipeline_proxy import load_pipeline, resolve_path


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    fn: Callable[[], None],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, float | list[float]]:
    for _ in range(max(0, int(warmup))):
        fn()
    synchronize(device)
    samples: list[float] = []
    for _ in range(max(1, int(repeats))):
        start = time.perf_counter()
        fn()
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    values = np.asarray(samples, dtype=np.float64)
    return {
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "samples_ms": [float(value) for value in values.tolist()],
    }


def module_parameters(module: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def pipeline_parameters(pipeline: object) -> int:
    total = 0
    for name in (
        "stage1",
        "presence",
        "kws",
        "kws_secondary",
        "target_kws",
        "target_kws_secondary",
        "intent",
    ):
        module = getattr(pipeline, name, None)
        if isinstance(module, torch.nn.Module):
            total += module_parameters(module)
    return int(total)


def checkpoint_bytes(root: Path, cfg: dict) -> int:
    total = 0
    for relative in cfg.get("checkpoints", {}).values():
        if not relative:
            continue
        path = resolve_path(root / str(relative))
        if path.exists():
            total += int(path.stat().st_size)
    encoder = cfg.get("intent_encoder_checkpoint")
    if encoder:
        path = Path(str(encoder))
        if not path.is_absolute():
            path = root / path
        path = resolve_path(path)
        if path.exists():
            total += int(path.stat().st_size)
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "configs"
        / "eval_end_to_end_cavemove_target_kws_dual_domain_stage2_hardneg_aishell5_neg_safety_splitgate_candidate.yaml",
    )
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="fp32")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available")
    device = get_device() if args.device == "auto" else torch.device(args.device)
    if args.precision in {"fp16", "int8"}:
        expected_device = "cuda" if args.precision == "fp16" else "cpu"
        if device.type != expected_device:
            raise ValueError(f"--precision {args.precision} requires --device {expected_device}")
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        # Keep the benchmark representative of a bounded deployment process.
        # cuDNN autotuning can allocate multi-gigabyte workspaces for this
        # variable-length cascade, which is not a safe default for edge use.
        torch.backends.cudnn.benchmark = False
    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    pipeline = load_pipeline(cfg, paths, device)
    pre_quantized_parameters = pipeline_parameters(pipeline)
    quantization_note = "not applied"
    if args.precision == "int8":
        # Dynamic quantization is limited to the CPU intent branch.  Packed
        # weights are not exposed through ``parameters()``, so retain the
        # original count separately for an honest model-size report.
        pipeline.intent = torch.ao.quantization.quantize_dynamic(
            pipeline.intent,
            {torch.nn.Linear, torch.nn.LSTM},
            dtype=torch.qint8,
        )
        pipeline.intent.eval()
        quantization_note = "dynamic int8 applied to intent Linear/LSTM modules"

    sample_rate = int(cfg.get("sample_rate", 16000))
    context_seconds = float(cfg.get("segment_seconds", 3.0))
    samples = max(1, int(round(sample_rate * context_seconds)))
    num_mics = int(cfg.get("num_mics", 4))
    rng = np.random.default_rng(20261011)
    multichannel_np = rng.standard_normal((num_mics, samples), dtype=np.float32)
    mixture_np = multichannel_np.mean(axis=0).astype(np.float32)
    enrollment_np = rng.standard_normal(max(1, int(round(3.0 * sample_rate))), dtype=np.float32)
    mixture = torch.from_numpy(mixture_np).to(device).unsqueeze(0)
    multichannel = torch.from_numpy(multichannel_np).to(device).unsqueeze(0)
    enrollment = torch.from_numpy(enrollment_np).to(device).unsqueeze(0)

    # Enrollment speaker embedding is a one-time setup cost in the streaming
    # application.  Measure it separately and reuse it for steady-state calls.
    with torch.inference_mode():
        enroll_start = time.perf_counter()
        enrollment_embedding = pipeline.stage1.speaker_embedding(enrollment).detach()
        synchronize(device)
        enrollment_embedding_ms = (time.perf_counter() - enroll_start) * 1000.0

    steady_state = PipelineState(target_kws_enrollment_embedding=enrollment_embedding)

    def inference_context():
        if args.precision == "fp16":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def run_wake() -> None:
        state = PipelineState(
            target_kws_enrollment_embedding=steady_state.target_kws_enrollment_embedding
        )
        with torch.inference_mode():
            with inference_context():
                pipeline.forward(mixture, enrollment, state, multichannel=multichannel)

    # Make the command branch deterministic for the compute benchmark.  This
    # does not alter the saved configuration or release thresholds.
    saved = (
        pipeline.command_presence_threshold,
        pipeline.speaker_gate_threshold,
        pipeline.command_speaker_gate_threshold,
        pipeline.intent_confidence_threshold,
    )
    pipeline.command_presence_threshold = 0.0
    pipeline.speaker_gate_threshold = None
    pipeline.command_speaker_gate_threshold = None
    pipeline.intent_confidence_threshold = None

    def run_command() -> None:
        state = PipelineState(
            awaiting_command=True,
            target_kws_enrollment_embedding=steady_state.target_kws_enrollment_embedding,
        )
        with torch.inference_mode():
            with inference_context():
                pipeline.forward(mixture, enrollment, state, multichannel=multichannel)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    wake = benchmark(run_wake, device=device, warmup=args.warmup, repeats=args.repeats)
    command = benchmark(run_command, device=device, warmup=args.warmup, repeats=args.repeats)
    wake["mean_rtf"] = float(wake["mean_ms"]) / 1000.0 / context_seconds
    command["mean_rtf"] = float(command["mean_ms"]) / 1000.0 / context_seconds
    pipeline.command_presence_threshold, pipeline.speaker_gate_threshold, pipeline.command_speaker_gate_threshold, pipeline.intent_confidence_threshold = saved

    total_parameters = pipeline_parameters(pipeline)
    checkpoint_root = Path(paths["checkpoint_root"])
    metrics = {
        "benchmark": "full_in_car_pipeline_forward",
        "timestamp": "2026-10-11",
        "config": str(args.config),
        "device": str(device),
        "precision": args.precision,
        "pre_quantized_parameters": pre_quantized_parameters,
        "quantization_note": quantization_note,
        "sample_rate": sample_rate,
        "context_seconds": context_seconds,
        "num_mics": num_mics,
        "warmup": int(args.warmup),
        "repeats": int(args.repeats),
        "wake_stage": wake,
        "post_wake_command_stage": command,
        "enrollment_embedding_ms": float(enrollment_embedding_ms),
        "total_parameters": total_parameters if args.precision != "int8" else pre_quantized_parameters,
        "checkpoint_bytes": checkpoint_bytes(checkpoint_root, cfg),
        "checkpoint_megabytes_decimal": checkpoint_bytes(checkpoint_root, cfg) / 1_000_000.0,
        "peak_cuda_allocated_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
            if device.type == "cuda"
            else None
        ),
        "notes": [
            "The wake-stage measurement runs the configured forward path with target-KWS and smoothing state freshly initialized per call.",
            "The post-wake measurement forces the intent branch and disables decision gates only to measure compute; it is not a quality result.",
            "The synthetic input is random and must not be used as a speech or noise accuracy claim.",
        ],
    }
    output = args.output or (Path(paths["log_root"]) / "full_pipeline_latency_2026-10-11.json")
    write_metrics(output, metrics)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
