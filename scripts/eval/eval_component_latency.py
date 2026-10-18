#!/usr/bin/env python
"""Per-component latency benchmark for the promoted in-car configuration.

Each module is timed in isolation on the same synthetic inputs (a 3-second
four-channel buffer plus a scripted enrollment) so the report can separate
extraction, activity, keyword, verifier, and intent costs.  The composite
wake-stage and post-wake command-stage numbers mirror
``eval_full_pipeline_latency.py`` so both can be compared directly.

Measurements are model-forward times only: microphone I/O, resampling,
ring-buffer scheduling, and vehicle action execution are excluded, and the
inputs are random so nothing here is an accuracy result.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.pipeline.in_car_pipeline import (
    PipelineState,
    normalize_extracted_waveform,
)
from car_tse.train.utils import get_device
from eval_continuous_full_pipeline_proxy import load_pipeline, resolve_path


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def bench(fn, device: torch.device, warmup: int, repeats: int) -> dict[str, float]:
    for _ in range(max(0, warmup)):
        fn()
    synchronize(device)
    times: list[float] = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        fn()
        synchronize(device)
        times.append((time.perf_counter() - start) * 1000.0)
    return {
        "mean_ms": float(np.mean(times)),
        "p50_ms": percentile(times, 50),
        "p95_ms": percentile(times, 95),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "configs/eval_end_to_end_cavemove_target_kws_smart_speaker_compact_cabin_smart_thr030_2026-10-12.yaml",
    )
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.yaml")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.precision == "fp16" and args.device == "cpu":
        raise ValueError("fp16 requires --device cuda")
    device = get_device() if args.device == "auto" else torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False

    cfg = load_stage_config(args.config, args.paths)
    pipeline = load_pipeline(cfg, cfg["paths"], device)
    sample_rate = int(cfg.get("sample_rate", 16000))
    context_seconds = float(cfg.get("segment_seconds", 3.0))
    num_mics = int(cfg.get("num_mics", 4))

    rng = np.random.default_rng(20261012)
    context_samples = int(round(sample_rate * context_seconds))
    multichannel_np = rng.standard_normal((num_mics, context_samples), dtype=np.float32)
    mixture_np = multichannel_np.mean(axis=0).astype(np.float32)
    enrollment_np = rng.standard_normal(int(round(3.0 * sample_rate)), dtype=np.float32)
    mixture = torch.from_numpy(mixture_np).to(device).unsqueeze(0)
    multichannel = torch.from_numpy(multichannel_np).to(device).unsqueeze(0)
    enrollment = torch.from_numpy(enrollment_np).to(device).unsqueeze(0)

    def context():
        if args.precision == "fp16":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    with torch.inference_mode(), context():
        s1, _, _, _ = pipeline.stage1(mixture, enrollment, multichannel=multichannel)
        extracted_embedding = pipeline.stage1.speaker_embedding(s1)
        enrollment_embedding = pipeline.stage1.speaker_embedding(enrollment)
        normalized = normalize_extracted_waveform(s1)
        kws_wave = normalized[..., : sample_rate].contiguous()
        verifier_enrollment = enrollment[..., :sample_rate].contiguous()
        verifier_extracted = normalized[..., :sample_rate].contiguous()
        intent_input_seconds = float(getattr(pipeline, "intent_input_seconds", 2.0))
        intent_samples = max(1, int(round(sample_rate * intent_input_seconds)))
        intent_wave = normalized[..., :intent_samples].contiguous()
        if intent_wave.shape[-1] < intent_samples:
            intent_wave = F.pad(intent_wave, (0, intent_samples - intent_wave.shape[-1]))
    s1 = s1.detach()
    extracted_embedding = extracted_embedding.detach()
    enrollment_embedding = enrollment_embedding.detach()

    components: dict[str, dict[str, float]] = {}

    def timed(name: str, fn) -> None:
        def runner():
            with torch.inference_mode(), context():
                fn()

        components[name] = bench(runner, device, int(args.warmup), int(args.repeats))

    timed("stage1_extraction_3s_4ch", lambda: pipeline.stage1(mixture, enrollment, multichannel=multichannel))
    timed("enrollment_speaker_embedding", lambda: pipeline.stage1.speaker_embedding(enrollment))
    timed("extracted_speaker_embedding", lambda: pipeline.stage1.speaker_embedding(s1))
    timed("presence_head_3s", lambda: pipeline.presence(s1))
    timed(
        "target_kws_1s",
        lambda: pipeline.target_kws(kws_wave, enrollment_embedding, extracted_embedding),
    )
    if pipeline.kws is not None:
        timed("general_kws_1s", lambda: pipeline.kws(kws_wave))
    if pipeline.compact_speaker_verifier is not None:
        timed(
            "compact_speaker_verifier_2x1s",
            lambda: pipeline.compact_speaker_verifier(verifier_enrollment, verifier_extracted),
        )
    intent_component_name = f"intent_wav2vec_{intent_input_seconds:g}s"
    timed(intent_component_name, lambda: pipeline.intent(intent_wave))

    steady_state = PipelineState(target_kws_enrollment_embedding=enrollment_embedding)

    def run_wake() -> None:
        state = PipelineState(target_kws_enrollment_embedding=enrollment_embedding)
        with torch.inference_mode(), context():
            pipeline.forward(mixture, enrollment, state, multichannel=multichannel)

    saved = (
        pipeline.command_presence_threshold,
        pipeline.speaker_gate_threshold,
        pipeline.command_speaker_gate_threshold,
        pipeline.intent_confidence_threshold,
    )

    def run_command() -> None:
        state = PipelineState(
            awaiting_command=True,
            target_kws_enrollment_embedding=enrollment_embedding,
        )
        with torch.inference_mode(), context():
            pipeline.forward(mixture, enrollment, state, multichannel=multichannel)

    pipeline.command_presence_threshold = 0.0
    pipeline.speaker_gate_threshold = None
    pipeline.command_speaker_gate_threshold = None
    pipeline.intent_confidence_threshold = None
    components["composite_wake_stage"] = bench(run_wake, device, int(args.warmup), int(args.repeats))
    components["composite_command_stage"] = bench(run_command, device, int(args.warmup), int(args.repeats))
    (
        pipeline.command_presence_threshold,
        pipeline.speaker_gate_threshold,
        pipeline.command_speaker_gate_threshold,
        pipeline.intent_confidence_threshold,
    ) = saved

    parameters = {
        "stage1": int(sum(p.numel() for p in pipeline.stage1.parameters())),
        "presence": int(sum(p.numel() for p in pipeline.presence.parameters())),
        "kws": int(sum(p.numel() for p in pipeline.kws.parameters())) if pipeline.kws is not None else 0,
        "target_kws": (
            int(sum(p.numel() for p in pipeline.target_kws.parameters()))
            if pipeline.target_kws is not None
            else 0
        ),
        "compact_speaker_verifier": (
            int(sum(p.numel() for p in pipeline.compact_speaker_verifier.parameters()))
            if pipeline.compact_speaker_verifier is not None
            else 0
        ),
        "intent": int(sum(p.numel() for p in pipeline.intent.parameters())),
    }
    result = {
        "benchmark": "per_component_latency",
        "config": str(resolve_path(args.config)),
        "device": str(device),
        "precision": args.precision,
        "sample_rate": sample_rate,
        "context_seconds": context_seconds,
        "intent_input_seconds": intent_input_seconds,
        "num_mics": num_mics,
        "warmup": int(args.warmup),
        "repeats": int(args.repeats),
        "components": components,
        "parameters": parameters,
        "notes": [
            "Model-forward times only; microphone I/O, resampling, ring scheduling, and action execution excluded.",
            "Random synthetic inputs: not an accuracy measurement.",
            "Composite command stage forces the intent branch and disables decision gates only for compute measurement.",
        ],
    }
    output = args.output or (
        Path(cfg["paths"]["log_root"])
        / f"component_latency_{device.type}_{args.precision}_2026-10-12.json"
    )
    output = resolve_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        f"{'component':38s} {'mean':>8s} {'p50':>8s} {'p95':>8s} {'params':>12s}",
        flush=True,
    )
    for name, values in components.items():
        parameter_key = {
            "stage1_extraction_3s_4ch": "stage1",
            "enrollment_speaker_embedding": "stage1",
            "extracted_speaker_embedding": "stage1",
            "presence_head_3s": "presence",
            "target_kws_1s": "target_kws",
            "general_kws_1s": "kws",
            "compact_speaker_verifier_2x1s": "compact_speaker_verifier",
        intent_component_name: "intent",
        }.get(name)
        count = parameters.get(parameter_key, 0) if parameter_key else 0
        print(
            f"{name:38s} {values['mean_ms']:8.2f} {values['p50_ms']:8.2f} {values['p95_ms']:8.2f} {count:12,d}",
            flush=True,
        )
    print(json.dumps({"output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
