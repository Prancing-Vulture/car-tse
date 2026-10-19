#!/usr/bin/env python
"""Measure stage-wise and end-to-end latency on a four-channel WAV.

The stage measurements time the configured Stage 1 extraction, Stage 2
presence, Stage 3 wake-word, and Stage 4 intent models separately.  The
four-stage total is the sum of their mean model times.  A separate measurement
times one real steady-state pipeline wake-window call, including state-machine
and Python overhead.  File I/O, WAV resampling, and one-time model loading are
not included in the steady-state call.

Compatible with the Jetson Xavier AGX Python 3.8 / CUDA PyTorch environment.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from car_tse.config import load_stage_config
from car_tse.io.audio import read_mono, read_multichannel
from car_tse.train.utils import load_checkpoint


def repo_path(value: Any) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    fn: Callable[[], Any], device: torch.device, warmup: int, repeats: int
) -> Dict[str, Any]:
    for _ in range(warmup):
        fn()
    synchronize(device)
    samples = []
    for _ in range(repeats):
        synchronize(device)
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
        "samples_ms": [float(item) for item in values.tolist()],
    }


def load_label_map(cfg: Dict[str, Any], paths: Dict[str, Any]) -> Dict[str, str]:
    value = cfg.get("intent_label_map")
    if isinstance(value, dict):
        if value and all(str(key).isdigit() for key in value):
            return {str(key): str(label) for key, label in value.items()}
        return {str(index): str(label) for label, index in value.items()}
    candidates = []
    if value:
        candidates.append(repo_path(value))
    candidates.append(repo_path(paths["manifest_root"]) / "meta" / "intent_label_map.json")
    for path in candidates:
        if path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
            return {str(index): str(label) for label, index in raw.items()}
    return {}


def load_pipeline(
    cfg: Dict[str, Any], paths_file: Path, base_checkpoint: Path, device: torch.device
) -> Any:
    # Import here so ``--help`` still works when optional Transformer
    # dependencies are not installed in a CPU-only inspection environment.
    from car_tse.models.stage1_system import Stage1System
    from car_tse.pipeline.in_car_pipeline import InCarPipeline

    paths = cfg["paths"]
    stage1_config_path = repo_path(cfg["stage1_config"])
    stage1_cfg = load_stage_config(stage1_config_path, paths_file)

    stage1 = Stage1System(
        spex_cfg=stage1_cfg["spex"],
        sample_rate=int(stage1_cfg.get("sample_rate", cfg.get("sample_rate", 16000))),
        use_cabin_frontend=bool(stage1_cfg.get("use_cabin_frontend", True)),
        num_mics=int(stage1_cfg.get("num_mics", cfg.get("num_mics", 4))),
        frontend_cfg=stage1_cfg.get("frontend"),
        pretrained_dir=str(base_checkpoint),
        backend=stage1_cfg.get("spex_backend", "transfer_spexplus_16k"),
    ).to(device)

    checkpoint_root = repo_path(paths["checkpoint_root"])
    checkpoints = cfg["checkpoints"]
    stage1_checkpoint = repo_path(checkpoint_root / str(checkpoints["spex"]))
    load_checkpoint(stage1_checkpoint, stage1)

    sample_rate = int(cfg.get("sample_rate", stage1_cfg.get("sample_rate", 16000)))
    model_cfg = cfg.get("model", {}) or {}
    intent_encoder_checkpoint = cfg.get("intent_encoder_checkpoint")
    if intent_encoder_checkpoint:
        intent_encoder_checkpoint = repo_path(checkpoint_root / str(intent_encoder_checkpoint))

    pipeline = InCarPipeline.from_checkpoints(
        stage1,
        repo_path(checkpoint_root / str(checkpoints["presence"])),
        repo_path(checkpoint_root / str(checkpoints["kws"])),
        repo_path(checkpoint_root / str(checkpoints["intent"])),
        kws_sample_rate=int(cfg.get("kws_sample_rate", sample_rate)),
        kws_n_mels=int(cfg.get("kws_n_mels", model_cfg.get("n_mels", 40))),
        kws_channels=cfg.get("kws_channels", model_cfg.get("channels", [32, 64, 128])),
        kws_num_classes=int(cfg.get("kws_num_classes", 2)),
        intent_sample_rate=int(cfg.get("intent_sample_rate", sample_rate)),
        intent_type=str(cfg.get("intent_type", "crnn")),
        intent_normalize_input=bool(cfg.get("intent_normalize_input", False)),
        intent_bidirectional=bool(cfg.get("intent_bidirectional", True)),
        intent_n_mels=int(cfg.get("intent_n_mels", model_cfg.get("n_mels", 80))),
        intent_hidden=int(cfg.get("intent_hidden", model_cfg.get("hidden", 256))),
        intent_num_layers=int(cfg.get("intent_num_layers", model_cfg.get("num_layers", 2))),
        intent_encoder=str(cfg.get("intent_encoder", "facebook/wav2vec2-base-960h")),
        intent_encoder_checkpoint=intent_encoder_checkpoint,
        intent_num_classes=int(cfg.get("intent_num_classes", 31)),
        presence_type=str(cfg.get("presence_type", "scalar")),
        presence_sample_rate=int(cfg.get("presence_sample_rate", sample_rate)),
        presence_n_mels=int(cfg.get("presence_n_mels", 40)),
        presence_hidden=int(cfg.get("presence_hidden", 64)),
        presence_n_fft=int(cfg.get("presence_n_fft", 512)),
        presence_hop=int(cfg.get("presence_hop", 160)),
        presence_speaker_embedding_dim=int(cfg.get("presence_speaker_embedding_dim", 256)),
        presence_threshold=float(cfg.get("presence_threshold", 0.5)),
        kws_threshold=float(cfg.get("kws_threshold", 0.5)),
        wake_class_id=int(cfg.get("wake_class_id", 1)),
        kws_input_seconds=float(cfg.get("kws_input_seconds", 1.0)),
        intent_input_seconds=float(cfg.get("intent_input_seconds", 2.0)),
        command_timeout_calls=int(cfg.get("command_timeout_calls", 4)),
        label_map=load_label_map(cfg, paths),
        device=device,
    )
    return pipeline


def stage3_forward(
    pipeline: Any,
    kws_wave: torch.Tensor,
    enrollment_embedding: Optional[torch.Tensor],
    extracted_embedding: Optional[torch.Tensor],
    presence_score: torch.Tensor,
    speaker_similarity: Optional[torch.Tensor],
) -> None:
    """Run the configured KWS branch, including optional ensemble heads."""
    kws_input = pipeline._prepare_kws_input(kws_wave)
    score = None
    if pipeline.target_kws is not None:
        if enrollment_embedding is None or extracted_embedding is None:
            raise RuntimeError("Configured target-KWS needs SpEx speaker embeddings")
        logits = pipeline.target_kws(kws_input, enrollment_embedding, extracted_embedding)
        score = torch.softmax(logits, dim=-1)[..., pipeline.wake_class_id].mean()
        if pipeline.target_kws_secondary is not None:
            secondary = pipeline.target_kws_secondary(
                kws_input, enrollment_embedding, extracted_embedding
            )
            secondary_score = torch.softmax(secondary, dim=-1)[..., pipeline.wake_class_id].mean()
            if pipeline.target_kws_fusion_mode == "mean":
                score = 0.5 * (score + secondary_score)
            elif pipeline.target_kws_fusion_mode == "geomean":
                score = torch.sqrt(score.clamp_min(0.0) * secondary_score.clamp_min(0.0))
            else:
                score = torch.minimum(score, secondary_score)
        if pipeline.speaker_fusion_mode != "none" and speaker_similarity is not None:
            score = torch.as_tensor(
                pipeline._fuse_speaker_score(float(score.item()), speaker_similarity),
                dtype=torch.float32,
                device=score.device,
            )
        if pipeline.kws_secondary is not None:
            secondary = pipeline.kws_secondary(kws_input)
            secondary_score = torch.softmax(secondary, dim=-1)[..., pipeline.wake_class_id].mean()
            if pipeline.kws_fusion_mode == "min":
                score = torch.minimum(score, secondary_score)
            elif pipeline.kws_fusion_mode == "max":
                score = torch.maximum(score, secondary_score)
            elif pipeline.kws_fusion_mode == "geomean":
                score = torch.sqrt(score.clamp_min(0.0) * secondary_score.clamp_min(0.0))
            else:
                alpha = float(pipeline.kws_fusion_alpha)
                score = alpha * score + (1.0 - alpha) * secondary_score
    else:
        score = torch.as_tensor(
            pipeline._regular_kws_score(kws_wave),
            dtype=torch.float32,
            device=presence_score.device,
        )
    if pipeline.stage23_fusion is not None:
        pipeline.stage23_fusion(presence_score.mean(), score, speaker_similarity)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "eval_end_to_end_cavemove.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--mixture", type=Path, default=REPO_ROOT / "data" / "cavemove_mixture_4ch.wav")
    parser.add_argument("--enrollment", type=Path, default=REPO_ROOT / "data" / "enrollment.wav")
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "car_tse_spexplus_from_stage1.pt",
        help="Base SpEx+ weights used to construct Stage 1 (overrides the path in the stage config).",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "logs" / "stagewise_latency.json")
    args = parser.parse_args()

    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be >= 0 and --repeats must be >= 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )

    config_path = repo_path(args.config)
    paths_file = repo_path(args.paths)
    mixture_path = repo_path(args.mixture)
    enrollment_path = repo_path(args.enrollment)
    base_checkpoint = repo_path(args.base_checkpoint)
    for path, label in (
        (config_path, "pipeline config"),
        (mixture_path, "four-channel mixture WAV"),
        (enrollment_path, "enrollment WAV"),
        (base_checkpoint, "base SpEx+ checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError("Missing {}: {}".format(label, path))
    if not paths_file.is_file():
        example = REPO_ROOT / "configs" / "paths.example.yaml"
        if example.is_file():
            paths_file = example

    cfg = load_stage_config(config_path, paths_file)
    sample_rate = int(cfg.get("sample_rate", 16000))
    num_mics = int(cfg.get("num_mics", 4))
    context_seconds = float(cfg.get("segment_seconds", 3.0))
    if context_seconds <= 0.0:
        raise ValueError("segment_seconds in the evaluation config must be positive")

    mixture_source_rate = int(sf.info(str(mixture_path)).samplerate)
    enrollment_source_rate = int(sf.info(str(enrollment_path)).samplerate)
    multichannel_np, _ = read_multichannel(mixture_path, sample_rate)
    enrollment_np, _ = read_mono(enrollment_path, sample_rate)
    if multichannel_np.shape[1] != num_mics:
        raise ValueError(
            "{} has {} channels, but the selected config expects {}. "
            "Use the four-channel mixture WAV.".format(
                mixture_path, multichannel_np.shape[1], num_mics
            )
        )
    if multichannel_np.shape[0] == 0 or enrollment_np.size == 0:
        raise ValueError("Mixture and enrollment WAVs must contain audio")
    if not np.isfinite(multichannel_np).all() or not np.isfinite(enrollment_np).all():
        raise ValueError("Mixture or enrollment WAV contains NaN or infinite samples")

    window_samples = max(1, int(round(sample_rate * context_seconds)))
    clip = multichannel_np[:window_samples]
    if clip.shape[0] < window_samples:
        clip = np.pad(clip, ((0, window_samples - clip.shape[0]), (0, 0)))
    multichannel = torch.from_numpy(np.ascontiguousarray(clip.T)).unsqueeze(0).to(device)
    mixture = multichannel.mean(dim=1)
    enrollment = torch.from_numpy(np.ascontiguousarray(enrollment_np)).unsqueeze(0).to(device)

    from car_tse.pipeline.in_car_pipeline import PipelineState, normalize_extracted_waveform

    pipeline = load_pipeline(cfg, paths_file, base_checkpoint, device)
    with torch.inference_mode():
        # One-time speaker setup is reported separately from steady-state audio calls.
        enrollment_embedding_ms = None
        enrollment_embedding = None
        if callable(getattr(pipeline.stage1, "speaker_embedding", None)):
            synchronize(device)
            setup_start = time.perf_counter()
            enrollment_embedding = pipeline.stage1.speaker_embedding(enrollment)
            synchronize(device)
            enrollment_embedding_ms = (time.perf_counter() - setup_start) * 1000.0
            if enrollment_embedding is not None:
                enrollment_embedding = enrollment_embedding.detach()

        stage1_cache = pipeline.stage1(mixture, enrollment, multichannel=multichannel)
        extracted = stage1_cache[0].detach()
        normalized = normalize_extracted_waveform(extracted)
        presence_embedding = (
            enrollment_embedding if getattr(pipeline.presence, "requires_enrollment", False) else None
        )
        presence_cache = pipeline.presence_activity_scores(extracted, presence_embedding)[0].detach()
        if callable(getattr(pipeline.stage1, "speaker_embedding", None)):
            extracted_embedding = pipeline.stage1.speaker_embedding(extracted)
            if extracted_embedding is not None:
                extracted_embedding = extracted_embedding.detach()
        else:
            extracted_embedding = None
        speaker_similarity = None
        if enrollment_embedding is not None and extracted_embedding is not None:
            speaker_similarity = torch.nn.functional.cosine_similarity(
                enrollment_embedding, extracted_embedding, dim=-1
            ).detach()
        kws_wave = normalized
        intent_wave = pipeline._prepare_intent_input(normalized)

        stage1_stats = benchmark(
            lambda: pipeline.stage1(mixture, enrollment, multichannel=multichannel),
            device,
            args.warmup,
            args.repeats,
        )
        stage2_stats = benchmark(
            lambda: pipeline.presence_activity_scores(extracted, presence_embedding),
            device,
            args.warmup,
            args.repeats,
        )
        stage3_stats = benchmark(
            lambda: stage3_forward(
                pipeline,
                kws_wave,
                enrollment_embedding,
                extracted_embedding,
                presence_cache,
                speaker_similarity,
            ),
            device,
            args.warmup,
            args.repeats,
        )
        stage4_stats = benchmark(
            lambda: pipeline.intent(intent_wave), device, args.warmup, args.repeats
        )

        def steady_state_wake_step() -> None:
            state = PipelineState(
                reference_speaker_embedding=enrollment_embedding,
                presence_enrollment_embedding=presence_embedding,
                target_kws_enrollment_embedding=enrollment_embedding,
            )
            pipeline.forward(mixture, enrollment, state, multichannel=multichannel)

        full_step_stats = benchmark(
            steady_state_wake_step, device, args.warmup, args.repeats
        )

    stage_means = [
        stage1_stats["mean_ms"],
        stage2_stats["mean_ms"],
        stage3_stats["mean_ms"],
        stage4_stats["mean_ms"],
    ]
    serial_stage_sum_ms = float(sum(stage_means))
    result = {
        "benchmark": "stagewise_latency",
        "device": str(device),
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "config": str(config_path),
        "mixture": str(mixture_path),
        "enrollment": str(enrollment_path),
        "sample_rate": sample_rate,
        "mixture_channels": int(num_mics),
        "measured_window_seconds": float(window_samples / float(sample_rate)),
        "source_mixture_sample_rate_hz": mixture_source_rate,
        "source_enrollment_sample_rate_hz": enrollment_source_rate,
        "warmup": int(args.warmup),
        "repeats": int(args.repeats),
        "stages": {
            "stage1_extraction": stage1_stats,
            "stage2_presence": stage2_stats,
            "stage3_wake_word": stage3_stats,
            "stage4_intent": stage4_stats,
        },
        "one_time_enrollment_embedding_ms": enrollment_embedding_ms,
        "four_stage_serial_sum_mean_ms": serial_stage_sum_ms,
        "four_stage_serial_sum_rtf": serial_stage_sum_ms / (1000.0 * context_seconds),
        "steady_state_pipeline_wake_window": full_step_stats,
        "notes": [
            "four_stage_serial_sum_mean_ms is the sum of isolated Stage 1-4 mean times; it is not a measured single call.",
            "steady_state_pipeline_wake_window is the actual configured InCarPipeline.forward wall time for one audio window; normal gates may skip later stages.",
            "Stage 4 is timed directly because the wake-word window normally does not run intent inference.",
            "WAV reading, sample-rate conversion, checkpoint loading, and enrollment embedding setup are reported separately or excluded from steady-state timing.",
        ],
    }

    output_path = repo_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("Stage-wise latency (milliseconds; {} warm-ups, {} repeats)".format(args.warmup, args.repeats))
    print("{:<30s} {:>10s} {:>10s} {:>10s}".format("Stage", "mean", "p50", "p95"))
    for name, stats in (
        ("Stage 1 - extraction", stage1_stats),
        ("Stage 2 - presence", stage2_stats),
        ("Stage 3 - wake word", stage3_stats),
        ("Stage 4 - intent", stage4_stats),
    ):
        print("{:<30s} {:>10.2f} {:>10.2f} {:>10.2f}".format(
            name, stats["mean_ms"], stats["p50_ms"], stats["p95_ms"]
        ))
    print("Four-stage serial mean sum: {:.2f} ms ({:.3f}x real time)".format(
        serial_stage_sum_ms, result["four_stage_serial_sum_rtf"]
    ))
    if enrollment_embedding_ms is not None:
        print("One-time enrollment embedding: {:.2f} ms".format(enrollment_embedding_ms))
    print("Actual configured wake-window forward: {:.2f} ms mean, {:.2f} ms p95".format(
        full_step_stats["mean_ms"], full_step_stats["p95_ms"]
    ))
    print("JSON result: {}".format(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
