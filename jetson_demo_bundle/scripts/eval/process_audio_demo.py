#!/usr/bin/env python3
"""Run one four-channel recording through the trained car-tse pipeline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from scipy.signal import resample_poly

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from car_tse.config import load_stage_config
from car_tse.models.stage1_system import Stage1System
from car_tse.pipeline.in_car_pipeline import InCarPipeline, PipelineState
from car_tse.train.utils import get_device, load_checkpoint


def read_audio(path: Path, target_rate: int, require_channels: int | None = None) -> np.ndarray:
    audio, source_rate = sf.read(str(path), always_2d=True, dtype="float32")
    if require_channels is not None and audio.shape[1] != require_channels:
        raise ValueError(
            f"{path} has {audio.shape[1]} channels; this model expects {require_channels}. "
            "Use a four-microphone recording."
        )
    if audio.shape[0] == 0:
        raise ValueError(f"Audio file is empty: {path}")
    if source_rate != target_rate:
        divisor = math.gcd(int(source_rate), int(target_rate))
        audio = resample_poly(
            audio,
            target_rate // divisor,
            source_rate // divisor,
            axis=0,
        ).astype(np.float32)
    return audio


def load_demo_pipeline(args: argparse.Namespace):
    eval_config = REPO_ROOT / "configs" / "eval_end_to_end_cavemove.yaml"
    cfg = load_stage_config(eval_config, args.paths)
    paths = cfg["paths"]

    stage1_config = Path(cfg["stage1_config"])
    if not stage1_config.is_absolute():
        stage1_config = REPO_ROOT / stage1_config
    stage1_cfg = load_stage_config(stage1_config, args.paths)
    stage1_cfg["pretrained_checkpoint"] = str(args.base_checkpoint.resolve())

    stage1 = Stage1System(
        spex_cfg=stage1_cfg["spex"],
        sample_rate=stage1_cfg["sample_rate"],
        use_cabin_frontend=stage1_cfg.get("use_cabin_frontend", True),
        num_mics=int(stage1_cfg.get("num_mics", cfg.get("num_mics", 4))),
        frontend_cfg=stage1_cfg.get("frontend"),
        pretrained_dir=str(args.base_checkpoint.resolve()),
        backend=stage1_cfg.get("spex_backend", "transfer_spexplus_16k"),
    ).to(args.device)
    checkpoint_root = Path(paths["checkpoint_root"])
    load_checkpoint(checkpoint_root / cfg["checkpoints"]["spex"], stage1)

    label_map_path = Path(paths["manifest_root"]) / "meta" / "intent_label_map.json"
    label_map = {}
    if label_map_path.exists():
        raw_map = json.loads(label_map_path.read_text(encoding="utf-8"))
        label_map = {str(value): key for key, value in raw_map.items()}

    pipeline = InCarPipeline.from_checkpoints(
        stage1,
        checkpoint_root / cfg["checkpoints"]["presence"],
        checkpoint_root / cfg["checkpoints"]["kws"],
        checkpoint_root / cfg["checkpoints"]["intent"],
        kws_sample_rate=int(cfg["sample_rate"]),
        kws_num_classes=int(cfg.get("kws_num_classes", 2)),
        intent_sample_rate=int(cfg["sample_rate"]),
        intent_type=str(cfg.get("intent_type", "crnn")),
        intent_num_classes=int(cfg.get("intent_num_classes", 31)),
        presence_threshold=float(cfg["presence_threshold"]),
        kws_threshold=float(cfg["kws_threshold"]),
        wake_class_id=int(cfg.get("wake_class_id", 1)),
        intent_input_seconds=float(cfg.get("intent_input_seconds", 2.0)),
        command_timeout_calls=int(cfg.get("command_timeout_calls", 4)),
        presence_type=str(cfg.get("presence_type", "scalar")),
        label_map=label_map,
        device=args.device,
    )
    return cfg, pipeline


def make_multichannel_input(
    speech_channels: np.ndarray,
    noise_channels: np.ndarray | None,
    channel_indices: list[int],
    num_mics: int,
    snr_db: float,
) -> np.ndarray:
    speech = speech_channels.mean(axis=1).astype(np.float32)
    if noise_channels is None:
        if speech_channels.shape[1] != num_mics:
            raise ValueError(
                f"Mixture has {speech_channels.shape[1]} channels; expected {num_mics}. "
                "For mono speech mixed with cabin noise, provide --cabin-noise."
            )
        return speech_channels.astype(np.float32)

    valid = [index for index in channel_indices if 0 <= index < noise_channels.shape[1]]
    if not valid:
        valid = list(range(min(num_mics, noise_channels.shape[1])))
    noise = noise_channels[:, valid]
    if noise.shape[1] < num_mics:
        repeats = int(np.ceil(num_mics / noise.shape[1]))
        noise = np.tile(noise, (1, repeats))[:, :num_mics]
    else:
        noise = noise[:, :num_mics]

    if noise.shape[0] >= len(speech):
        start = (noise.shape[0] - len(speech)) // 2
        noise = noise[start : start + len(speech)]
    else:
        repeats = int(np.ceil(len(speech) / max(1, noise.shape[0])))
        noise = np.tile(noise, (repeats, 1))[: len(speech)]

    speech_rms = float(np.sqrt(np.mean(speech * speech) + 1e-8))
    noise_rms = float(np.sqrt(np.mean(noise * noise) + 1e-8))
    if speech_rms <= 1e-6 or noise_rms <= 1e-6:
        raise ValueError("Speech mixture or cabin noise is silent; cannot construct demo input")
    scale = speech_rms * (10.0 ** (-snr_db / 20.0)) / noise_rms
    return (speech[:, None] + noise * scale).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixture", required=True, type=Path, help="Speech mixture WAV")
    parser.add_argument(
        "--cabin-noise",
        type=Path,
        default=None,
        help="Optional multichannel cabin noise WAV; combines with a mono speech mixture",
    )
    parser.add_argument("--cabin-snr-db", type=float, default=0.0)
    parser.add_argument("--enrollment", required=True, type=Path, help="Target speaker enrollment WAV")
    parser.add_argument(
        "--base-checkpoint",
        required=True,
        type=Path,
        help="SpEx+ base model, e.g. checkpoints/car_tse_spexplus_from_stage1.pt",
    )
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "logs" / "jetson_demo")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.device == "auto":
        args.device = get_device()
    else:
        args.device = torch.device(args.device)
    if args.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    if not args.base_checkpoint.is_file():
        raise FileNotFoundError(f"Base SpEx+ checkpoint not found: {args.base_checkpoint}")

    cfg, pipeline = load_demo_pipeline(args)
    sample_rate = int(cfg["sample_rate"])
    channels = int(cfg.get("num_mics", 4))
    speech_channels = read_audio(args.mixture, sample_rate)
    noise_channels = (
        read_audio(args.cabin_noise, sample_rate) if args.cabin_noise is not None else None
    )
    mixture_channels = make_multichannel_input(
        speech_channels,
        noise_channels,
        [int(x) for x in cfg.get("cabin_noise_channels", [0, 2, 4, 6])],
        channels,
        args.cabin_snr_db,
    )
    enrollment_channels = read_audio(args.enrollment, sample_rate)
    enrollment = enrollment_channels.mean(axis=1)

    segment_samples = max(1, int(round(float(cfg.get("segment_seconds", 3.0)) * sample_rate)))
    original_samples = mixture_channels.shape[0]
    state = PipelineState()
    extracted_parts = []
    last_output = None
    with torch.inference_mode():
        for start in range(0, original_samples, segment_samples):
            stop = min(start + segment_samples, original_samples)
            actual_count = stop - start
            chunk = mixture_channels[start:stop]
            if actual_count < segment_samples:
                chunk = np.pad(chunk, ((0, segment_samples - actual_count), (0, 0)))
            multichannel = torch.from_numpy(chunk.T.copy())
            mono = torch.from_numpy(chunk.mean(axis=1).copy()).unsqueeze(0)
            enroll = torch.from_numpy(enrollment.copy()).unsqueeze(0)
            last_output, state = pipeline.forward(
                mono,
                enroll,
                state,
                multichannel=multichannel,
                audio_start_sample=start,
            )
            if last_output.s1 is not None:
                extracted_parts.append(last_output.s1[0, :actual_count].detach().cpu().numpy())

    if not extracted_parts:
        raise RuntimeError("Stage 1 produced no extracted audio")
    extracted = np.concatenate(extracted_parts).astype(np.float32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extracted_path = args.output_dir / "extracted_speech.wav"
    sf.write(str(extracted_path), extracted, sample_rate, subtype="PCM_16")

    result = {
        "device": str(args.device),
        "sample_rate": sample_rate,
        "input_channels": channels,
        "input_seconds": original_samples / float(sample_rate),
        "extracted_wav": str(extracted_path.resolve()),
        "accepted": bool(last_output.accepted),
        "wake_detected": bool(last_output.wake_detected),
        "presence_probability": last_output.presence_probability,
        "wake_probability": last_output.wake_probability,
        "intent_ready": bool(last_output.intent_ready),
        "predicted_intent_id": (
            int(last_output.intent_logits.argmax(dim=-1).item())
            if last_output.intent_logits is not None
            else None
        ),
        "vehicle_action": last_output.action.action if last_output.action is not None else None,
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
