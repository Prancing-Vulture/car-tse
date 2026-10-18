#!/usr/bin/env python
"""Run the retained target-KWS path on ANIR noise-only microphone groups.

This is a target-absent, noise-only diagnostic. It is intentionally separate
from the AISHELL speech-bearing false-accept benchmark and must not be called a
full real-driver evaluation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import soundfile as sf
import torch

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.segment import segment_waveform
from car_tse.io.audio import read_mono
from car_tse.models.kws_model import KWSModel, TargetSpeakerKWSModel
from car_tse.models.presence_head import FramePresenceHead
from car_tse.models.stage1_system import Stage1System
from car_tse.train.utils import get_device, load_checkpoint, write_metrics
from eval_aishell5_continuous_kws import evaluate_stream


NAME_RE = re.compile(r"^con_(?P<condition>[^_]+_[^_]+)_mic_(?P<microphone>\d+)\.wav$", re.IGNORECASE)


def resolve_path(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def select_groups(
    root: Path,
    microphones: list[int],
    min_duration: float,
    max_streams: int,
    max_seconds: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[int, Path]] = defaultdict(dict)
    for path in sorted(root.rglob("*.wav")):
        match = NAME_RE.match(path.name)
        if match is None:
            continue
        grouped[match.group("condition")][int(match.group("microphone"))] = path

    selected: list[dict[str, Any]] = []
    total_seconds = 0.0
    for condition in sorted(grouped):
        paths = grouped[condition]
        if not all(microphone in paths for microphone in microphones):
            continue
        infos = [sf.info(str(paths[microphone])) for microphone in microphones]
        duration = min(float(info.duration) for info in infos)
        if duration < float(min_duration):
            continue
        if selected and total_seconds + duration > float(max_seconds):
            break
        selected.append(
            {
                "condition": condition,
                "microphones": list(microphones),
                "paths": [str(paths[microphone]) for microphone in microphones],
                "duration_seconds": duration,
                "sample_rates_hz": [int(info.samplerate) for info in infos],
            }
        )
        total_seconds += duration
        if len(selected) >= int(max_streams):
            break
    return selected


def load_group(row: dict[str, Any]) -> np.ndarray:
    arrays = [read_mono(Path(path), 16000)[0] for path in row["paths"]]
    length = min(len(array) for array in arrays)
    return np.stack([array[:length] for array in arrays], axis=1).astype(np.float32)


def load_models(args: argparse.Namespace, device: torch.device) -> tuple[Any, Any, Any, Any]:
    paths_cfg = load_stage_config(args.stage1_config, args.paths)
    stage1_cfg = paths_cfg
    kws_cfg = load_stage_config(args.kws_config, args.paths)
    target_cfg = load_stage_config(args.target_kws_config, args.paths)
    pretrained = resolve_path(stage1_cfg["pretrained_checkpoint"])
    stage1 = Stage1System(
        spex_cfg=stage1_cfg["spex"],
        sample_rate=16000,
        use_cabin_frontend=True,
        num_mics=len(args.microphones),
        frontend_cfg=stage1_cfg.get("frontend"),
        pretrained_dir=str(pretrained),
        backend=stage1_cfg.get("spex_backend", "transfer_spexplus_16k"),
    ).to(device)
    load_checkpoint(resolve_path(args.stage1_checkpoint), stage1)
    stage1.eval()

    presence = FramePresenceHead(sample_rate=16000, n_mels=40, hidden=64, n_fft=512, hop=160).to(device)
    load_checkpoint(resolve_path(args.presence_checkpoint), presence)
    presence.eval()

    kws = KWSModel(
        sample_rate=16000,
        n_mels=int(kws_cfg["model"]["n_mels"]),
        channels=list(kws_cfg["model"]["channels"]),
        num_classes=int(kws_cfg["model"]["num_classes"]),
    ).to(device)
    load_checkpoint(resolve_path(args.kws_checkpoint), kws)
    kws.eval()

    target_model_cfg = target_cfg["model"]
    target_kws = TargetSpeakerKWSModel(
        sample_rate=16000,
        n_mels=int(target_model_cfg["n_mels"]),
        channels=list(target_model_cfg["channels"]),
        num_classes=int(target_model_cfg["num_classes"]),
        speaker_embedding_dim=int(target_model_cfg.get("speaker_embedding_dim", 256)),
        hidden=int(target_model_cfg.get("hidden", 256)),
    ).to(device)
    load_checkpoint(resolve_path(args.target_kws_checkpoint), target_kws)
    target_kws.eval()
    return stage1, presence, kws, target_kws


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.yaml")
    parser.add_argument("--noise-root", type=Path, required=True)
    parser.add_argument("--stage1-config", type=Path, default=REPO_ROOT / "configs/stage1_spex_dual_domain_rms010_ft1.yaml")
    parser.add_argument("--stage1-checkpoint", type=Path, default=REPO_ROOT / "checkpoints/stage1_spex_dual_domain_rms010_ft1/best.pt")
    parser.add_argument("--kws-config", type=Path, default=REPO_ROOT / "configs/stage3_kws_tse_hybrid.yaml")
    parser.add_argument("--kws-checkpoint", type=Path, default=REPO_ROOT / "checkpoints/stage3_kws_gsc_marvin_tse_ft2/best.pt")
    parser.add_argument("--target-kws-config", type=Path, default=REPO_ROOT / "configs/stage3_target_kws_gsc_marvin_hardneg_aishell5.yaml")
    parser.add_argument("--target-kws-checkpoint", type=Path, default=REPO_ROOT / "checkpoints/stage3_target_kws_gsc_marvin_hardneg_aishell5_ft1/best.pt")
    parser.add_argument("--presence-checkpoint", type=Path, default=REPO_ROOT / "checkpoints/stage2_frame_presence_dual_domain_nonwake_ft1/best.pt")
    parser.add_argument("--enrollment-wav", type=Path, default=REPO_ROOT / "data/raw/librispeech/extracted/LibriSpeech/test-clean/1089/134686/1089-134686-0014.flac")
    parser.add_argument("--microphones", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--min-duration", type=float, default=30.0)
    parser.add_argument("--max-streams", type=int, default=12)
    parser.add_argument("--max-seconds", type=float, default=1200.0)
    parser.add_argument("--hop-seconds", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--presence-threshold", type=float, default=0.15)
    parser.add_argument("--speaker-gate-threshold", type=float, default=0.24)
    parser.add_argument("--kws-threshold", type=float, default=0.50)
    parser.add_argument("--smooth-window", type=int, default=2)
    parser.add_argument("--required-hits", type=int, default=2)
    parser.add_argument("--refractory-calls", type=int, default=2)
    parser.add_argument("--kws-input-seconds", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = get_device()
    stage1, presence, kws, target_kws = load_models(args, device)
    enrollment, _ = read_mono(resolve_path(args.enrollment_wav), 16000)
    enrollment = segment_waveform(enrollment, 24000, training=False)
    groups = select_groups(
        resolve_path(args.noise_root),
        list(args.microphones),
        float(args.min_duration),
        int(args.max_streams),
        float(args.max_seconds),
    )
    if not groups:
        raise RuntimeError("No complete ANIR microphone groups matched the requested conditions")

    results: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        audio = load_group(group)
        metrics = evaluate_stream(
            stage1,
            presence,
            kws,
            target_kws,
            None,
            device,
            audio,
            enrollment,
            3.0,
            float(args.hop_seconds),
            int(args.batch_size),
            float(args.presence_threshold),
            float(args.kws_threshold),
            float(args.speaker_gate_threshold),
            int(args.smooth_window),
            int(args.required_hits),
            int(args.refractory_calls),
            float(args.kws_input_seconds),
            0.20,
            None,
        )
        row = {**group, **metrics}
        results.append(row)
        print(
            f"{index}/{len(groups)} condition={group['condition']} duration={metrics['duration_seconds']:.1f}s "
            f"full={metrics['full_path_smoothed_event_rate_per_hour']:.2f}/h "
            f"kws={metrics['kws_only_smoothed_event_rate_per_hour']:.2f}/h",
            flush=True,
        )

    duration = sum(float(row["duration_seconds"]) for row in results)
    output = {
        "benchmark": "ANIR_noise_only_target_absent_target_kws",
        "dataset": "ANIR Corpus V class",
        "device": str(device),
        "microphones": list(args.microphones),
        "stage1_checkpoint": str(resolve_path(args.stage1_checkpoint)),
        "presence_checkpoint": str(resolve_path(args.presence_checkpoint)),
        "kws_checkpoint": str(resolve_path(args.kws_checkpoint)),
        "target_kws_checkpoint": str(resolve_path(args.target_kws_checkpoint)),
        "enrollment_wav": str(resolve_path(args.enrollment_wav)),
        "presence_threshold": float(args.presence_threshold),
        "speaker_gate_threshold": float(args.speaker_gate_threshold),
        "kws_threshold": float(args.kws_threshold),
        "smoothing": {"window": int(args.smooth_window), "required_hits": int(args.required_hits), "refractory_calls": int(args.refractory_calls)},
        "groups": len(results),
        "duration_seconds": duration,
        "duration_hours": duration / 3600.0,
        "full_path_smoothed_events": sum(int(row["full_path_smoothed_events"]) for row in results),
        "full_path_smoothed_event_rate_per_hour": sum(int(row["full_path_smoothed_events"]) for row in results) * 3600.0 / max(duration, 1e-6),
        "kws_only_smoothed_events": sum(int(row["kws_only_smoothed_events"]) for row in results),
        "kws_only_smoothed_event_rate_per_hour": sum(int(row["kws_only_smoothed_events"]) for row in results) * 3600.0 / max(duration, 1e-6),
        "selected_groups": results,
        "limitation": "Noise-only target-absent diagnostic; no speech, wake positives, commands, or real-driver continuity.",
    }
    write_metrics(resolve_path(args.output), output)
    print(json.dumps({key: output[key] for key in ("groups", "duration_hours", "full_path_smoothed_event_rate_per_hour", "kws_only_smoothed_event_rate_per_hour")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
