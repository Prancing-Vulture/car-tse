#!/usr/bin/env python
"""Run the continuous cabin proxy across multiple wake-word speakers."""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.segment import segment_waveform
from car_tse.io.audio import read_mono
from car_tse.io.manifest import read_jsonl
from car_tse.models.kws_model import KWSModel
from car_tse.models.presence_head import PresenceHead
from car_tse.models.stage1_system import Stage1System
from car_tse.train.utils import get_device, load_checkpoint, write_metrics
from eval_synthetic_continuous_speech import choose_noise_files, evaluate_scene, make_scene, speaker_key


def resolve_path(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def choose_speaker_groups(root: Path, speaker_count: int, clips_per_speaker: int) -> list[tuple[str, list[Path]]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(root.glob("*.wav")):
        grouped[speaker_key(path)].append(path)
    eligible = [(speaker, paths) for speaker, paths in sorted(grouped.items()) if len(paths) >= clips_per_speaker]
    if len(eligible) < speaker_count:
        raise RuntimeError(f"Only {len(eligible)} speakers have {clips_per_speaker}+ clips")
    return [(speaker, paths[:clips_per_speaker]) for speaker, paths in eligible[:speaker_count]]


def build_enrollment(paths: list[Path], sample_rate: int, samples: int, clip_count: int) -> np.ndarray:
    count = max(1, min(int(clip_count), len(paths)))
    per_clip = max(1, samples // count)
    chunks = [segment_waveform(read_mono(path, sample_rate)[0], per_clip, training=False) for path in paths[:count]]
    return segment_waveform(np.concatenate(chunks), samples, training=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.yaml")
    parser.add_argument("--stage1-config", type=Path, default=REPO_ROOT / "configs/stage1_spex_transfer_mc_cavemove.yaml")
    parser.add_argument("--stage1-checkpoint", type=Path, default=REPO_ROOT / "checkpoints/stage1_spex_transfer_mc_cavemove_cabin/best.pt")
    parser.add_argument("--presence-checkpoint", type=Path, default=REPO_ROOT / "checkpoints/stage2_presence_cavemove_cabin/best.pt")
    parser.add_argument("--kws-config", type=Path, default=REPO_ROOT / "configs/stage3_kws_tse_aishell.yaml")
    parser.add_argument("--kws-checkpoint", type=Path, default=REPO_ROOT / "checkpoints/stage3_kws_tse_aishell_ft1/best.pt")
    parser.add_argument("--positive-root", type=Path, default=REPO_ROOT / "data/raw/speech_commands_v2/extracted/marvin")
    parser.add_argument("--negative-manifest", type=Path, default=REPO_ROOT / "manifests/test/stage3_kws_tse_hardneg_test.jsonl")
    parser.add_argument("--noise-root", type=Path, default=REPO_ROOT / "data/raw/cavemove/extracted/cavemove_dataset")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "logs/synthetic_multispeaker_proxy.json")
    parser.add_argument("--speaker-count", type=int, default=8)
    parser.add_argument("--clips-per-speaker", type=int, default=4)
    parser.add_argument("--enrollment-clips", type=int, default=1)
    parser.add_argument("--negative-count", type=int, default=20)
    parser.add_argument("--slot-seconds", type=float, default=4.0)
    parser.add_argument("--speech-seconds", type=float, default=1.5)
    parser.add_argument("--speech-offset-seconds", type=float, default=1.0)
    parser.add_argument("--snr-db", nargs="+", type=float, default=[-5.0, 0.0, 10.0, 20.0])
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, nargs="+", default=[0, 2, 4, 6])
    parser.add_argument("--num-mics", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--presence-threshold", type=float, default=0.75)
    parser.add_argument("--kws-threshold", type=float, default=0.81)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--required-hits", type=int, default=2)
    parser.add_argument("--refractory-calls", type=int, default=2)
    parser.add_argument("--kws-input-seconds", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()

    stage1_cfg = load_stage_config(args.stage1_config, args.paths)
    kws_cfg = load_stage_config(args.kws_config, args.paths)
    device = get_device()
    sample_rate = int(args.sample_rate)
    stage1_seconds = float(stage1_cfg.get("segment_seconds", 3.0))
    pretrained = resolve_path(stage1_cfg["pretrained_checkpoint"])
    stage1 = Stage1System(
        spex_cfg=stage1_cfg["spex"], sample_rate=sample_rate, use_cabin_frontend=True,
        num_mics=int(args.num_mics), frontend_cfg=stage1_cfg.get("frontend"),
        pretrained_dir=str(pretrained), backend=stage1_cfg.get("spex_backend", "transfer_spexplus_16k"),
    ).to(device)
    load_checkpoint(resolve_path(args.stage1_checkpoint), stage1)
    stage1.eval()
    presence = PresenceHead().to(device)
    load_checkpoint(resolve_path(args.presence_checkpoint), presence)
    presence.eval()
    kws = KWSModel(
        sample_rate=int(kws_cfg["sample_rate"]), n_mels=int(kws_cfg["model"]["n_mels"]),
        channels=list(kws_cfg["model"]["channels"]), num_classes=int(kws_cfg["model"]["num_classes"]),
    ).to(device)
    load_checkpoint(resolve_path(args.kws_checkpoint), kws)
    kws.eval()

    groups = choose_speaker_groups(resolve_path(args.positive_root), int(args.speaker_count), int(args.clips_per_speaker))
    all_negative_rows = [row for row in read_jsonl(resolve_path(args.negative_manifest)) if int(row.get("label", 0)) == 0]
    if len(all_negative_rows) < int(args.negative_count):
        raise RuntimeError("Not enough hard-negative rows")
    rng = random.Random(int(args.seed))
    rng.shuffle(all_negative_rows)
    negative_files = [resolve_path(row["wav"]) for row in all_negative_rows[: int(args.negative_count)]]
    noise_files = choose_noise_files(resolve_path(args.noise_root), min_duration=30.0)
    scene_rows: list[dict[str, Any]] = []
    for speaker_index, (speaker, positive_files) in enumerate(groups):
        enrollment = build_enrollment(
            positive_files,
            sample_rate,
            int(round(stage1_seconds * sample_rate)) // 2,
            int(args.enrollment_clips),
        )
        for snr_index, snr_db in enumerate(args.snr_db):
            noise_path = noise_files[(speaker_index + snr_index) % len(noise_files)]
            mixture, multichannel, expected_centers, duration = make_scene(
                positive_files, negative_files, noise_path, sample_rate, int(args.num_mics), list(args.channels),
                float(args.slot_seconds), float(args.speech_seconds), float(args.speech_offset_seconds), float(snr_db),
            )
            metrics = evaluate_scene(
                stage1, presence, kws, device, mixture, multichannel, enrollment, expected_centers, duration,
                sample_rate, stage1_seconds, 1.0, int(args.batch_size), float(args.presence_threshold),
                float(args.kws_threshold), int(args.smooth_window), int(args.required_hits),
                int(args.refractory_calls), float(args.kws_input_seconds), event_tolerance_seconds=max(2.0, stage1_seconds / 2.0),
            )
            metrics.update({"speaker": speaker, "snr_db": float(snr_db), "noise_wav": str(noise_path)})
            scene_rows.append(metrics)
            print(
                f"speaker={speaker} SNR={snr_db:+.1f}: presence={metrics['presence_pass_rate']:.3f} "
                f"raw={metrics['raw_wake_recall']:.3f} smooth={metrics['smoothed_wake_recall']:.3f} "
                f"FA/h={metrics['smoothed_false_accept_rate_per_hour']:.1f}", flush=True,
            )

    aggregate: dict[str, dict[str, float | int]] = {}
    for snr_db in args.snr_db:
        rows = [row for row in scene_rows if float(row["snr_db"]) == float(snr_db)]
        total_duration = sum(float(row["duration_seconds"]) for row in rows)
        total_wakes = sum(int(row["expected_wake_events"]) for row in rows)
        aggregate[str(float(snr_db))] = {
            "scenes": len(rows),
            "expected_wakes": total_wakes,
            "presence_pass_rate_mean": float(np.mean([row["presence_pass_rate"] for row in rows])),
            "raw_wake_recall": sum(int(row["raw_matched_wakes"]) for row in rows) / max(total_wakes, 1),
            "smoothed_wake_recall": sum(int(row["smoothed_matched_wakes"]) for row in rows) / max(total_wakes, 1),
            "smoothed_false_accept_events": sum(int(row["smoothed_false_accept_events"]) for row in rows),
            "smoothed_false_accept_rate_per_hour": sum(int(row["smoothed_false_accept_events"]) for row in rows) * 3600.0 / max(total_duration, 1e-6),
        }

    result = {
        "benchmark": "multi_speaker_synthetic_continuous_driver_passenger_proxy",
        "device": str(device),
        "stage1_checkpoint": str(resolve_path(args.stage1_checkpoint)),
        "presence_checkpoint": str(resolve_path(args.presence_checkpoint)),
        "kws_checkpoint": str(resolve_path(args.kws_checkpoint)),
        "speakers": [speaker for speaker, _ in groups],
        "speaker_count": len(groups),
        "clips_per_speaker": int(args.clips_per_speaker),
        "enrollment_clips": int(args.enrollment_clips),
        "negative_clips_per_scene": int(args.negative_count),
        "presence_threshold": float(args.presence_threshold),
        "kws_threshold": float(args.kws_threshold),
        "smoothing": {"window": int(args.smooth_window), "required_hits": int(args.required_hits), "refractory_calls": int(args.refractory_calls)},
        "aggregate_by_snr": aggregate,
        "scenes": scene_rows,
        "note": "Synthetic multi-speaker proxy; use for engineering calibration, not a real-cabin claim.",
    }
    output = resolve_path(args.output)
    write_metrics(output, result)
    print(f"Wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
