#!/usr/bin/env python
"""Measure KWS false accepts on real continuous AISHELL-5 cabin recordings.

AISHELL-5 provides long synchronized far-field channels and TextGrid speech
annotations.  The selected sessions are target-absent for the current system:
the recording speaker is not the fixed LibriSpeech enrollment voice.  This is
therefore a real-cabin false-accept benchmark, not a wake-recall benchmark.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import soundfile as sf
import torch

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.segment import segment_waveform
from car_tse.io.audio import read_mono
from car_tse.models.kws_model import KWSModel
from car_tse.models.presence_head import FramePresenceHead, PresenceHead
from car_tse.models.stage1_system import Stage1System
from car_tse.pipeline.in_car_pipeline import PipelineState, normalize_extracted_waveform, update_kws_state
from car_tse.train.utils import get_device, load_checkpoint, write_metrics
from eval_continuous_kws import make_windows


def resolve_path(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def parse_textgrid_speech_seconds(path: Path, limit_seconds: float) -> float:
    if not path.exists():
        return 0.0
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    intervals: list[tuple[float, float]] = []
    for index, line in enumerate(lines):
        if not line.strip().startswith("intervals ["):
            continue
        values = lines[index + 1 : index + 4]
        if len(values) < 3:
            continue
        try:
            start = float(re.search(r"=\s*([0-9.eE+-]+)", values[0]).group(1))
            end = float(re.search(r"=\s*([0-9.eE+-]+)", values[1]).group(1))
        except (AttributeError, ValueError):
            continue
        text_match = re.search(r'text\s*=\s*"(.*)"', values[2])
        if text_match and text_match.group(1).strip():
            clipped_start = min(max(start, 0.0), limit_seconds)
            clipped_end = min(max(end, 0.0), limit_seconds)
            if clipped_end > clipped_start:
                intervals.append((clipped_start, clipped_end))
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def load_session(root: Path, session_id: str, max_seconds: float, channels: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    session_root = root / session_id
    paths = [session_root / name for name in channels]
    if not all(path.exists() for path in paths):
        raise FileNotFoundError(f"Missing AISHELL-5 channels for {session_id}: {paths}")
    arrays = []
    sample_rate = None
    for path in paths:
        wav, sr = read_mono(path, 16000)
        arrays.append(np.asarray(wav, dtype=np.float32))
        sample_rate = sr
    length = min(len(wav) for wav in arrays)
    limit_samples = min(length, int(round(max_seconds * 16000)))
    multichannel = np.stack([wav[:limit_samples] for wav in arrays], axis=1)
    actual_seconds = limit_samples / 16000.0
    textgrid = session_root / f"{Path(channels[0]).stem}.TextGrid"
    speech_seconds = parse_textgrid_speech_seconds(textgrid, actual_seconds)
    return multichannel, {
        "session_id": session_id,
        "channels": [str(path) for path in paths],
        "textgrid": str(textgrid),
        "duration_seconds": actual_seconds,
        "speech_seconds": speech_seconds,
        "speech_occupancy": speech_seconds / max(actual_seconds, 1e-6),
        "sample_rate": int(sample_rate or 16000),
    }


def event_indices(flags: np.ndarray, merge_gap_steps: int) -> list[int]:
    result: list[int] = []
    last = -10**9
    for index, flag in enumerate(flags.tolist()):
        if flag and index - last > merge_gap_steps:
            result.append(index)
            last = index
    return result


def evaluate_stream(
    stage1: Stage1System,
    presence: torch.nn.Module,
    kws: KWSModel,
    device: torch.device,
    multichannel: np.ndarray,
    enrollment: np.ndarray,
    stage1_seconds: float,
    hop_seconds: float,
    batch_size: int,
    presence_threshold: float,
    kws_threshold: float,
    smooth_window: int,
    required_hits: int,
    refractory_calls: int,
    kws_input_seconds: float,
    presence_top_fraction: float = 0.20,
) -> dict[str, Any]:
    sample_rate = 16000
    window_samples = int(round(stage1_seconds * sample_rate))
    hop_samples = int(round(hop_seconds * sample_rate))
    windows, count = make_windows(multichannel, window_samples, hop_samples)
    mixture_windows, _ = make_windows(multichannel.mean(axis=1)[:, None], window_samples, hop_samples)
    presence_scores: list[float] = []
    kws_scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, count, max(1, int(batch_size))):
            batch_mc = torch.from_numpy(windows[start : start + batch_size]).to(device)
            batch_mix = torch.from_numpy(mixture_windows[start : start + batch_size, 0, :]).to(device)
            batch_enroll = torch.from_numpy(np.repeat(enrollment[None, :], batch_mc.shape[0], axis=0)).to(device)
            extracted, _, _, _ = stage1(batch_mix, batch_enroll, multichannel=batch_mc)
            pres_logits = presence(extracted)
            if getattr(presence, "is_framewise", False):
                pres = FramePresenceHead.aggregate_probability(
                    torch.sigmoid(pres_logits),
                    top_fraction=presence_top_fraction,
                )
            else:
                pres = torch.sigmoid(pres_logits)
            conditioned = normalize_extracted_waveform(extracted)
            if kws_input_seconds > 0:
                kws_samples = max(1, int(round(sample_rate * kws_input_seconds)))
                if conditioned.shape[-1] >= kws_samples:
                    conditioned = conditioned[..., :kws_samples]
                else:
                    conditioned = torch.nn.functional.pad(conditioned, (0, kws_samples - conditioned.shape[-1]))
            scores = torch.softmax(kws(conditioned), dim=-1)[:, 1]
            presence_scores.extend(pres.cpu().tolist())
            kws_scores.extend(scores.cpu().tolist())

    presence_array = np.asarray(presence_scores, dtype=np.float32)
    score_array = np.asarray(kws_scores, dtype=np.float32)
    presence_flags = presence_array >= float(presence_threshold)
    raw_flags = presence_flags & (score_array >= float(kws_threshold))
    raw_kws_flags = score_array >= float(kws_threshold)

    state = PipelineState()
    smooth_flags = np.zeros(count, dtype=bool)
    smooth_kws_only_flags = np.zeros(count, dtype=bool)
    for index, score in enumerate(score_array.tolist()):
        if not presence_flags[index]:
            state.kws_scores.clear()
            state.kws_refractory_left = 0
        else:
            smooth_flags[index] = update_kws_state(
                state, float(score), float(kws_threshold), int(smooth_window), int(required_hits), int(refractory_calls)
            )
    kws_only_state = PipelineState()
    for index, score in enumerate(score_array.tolist()):
        smooth_kws_only_flags[index] = update_kws_state(
            kws_only_state, float(score), float(kws_threshold), int(smooth_window), int(required_hits), int(refractory_calls)
        )

    duration = len(multichannel) / sample_rate
    raw_events = event_indices(raw_flags, max(1, int(np.floor(2.0 / hop_seconds))))
    raw_kws_only_events = event_indices(raw_kws_flags, max(1, int(np.floor(2.0 / hop_seconds))))
    return {
        "duration_seconds": duration,
        "windows": count,
        "presence_pass_rate": float(presence_flags.mean()),
        "max_presence_score": float(presence_array.max(initial=0.0)),
        "max_kws_score": float(score_array.max(initial=0.0)),
        "full_path_raw_positive_windows": int(raw_flags.sum()),
        "full_path_raw_event_rate_per_hour": len(raw_events) * 3600.0 / max(duration, 1e-6),
        "full_path_smoothed_events": int(smooth_flags.sum()),
        "full_path_smoothed_event_rate_per_hour": int(smooth_flags.sum()) * 3600.0 / max(duration, 1e-6),
        "kws_only_raw_positive_windows": int(raw_kws_flags.sum()),
        "kws_only_raw_window_rate_per_hour": int(raw_kws_flags.sum()) * 3600.0 / max(duration, 1e-6),
        "kws_only_raw_event_rate_per_hour": len(raw_kws_only_events) * 3600.0 / max(duration, 1e-6),
        "kws_only_smoothed_events": int(smooth_kws_only_flags.sum()),
        "kws_only_smoothed_event_rate_per_hour": int(smooth_kws_only_flags.sum()) * 3600.0 / max(duration, 1e-6),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--split", default="Eval2", choices=("train", "Dev", "Eval1", "Eval2"))
    parser.add_argument("--session-ids", nargs="+", default=["001", "002"])
    parser.add_argument("--max-seconds", type=float, default=1200.0)
    parser.add_argument("--stage1-config", type=Path, default=REPO_ROOT / "configs" / "stage1_spex_transfer_mc_cavemove.yaml")
    parser.add_argument("--stage1-checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "stage1_spex_transfer_mc_cavemove_cabin" / "best.pt")
    parser.add_argument("--kws-config", type=Path, default=REPO_ROOT / "configs" / "stage3_kws_tse_hybrid.yaml")
    parser.add_argument("--kws-checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "stage3_kws_tse_hybrid_ft1" / "best.pt")
    parser.add_argument("--presence-checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "stage2_presence_cavemove_cabin" / "best.pt")
    parser.add_argument("--presence-type", choices=("scalar", "frame"), default="scalar")
    parser.add_argument("--presence-top-fraction", type=float, default=0.20)
    parser.add_argument("--enrollment-wav", type=Path, default=REPO_ROOT / "data" / "raw" / "librispeech" / "extracted" / "LibriSpeech" / "test-clean" / "1089" / "134686" / "1089-134686-0014.flac")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "logs" / "aishell5_continuous_kws_eval2.json")
    parser.add_argument("--presence-threshold", type=float, default=0.90)
    parser.add_argument("--kws-threshold", type=float, default=0.29)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--required-hits", type=int, default=2)
    parser.add_argument("--refractory-calls", type=int, default=2)
    parser.add_argument("--kws-input-seconds", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--channels", nargs="+", default=["DX01C01.wav", "DX02C01.wav", "DX03C01.wav", "DX04C01.wav"])
    args = parser.parse_args()

    stage1_cfg = load_stage_config(args.stage1_config, args.paths)
    kws_cfg = load_stage_config(args.kws_config, args.paths)
    device = get_device()
    stage1_checkpoint = resolve_path(args.stage1_checkpoint)
    kws_checkpoint = resolve_path(args.kws_checkpoint)
    presence_checkpoint = resolve_path(args.presence_checkpoint)
    pretrained = resolve_path(stage1_cfg["pretrained_checkpoint"])
    stage1 = Stage1System(
        spex_cfg=stage1_cfg["spex"], sample_rate=16000, use_cabin_frontend=True,
        num_mics=len(args.channels), frontend_cfg=stage1_cfg.get("frontend"),
        pretrained_dir=str(pretrained), backend=stage1_cfg.get("spex_backend", "transfer_spexplus_16k"),
    ).to(device)
    load_checkpoint(stage1_checkpoint, stage1)
    stage1.eval()
    if args.presence_type == "frame":
        presence = FramePresenceHead(sample_rate=16000, n_mels=40, hidden=64, n_fft=512, hop=160).to(device)
    else:
        presence = PresenceHead().to(device)
    load_checkpoint(presence_checkpoint, presence)
    presence.eval()
    kws = KWSModel(
        sample_rate=16000, n_mels=int(kws_cfg["model"]["n_mels"]),
        channels=list(kws_cfg["model"]["channels"]), num_classes=int(kws_cfg["model"]["num_classes"]),
    ).to(device)
    load_checkpoint(kws_checkpoint, kws)
    kws.eval()

    enrollment, _ = read_mono(resolve_path(args.enrollment_wav), 16000)
    enrollment = segment_waveform(enrollment, 24000, training=False)
    aishell_root = resolve_path(Path("data/raw/aishell5/extracted") / args.split)
    remaining = float(args.max_seconds)
    session_results: list[dict[str, Any]] = []
    total_duration = 0.0
    for session_id in args.session_ids:
        if remaining <= 0:
            break
        multichannel, metadata = load_session(aishell_root, session_id, remaining, list(args.channels))
        metrics = evaluate_stream(
            stage1, presence, kws, device, multichannel, enrollment,
            float(stage1_cfg.get("segment_seconds", 3.0)), 1.0, int(args.batch_size),
            float(args.presence_threshold), float(args.kws_threshold), int(args.smooth_window),
            int(args.required_hits), int(args.refractory_calls), float(args.kws_input_seconds),
            float(args.presence_top_fraction),
        )
        row = {**metadata, **metrics}
        session_results.append(row)
        remaining -= metadata["duration_seconds"]
        total_duration += metadata["duration_seconds"]
        print(
            f"{args.split}/{session_id}: {metadata['duration_seconds']:.1f}s, "
            f"speech={metadata['speech_occupancy']:.3f}, "
            f"full_smooth={metrics['full_path_smoothed_event_rate_per_hour']:.1f}/h, "
            f"kws_only_raw={metrics['kws_only_raw_window_rate_per_hour']:.1f}/h, "
            f"kws_only_smooth={metrics['kws_only_smoothed_event_rate_per_hour']:.1f}/h",
            flush=True,
        )

    if not session_results:
        raise RuntimeError("No AISHELL-5 sessions were evaluated")
    result = {
        "benchmark": "real_continuous_aishell5_target_absent_kws",
        "dataset": "AISHELL-5",
        "split": args.split,
        "device": str(device),
        "stage1_checkpoint": str(stage1_checkpoint),
        "presence_checkpoint": str(presence_checkpoint),
        "presence_type": args.presence_type,
        "presence_top_fraction": float(args.presence_top_fraction),
        "kws_checkpoint": str(kws_checkpoint),
        "enrollment_wav": str(resolve_path(args.enrollment_wav)),
        "presence_threshold": float(args.presence_threshold),
        "kws_threshold": float(args.kws_threshold),
        "kws_input_seconds": float(args.kws_input_seconds),
        "smoothing": {"window": int(args.smooth_window), "required_hits": int(args.required_hits), "refractory_calls": int(args.refractory_calls)},
        "sessions": len(session_results),
        "duration_seconds": total_duration,
        "duration_hours": total_duration / 3600.0,
        "speech_seconds": sum(row["speech_seconds"] for row in session_results),
        "speech_occupancy": sum(row["speech_seconds"] for row in session_results) / max(total_duration, 1e-6),
        "full_path_smoothed_events": sum(row["full_path_smoothed_events"] for row in session_results),
        "full_path_smoothed_event_rate_per_hour": sum(row["full_path_smoothed_events"] for row in session_results) * 3600.0 / max(total_duration, 1e-6),
        "kws_only_raw_positive_windows": sum(row["kws_only_raw_positive_windows"] for row in session_results),
        "kws_only_raw_window_rate_per_hour": sum(row["kws_only_raw_positive_windows"] for row in session_results) * 3600.0 / max(total_duration, 1e-6),
        "kws_only_smoothed_events": sum(row["kws_only_smoothed_events"] for row in session_results),
        "kws_only_smoothed_event_rate_per_hour": sum(row["kws_only_smoothed_events"] for row in session_results) * 3600.0 / max(total_duration, 1e-6),
        "session_results": session_results,
        "note": "All recordings are target-absent relative to the fixed enrollment. This is a real continuous in-car speech/noise false-accept benchmark; it contains no wake-word positives.",
    }
    output = resolve_path(args.output)
    write_metrics(output, result)
    print({"output": str(output), "duration_hours": result["duration_hours"], "full_path_smoothed_event_rate_per_hour": result["full_path_smoothed_event_rate_per_hour"], "kws_only_smoothed_event_rate_per_hour": result["kws_only_smoothed_event_rate_per_hour"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
