#!/usr/bin/env python
"""Benchmark KWS false accepts on continuous CAVEMOVE noise streams.

The ordinary KWS test set is made of independent one-second examples.  That is
useful for classifier metrics, but it cannot answer the runtime question:
"How often does the assistant wake during a real uninterrupted cabin-noise
stream?"  This evaluator keeps the production Stage 1 -> presence -> KWS path,
slides a three-second Stage 1 context through long CAVEMOVE recordings, and
reports both window-level and event-level false accepts.

Only noise/ventilation recordings are used here, so every wake is a false
accept.  The selected stream list and per-stream results are written to the
output JSON for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import soundfile as sf
import torch

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.segment import segment_waveform
from car_tse.io.audio import read_mono, read_multichannel
from car_tse.models.kws_model import KWSModel
from car_tse.models.presence_head import PresenceHead
from car_tse.models.stage1_system import Stage1System
from car_tse.pipeline.in_car_pipeline import PipelineState, normalize_extracted_waveform, update_kws_state
from car_tse.pipeline.windowing import full_context_window_starts
from car_tse.train.utils import get_device, load_checkpoint, write_metrics


def resolve_path(path: Path | str, root: Path = REPO_ROOT) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def stream_metadata(path: Path) -> tuple[str, str, str]:
    parts = set(path.parts)
    vehicle = next(
        (name for name in ("Volkswagen_Golf", "Smart_forfour", "AlfaRomeo_146") if name in parts),
        "unknown",
    )
    signal_type = "ventilation" if "ventilation" in parts else "noise"
    setup = "array" if "array" in parts else "distributed"
    return vehicle, signal_type, setup


def select_streams(
    root: Path,
    min_duration: float,
    max_streams: int,
    max_seconds: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(root.rglob("*.wav")):
        if "noise" not in path.parts and "ventilation" not in path.parts:
            continue
        try:
            info = sf.info(str(path))
        except (OSError, RuntimeError):
            continue
        if info.duration < min_duration:
            continue
        vehicle, signal_type, setup = stream_metadata(path)
        grouped[(vehicle, signal_type)].append(
            {
                "path": str(path),
                "duration_seconds": float(info.duration),
                "vehicle": vehicle,
                "noise_type": signal_type,
                "microphone_setup": setup,
                "sample_rate": int(info.samplerate),
                "channels": int(info.channels),
            }
        )
    for rows in grouped.values():
        rows.sort(key=lambda row: (-row["duration_seconds"], row["path"]))

    # Round-robin over vehicle/noise-type groups so the benchmark is not just
    # the longest road-noise recordings from one vehicle.
    selected: list[dict[str, Any]] = []
    group_keys = sorted(grouped)
    positions = {key: 0 for key in group_keys}
    total_seconds = 0.0
    while len(selected) < max_streams:
        added = False
        for key in group_keys:
            pos = positions[key]
            rows = grouped[key]
            if pos >= len(rows):
                continue
            candidate = rows[pos]
            positions[key] += 1
            if selected and total_seconds + candidate["duration_seconds"] > max_seconds:
                continue
            selected.append(candidate)
            total_seconds += candidate["duration_seconds"]
            added = True
            if len(selected) >= max_streams:
                break
        if not added:
            break
    return selected


def prepare_channels(data: np.ndarray, channels: list[int], num_mics: int) -> np.ndarray:
    if data.ndim == 1:
        data = data[:, None]
    valid = [idx for idx in channels if 0 <= idx < data.shape[1]]
    if not valid:
        valid = list(range(min(num_mics, data.shape[1])))
    selected = data[:, valid]
    if selected.shape[1] < num_mics:
        repeats = int(np.ceil(num_mics / max(1, selected.shape[1])))
        selected = np.tile(selected, (1, repeats))[:, :num_mics]
    elif selected.shape[1] > num_mics:
        selected = selected[:, :num_mics]
    return np.asarray(selected, dtype=np.float32)


def make_windows(data: np.ndarray, window_samples: int, hop_samples: int) -> tuple[np.ndarray, int]:
    """Return [windows, channels, time] and the number of valid starts."""
    starts = list(full_context_window_starts(len(data), window_samples, hop_samples))
    windows = np.zeros((len(starts), data.shape[1], window_samples), dtype=np.float32)
    for index, start in enumerate(starts):
        end = min(start + window_samples, len(data))
        if end > start:
            windows[index, :, : end - start] = data[start:end].T
    return windows, len(starts)


def count_events(flags: np.ndarray, hop_seconds: float, merge_gap_seconds: float) -> int:
    event_count = 0
    last_event_index = -10**9
    merge_gap_steps = max(0, int(np.floor(merge_gap_seconds / max(hop_seconds, 1e-6))))
    for index, flag in enumerate(flags.tolist()):
        if not flag:
            continue
        if index - last_event_index > merge_gap_steps:
            event_count += 1
            last_event_index = index
    return event_count


def rate_per_hour(count: int, seconds: float) -> float:
    return float(count) * 3600.0 / max(float(seconds), 1e-6)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--stage1-config", type=Path, default=REPO_ROOT / "configs" / "stage1_spex_transfer_mc_cavemove.yaml")
    parser.add_argument("--stage1-checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "stage1_spex_transfer_mc_cavemove_cabin" / "best.pt")
    parser.add_argument("--kws-config", type=Path, default=REPO_ROOT / "configs" / "stage3_kws_tse_hybrid.yaml")
    parser.add_argument("--kws-checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "stage3_kws_tse_hybrid_ft1" / "best.pt")
    parser.add_argument("--presence-checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "stage2_presence_cavemove_cabin" / "best.pt")
    parser.add_argument("--noise-root", type=Path, default=REPO_ROOT / "data" / "raw" / "cavemove" / "extracted" / "cavemove_dataset")
    parser.add_argument("--enrollment-wav", type=Path, default=REPO_ROOT / "data" / "raw" / "librispeech" / "extracted" / "LibriSpeech" / "test-clean" / "1089" / "134686" / "1089-134686-0014.flac")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "logs" / "continuous_kws_cavemove_tse_presence90.json")
    parser.add_argument("--presence-threshold", type=float, default=0.90)
    parser.add_argument("--kws-threshold", type=float, default=0.29)
    parser.add_argument("--smoothing-window", type=int, default=3)
    parser.add_argument("--required-hits", type=int, default=2)
    parser.add_argument("--refractory-calls", type=int, default=2)
    parser.add_argument("--hop-seconds", type=float, default=1.0)
    parser.add_argument("--event-merge-gap-seconds", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-duration", type=float, default=30.0)
    parser.add_argument("--max-streams", type=int, default=24)
    parser.add_argument("--max-seconds", type=float, default=1200.0)
    parser.add_argument("--channels", type=int, nargs="+", default=[0, 2, 4, 6])
    parser.add_argument("--kws-input-seconds", type=float, default=1.0, help="0 uses the full Stage 1 context")
    args = parser.parse_args()

    paths = load_stage_config(args.stage1_config, args.paths)["paths"]
    stage1_cfg = load_stage_config(args.stage1_config, args.paths)
    kws_cfg = load_stage_config(args.kws_config, args.paths)
    device = get_device()
    stage1_checkpoint = resolve_path(args.stage1_checkpoint)
    kws_checkpoint = resolve_path(args.kws_checkpoint)
    presence_checkpoint = resolve_path(args.presence_checkpoint)
    enrollment_path = resolve_path(args.enrollment_wav)

    pretrained = resolve_path(stage1_cfg["pretrained_checkpoint"])
    stage1 = Stage1System(
        spex_cfg=stage1_cfg["spex"],
        sample_rate=int(stage1_cfg["sample_rate"]),
        use_cabin_frontend=True,
        num_mics=int(stage1_cfg.get("num_mics", 4)),
        frontend_cfg=stage1_cfg.get("frontend"),
        pretrained_dir=str(pretrained),
        backend=stage1_cfg.get("spex_backend", "transfer_spexplus_16k"),
    ).to(device)
    load_checkpoint(stage1_checkpoint, stage1)
    stage1.eval()

    presence = PresenceHead().to(device)
    load_checkpoint(presence_checkpoint, presence)
    presence.eval()
    kws = KWSModel(
        sample_rate=int(kws_cfg["sample_rate"]),
        n_mels=int(kws_cfg["model"]["n_mels"]),
        channels=list(kws_cfg["model"]["channels"]),
        num_classes=int(kws_cfg["model"]["num_classes"]),
    ).to(device)
    load_checkpoint(kws_checkpoint, kws)
    kws.eval()

    enrollment, _ = read_mono(enrollment_path, int(stage1_cfg["sample_rate"]))
    window_seconds = float(stage1_cfg.get("segment_seconds", 3.0))
    sample_rate = int(stage1_cfg["sample_rate"])
    window_samples = int(round(window_seconds * sample_rate))
    enrollment = segment_waveform(enrollment, window_samples // 2, training=False)
    streams = select_streams(
        resolve_path(args.noise_root),
        min_duration=float(args.min_duration),
        max_streams=int(args.max_streams),
        max_seconds=float(args.max_seconds),
    )
    if not streams:
        raise RuntimeError("No eligible CAVEMOVE noise/ventilation streams were found")

    hop_samples = max(1, int(round(float(args.hop_seconds) * sample_rate)))
    per_stream: list[dict[str, Any]] = []
    all_presence: list[bool] = []
    all_raw: list[bool] = []
    all_smoothed: list[bool] = []
    total_duration = 0.0
    total_windows = 0
    for stream_index, stream in enumerate(streams, start=1):
        path = Path(stream["path"])
        raw_data, _ = read_multichannel(path, sample_rate)
        data = prepare_channels(raw_data, list(args.channels), int(stage1_cfg.get("num_mics", 4)))
        windows, window_count = make_windows(data, window_samples, hop_samples)
        presence_scores: list[float] = []
        kws_scores: list[float] = []
        for start in range(0, window_count, max(1, int(args.batch_size))):
            batch_mc = torch.from_numpy(windows[start : start + args.batch_size]).to(device)
            batch_mix = torch.zeros((batch_mc.shape[0], window_samples), dtype=torch.float32, device=device)
            batch_enroll = torch.from_numpy(np.repeat(enrollment[None, :], batch_mc.shape[0], axis=0)).to(device)
            with torch.inference_mode():
                extracted, _, _, _ = stage1(batch_mix, batch_enroll, multichannel=batch_mc)
                pres = torch.sigmoid(presence(extracted)).detach()
                kws_input = normalize_extracted_waveform(extracted)
                if args.kws_input_seconds > 0:
                    kws_samples = int(round(args.kws_input_seconds * sample_rate))
                    if kws_input.shape[-1] >= kws_samples:
                        kws_input = kws_input[..., :kws_samples]
                    else:
                        kws_input = torch.nn.functional.pad(kws_input, (0, kws_samples - kws_input.shape[-1]))
                score = torch.softmax(kws(kws_input), dim=-1)[:, 1].detach()
            presence_scores.extend(pres.cpu().tolist())
            kws_scores.extend(score.cpu().tolist())
        presence_array = np.asarray(presence_scores, dtype=np.float32)
        kws_array = np.asarray(kws_scores, dtype=np.float32)
        presence_flags = presence_array >= float(args.presence_threshold)
        raw_flags = presence_flags & (kws_array >= float(args.kws_threshold))

        state = PipelineState()
        smooth_flags = np.zeros(window_count, dtype=bool)
        for index, (present, score) in enumerate(zip(presence_flags.tolist(), kws_array.tolist())):
            if not present:
                state.kws_scores.clear()
                state.kws_refractory_left = 0
            else:
                smooth_flags[index] = update_kws_state(
                    state,
                    float(score),
                    float(args.kws_threshold),
                    int(args.smoothing_window),
                    int(args.required_hits),
                    int(args.refractory_calls),
                )
        duration = float(stream["duration_seconds"])
        total_duration += duration
        total_windows += window_count
        all_presence.extend(presence_flags.tolist())
        all_raw.extend(raw_flags.tolist())
        all_smoothed.extend(smooth_flags.tolist())
        per_stream.append(
            {
                **stream,
                "windows": window_count,
                "presence_pass_windows": int(presence_flags.sum()),
                "raw_kws_positive_windows": int(raw_flags.sum()),
                "raw_kws_events": count_events(raw_flags, args.hop_seconds, args.event_merge_gap_seconds),
                "smoothed_events": int(smooth_flags.sum()),
                "presence_pass_rate": float(presence_flags.mean()),
                "raw_kws_window_rate_per_hour": rate_per_hour(int(raw_flags.sum()), duration),
                "raw_kws_event_rate_per_hour": rate_per_hour(
                    count_events(raw_flags, args.hop_seconds, args.event_merge_gap_seconds), duration
                ),
                "smoothed_event_rate_per_hour": rate_per_hour(int(smooth_flags.sum()), duration),
                "max_presence_score": float(presence_array.max(initial=0.0)),
                "max_kws_score": float(kws_array.max(initial=0.0)),
            }
        )
        print(
            f"{stream_index}/{len(streams)} {stream['vehicle']} {stream['noise_type']} "
            f"{duration:.1f}s: presence={presence_flags.mean():.3f}, "
            f"raw_events={per_stream[-1]['raw_kws_events']}, "
            f"smooth_events={int(smooth_flags.sum())}",
            flush=True,
        )

    all_presence_array = np.asarray(all_presence, dtype=bool)
    all_raw_array = np.asarray(all_raw, dtype=bool)
    all_smoothed_array = np.asarray(all_smoothed, dtype=bool)
    by_condition: dict[str, dict[str, Any]] = {}
    for condition in sorted({(row["vehicle"], row["noise_type"]) for row in per_stream}):
        rows = [row for row in per_stream if (row["vehicle"], row["noise_type"]) == condition]
        duration = sum(row["duration_seconds"] for row in rows)
        by_condition[f"{condition[0]}|{condition[1]}"] = {
            "streams": len(rows),
            "duration_seconds": duration,
            "presence_pass_windows": sum(row["presence_pass_windows"] for row in rows),
            "raw_kws_positive_windows": sum(row["raw_kws_positive_windows"] for row in rows),
            "raw_kws_events": sum(row["raw_kws_events"] for row in rows),
            "smoothed_events": sum(row["smoothed_events"] for row in rows),
            "raw_kws_event_rate_per_hour": rate_per_hour(sum(row["raw_kws_events"] for row in rows), duration),
            "smoothed_event_rate_per_hour": rate_per_hour(sum(row["smoothed_events"] for row in rows), duration),
        }

    result = {
        "benchmark": "continuous_cavemove_noise_false_accepts",
        "device": str(device),
        "stage1_checkpoint": str(stage1_checkpoint),
        "presence_checkpoint": str(presence_checkpoint),
        "kws_checkpoint": str(kws_checkpoint),
        "enrollment_wav": str(enrollment_path),
        "sample_rate": sample_rate,
        "stage1_context_seconds": window_seconds,
        "kws_input_seconds": float(args.kws_input_seconds or window_seconds),
        "hop_seconds": float(args.hop_seconds),
        "presence_threshold": float(args.presence_threshold),
        "kws_threshold": float(args.kws_threshold),
        "smoothing": {
            "window": int(args.smoothing_window),
            "required_hits": int(args.required_hits),
            "refractory_calls": int(args.refractory_calls),
        },
        "event_merge_gap_seconds": float(args.event_merge_gap_seconds),
        "streams": len(streams),
        "duration_seconds": total_duration,
        "duration_hours": total_duration / 3600.0,
        "windows": total_windows,
        "presence_pass_windows": int(all_presence_array.sum()),
        "presence_pass_rate": float(all_presence_array.mean()),
        "raw_kws_positive_windows": int(all_raw_array.sum()),
        "raw_kws_window_rate_per_hour": rate_per_hour(int(all_raw_array.sum()), total_duration),
        "raw_kws_events": sum(row["raw_kws_events"] for row in per_stream),
        "raw_kws_event_rate_per_hour": rate_per_hour(sum(row["raw_kws_events"] for row in per_stream), total_duration),
        "smoothed_events": int(all_smoothed_array.sum()),
        "smoothed_event_rate_per_hour": rate_per_hour(int(all_smoothed_array.sum()), total_duration),
        "condition_summary": by_condition,
        "selected_streams": per_stream,
        "interpretation": (
            "All streams are target-absent CAVEMOVE recordings; any wake is a false accept. "
            "The event rate is the runtime-relevant metric. The independent-window rate is "
            "also reported to make comparison with the ordinary KWS test explicit."
        ),
    }
    output = resolve_path(args.output)
    write_metrics(output, result)
    print(json.dumps({key: result[key] for key in (
        "device", "streams", "duration_hours", "windows", "presence_pass_rate",
        "raw_kws_window_rate_per_hour", "raw_kws_event_rate_per_hour",
        "smoothed_event_rate_per_hour", "output",
    ) if key in result} | {"output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
