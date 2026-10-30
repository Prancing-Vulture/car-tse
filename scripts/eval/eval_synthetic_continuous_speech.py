#!/usr/bin/env python
"""Evaluate wake recall and speech-triggered FAR in a synthetic cabin scene.

This benchmark creates a deterministic multi-channel scene from one Google
Speech Commands speaker (the enrolled target), held-out non-wake speech clips
(passenger hard negatives), and long CAVEMOVE noise recordings.  It then runs
the actual Stage 1 -> presence -> KWS temporal path on overlapping windows.

It is deliberately labelled a proxy: the speech is mixed identically into
each microphone and the scene is assembled from short public clips.  It is
still closer to the runtime question than independent one-second examples,
because target and passenger speech share a continuous noise bed and the
temporal decoder sees adjacent windows.
"""

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
import soundfile as sf
import torch

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.segment import segment_waveform
from car_tse.io.audio import read_mono, read_multichannel
from car_tse.io.manifest import read_jsonl
from car_tse.models.kws_model import KWSModel
from car_tse.models.presence_head import PresenceHead
from car_tse.models.stage1_system import Stage1System
from car_tse.pipeline.in_car_pipeline import PipelineState, normalize_extracted_waveform, update_kws_state
from car_tse.train.utils import get_device, load_checkpoint, write_metrics
from eval_continuous_kws import make_windows, prepare_channels


def resolve_path(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def speaker_key(path: Path) -> str:
    return path.stem.split("_nohash_")[0]


def load_clip(path: Path, sample_rate: int, max_samples: int) -> np.ndarray:
    wav, _ = read_mono(path, sample_rate)
    wav = np.asarray(wav, dtype=np.float32)
    if len(wav) >= max_samples:
        start = (len(wav) - max_samples) // 2
        return wav[start : start + max_samples]
    out = np.zeros(max_samples, dtype=np.float32)
    start = (max_samples - len(wav)) // 2
    out[start : start + len(wav)] = wav
    return out


def choose_positive_files(root: Path, count: int) -> list[Path]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(root.glob("*.wav")):
        grouped[speaker_key(path)].append(path)
    if not grouped:
        raise RuntimeError(f"No wake-word recordings found under {root}")
    chosen_speaker, files = max(grouped.items(), key=lambda item: (len(item[1]), item[0]))
    if len(files) < count:
        raise RuntimeError(f"Speaker {chosen_speaker} has only {len(files)} wake clips; need {count}")
    return files[:count]


def choose_noise_files(root: Path, min_duration: float) -> list[Path]:
    best: dict[str, tuple[float, Path]] = {}
    for path in sorted(root.rglob("*.wav")):
        if "noise" not in path.parts and "ventilation" not in path.parts:
            continue
        try:
            info = sf.info(str(path))
        except (OSError, RuntimeError):
            continue
        if info.duration < min_duration:
            continue
        vehicle = next(
            (name for name in ("AlfaRomeo_146", "Smart_forfour", "Volkswagen_Golf") if name in path.parts),
            "unknown",
        )
        current = best.get(vehicle)
        if current is None or info.duration > current[0]:
            best[vehicle] = (float(info.duration), path)
    if len(best) < 3:
        raise RuntimeError("Could not find one long CAVEMOVE recording for each vehicle")
    return [best[name][1] for name in ("AlfaRomeo_146", "Smart_forfour", "Volkswagen_Golf")]


def make_scene(
    positive_files: list[Path],
    negative_files: list[Path],
    noise_path: Path,
    sample_rate: int,
    num_mics: int,
    channels: list[int],
    slot_seconds: float,
    speech_seconds: float,
    speech_offset_seconds: float,
    snr_db: float,
) -> tuple[np.ndarray, np.ndarray, list[float], float]:
    if len(negative_files) < len(positive_files) * 5:
        raise ValueError("Need at least five passenger clips per wake event")
    slot_samples = int(round(slot_seconds * sample_rate))
    speech_samples = int(round(speech_seconds * sample_rate))
    total_slots = len(positive_files) * 6
    total_samples = total_slots * slot_samples
    noise, _ = read_multichannel(noise_path, sample_rate)
    noise = prepare_channels(noise, channels, num_mics)
    if len(noise) < total_samples:
        noise = np.tile(noise, (int(np.ceil(total_samples / max(1, len(noise)))), 1))
    noise = noise[:total_samples].copy()

    speech_items: list[tuple[np.ndarray, bool, int]] = []
    neg_index = 0
    positive_centers: list[float] = []
    for positive_index, positive in enumerate(positive_files):
        for _ in range(5):
            speech_items.append((load_clip(negative_files[neg_index], sample_rate, speech_samples), False, neg_index))
            neg_index += 1
        speech_items.append((load_clip(positive, sample_rate, speech_samples), True, positive_index))

    speech_rms = [float(np.sqrt(np.mean(wav * wav) + 1e-8)) for wav, _, _ in speech_items]
    noise_rms = float(np.sqrt(np.mean(noise * noise) + 1e-8))
    scale = float(np.median(speech_rms)) * (10.0 ** (-float(snr_db) / 20.0)) / max(noise_rms, 1e-6)
    scene = noise * scale
    speech_offset = int(round(speech_offset_seconds * sample_rate))
    for slot_index, (speech, is_positive, source_index) in enumerate(speech_items):
        start = slot_index * slot_samples + speech_offset
        end = min(start + speech_samples, total_samples)
        if end <= start:
            continue
        scene[start:end] += speech[: end - start, None]
        if is_positive:
            positive_centers.append((start + 0.5 * speech_samples) / sample_rate)
    mixture = scene.mean(axis=1).astype(np.float32)
    return mixture, scene.astype(np.float32), positive_centers, total_samples / sample_rate


def event_indices(flags: np.ndarray, merge_gap_steps: int) -> list[int]:
    indices: list[int] = []
    last = -10**9
    for index, flag in enumerate(flags.tolist()):
        if flag and index - last > merge_gap_steps:
            indices.append(index)
            last = index
    return indices


def match_events(detection_times: list[float], expected_times: list[float], tolerance: float) -> tuple[int, list[float]]:
    remaining = list(detection_times)
    matched = 0
    false_times: list[float] = []
    for expected in expected_times:
        candidates = [(abs(value - expected), index) for index, value in enumerate(remaining) if abs(value - expected) <= tolerance]
        if not candidates:
            continue
        _, index = min(candidates)
        remaining.pop(index)
        matched += 1
    false_times.extend(remaining)
    return matched, false_times


def evaluate_scene(
    stage1: Stage1System,
    presence: PresenceHead,
    kws: KWSModel,
    device: torch.device,
    mixture: np.ndarray,
    multichannel: np.ndarray,
    enrollment: np.ndarray,
    expected_centers: list[float],
    duration_seconds: float,
    sample_rate: int,
    stage1_seconds: float,
    hop_seconds: float,
    batch_size: int,
    presence_threshold: float,
    kws_threshold: float,
    smooth_window: int,
    required_hits: int,
    refractory_calls: int,
    kws_input_seconds: float,
    event_tolerance_seconds: float,
) -> dict[str, Any]:
    window_samples = int(round(stage1_seconds * sample_rate))
    hop_samples = int(round(hop_seconds * sample_rate))
    windows, window_count = make_windows(multichannel, window_samples, hop_samples)
    mixture_windows, _ = make_windows(mixture[:, None], window_samples, hop_samples)
    presence_scores: list[float] = []
    kws_scores: list[float] = []
    for start in range(0, window_count, max(1, int(batch_size))):
        batch_mc = torch.from_numpy(windows[start : start + batch_size]).to(device)
        batch_mix = torch.from_numpy(mixture_windows[start : start + batch_size, 0, :]).to(device)
        batch_enroll = torch.from_numpy(np.repeat(enrollment[None, :], batch_mc.shape[0], axis=0)).to(device)
        with torch.inference_mode():
            extracted, _, _, _ = stage1(batch_mix, batch_enroll, multichannel=batch_mc)
            pres = torch.sigmoid(presence(extracted)).detach()
            kws_input = normalize_extracted_waveform(extracted)
            if kws_input_seconds > 0:
                kws_samples = int(round(kws_input_seconds * sample_rate))
                if kws_input.shape[-1] >= kws_samples:
                    kws_input = kws_input[..., :kws_samples]
                else:
                    kws_input = torch.nn.functional.pad(kws_input, (0, kws_samples - kws_input.shape[-1]))
            score = torch.softmax(kws(kws_input), dim=-1)[:, 1].detach()
        presence_scores.extend(pres.cpu().tolist())
        kws_scores.extend(score.cpu().tolist())

    presence_array = np.asarray(presence_scores, dtype=np.float32)
    kws_array = np.asarray(kws_scores, dtype=np.float32)
    presence_flags = presence_array >= presence_threshold
    raw_flags = presence_flags & (kws_array >= kws_threshold)
    state = PipelineState()
    smooth_flags = np.zeros(window_count, dtype=bool)
    for index, (present, score) in enumerate(zip(presence_flags.tolist(), kws_array.tolist())):
        if not present:
            state.kws_scores.clear()
            state.kws_refractory_left = 0
        else:
            smooth_flags[index] = update_kws_state(
                state, float(score), kws_threshold, smooth_window, required_hits, refractory_calls
            )

    raw_indices = event_indices(raw_flags, merge_gap_steps=max(1, int(np.floor(2.0 / hop_seconds))))
    smooth_indices = np.flatnonzero(smooth_flags).tolist()
    raw_times = [index * hop_seconds + stage1_seconds / 2.0 for index in raw_indices]
    smooth_times = [index * hop_seconds + stage1_seconds / 2.0 for index in smooth_indices]
    raw_matched, raw_false = match_events(raw_times, expected_centers, event_tolerance_seconds)
    smooth_matched, smooth_false = match_events(smooth_times, expected_centers, event_tolerance_seconds)

    presence_hits = sum(
        any(abs(index * hop_seconds + stage1_seconds / 2.0 - expected) <= event_tolerance_seconds for index, flag in enumerate(presence_flags) if flag)
        for expected in expected_centers
    )
    return {
        "duration_seconds": duration_seconds,
        "windows": window_count,
        "expected_wake_events": len(expected_centers),
        "presence_event_coverage": presence_hits / max(len(expected_centers), 1),
        "presence_pass_rate": float(presence_flags.mean()),
        "raw_positive_windows": int(raw_flags.sum()),
        "raw_events": len(raw_times),
        "raw_matched_wakes": raw_matched,
        "raw_wake_recall": raw_matched / max(len(expected_centers), 1),
        "raw_false_accept_events": len(raw_false),
        "raw_false_accept_rate_per_hour": len(raw_false) * 3600.0 / max(duration_seconds, 1e-6),
        "smoothed_events": len(smooth_times),
        "smoothed_matched_wakes": smooth_matched,
        "smoothed_wake_recall": smooth_matched / max(len(expected_centers), 1),
        "smoothed_false_accept_events": len(smooth_false),
        "smoothed_false_accept_rate_per_hour": len(smooth_false) * 3600.0 / max(duration_seconds, 1e-6),
        "max_presence_score": float(presence_array.max(initial=0.0)),
        "max_kws_score": float(kws_array.max(initial=0.0)),
        "mean_kws_score": float(kws_array.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--stage1-config", type=Path, default=REPO_ROOT / "configs" / "stage1_spex_transfer_mc_cavemove.yaml")
    parser.add_argument("--stage1-checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "stage1_spex_transfer_mc_cavemove_cabin" / "best.pt")
    parser.add_argument("--kws-config", type=Path, default=REPO_ROOT / "configs" / "stage3_kws_tse_hybrid.yaml")
    parser.add_argument("--kws-checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "stage3_kws_tse_hybrid_ft1" / "best.pt")
    parser.add_argument("--presence-checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "stage2_presence_cavemove_cabin" / "best.pt")
    parser.add_argument("--positive-root", type=Path, default=REPO_ROOT / "data" / "raw" / "speech_commands_v2" / "extracted" / "marvin")
    parser.add_argument("--negative-manifest", type=Path, default=REPO_ROOT / "manifests" / "test" / "stage3_kws_tse_hardneg_test.jsonl")
    parser.add_argument("--noise-root", type=Path, default=REPO_ROOT / "data" / "raw" / "cavemove" / "extracted" / "cavemove_dataset")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "logs" / "synthetic_continuous_speech_proxy.json")
    parser.add_argument("--positive-count", type=int, default=6)
    parser.add_argument("--negative-count", type=int, default=30)
    parser.add_argument("--slot-seconds", type=float, default=4.0)
    parser.add_argument("--speech-seconds", type=float, default=1.5)
    parser.add_argument("--speech-offset-seconds", type=float, default=1.0)
    parser.add_argument("--snr-db", nargs="+", type=float, default=[-5.0, 0.0, 10.0, 20.0])
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, nargs="+", default=[0, 2, 4, 6])
    parser.add_argument("--num-mics", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hop-seconds", type=float, default=1.0)
    parser.add_argument("--presence-threshold", type=float, default=0.90)
    parser.add_argument("--kws-threshold", type=float, default=0.29)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--required-hits", type=int, default=2)
    parser.add_argument("--refractory-calls", type=int, default=2)
    parser.add_argument("--kws-input-seconds", type=float, default=1.0, help="0 uses the full Stage 1 context")
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()

    stage1_cfg = load_stage_config(args.stage1_config, args.paths)
    kws_cfg = load_stage_config(args.kws_config, args.paths)
    device = get_device()
    sample_rate = int(args.sample_rate)
    stage1_checkpoint = resolve_path(args.stage1_checkpoint)
    presence_checkpoint = resolve_path(args.presence_checkpoint)
    kws_checkpoint = resolve_path(args.kws_checkpoint)
    pretrained = resolve_path(stage1_cfg["pretrained_checkpoint"])

    stage1 = Stage1System(
        spex_cfg=stage1_cfg["spex"],
        sample_rate=sample_rate,
        use_cabin_frontend=True,
        num_mics=int(args.num_mics),
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

    positive_files = choose_positive_files(resolve_path(args.positive_root), int(args.positive_count))
    enrollment, _ = read_mono(positive_files[0], sample_rate)
    stage1_seconds = float(stage1_cfg.get("segment_seconds", 3.0))
    enrollment = segment_waveform(enrollment, int(round(stage1_seconds * sample_rate)) // 2, training=False)
    all_negative_rows = [row for row in read_jsonl(resolve_path(args.negative_manifest)) if int(row.get("label", 0)) == 0]
    rng = random.Random(int(args.seed))
    rng.shuffle(all_negative_rows)
    required_negatives = int(args.positive_count) * 5
    if len(all_negative_rows) < required_negatives:
        raise RuntimeError("Negative manifest does not contain enough hard-negative rows")
    negative_files = [resolve_path(row["wav"]) for row in all_negative_rows[: max(int(args.negative_count), required_negatives)]]
    negative_files = negative_files[:required_negatives]
    noise_files = choose_noise_files(resolve_path(args.noise_root), min_duration=30.0)

    results: list[dict[str, Any]] = []
    for index, snr_db in enumerate(args.snr_db):
        noise_path = noise_files[index % len(noise_files)]
        mixture, multichannel, expected_centers, duration = make_scene(
            positive_files,
            negative_files,
            noise_path,
            sample_rate,
            int(args.num_mics),
            list(args.channels),
            float(args.slot_seconds),
            float(args.speech_seconds),
            float(args.speech_offset_seconds),
            float(snr_db),
        )
        scene_result = evaluate_scene(
            stage1,
            presence,
            kws,
            device,
            mixture,
            multichannel,
            enrollment,
            expected_centers,
            duration,
            sample_rate,
            stage1_seconds,
            float(args.hop_seconds),
            int(args.batch_size),
            float(args.presence_threshold),
            float(args.kws_threshold),
            int(args.smooth_window),
            int(args.required_hits),
            int(args.refractory_calls),
            float(args.kws_input_seconds),
            event_tolerance_seconds=max(2.0, stage1_seconds / 2.0),
        )
        scene_result.update({"snr_db": float(snr_db), "noise_wav": str(noise_path), "vehicle": next((name for name in ("AlfaRomeo_146", "Smart_forfour", "Volkswagen_Golf") if name in noise_path.parts), "unknown")})
        results.append(scene_result)
        print(
            f"SNR {snr_db:+.1f} dB ({scene_result['vehicle']}): "
            f"presence={scene_result['presence_pass_rate']:.3f}, "
            f"raw_recall={scene_result['raw_wake_recall']:.3f}, "
            f"smooth_recall={scene_result['smoothed_wake_recall']:.3f}, "
            f"smooth_FA/h={scene_result['smoothed_false_accept_rate_per_hour']:.1f}",
            flush=True,
        )

    result = {
        "benchmark": "synthetic_continuous_driver_passenger_speech_proxy",
        "device": str(device),
        "stage1_checkpoint": str(stage1_checkpoint),
        "presence_checkpoint": str(presence_checkpoint),
        "kws_checkpoint": str(kws_checkpoint),
        "positive_speaker": speaker_key(positive_files[0]),
        "positive_files": [str(path) for path in positive_files],
        "negative_examples": [str(path) for path in negative_files],
        "stage1_context_seconds": stage1_seconds,
        "kws_input_seconds": float(args.kws_input_seconds or stage1_seconds),
        "hop_seconds": float(args.hop_seconds),
        "presence_threshold": float(args.presence_threshold),
        "kws_threshold": float(args.kws_threshold),
        "smoothing": {"window": int(args.smooth_window), "required_hits": int(args.required_hits), "refractory_calls": int(args.refractory_calls)},
        "scenes": results,
        "note": (
            "Proxy benchmark: target and passenger clips are assembled into a continuous "
            "CAVEMOVE noise bed and identically mixed into the selected microphones. "
            "Use for engineering calibration, not as a real-cabin claim."
        ),
    }
    output = resolve_path(args.output)
    write_metrics(output, result)
    print({"output": str(output), "scenes": len(results)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
