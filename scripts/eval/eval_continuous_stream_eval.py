#!/usr/bin/env python
"""Evaluate the cascade on one long continuous synthetic stream.

Unlike the 14-second scene matrix, this builds a single timeline (default 15
minutes) with a continuous noise bed, empty/silent stretches, and sparse
speech events placed at random times: target ``Marvin`` + command, passenger
``Marvin``, passenger non-wake speech, and command-like speech without a wake
word.  The runtime pipeline processes the whole stream in one stateful
pass, so wake association, command capture, smoothing/refractory, and
false-action accounting behave as they would over a long deployment window.

The construction reuses the cross-corpus proxy contract: the Marvin wake clip
and the FSC command come from different corpora and the wake speaker is not
verified to match the enrollment speaker.  This is labelled synthetic
cross-corpus evidence, never real-driver or production evidence.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import soundfile as sf
import torch

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.io.audio import read_mono, read_multichannel
from car_tse.pipeline.in_car_pipeline import PipelineState
from car_tse.pipeline.windowing import full_context_window_starts
from car_tse.train.utils import get_device
from eval_continuous_full_pipeline_proxy import (
    add_at,
    fit_start,
    load_jsonl,
    load_pipeline,
    resolve_path,
    rms,
    select_noise_files,
    speaker_key,
)
from eval_continuous_kws import prepare_channels


DEFAULT_NOISE_BED = Path("D:/datasets/remasc/stress_bed/ReMASC/noise/remasc_vehicle_replayed_6ch_tiled.wav")


def interval_conflicts(
    intervals: list[tuple[float, float]], start: float, end: float
) -> bool:
    return any(start < other_end and end > other_start for other_start, other_end in intervals)


def place_event(
    rng: random.Random,
    duration: float,
    needed_seconds: float,
    intervals: list[tuple[float, float]],
    margin_seconds: float,
) -> float | None:
    for _ in range(400):
        start = rng.uniform(margin_seconds, max(margin_seconds, duration - needed_seconds - margin_seconds))
        end = start + needed_seconds
        if not interval_conflicts(intervals, start - margin_seconds, end + margin_seconds):
            return start
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "configs/eval_end_to_end_cavemove_target_kws_smart_speaker_compact_cabin_smart_thr030_2026-10-12.yaml",
    )
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.yaml")
    parser.add_argument("--duration-seconds", type=float, default=900.0)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--hop-seconds", type=float, default=1.0)
    parser.add_argument(
        "--command-delay-calls",
        type=int,
        default=None,
        help="Override the post-wake command delay (needed when hop_seconds < 1.0).",
    )
    parser.add_argument("--target-events", type=int, default=12)
    parser.add_argument("--passenger-wake-events", type=int, default=4)
    parser.add_argument("--passenger-speech-events", type=int, default=15)
    parser.add_argument("--command-like-negative-events", type=int, default=6)
    parser.add_argument("--cabin-snr-db-min", type=float, default=0.0)
    parser.add_argument("--cabin-snr-db-max", type=float, default=15.0)
    parser.add_argument("--passenger-snr-db-min", type=float, default=-5.0)
    parser.add_argument("--passenger-snr-db-max", type=float, default=5.0)
    parser.add_argument("--wake-manifest", type=Path, default=REPO_ROOT / "manifests/test/stage3_kws_cabin_test.jsonl")
    parser.add_argument(
        "--intent-manifest",
        type=Path,
        default=REPO_ROOT / "manifests/official/stage4_intent_fsc_test.jsonl",
    )
    parser.add_argument("--noise-bed", type=Path, default=DEFAULT_NOISE_BED)
    parser.add_argument("--noise-root", type=Path, default=REPO_ROOT / "data/raw/cavemove/extracted/cavemove_dataset")
    parser.add_argument(
        "--enrollment-source",
        choices=("wake", "command", "same_speaker_other", "same_wake_speaker"),
        default="same_wake_speaker",
        help=(
            "same_wake_speaker: enrollment is a different Marvin utterance by the "
            "wake speaker (speaker-consistent wake leg; the FSC command leg is not). "
            "same_speaker_other: enrollment is another FSC utterance by the command "
            "speaker (speaker-consistent command leg; the wake leg is not)."
        ),
    )
    parser.add_argument("--save-audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "logs/continuous_stream_eval_2026-10-12.json",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=REPO_ROOT / "output/artifacts/continuous_stream_eval_2026-10-12",
    )
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    np.random.seed(int(args.seed) % (2**32))
    torch.manual_seed(int(args.seed))

    cfg = load_stage_config(args.config, args.paths)
    if args.command_delay_calls is not None:
        cfg["command_delay_calls"] = int(args.command_delay_calls)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    pipeline = load_pipeline(cfg, cfg["paths"], device)
    sample_rate = int(cfg["sample_rate"])
    stage1_seconds = float(cfg.get("segment_seconds", 3.0))
    duration = float(args.duration_seconds)
    total_samples = int(round(duration * sample_rate))

    wake_rows = [
        row
        for row in load_jsonl(resolve_path(args.wake_manifest))
        if int(row.get("label", 0)) == 1 and not bool(row.get("noise_only", False))
    ]
    negative_rows = [
        row
        for row in load_jsonl(resolve_path(args.wake_manifest))
        if int(row.get("label", 0)) == 0 and not bool(row.get("noise_only", False))
    ]
    if not wake_rows or not negative_rows:
        raise RuntimeError("Wake manifest lacks positive or non-wake negative rows")

    wake_rows.sort(key=lambda row: str(row.get("wav", "")))
    speaker_groups: dict[str, list[dict[str, Any]]] = {}
    for row in wake_rows:
        speaker_groups.setdefault(speaker_key(resolve_path(row["wav"])), []).append(row)
    speaker_ids = sorted(speaker_groups)

    all_intent_rows = load_jsonl(resolve_path(args.intent_manifest))
    intent_by_speaker: dict[str, list[dict[str, Any]]] = {}
    for row in all_intent_rows:
        intent_by_speaker.setdefault(str(row.get("speaker_id", "")), []).append(row)
    intent_rows = sorted(all_intent_rows, key=lambda row: str(row.get("wav", "")))
    if not intent_rows:
        raise RuntimeError("No intent rows found")

    noise_bed_path = resolve_path(args.noise_bed) if args.noise_bed is not None else None
    if noise_bed_path is not None and not noise_bed_path.exists():
        noise_bed_path = None
    if noise_bed_path is None:
        candidates = select_noise_files(resolve_path(args.noise_root), 20.0)
        if not candidates:
            raise RuntimeError("No noise bed available")
        noise_bed_path = candidates[0]

    raw_noise, _ = read_multichannel(noise_bed_path, sample_rate)
    noise = prepare_channels(
        raw_noise,
        list(cfg.get("cabin_noise_channels", [0, 2, 4, 6])),
        int(cfg.get("num_mics", 4)),
    )
    if len(noise) < total_samples:
        repeats = int(np.ceil(total_samples / max(1, len(noise))))
        noise = np.tile(noise, (repeats, 1))
    noise = np.asarray(noise[:total_samples], dtype=np.float32)
    noise_rms = rms(noise.mean(axis=1))

    speech = np.zeros(total_samples, dtype=np.float32)
    intervals: list[tuple[float, float]] = []
    events: list[dict[str, Any]] = []

    multi_utterance_speakers = [speaker for speaker in speaker_ids if len(speaker_groups[speaker]) > 1]
    fsc_speakers = sorted(
        speaker for speaker, rows in intent_by_speaker.items() if len(rows) > 1
    )
    if not fsc_speakers:
        raise RuntimeError("No FSC speaker has multiple utterances for enrollment")
    command_speaker = fsc_speakers[0]
    command_clips = sorted(
        intent_by_speaker[command_speaker], key=lambda row: str(row.get("wav", ""))
    )

    target_samples = int(round(sample_rate * stage1_seconds / 2.0))
    wave_cache: dict[str, np.ndarray] = {}

    def load_wave(path_value: str, samples: int) -> np.ndarray:
        if path_value not in wave_cache:
            wave_cache[path_value] = read_mono(resolve_path(path_value), sample_rate)[0]
        return fit_start(wave_cache[path_value], samples)

    target_speaker: str | None = None
    target_clips: list[dict[str, Any]] = []
    if args.enrollment_source == "same_wake_speaker":
        if not multi_utterance_speakers:
            raise RuntimeError("No multi-utterance Marvin speakers available for same_wake_speaker")
        target_speaker = multi_utterance_speakers[0]
        target_clips = sorted(speaker_groups[target_speaker], key=lambda row: str(row["wav"]))
        enrollment_source_path = str(resolve_path(target_clips[1]["wav"]))
        enrollment = load_wave(target_clips[1]["wav"], target_samples)
    elif args.enrollment_source == "wake":
        target_speaker = speaker_ids[0]
        target_clips = sorted(speaker_groups[target_speaker], key=lambda row: str(row["wav"]))
        enrollment_source_path = str(resolve_path(target_clips[0]["wav"]))
        enrollment = load_wave(target_clips[0]["wav"], target_samples)
    elif args.enrollment_source == "command":
        enrollment_source_path = str(resolve_path(command_clips[0]["wav"]))
        enrollment = load_wave(command_clips[0]["wav"], target_samples)
    else:
        enrollment_source_path = str(resolve_path(command_clips[1]["wav"]))
        enrollment = load_wave(command_clips[1]["wav"], target_samples)
    enrollment_bank_paths = [enrollment_source_path]
    reference_speaker_embedding = None

    target_event_speakers: list[str] = []
    for event_index in range(max(0, int(args.target_events))):
        if args.enrollment_source == "same_wake_speaker" and target_speaker is not None:
            wake_row = target_clips[event_index % len(target_clips)]
            speaker = target_speaker
        else:
            speaker = speaker_ids[event_index % len(speaker_ids)]
            wake_row = speaker_groups[speaker][
                event_index // len(speaker_ids) % len(speaker_groups[speaker])
            ]
        wake_path = str(resolve_path(wake_row["wav"]))
        wake_wav = load_wave(wake_row["wav"], sample_rate)
        if args.enrollment_source == "same_speaker_other":
            intent_row = command_clips[event_index % len(command_clips)]
        else:
            intent_row = intent_rows[event_index % len(intent_rows)]
        command_wav, _ = read_mono(resolve_path(intent_row["wav"]), sample_rate)
        command_wav = fit_start(command_wav, 2 * sample_rate)
        gap = rng.uniform(0.3, 1.2)
        needed = (len(wake_wav) + len(command_wav)) / sample_rate + gap + 3.5
        start = place_event(rng, duration, needed, intervals, margin_seconds=2.0)
        if start is None:
            continue
        wake_start = start
        command_start = wake_start + len(wake_wav) / sample_rate + gap
        command_end = command_start + len(command_wav) / sample_rate
        event_snr = rng.uniform(float(args.cabin_snr_db_min), float(args.cabin_snr_db_max))
        scale = noise_rms * (10.0 ** (event_snr / 20.0)) / max(rms(wake_wav), 1e-6)
        add_at(speech, wake_wav * scale, int(round(wake_start * sample_rate)))
        add_at(speech, command_wav * scale, int(round(command_start * sample_rate)))
        intervals.append((wake_start - 2.0, command_end + 3.5))
        target_event_speakers.append(speaker)
        events.append(
            {
                "type": "target_wake_command",
                "start_seconds": wake_start,
                "wake_end_seconds": wake_start + len(wake_wav) / sample_rate,
                "command_start_seconds": command_start,
                "command_end_seconds": command_end,
                "cabin_snr_db": event_snr,
                "expected_intent": str(intent_row["intent"]),
                "wake_source": str(resolve_path(wake_row["wav"])),
                "command_source": str(resolve_path(intent_row["wav"])),
                "target_speaker": speaker,
            }
        )

    for _ in range(max(0, int(args.passenger_wake_events))):
        candidates = [row for row in wake_rows if speaker_key(resolve_path(row["wav"])) not in target_event_speakers]
        if not candidates:
            break
        row = rng.choice(candidates)
        wave, _ = read_mono(resolve_path(row["wav"]), sample_rate)
        wave = fit_start(wave, sample_rate)
        needed = len(wave) / sample_rate + 3.5
        start = place_event(rng, duration, needed, intervals, margin_seconds=2.0)
        if start is None:
            continue
        snr = rng.uniform(float(args.passenger_snr_db_min), float(args.passenger_snr_db_max))
        scale = noise_rms * (10.0 ** (snr / 20.0)) / max(rms(wave), 1e-6)
        add_at(speech, wave * scale, int(round(start * sample_rate)))
        intervals.append((start - 2.0, start + len(wave) / sample_rate + 3.5))
        events.append(
            {
                "type": "passenger_wake",
                "start_seconds": start,
                "end_seconds": start + len(wave) / sample_rate,
                "snr_db": snr,
                "source": str(resolve_path(row["wav"])),
                "speaker": speaker_key(resolve_path(row["wav"])),
            }
        )

    for _ in range(max(0, int(args.passenger_speech_events))):
        row = rng.choice(negative_rows)
        wave, _ = read_mono(resolve_path(row["wav"]), sample_rate)
        wave = fit_start(wave, int(1.5 * sample_rate))
        needed = len(wave) / sample_rate + 3.5
        start = place_event(rng, duration, needed, intervals, margin_seconds=2.0)
        if start is None:
            continue
        snr = rng.uniform(float(args.passenger_snr_db_min), float(args.passenger_snr_db_max))
        scale = noise_rms * (10.0 ** (snr / 20.0)) / max(rms(wave), 1e-6)
        add_at(speech, wave * scale, int(round(start * sample_rate)))
        intervals.append((start - 2.0, start + len(wave) / sample_rate + 3.5))
        events.append(
            {
                "type": "passenger_speech",
                "start_seconds": start,
                "end_seconds": start + len(wave) / sample_rate,
                "snr_db": snr,
                "source": str(resolve_path(row["wav"])),
            }
        )

    for _ in range(max(0, int(args.command_like_negative_events))):
        row = rng.choice(intent_rows)
        wave, _ = read_mono(resolve_path(row["wav"]), sample_rate)
        wave = fit_start(wave, int(2.0 * sample_rate))
        needed = len(wave) / sample_rate + 3.5
        start = place_event(rng, duration, needed, intervals, margin_seconds=2.0)
        if start is None:
            continue
        snr = rng.uniform(float(args.passenger_snr_db_min), float(args.passenger_snr_db_max))
        scale = noise_rms * (10.0 ** (snr / 20.0)) / max(rms(wave), 1e-6)
        add_at(speech, wave * scale, int(round(start * sample_rate)))
        intervals.append((start - 2.0, start + len(wave) / sample_rate + 3.5))
        events.append(
            {
                "type": "command_like_without_wake",
                "start_seconds": start,
                "end_seconds": start + len(wave) / sample_rate,
                "snr_db": snr,
                "source": str(resolve_path(row["wav"])),
                "intent": str(row["intent"]),
            }
        )

    with torch.inference_mode():
        embedding = pipeline.stage1.speaker_embedding(
            torch.from_numpy(enrollment).to(device).unsqueeze(0)
        )
        if embedding is not None:
            reference_speaker_embedding = embedding.detach()

    multichannel = (speech[:, None] + noise).astype(np.float32)
    mixture = multichannel.mean(axis=1).astype(np.float32)

    artifact_dir = resolve_path(args.artifact_dir)
    if args.enrollment_source == "same_wake_speaker":
        consistency_note = (
            "The enrollment and wake clips are different Marvin utterances by the same "
            "Speech Commands speaker (speaker-consistent wake leg). The FSC command is a "
            "different speaker, so the command/intent leg is a cross-corpus proxy. "
            "Not real-driver evidence."
        )
    elif args.enrollment_source == "same_speaker_other":
        consistency_note = (
            "The enrollment and command are different utterances by the same FSC speaker "
            "(speaker-consistent command leg). The Marvin wake clip speaker is not "
            "verified to match, so the wake leg is a cross-corpus proxy. "
            "Not real-driver evidence."
        )
    else:
        consistency_note = (
            "Synthetic cross-corpus stream: Marvin clips come from Speech Commands and "
            "commands from Fluent Speech Commands; the wake speaker is not verified to "
            "match the enrollment speaker. Not real-driver evidence."
        )
    timeline = {
        "seed": int(args.seed),
        "duration_seconds": duration,
        "sample_rate": sample_rate,
        "config": str(resolve_path(args.config)),
        "noise_bed": str(noise_bed_path),
        "noise_rms": noise_rms,
        "enrollment_source": str(args.enrollment_source),
        "enrollment_source_wav": enrollment_source_path,
        "enrollment_bank_wavs": enrollment_bank_paths,
        "events": events,
        "limitation": consistency_note,
    }
    if bool(args.save_audio):
        artifact_dir.mkdir(parents=True, exist_ok=True)
        pcm = np.clip(multichannel, -1.0, 1.0)
        sf.write(
            str(artifact_dir / "stream_4ch.wav"),
            (pcm * 32767.0).astype(np.int16),
            sample_rate,
            subtype="PCM_16",
        )
        (artifact_dir / "stream_timeline.json").write_text(
            json.dumps(timeline, indent=2), encoding="utf-8"
        )

    window_samples = int(round(stage1_seconds * sample_rate))
    hop_samples = max(1, int(round(float(args.hop_seconds) * sample_rate)))
    state = PipelineState(reference_speaker_embedding=reference_speaker_embedding)
    wake_events: list[dict[str, Any]] = []
    command_events: list[dict[str, Any]] = []
    window_count = 0

    start_time = time.perf_counter()
    for start in full_context_window_starts(len(mixture), window_samples, hop_samples):
        end = min(start + window_samples, len(mixture))
        mix_window = np.zeros(window_samples, dtype=np.float32)
        mc_window = np.zeros((multichannel.shape[1], window_samples), dtype=np.float32)
        if end > start:
            mix_window[: end - start] = mixture[start:end]
            mc_window[:, : end - start] = multichannel[start:end].T
        with torch.inference_mode():
            output, state = pipeline.forward(
                torch.from_numpy(mix_window).to(device),
                torch.from_numpy(enrollment).to(device),
                state,
                multichannel=torch.from_numpy(mc_window).to(device),
            )
        window_start = start / sample_rate
        row = {
            "window_start_seconds": window_start,
            "window_center_seconds": window_start + stage1_seconds / 2.0,
            "speaker_similarity": output.speaker_similarity,
            "wake_probability": output.wake_probability,
            "target_kws_probability": output.target_kws_probability,
            "wake_detected": bool(output.wake_detected),
            "intent_ready": bool(output.intent_ready),
            "command_suppressed": bool(output.command_suppressed),
            "intent": None,
            "intent_confidence": None,
        }
        if output.intent_logits is not None:
            probabilities = torch.softmax(output.intent_logits, dim=-1)[0]
            predicted_id = str(int(probabilities.argmax().item()))
            row["intent"] = (
                pipeline.label_map.get(predicted_id, predicted_id) if pipeline.label_map else predicted_id
            )
            row["intent_confidence"] = float(probabilities.max().cpu())
        window_count += 1
        if output.wake_detected:
            wake_events.append(row)
        if output.intent_ready:
            command_events.append(row)
    elapsed = time.perf_counter() - start_time

    target_events = [event for event in events if event["type"] == "target_wake_command"]
    matched_wake_indices: set[int] = set()
    matched_command_indices: set[int] = set()
    per_event: list[dict[str, Any]] = []
    for event in target_events:
        wake_start = float(event["start_seconds"])
        wake_end = float(event["wake_end_seconds"])
        command_start = float(event["command_start_seconds"])
        command_end = float(event["command_end_seconds"])
        wake_match = None
        for index, row in enumerate(wake_events):
            if index in matched_wake_indices:
                continue
            center = float(row["window_center_seconds"])
            if wake_start - 0.25 <= center <= wake_end + 3.0:
                wake_match = (index, row)
                break
        command_match = None
        if wake_match is not None:
            for index, row in enumerate(command_events):
                if index in matched_command_indices:
                    continue
                center = float(row["window_center_seconds"])
                if command_start - 0.5 <= center <= command_end + 3.0:
                    command_match = (index, row)
                    break
        if wake_match is not None:
            matched_wake_indices.add(wake_match[0])
        if command_match is not None:
            matched_command_indices.add(command_match[0])
        woke = wake_match is not None
        intent_correct = bool(
            command_match is not None
            and command_match[1].get("intent") == str(event["expected_intent"])
        )
        per_event.append(
            {
                "start_seconds": wake_start,
                "cabin_snr_db": float(event["cabin_snr_db"]),
                "expected_intent": str(event["expected_intent"]),
                "wake_detected": woke,
                "wake_latency_seconds": (
                    float(wake_match[1]["window_center_seconds"]) - wake_start if woke else None
                ),
                "intent_correct": intent_correct,
                "predicted_intent": (
                    command_match[1].get("intent") if command_match is not None else None
                ),
                "wake_source": event["wake_source"],
                "command_source": event["command_source"],
            }
        )

    false_wakes = [row for index, row in enumerate(wake_events) if index not in matched_wake_indices]
    false_actions = [row for index, row in enumerate(command_events) if index not in matched_command_indices]
    detected = [row for row in per_event if row["wake_detected"]]
    latencies = [float(row["wake_latency_seconds"]) for row in detected if row["wake_latency_seconds"] is not None]
    total_targets = len(per_event)
    result = {
        "benchmark": "continuous_stream_eval",
        "device": str(device),
        "config": str(resolve_path(args.config)),
        "duration_seconds": duration,
        "duration_hours": duration / 3600.0,
        "hop_seconds": float(args.hop_seconds),
        "windows": window_count,
        "wall_seconds": elapsed,
        "real_time_factor": elapsed / max(duration, 1e-6),
        "target_events": total_targets,
        "target_wake_recall": (len(detected) / total_targets) if total_targets else None,
        "target_intent_accuracy_overall": (
            sum(1 for row in per_event if row["intent_correct"]) / total_targets if total_targets else None
        ),
        "target_intent_accuracy_conditioned_on_wake": (
            sum(1 for row in per_event if row["intent_correct"]) / len(detected) if detected else None
        ),
        "false_wake_events": len(false_wakes),
        "false_wake_rate_per_hour": len(false_wakes) * 3600.0 / max(duration, 1e-6),
        "false_action_events": len(false_actions),
        "false_action_rate_per_hour": len(false_actions) * 3600.0 / max(duration, 1e-6),
        "target_wake_latency_mean_seconds": float(np.mean(latencies)) if latencies else None,
        "target_wake_latency_p95_seconds": float(np.percentile(latencies, 95)) if latencies else None,
        "per_event": per_event,
        "false_wake_windows": [
            {"window_center_seconds": row["window_center_seconds"], "wake_probability": row["wake_probability"]}
            for row in false_wakes
        ],
        "false_action_windows": [
            {
                "window_center_seconds": row["window_center_seconds"],
                "intent": row.get("intent"),
                "intent_confidence": row.get("intent_confidence"),
            }
            for row in false_actions
        ],
        "events": events,
        "production_valid": False,
        "limitation": timeline["limitation"],
    }
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "artifact_dir": str(artifact_dir) if args.save_audio else None,
                "duration_hours": result["duration_hours"],
                "target_events": total_targets,
                "target_wake_recall": result["target_wake_recall"],
                "target_intent_overall": result["target_intent_accuracy_overall"],
                "target_intent_conditioned_on_wake": result["target_intent_accuracy_conditioned_on_wake"],
                "false_wake_rate_per_hour": result["false_wake_rate_per_hour"],
                "false_action_rate_per_hour": result["false_action_rate_per_hour"],
                "latency_mean": result["target_wake_latency_mean_seconds"],
                "real_time_factor": result["real_time_factor"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
