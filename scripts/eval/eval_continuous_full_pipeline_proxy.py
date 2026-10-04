#!/usr/bin/env python
"""Evaluate the complete in-car cascade on continuous synthetic scenes.

This evaluator is intentionally honest about the limitation of the public
corpora used by the project: Speech Commands contains ``marvin`` recordings
and Fluent Speech Commands contains intent recordings, but there is no
natural same-speaker recording containing both.  The script therefore builds
an explicitly labelled *cross-corpus full-cascade proxy*.

Unlike the ordinary one-second KWS and isolated intent tests, each scene is a
continuous multichannel stream containing:

* a target wake word followed by a labelled command;
* a passenger wake word that must not trigger an action;
* passenger non-wake speech;
* a CAVEMOVE multichannel cabin-noise bed.

The actual runtime path is used for every overlapping window:
Stage 1 -> presence -> temporal KWS -> post-wake command capture -> intent.
Results must not be called a real-cabin or same-speaker benchmark until
recordings with that structure are available.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import soundfile as sf
import torch

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.io.audio import read_mono, read_multichannel
from car_tse.models.kws_model import KWSModel
from car_tse.models.presence_head import PresenceHead
from car_tse.models.stage1_system import Stage1System
from car_tse.pipeline.in_car_pipeline import InCarPipeline, PipelineState, normalize_extracted_waveform
from car_tse.train.utils import get_device, load_checkpoint, write_metrics
from eval_continuous_kws import prepare_channels


def resolve_path(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def speaker_key(path: Path) -> str:
    return path.stem.split("_nohash_")[0]


def fit_start(wav: np.ndarray, samples: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    out = np.zeros(max(1, int(samples)), dtype=np.float32)
    out[: min(len(wav), len(out))] = wav[: len(out)]
    return out


def rms(wav: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(wav, dtype=np.float32) ** 2) + 1.0e-8))


def add_at(destination: np.ndarray, source: np.ndarray, start: int) -> None:
    if start >= len(destination) or len(source) == 0:
        return
    start = max(0, int(start))
    end = min(len(destination), start + len(source))
    destination[start:end] += source[: end - start]


def select_noise_files(root: Path, minimum_seconds: float) -> list[Path]:
    selected: dict[str, tuple[float, Path]] = {}
    for path in sorted(root.rglob("*.wav")):
        if "noise" not in path.parts and "ventilation" not in path.parts:
            continue
        try:
            info = sf.info(str(path))
        except (OSError, RuntimeError):
            continue
        if info.duration < minimum_seconds:
            continue
        vehicle = next(
            (name for name in ("AlfaRomeo_146", "Smart_forfour", "Volkswagen_Golf") if name in path.parts),
            "unknown",
        )
        current = selected.get(vehicle)
        if current is None or info.duration > current[0]:
            selected[vehicle] = (float(info.duration), path)
    return [selected[key][1] for key in sorted(selected)]


def load_pipeline(cfg: dict[str, Any], paths: dict[str, Any], device: torch.device) -> InCarPipeline:
    stage1_cfg_path = resolve_path(cfg["stage1_config"])
    stage1_cfg = load_stage_config(stage1_cfg_path, REPO_ROOT / "configs" / "paths.yaml")
    pretrained = resolve_path(stage1_cfg["pretrained_checkpoint"])
    stage1 = Stage1System(
        spex_cfg=stage1_cfg["spex"],
        sample_rate=int(stage1_cfg["sample_rate"]),
        use_cabin_frontend=True,
        num_mics=int(cfg.get("num_mics", 4)),
        frontend_cfg=stage1_cfg.get("frontend"),
        pretrained_dir=str(pretrained),
        backend=stage1_cfg.get("spex_backend", "transfer_spexplus_16k"),
    ).to(device)
    load_checkpoint(resolve_path(Path(paths["checkpoint_root"]) / cfg["checkpoints"]["spex"]), stage1)

    label_map_path = Path(paths["manifest_root"]) / "meta" / "intent_label_map.json"
    raw_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    label_map = {str(value): key for key, value in raw_map.items()}

    encoder_checkpoint = cfg.get("intent_encoder_checkpoint")
    if encoder_checkpoint:
        encoder_checkpoint = Path(encoder_checkpoint)
        if not encoder_checkpoint.is_absolute():
            encoder_checkpoint = Path(paths["checkpoint_root"]) / encoder_checkpoint
        encoder_checkpoint = resolve_path(encoder_checkpoint)

    secondary_kws_checkpoint = cfg.get("checkpoints", {}).get("kws_secondary")
    if secondary_kws_checkpoint:
        secondary_kws_checkpoint = resolve_path(
            Path(paths["checkpoint_root"]) / secondary_kws_checkpoint
        )
    target_kws_checkpoint = cfg.get("checkpoints", {}).get("target_kws")
    if target_kws_checkpoint:
        target_kws_checkpoint = resolve_path(
            Path(paths["checkpoint_root"]) / target_kws_checkpoint
        )

    pipeline = InCarPipeline.from_checkpoints(
        stage1,
        resolve_path(Path(paths["checkpoint_root"]) / cfg["checkpoints"]["presence"]),
        resolve_path(Path(paths["checkpoint_root"]) / cfg["checkpoints"]["kws"]),
        resolve_path(Path(paths["checkpoint_root"]) / cfg["checkpoints"]["intent"]),
        kws_sample_rate=int(cfg["sample_rate"]),
        kws_num_classes=int(cfg.get("kws_num_classes", 2)),
        intent_sample_rate=int(cfg["sample_rate"]),
        intent_type=str(cfg.get("intent_type", "crnn")),
        intent_encoder=str(cfg.get("intent_encoder", "facebook/wav2vec2-base-960h")),
        intent_encoder_checkpoint=encoder_checkpoint,
        intent_num_classes=int(cfg.get("intent_num_classes", 31)),
        presence_threshold=float(cfg.get("presence_threshold", 0.5)),
        kws_threshold=float(cfg.get("kws_threshold", 0.5)),
        kws_secondary_checkpoint=secondary_kws_checkpoint,
        target_kws_checkpoint=target_kws_checkpoint,
        kws_fusion_mode=str(cfg.get("kws_fusion_mode", "mean")),
        kws_fusion_alpha=float(cfg.get("kws_fusion_alpha", 0.5)),
        wake_class_id=int(cfg.get("wake_class_id", 1)),
        kws_input_seconds=float(cfg.get("kws_input_seconds", 1.0)),
        kws_smoothing_window=int(cfg.get("kws_smoothing_window", 1)),
        kws_required_hits=int(cfg.get("kws_required_hits", 1)),
        kws_refractory_calls=int(cfg.get("kws_refractory_calls", 0)),
        intent_input_seconds=float(cfg.get("intent_input_seconds", 2.0)),
        command_timeout_calls=int(cfg.get("command_timeout_calls", 4)),
        command_presence_threshold=(
            float(cfg["command_presence_threshold"])
            if cfg.get("command_presence_threshold") is not None
            else None
        ),
        speaker_gate_threshold=(
            float(cfg["speaker_gate_threshold"])
            if cfg.get("speaker_gate_threshold") is not None
            else None
        ),
        speaker_gate_backend=str(cfg.get("speaker_gate_backend", "spex")),
        presence_type=str(cfg.get("presence_type", "scalar")),
        presence_sample_rate=int(cfg.get("presence_sample_rate", cfg["sample_rate"])),
        presence_n_mels=int(cfg.get("presence_n_mels", 40)),
        presence_hidden=int(cfg.get("presence_hidden", 64)),
        presence_n_fft=int(cfg.get("presence_n_fft", 512)),
        presence_hop=int(cfg.get("presence_hop", 160)),
        presence_top_fraction=float(cfg.get("presence_top_fraction", 0.20)),
        label_map=label_map,
        device=device,
    )
    return pipeline


def build_scene(
    target_wake: np.ndarray,
    target_command: np.ndarray,
    passenger_wake: np.ndarray,
    passenger_nonwake: np.ndarray,
    cabin_noise: np.ndarray,
    sample_rate: int,
    cabin_snr_db: float,
    passenger_snr_db: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build one 14-second continuous scene and its ground-truth components."""

    duration_seconds = 14.0
    total_samples = int(round(duration_seconds * sample_rate))
    target = np.zeros(total_samples, dtype=np.float32)
    passenger = np.zeros(total_samples, dtype=np.float32)

    wake_start = int(round(1.0 * sample_rate))
    command_start = int(round(2.1 * sample_rate))
    passenger_wake_start = int(round(6.0 * sample_rate))
    passenger_nonwake_start = int(round(9.0 * sample_rate))
    add_at(target, target_wake, wake_start)
    add_at(target, target_command, command_start)
    add_at(passenger, passenger_wake, passenger_wake_start)
    add_at(passenger, passenger_nonwake, passenger_nonwake_start)

    target_rms = rms(target)
    passenger_scaled = passenger * (
        target_rms * (10.0 ** (-float(passenger_snr_db) / 20.0)) / max(rms(passenger), 1.0e-6)
    )
    speech = target + passenger_scaled

    noise = np.asarray(cabin_noise, dtype=np.float32)
    if noise.ndim == 1:
        noise = noise[:, None]
    if len(noise) < total_samples:
        repeats = int(np.ceil(total_samples / max(1, len(noise))))
        noise = np.tile(noise, (repeats, 1))
    noise = noise[:total_samples]
    noise_scale = target_rms * (10.0 ** (-float(cabin_snr_db) / 20.0)) / max(rms(noise), 1.0e-6)
    noise_scaled = noise * noise_scale
    multichannel = speech[:, None] + noise_scaled
    mixture = multichannel.mean(axis=1).astype(np.float32)
    truth = {
        "duration_seconds": duration_seconds,
        "target_wake_start_seconds": wake_start / sample_rate,
        "target_wake_end_seconds": (wake_start + len(target_wake)) / sample_rate,
        "target_command_start_seconds": command_start / sample_rate,
        "target_command_end_seconds": (command_start + len(target_command)) / sample_rate,
        "passenger_wake_start_seconds": passenger_wake_start / sample_rate,
        "passenger_nonwake_start_seconds": passenger_nonwake_start / sample_rate,
        "cabin_snr_db": float(cabin_snr_db),
        "passenger_snr_db": float(passenger_snr_db),
    }
    return mixture, multichannel.astype(np.float32), target, speech.astype(np.float32), truth


def save_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sample_rate, subtype="FLOAT")


def evaluate_scene(
    pipeline: InCarPipeline,
    mixture: np.ndarray,
    multichannel: np.ndarray,
    enrollment: np.ndarray,
    truth: dict[str, Any],
    sample_rate: int,
    stage1_seconds: float,
    hop_seconds: float,
    device: torch.device,
) -> dict[str, Any]:
    window_samples = int(round(stage1_seconds * sample_rate))
    hop_samples = max(1, int(round(hop_seconds * sample_rate)))
    state = PipelineState()
    windows: list[dict[str, Any]] = []
    wake_events: list[dict[str, Any]] = []
    command_events: list[dict[str, Any]] = []

    for start in range(0, max(1, len(mixture)), hop_samples):
        end = min(start + window_samples, len(mixture))
        mix_window = np.zeros(window_samples, dtype=np.float32)
        mc_window = np.zeros((multichannel.shape[1], window_samples), dtype=np.float32)
        if end > start:
            mix_window[: end - start] = mixture[start:end]
            mc_window[:, : end - start] = multichannel[start:end].T
        mix_tensor = torch.from_numpy(mix_window).to(device)
        enrollment_tensor = torch.from_numpy(enrollment).to(device)
        mc_tensor = torch.from_numpy(mc_window).to(device)
        with torch.inference_mode():
            output, state = pipeline.forward(mix_tensor, enrollment_tensor, state, multichannel=mc_tensor)
            presence_probability = float(
                output.presence_probability
                if output.presence_probability is not None
                else pipeline.presence_probability(output.s1).mean().cpu()
            )
            wake_probability = output.wake_probability
        window_start = start / sample_rate
        window_center = window_start + stage1_seconds / 2.0
        row: dict[str, Any] = {
            "window_start_seconds": window_start,
            "window_center_seconds": window_center,
            "presence_probability": presence_probability,
            "speaker_similarity": output.speaker_similarity,
            "wake_probability": wake_probability,
            "accepted": bool(output.accepted),
            "wake_detected": bool(output.wake_detected),
            "command_pending": bool(output.command_pending),
            "intent_ready": bool(output.intent_ready),
            "intent": None,
            "intent_confidence": None,
            "action": None,
        }
        if output.intent_logits is not None:
            probabilities = torch.softmax(output.intent_logits, dim=-1)[0]
            predicted_id = str(int(probabilities.argmax().item()))
            row["intent_id"] = predicted_id
            row["intent"] = pipeline.label_map.get(predicted_id, predicted_id) if pipeline.label_map else predicted_id
            row["intent_confidence"] = float(probabilities.max().cpu())
            row["action"] = output.action.action if output.action is not None else None
        windows.append(row)
        if output.wake_detected:
            wake_events.append(row)
        if output.intent_ready:
            command_events.append(row)

    target_start = float(truth["target_wake_start_seconds"])
    target_end = float(truth["target_wake_end_seconds"])
    command_intent = str(truth["expected_intent"])
    target_wake_events = [
        row for row in wake_events
        if row["window_center_seconds"] >= target_start - 0.25
        and row["window_center_seconds"] <= target_end + 3.0
    ]
    false_events = [row for row in wake_events if row not in target_wake_events]
    target_wake_event = target_wake_events[0] if target_wake_events else None
    command_start = float(truth["target_command_start_seconds"])
    command_end = float(truth["target_command_end_seconds"])
    target_command_events = [
        row for row in command_events
        if row["window_center_seconds"] >= command_start - 0.5
        and row["window_center_seconds"] <= command_end + 3.0
    ]
    target_command_event = target_command_events[0] if target_command_events else None
    false_command_events = [row for row in command_events if row not in target_command_events]
    intent_correct = bool(target_command_event and target_command_event.get("intent") == command_intent)
    target_wake_detected = target_wake_event is not None
    latency = None
    if target_wake_event is not None:
        latency = float(target_wake_event["window_center_seconds"] - target_start)
    duration = float(truth["duration_seconds"])
    return {
        "duration_seconds": duration,
        "windows": windows,
        "wake_events": wake_events,
        "command_events": command_events,
        "target_wake_detected": target_wake_detected,
        "target_intent_correct": intent_correct,
        "target_wake_event": target_wake_event,
        "target_command_event": target_command_event,
        "false_wake_events": len(false_events),
        "false_wake_rate_per_hour": len(false_events) * 3600.0 / max(duration, 1.0e-6),
        "false_action_events": len(false_command_events),
        "false_action_rate_per_hour": len(false_command_events) * 3600.0 / max(duration, 1.0e-6),
        "target_wake_latency_seconds": latency,
        "expected_intent": command_intent,
        "predicted_intent": target_command_event.get("intent") if target_command_event else None,
        "passenger_wake_ground_truth_present": True,
        "truth": truth,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/eval_end_to_end_cavemove_wav2vec_ft8_kws_hybrid_presence90_norm_smooth.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.yaml")
    parser.add_argument("--wake-manifest", type=Path, default=REPO_ROOT / "manifests/test/stage3_kws_cabin_test.jsonl")
    parser.add_argument("--intent-manifest", type=Path, default=REPO_ROOT / "manifests/official/stage4_intent_fsc_test.jsonl")
    parser.add_argument("--noise-root", type=Path, default=REPO_ROOT / "data/raw/cavemove/extracted/cavemove_dataset")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "logs/continuous_full_pipeline_proxy.json")
    parser.add_argument("--artifact-dir", type=Path, default=REPO_ROOT / "reports/artifacts/continuous_full_pipeline_proxy")
    parser.add_argument("--scenes", type=int, default=4)
    parser.add_argument("--cabin-snr-db", nargs="+", type=float, default=[-5.0, 0.0])
    parser.add_argument("--passenger-snr-db", type=float, default=0.0)
    parser.add_argument("--hop-seconds", type=float, default=1.0)
    parser.add_argument("--presence-threshold", type=float, default=None)
    parser.add_argument("--kws-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    cfg = load_stage_config(args.config, args.paths)
    if args.presence_threshold is not None:
        cfg["presence_threshold"] = float(args.presence_threshold)
    if args.kws_threshold is not None:
        cfg["kws_threshold"] = float(args.kws_threshold)
    device = get_device()
    pipeline = load_pipeline(cfg, cfg["paths"], device)
    sample_rate = int(cfg["sample_rate"])
    stage1_seconds = float(cfg.get("segment_seconds", 3.0))

    wake_rows = [
        row for row in load_jsonl(resolve_path(args.wake_manifest))
        if int(row.get("label", 0)) == 1 and not bool(row.get("noise_only", False))
    ]
    if len(wake_rows) < 2:
        raise RuntimeError("Need at least two positive Marvin recordings")
    wake_rows.sort(key=lambda row: str(row.get("wav", "")))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in wake_rows:
        grouped[speaker_key(resolve_path(row["wav"]))].append(row)
    speaker_groups = [rows for _, rows in sorted(grouped.items()) if rows]
    target_wake_rows = [rows[0] for rows in speaker_groups[: max(1, int(args.scenes))]]
    passenger_wake_row = next(
        row for row in wake_rows if speaker_key(resolve_path(row["wav"])) != speaker_key(resolve_path(target_wake_rows[0]["wav"]))
    )

    intent_rows = load_jsonl(resolve_path(args.intent_manifest))
    by_intent: dict[str, dict[str, Any]] = {}
    for row in intent_rows:
        by_intent.setdefault(str(row["intent"]), row)
    intent_rows = [by_intent[key] for key in sorted(by_intent)]
    if not intent_rows:
        raise RuntimeError("No intent examples found")

    negative_rows = [
        row for row in load_jsonl(resolve_path(args.wake_manifest))
        if int(row.get("label", 0)) == 0 and not bool(row.get("noise_only", False))
    ]
    if not negative_rows:
        raise RuntimeError("No non-wake passenger examples found")
    passenger_nonwake_row = negative_rows[0]
    noise_files = select_noise_files(resolve_path(args.noise_root), 20.0)
    if not noise_files:
        raise RuntimeError("No sufficiently long CAVEMOVE noise streams found")

    results: list[dict[str, Any]] = []
    saved_audio = False
    for scene_index in range(min(int(args.scenes), len(target_wake_rows))):
        wake_row = target_wake_rows[scene_index]
        intent_row = intent_rows[scene_index % len(intent_rows)]
        target_wake, _ = read_mono(resolve_path(wake_row["wav"]), sample_rate)
        target_command, _ = read_mono(resolve_path(intent_row["wav"]), sample_rate)
        passenger_wake, _ = read_mono(resolve_path(passenger_wake_row["wav"]), sample_rate)
        passenger_nonwake, _ = read_mono(resolve_path(passenger_nonwake_row["wav"]), sample_rate)
        target_wake = fit_start(target_wake, sample_rate)
        target_command = fit_start(target_command, 2 * sample_rate)
        passenger_wake = fit_start(passenger_wake, sample_rate)
        passenger_nonwake = fit_start(passenger_nonwake, int(1.5 * sample_rate))
        enrollment = fit_start(target_wake, int(round(stage1_seconds * sample_rate / 2.0)))

        for snr_index, cabin_snr_db in enumerate(args.cabin_snr_db):
            noise_path = noise_files[(scene_index + snr_index) % len(noise_files)]
            raw_noise, _ = read_multichannel(noise_path, sample_rate)
            cabin_noise = prepare_channels(raw_noise, list(cfg.get("cabin_noise_channels", [0, 2, 4, 6])), int(cfg.get("num_mics", 4)))
            mixture, multichannel, target, speech, truth = build_scene(
                target_wake,
                target_command,
                passenger_wake,
                passenger_nonwake,
                cabin_noise,
                sample_rate,
                float(cabin_snr_db),
                float(args.passenger_snr_db),
            )
            truth["expected_intent"] = str(intent_row["intent"])
            truth["target_wake_source"] = str(resolve_path(wake_row["wav"]))
            truth["target_command_source"] = str(resolve_path(intent_row["wav"]))
            truth["passenger_wake_source"] = str(resolve_path(passenger_wake_row["wav"]))
            truth["passenger_nonwake_source"] = str(resolve_path(passenger_nonwake_row["wav"]))
            truth["noise_source"] = str(noise_path)
            truth["speaker_consistent"] = False
            truth["speaker_consistency_note"] = (
                "The target wake clip and target command come from different public corpora; "
                "the command speaker is not verified to be the Marvin speaker."
            )
            metrics = evaluate_scene(
                pipeline,
                mixture,
                multichannel,
                enrollment,
                truth,
                sample_rate,
                stage1_seconds,
                float(args.hop_seconds),
                device,
            )
            metrics.update({"scene_index": scene_index, "snr_db": float(cabin_snr_db)})
            results.append(metrics)
            print(
                f"scene={scene_index} cabin_snr={cabin_snr_db:+.1f}: "
                f"wake={metrics['target_wake_detected']} "
                f"intent={metrics['target_intent_correct']} "
                f"false_events={metrics['false_wake_events']} "
                f"pred={metrics['predicted_intent']}",
                flush=True,
            )
            if not saved_audio:
                artifact_dir = resolve_path(args.artifact_dir)
                save_audio(artifact_dir / "cabin_input_mean.wav", mixture, sample_rate)
                save_audio(artifact_dir / "cabin_input_4ch.wav", multichannel, sample_rate)
                save_audio(artifact_dir / "target_wake_plus_command.wav", target, sample_rate)
                save_audio(artifact_dir / "speech_with_passenger.wav", speech, sample_rate)
                save_audio(artifact_dir / "enrollment.wav", enrollment, sample_rate)
                saved_audio = True

    total_duration = sum(float(row["duration_seconds"]) for row in results)
    target_count = len(results)
    wake_successes = sum(bool(row["target_wake_detected"]) for row in results)
    intent_successes = sum(bool(row["target_intent_correct"]) for row in results)
    false_events = sum(int(row["false_wake_events"]) for row in results)
    false_actions = sum(int(row["false_action_events"]) for row in results)
    latencies = [float(row["target_wake_latency_seconds"]) for row in results if row["target_wake_latency_seconds"] is not None]
    report = {
        "benchmark": "continuous_full_pipeline_cross_corpus_proxy",
        "device": str(device),
        "config": str(resolve_path(args.config)),
        "checkpoints": cfg["checkpoints"],
        "sample_rate": sample_rate,
        "stage1_context_seconds": stage1_seconds,
        "hop_seconds": float(args.hop_seconds),
        "effective_presence_threshold": float(cfg.get("presence_threshold", 0.5)),
        "effective_kws_threshold": float(cfg.get("kws_threshold", 0.5)),
        "scenes": target_count,
        "duration_seconds": total_duration,
        "duration_hours": total_duration / 3600.0,
        "target_wake_recall": wake_successes / max(target_count, 1),
        "target_intent_accuracy_overall": intent_successes / max(target_count, 1),
        "target_intent_accuracy_conditioned_on_wake": (
            sum(bool(row["target_intent_correct"]) for row in results if row["target_wake_detected"])
            / max(wake_successes, 1)
        ),
        "false_wake_events": false_events,
        "false_wake_rate_per_hour": false_events * 3600.0 / max(total_duration, 1.0e-6),
        "false_action_events": false_actions,
        "false_action_rate_per_hour": false_actions * 3600.0 / max(total_duration, 1.0e-6),
        "target_wake_latency_mean_seconds": float(np.mean(latencies)) if latencies else None,
        "target_wake_latency_p95_seconds": float(np.percentile(latencies, 95)) if latencies else None,
        "speaker_consistent": False,
        "production_valid": False,
        "limitation": (
            "This is a continuous full-cascade engineering proxy. It uses the same runtime path and includes "
            "target wake, labelled command, passenger wake, passenger non-wake speech, and multichannel CAVEMOVE "
            "noise, but the target wake and command are cross-corpus clips rather than a natural same-speaker pair."
        ),
        "scenes_detail": results,
        "artifacts": {
            "directory": str(resolve_path(args.artifact_dir)),
            "files": [
                "cabin_input_mean.wav",
                "cabin_input_4ch.wav",
                "target_wake_plus_command.wav",
                "speech_with_passenger.wav",
                "enrollment.wav",
            ],
        },
    }
    output = resolve_path(args.output)
    write_metrics(output, report)
    print(json.dumps({key: report[key] for key in (
        "device", "scenes", "duration_seconds", "target_wake_recall",
        "target_intent_accuracy_overall", "target_intent_accuracy_conditioned_on_wake",
        "false_wake_events", "false_action_events", "false_wake_rate_per_hour",
        "false_action_rate_per_hour", "target_wake_latency_mean_seconds",
        "target_wake_latency_p95_seconds", "speaker_consistent", "production_valid", "output",
    ) if key in report} | {"output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
