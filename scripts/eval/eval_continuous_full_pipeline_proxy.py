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
import torch.nn.functional as F

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.io.audio import read_mono, read_multichannel
from car_tse.models.kws_model import KWSModel
from car_tse.models.presence_head import PresenceHead
from car_tse.models.stage1_system import Stage1System
from car_tse.pipeline.in_car_pipeline import InCarPipeline, PipelineState, normalize_extracted_waveform
from car_tse.pipeline.windowing import full_context_window_starts
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


def row_speaker_id(row: dict[str, Any], path: Path) -> str:
    """Return the strongest speaker identity available for a manifest row."""

    value = row.get("speaker_id")
    if value is not None and str(value).strip():
        return str(value)
    parent = path.parent.name.strip()
    return parent or speaker_key(path)


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


def intervals_overlap(
    first_start: float,
    first_end: float,
    second_start: float,
    second_end: float,
) -> bool:
    """Return whether two half-open time intervals share audio."""

    return min(float(first_end), float(second_end)) > max(float(first_start), float(second_start))


def event_overlaps_label(
    row: dict[str, Any],
    label_start: float,
    label_end: float,
    stage1_seconds: float,
    tolerance_before: float,
    tolerance_after: float,
) -> bool:
    """Associate an emitted event using window support, not only its centre."""

    window_start = float(row["window_start_seconds"])
    window_end = float(row.get("window_end_seconds", window_start + stage1_seconds))
    return intervals_overlap(
        window_start,
        window_end,
        float(label_start) - float(tolerance_before),
        float(label_end) + float(tolerance_after),
    )


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
        use_cabin_frontend=bool(stage1_cfg.get("use_cabin_frontend", True)),
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
    target_kws_secondary_checkpoint = cfg.get("checkpoints", {}).get("target_kws_secondary")
    if target_kws_secondary_checkpoint:
        target_kws_secondary_checkpoint = resolve_path(
            Path(paths["checkpoint_root"]) / target_kws_secondary_checkpoint
        )
    compact_speaker_verifier_checkpoint = cfg.get("checkpoints", {}).get(
        "compact_speaker_verifier"
    )
    if compact_speaker_verifier_checkpoint:
        compact_speaker_verifier_checkpoint = resolve_path(
            Path(paths["checkpoint_root"]) / compact_speaker_verifier_checkpoint
        )

    pipeline = InCarPipeline.from_checkpoints(
        stage1,
        resolve_path(Path(paths["checkpoint_root"]) / cfg["checkpoints"]["presence"]),
        resolve_path(Path(paths["checkpoint_root"]) / cfg["checkpoints"]["kws"]),
        resolve_path(Path(paths["checkpoint_root"]) / cfg["checkpoints"]["intent"]),
        kws_sample_rate=int(cfg["sample_rate"]),
        kws_n_mels=int(cfg.get("kws_n_mels", cfg.get("model", {}).get("n_mels", 40))),
        kws_channels=cfg.get("kws_channels", cfg.get("model", {}).get("channels", [32, 64, 128])),
        kws_num_classes=int(cfg.get("kws_num_classes", 2)),
        intent_sample_rate=int(cfg["sample_rate"]),
        intent_type=str(cfg.get("intent_type", "crnn")),
        intent_normalize_input=bool(cfg.get("intent_normalize_input", False)),
        intent_bidirectional=bool(cfg.get("intent_bidirectional", True)),
        intent_n_mels=int(cfg.get("model", {}).get("n_mels", cfg.get("intent_n_mels", 80))),
        intent_hidden=int(cfg.get("model", {}).get("hidden", cfg.get("intent_hidden", 256))),
        intent_num_layers=int(cfg.get("model", {}).get("num_layers", cfg.get("intent_num_layers", 2))),
        intent_encoder=str(cfg.get("intent_encoder", "facebook/wav2vec2-base-960h")),
        intent_encoder_checkpoint=encoder_checkpoint,
        intent_num_classes=int(cfg.get("intent_num_classes", 31)),
        presence_threshold=float(cfg.get("presence_threshold", 0.5)),
        kws_threshold=float(cfg.get("kws_threshold", 0.5)),
        kws_secondary_checkpoint=secondary_kws_checkpoint,
        target_kws_checkpoint=target_kws_checkpoint,
        target_kws_secondary_checkpoint=target_kws_secondary_checkpoint,
        target_kws_n_mels=int(cfg.get("target_kws_n_mels", 40)),
        target_kws_channels=cfg.get(
            "target_kws_channels",
            [32, 64, 128],
        ),
        target_kws_speaker_embedding_dim=int(cfg.get("target_kws_speaker_embedding_dim", 256)),
        target_kws_hidden=int(cfg.get("target_kws_hidden", 256)),
        compact_speaker_verifier_checkpoint=compact_speaker_verifier_checkpoint,
        compact_speaker_verifier_embedding_dim=int(
            cfg.get("compact_speaker_verifier_embedding_dim", 128)
        ),
        compact_speaker_verifier_hidden=int(
            cfg.get("compact_speaker_verifier_hidden", 128)
        ),
        compact_speaker_verifier_input_seconds=float(
            cfg.get("compact_speaker_verifier_input_seconds", 1.0)
        ),
        compact_speaker_verifier_sample_rate=int(
            cfg.get("compact_speaker_verifier_sample_rate", 16000)
        ),
        target_kws_fusion_mode=str(cfg.get("target_kws_fusion_mode", "min")),
        kws_fusion_mode=str(cfg.get("kws_fusion_mode", "mean")),
        kws_fusion_alpha=float(cfg.get("kws_fusion_alpha", 0.5)),
        wake_class_id=int(cfg.get("wake_class_id", 1)),
        kws_input_seconds=float(cfg.get("kws_input_seconds", 1.0)),
        kws_smoothing_window=int(cfg.get("kws_smoothing_window", 1)),
        kws_required_hits=int(cfg.get("kws_required_hits", 1)),
        kws_refractory_calls=int(cfg.get("kws_refractory_calls", 0)),
        intent_input_seconds=float(cfg.get("intent_input_seconds", 2.0)),
        intent_window_offset_seconds=float(cfg.get("intent_window_offset_seconds", 0.0)),
        command_timeout_calls=int(cfg.get("command_timeout_calls", 4)),
        command_delay_calls=int(cfg.get("command_delay_calls", 0)),
        command_window_strategy=str(cfg.get("command_window_strategy", "fixed")),
        command_buffer_calls=int(cfg.get("command_buffer_calls", 2)),
        command_confidence_commit_threshold=float(
            cfg.get("command_confidence_commit_threshold", 0.90)
        ),
        command_endpoint_kws_hop_seconds=float(cfg.get("command_endpoint_kws_hop_seconds", 0.25)),
        command_endpoint_margin_seconds=float(cfg.get("command_endpoint_margin_seconds", 0.50)),
        command_presence_threshold=(
            float(cfg["command_presence_threshold"])
            if cfg.get("command_presence_threshold") is not None
            else None
        ),
        presence_activity_margin_threshold=(
            float(cfg["presence_activity_margin_threshold"])
            if cfg.get("presence_activity_margin_threshold") is not None
            else None
        ),
        presence_activity_margin_mode=str(cfg.get("presence_activity_margin_mode", "independent_top")),
        intent_confidence_threshold=(
            float(cfg["intent_confidence_threshold"])
            if cfg.get("intent_confidence_threshold") is not None
            else None
        ),
        speaker_gate_threshold=(
            float(cfg["speaker_gate_threshold"])
            if cfg.get("speaker_gate_threshold") is not None
            else None
        ),
        command_speaker_gate_threshold=(
            float(cfg["command_speaker_gate_threshold"])
            if cfg.get("command_speaker_gate_threshold") is not None
            else None
        ),
        speaker_gate_backend=str(cfg.get("speaker_gate_backend", "spex")),
        speaker_fusion_mode=str(cfg.get("speaker_fusion_mode", "none")),
        speaker_fusion_center=float(cfg.get("speaker_fusion_center", 0.0)),
        speaker_fusion_scale=float(cfg.get("speaker_fusion_scale", 0.10)),
        speaker_fusion_alpha=float(cfg.get("speaker_fusion_alpha", 0.5)),
        presence_type=str(cfg.get("presence_type", "scalar")),
        presence_sample_rate=int(cfg.get("presence_sample_rate", cfg["sample_rate"])),
        presence_n_mels=int(cfg.get("presence_n_mels", 40)),
        presence_hidden=int(cfg.get("presence_hidden", 64)),
        presence_n_fft=int(cfg.get("presence_n_fft", 512)),
        presence_hop=int(cfg.get("presence_hop", 160)),
        presence_speaker_embedding_dim=int(cfg.get("presence_speaker_embedding_dim", 256)),
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
    clean: bool = False,
    passenger_only: bool = False,
    overlap: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build one 14-second scene with optional passenger/target overlap."""

    duration_seconds = 14.0
    total_samples = int(round(duration_seconds * sample_rate))
    target = np.zeros(total_samples, dtype=np.float32)
    passenger = np.zeros(total_samples, dtype=np.float32)

    wake_start = int(round(1.0 * sample_rate))
    command_start = int(round(2.1 * sample_rate))
    if overlap:
        # Put both passenger utterances inside the target command interval.
        # This is deliberately a co-channel stress test: all microphone
        # channels receive the same summed speech, while the target identity
        # remains available through enrollment.
        passenger_wake_start = int(round(2.4 * sample_rate))
        passenger_nonwake_start = int(round(3.2 * sample_rate))
    else:
        passenger_wake_start = int(round(6.0 * sample_rate))
        passenger_nonwake_start = int(round(9.0 * sample_rate))
    add_at(target, target_wake, wake_start)
    add_at(target, target_command, command_start)
    if not clean or passenger_only or overlap:
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
    if clean or passenger_only or overlap:
        noise_scaled = np.zeros_like(noise)
    else:
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
        "clean": bool(clean),
        "passenger_only": bool(passenger_only),
        "overlap": bool(overlap),
        "passenger_present": not bool(clean),
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
    reference_speaker_embedding: torch.Tensor | None = None,
    bypass_stage1: bool = False,
) -> dict[str, Any]:
    window_samples = int(round(stage1_seconds * sample_rate))
    hop_samples = max(1, int(round(hop_seconds * sample_rate)))
    state = PipelineState(reference_speaker_embedding=reference_speaker_embedding)
    windows: list[dict[str, Any]] = []
    wake_events: list[dict[str, Any]] = []
    command_events: list[dict[str, Any]] = []

    # Match the streaming runner: do not score a synthetic zero-padded tail
    # as if it were a real audio window. Keep one padded window only when the
    # complete input is shorter than the model context.
    for start in full_context_window_starts(len(mixture), window_samples, hop_samples):
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
            output, state = pipeline.forward(
                mix_tensor,
                enrollment_tensor,
                state,
                multichannel=mc_tensor,
                pre_extracted=mix_tensor if bypass_stage1 else None,
            )
            presence_probability = float(
                output.presence_probability
                if output.presence_probability is not None
                else pipeline.presence_probability(
                    output.s1,
                    pipeline.stage1.speaker_embedding(enrollment_tensor.unsqueeze(0))
                    if getattr(pipeline.presence, "requires_enrollment", False)
                    else None,
                ).mean().cpu()
            )
            wake_probability = output.wake_probability
        window_start = start / sample_rate
        window_center = window_start + stage1_seconds / 2.0
        row: dict[str, Any] = {
            "window_start_seconds": window_start,
            "window_end_seconds": window_start + stage1_seconds,
            "window_center_seconds": window_center,
            "presence_probability": presence_probability,
            "presence_non_target_probability": output.presence_non_target_probability,
            "presence_activity_margin": output.presence_activity_margin,
            "speaker_similarity": output.speaker_similarity,
            "wake_probability": wake_probability,
            "accepted": bool(output.accepted),
            "wake_detected": bool(output.wake_detected),
            "command_pending": bool(output.command_pending),
            "intent_ready": bool(output.intent_ready),
            "command_suppressed": bool(output.command_suppressed),
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
        row
        for row in wake_events
        if event_overlaps_label(
            row,
            target_start,
            target_end,
            stage1_seconds,
            tolerance_before=0.25,
            tolerance_after=3.0,
        )
    ]
    target_wake_events.sort(key=lambda row: float(row["window_start_seconds"]))
    false_events = [row for row in wake_events if row not in target_wake_events]
    target_wake_event = target_wake_events[0] if target_wake_events else None
    command_start = float(truth["target_command_start_seconds"])
    command_end = float(truth["target_command_end_seconds"])
    target_command_events = [
        row
        for row in command_events
        if event_overlaps_label(
            row,
            command_start,
            command_end,
            stage1_seconds,
            tolerance_before=0.5,
            tolerance_after=3.0,
        )
    ]
    target_command_events.sort(key=lambda row: float(row["window_start_seconds"]))
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
        "target_wake_supporting_windows": len(target_wake_events),
        "target_command_supporting_windows": len(target_command_events),
        "false_wake_events": len(false_events),
        "false_wake_rate_per_hour": len(false_events) * 3600.0 / max(duration, 1.0e-6),
        "false_action_events": len(false_command_events),
        "false_action_rate_per_hour": len(false_command_events) * 3600.0 / max(duration, 1.0e-6),
        "target_wake_latency_seconds": latency,
        "expected_intent": command_intent,
        "predicted_intent": target_command_event.get("intent") if target_command_event else None,
        "passenger_wake_ground_truth_present": bool(truth.get("passenger_present", True)),
        "truth": truth,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/eval_end_to_end_cavemove_wav2vec_ft8_kws_hybrid_presence90_norm_smooth.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.yaml")
    parser.add_argument("--wake-manifest", type=Path, default=REPO_ROOT / "manifests/test/stage3_kws_cabin_test.jsonl")
    parser.add_argument(
        "--passenger-manifest",
        type=Path,
        default=None,
        help=(
            "Manifest supplying passenger wake and non-wake controls. If omitted, "
            "matched_speaker uses the original held-out cabin KWS manifest while "
            "other modes reuse --wake-manifest."
        ),
    )
    parser.add_argument("--intent-manifest", type=Path, default=REPO_ROOT / "manifests/official/stage4_intent_fsc_test.jsonl")
    parser.add_argument("--noise-root", type=Path, default=REPO_ROOT / "data/raw/cavemove/extracted/cavemove_dataset")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "logs/continuous_full_pipeline_proxy.json")
    parser.add_argument("--artifact-dir", type=Path, default=REPO_ROOT / "reports/artifacts/continuous_full_pipeline_proxy")
    parser.add_argument("--scenes", type=int, default=4)
    parser.add_argument(
        "--scene-sampling",
        choices=("speaker_groups", "all_rows", "matched_speaker"),
        default="speaker_groups",
        help=(
            "Choose the clean/proxy scene source. 'speaker_groups' preserves the "
            "historical one-row-per-Marvin-source behavior; 'all_rows' uses the "
            "sorted wake and FSC test rows so a large benchmark does not repeat "
            "the same command and wake clips. 'matched_speaker' requires explicit "
            "speaker_id fields in both manifests and pairs wake/command rows by "
            "identity. It is a synthetic same-speaker proxy, not a continuous "
            "recording."
        ),
    )
    parser.add_argument("--cabin-snr-db", nargs="+", type=float, default=[-5.0, 0.0])
    parser.add_argument("--passenger-snr-db", type=float, default=0.0)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Run a clean English control with no passenger speech or CAVEMOVE noise.",
    )
    parser.add_argument(
        "--passenger-only",
        action="store_true",
        help="Add passenger wake/non-wake speech but force the cabin-noise bed to zero.",
    )
    parser.add_argument(
        "--overlap",
        action="store_true",
        help=(
            "Place passenger wake/non-wake speech inside the target command interval "
            "and force the cabin-noise bed to zero. This is a synthetic co-channel "
            "overlap stress test."
        ),
    )
    parser.add_argument(
        "--clean-bypass-stage1",
        action="store_true",
        help="For clean diagnostics only, feed the clean waveform directly to downstream stages.",
    )
    parser.add_argument("--hop-seconds", type=float, default=1.0)
    parser.add_argument("--presence-threshold", type=float, default=None)
    parser.add_argument("--presence-activity-margin-threshold", type=float, default=None)
    parser.add_argument(
        "--presence-activity-margin-mode",
        choices=("independent_top", "target_top", "framewise_top", "mean"),
        default=None,
    )
    parser.add_argument("--kws-threshold", type=float, default=None)
    parser.add_argument("--intent-input-seconds", type=float, default=None)
    parser.add_argument(
        "--proxy-command-seconds",
        type=float,
        default=None,
        help="Length used when placing each FSC command in the synthetic scene; keep this at or above the source utterance duration.",
    )
    parser.add_argument("--intent-confidence-threshold", type=float, default=None)
    parser.add_argument("--speaker-gate-threshold", type=float, default=None)
    parser.add_argument(
        "--speaker-gate-backend",
        choices=("spex", "wav2vec", "compact", "auto"),
        default=None,
    )
    parser.add_argument("--intent-window-offset-seconds", type=float, default=None)
    parser.add_argument("--command-delay-calls", type=int, default=None)
    parser.add_argument("--command-timeout-calls", type=int, default=None)
    parser.add_argument(
        "--command-window-strategy",
        choices=(
            "fixed",
            "kws_trim",
            "kws_trim_latest",
            "confidence_buffered",
            "confidence_buffered_hybrid",
        ),
        default=None,
    )
    parser.add_argument("--command-buffer-calls", type=int, default=None)
    parser.add_argument("--command-confidence-commit-threshold", type=float, default=None)
    parser.add_argument("--command-endpoint-kws-hop-seconds", type=float, default=None)
    parser.add_argument("--command-endpoint-margin-seconds", type=float, default=None)
    parser.add_argument("--speaker-fusion-mode", choices=("none", "min", "product", "mean"), default=None)
    parser.add_argument("--speaker-fusion-center", type=float, default=None)
    parser.add_argument("--speaker-fusion-scale", type=float, default=None)
    parser.add_argument("--speaker-fusion-alpha", type=float, default=None)
    parser.add_argument(
        "--enrollment-source",
        choices=("wake", "command", "same_speaker_other"),
        default="wake",
        help=(
            "Diagnostic enrollment source. 'wake' is the cross-corpus proxy; "
            "'command' reuses the command clip; 'same_speaker_other' uses a "
            "different FSC utterance from the command speaker."
        ),
    )
    parser.add_argument(
        "--enrollment-bank-size",
        type=int,
        default=1,
        help=(
            "Number of short same-speaker utterances used to build the optional "
            "SpEx reference embedding bank. The first utterance still conditions "
            "Stage 1 extraction; values above 1 affect speaker-conditioned gates/KWS."
        ),
    )
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    cfg = load_stage_config(args.config, args.paths)
    if args.presence_threshold is not None:
        cfg["presence_threshold"] = float(args.presence_threshold)
    if args.presence_activity_margin_threshold is not None:
        cfg["presence_activity_margin_threshold"] = float(args.presence_activity_margin_threshold)
    if args.presence_activity_margin_mode is not None:
        cfg["presence_activity_margin_mode"] = str(args.presence_activity_margin_mode)
    if args.kws_threshold is not None:
        cfg["kws_threshold"] = float(args.kws_threshold)
    if args.intent_input_seconds is not None:
        cfg["intent_input_seconds"] = float(args.intent_input_seconds)
    if args.intent_confidence_threshold is not None:
        cfg["intent_confidence_threshold"] = float(args.intent_confidence_threshold)
    if args.speaker_gate_threshold is not None:
        cfg["speaker_gate_threshold"] = float(args.speaker_gate_threshold)
        cfg["command_speaker_gate_threshold"] = float(args.speaker_gate_threshold)
    if args.speaker_gate_backend is not None:
        cfg["speaker_gate_backend"] = str(args.speaker_gate_backend)
    if args.intent_window_offset_seconds is not None:
        cfg["intent_window_offset_seconds"] = float(args.intent_window_offset_seconds)
    if args.command_delay_calls is not None:
        cfg["command_delay_calls"] = int(args.command_delay_calls)
    if args.command_timeout_calls is not None:
        cfg["command_timeout_calls"] = int(args.command_timeout_calls)
    if args.command_window_strategy is not None:
        cfg["command_window_strategy"] = str(args.command_window_strategy)
    if args.command_buffer_calls is not None:
        cfg["command_buffer_calls"] = int(args.command_buffer_calls)
    if args.command_confidence_commit_threshold is not None:
        cfg["command_confidence_commit_threshold"] = float(
            args.command_confidence_commit_threshold
        )
    if args.command_endpoint_kws_hop_seconds is not None:
        cfg["command_endpoint_kws_hop_seconds"] = float(args.command_endpoint_kws_hop_seconds)
    if args.command_endpoint_margin_seconds is not None:
        cfg["command_endpoint_margin_seconds"] = float(args.command_endpoint_margin_seconds)
    if args.speaker_fusion_mode is not None:
        cfg["speaker_fusion_mode"] = str(args.speaker_fusion_mode)
    if args.speaker_fusion_center is not None:
        cfg["speaker_fusion_center"] = float(args.speaker_fusion_center)
    if args.speaker_fusion_scale is not None:
        cfg["speaker_fusion_scale"] = float(args.speaker_fusion_scale)
    if args.speaker_fusion_alpha is not None:
        cfg["speaker_fusion_alpha"] = float(args.speaker_fusion_alpha)
    device = get_device()
    pipeline = load_pipeline(cfg, cfg["paths"], device)
    sample_rate = int(cfg["sample_rate"])
    stage1_seconds = float(cfg.get("segment_seconds", 3.0))
    proxy_command_seconds = float(
        args.proxy_command_seconds
        if args.proxy_command_seconds is not None
        else cfg.get("proxy_command_seconds", cfg.get("intent_input_seconds", 2.0))
    )
    if proxy_command_seconds <= 0.0:
        raise ValueError("proxy command duration must be positive")

    wake_rows = [
        row for row in load_jsonl(resolve_path(args.wake_manifest))
        if int(row.get("label", 0)) == 1 and not bool(row.get("noise_only", False))
    ]
    if len(wake_rows) < 2:
        raise RuntimeError("Need at least two positive Marvin recordings")
    wake_rows.sort(key=lambda row: str(row.get("wav", "")))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in wake_rows:
        path = resolve_path(row["wav"])
        # Speech Commands rows have no speaker_id and use the stable utterance
        # hash as their source-speaker key.  The kNN-VC manifest carries an
        # explicit FSC speaker_id, which must take precedence for matching.
        key = str(row.get("speaker_id", "")).strip() or speaker_key(path)
        grouped[key].append(row)
    speaker_groups = [rows for _, rows in sorted(grouped.items()) if rows]
    target_wake_rows = [rows[0] for rows in speaker_groups[: max(1, int(args.scenes))]]
    scene_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    scene_wake_rows: list[dict[str, Any]]
    passenger_manifest_path = resolve_path(args.passenger_manifest) if args.passenger_manifest else (
        REPO_ROOT / "manifests/test/stage3_kws_cabin_test.jsonl"
        if args.scene_sampling == "matched_speaker"
        else resolve_path(args.wake_manifest)
    )
    passenger_rows = load_jsonl(passenger_manifest_path)
    passenger_wake_rows = [
        row for row in passenger_rows
        if int(row.get("label", 0)) == 1 and not bool(row.get("noise_only", False))
    ]
    if not passenger_wake_rows:
        raise RuntimeError(f"No passenger wake examples found in {passenger_manifest_path}")
    passenger_wake_row = next(
        row
        for row in passenger_wake_rows
        if (str(row.get("speaker_id", "")).strip() or speaker_key(resolve_path(row["wav"])))
        != (str(target_wake_rows[0].get("speaker_id", "")).strip() or speaker_key(resolve_path(target_wake_rows[0]["wav"])))
    )

    all_intent_rows = load_jsonl(resolve_path(args.intent_manifest))
    by_intent: dict[str, dict[str, Any]] = {}
    for row in all_intent_rows:
        by_intent.setdefault(str(row["intent"]), row)
    intent_rows = [by_intent[key] for key in sorted(by_intent)]
    if not intent_rows:
        raise RuntimeError("No intent examples found")
    if args.scene_sampling == "matched_speaker":
        # This mode is deliberately strict: a generated wake row is accepted
        # only when its explicit speaker_id is also present in FSC.  Falling
        # back to path-derived IDs would make the benchmark look matched when
        # it is actually the old cross-corpus proxy.
        wake_by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
        intent_by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in wake_rows:
            speaker_id = str(row.get("speaker_id", "")).strip()
            if speaker_id:
                wake_by_speaker[speaker_id].append(row)
        for row in all_intent_rows:
            speaker_id = str(row.get("speaker_id", "")).strip()
            if speaker_id:
                intent_by_speaker[speaker_id].append(row)
        matched_pair_groups: list[list[tuple[dict[str, Any], dict[str, Any]]]] = []
        for speaker_id in sorted(set(wake_by_speaker) & set(intent_by_speaker)):
            speaker_wakes = sorted(wake_by_speaker[speaker_id], key=lambda row: str(row.get("wav", "")))
            speaker_commands = sorted(intent_by_speaker[speaker_id], key=lambda row: str(row.get("wav", "")))
            speaker_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for index, command_row in enumerate(speaker_commands):
                speaker_pairs.append((speaker_wakes[index % len(speaker_wakes)], command_row))
            matched_pair_groups.append(speaker_pairs)
        # Interleave speakers so a finite prefix such as the 1,000-scene
        # benchmark does not accidentally become a three-speaker study simply
        # because one speaker has more FSC rows than another.
        for index in range(max((len(group) for group in matched_pair_groups), default=0)):
            for group in matched_pair_groups:
                if index < len(group):
                    scene_pairs.append(group[index])
        if not scene_pairs:
            raise RuntimeError(
                "matched_speaker requires explicit speaker_id overlap between wake and intent manifests"
            )
        scene_wake_rows = []
        seen_wake_paths: set[str] = set()
        for wake_row, _ in scene_pairs:
            wake_path = str(resolve_path(wake_row["wav"]))
            if wake_path not in seen_wake_paths:
                scene_wake_rows.append(wake_row)
                seen_wake_paths.add(wake_path)
        scene_intent_rows = [command_row for _, command_row in scene_pairs]
    else:
        scene_wake_rows = wake_rows if args.scene_sampling == "all_rows" else target_wake_rows
        scene_intent_rows = all_intent_rows if args.scene_sampling == "all_rows" else intent_rows
    if not scene_wake_rows or not scene_intent_rows:
        raise RuntimeError("No rows available for the requested scene sampling mode")

    negative_rows = [
        row for row in passenger_rows
        if int(row.get("label", 0)) == 0 and not bool(row.get("noise_only", False))
    ]
    if not negative_rows:
        raise RuntimeError("No non-wake passenger examples found")
    passenger_nonwake_row = negative_rows[0]
    if sum(bool(value) for value in (args.clean, args.passenger_only, args.overlap)) > 1:
        raise ValueError("--clean, --passenger-only, and --overlap are mutually exclusive")
    noise_files = [] if (args.clean or args.passenger_only or args.overlap) else select_noise_files(resolve_path(args.noise_root), 20.0)
    if not args.clean and not args.passenger_only and not args.overlap and not noise_files:
        raise RuntimeError("No sufficiently long CAVEMOVE noise streams found")

    results: list[dict[str, Any]] = []
    saved_audio = False
    requested_scene_count = max(0, int(args.scenes))
    # ``all_rows`` is intended for large command-coverage studies.  It may
    # cycle the finite held-out wake pool, but the report records the number of
    # unique source rows so repeated wake clips cannot be mistaken for
    # independent speaker evidence.  The historical speaker-group mode keeps
    # its one-row-per-source behavior.
    scene_count = (
        requested_scene_count
        if args.scene_sampling in ("all_rows", "matched_speaker")
        else min(requested_scene_count, len(scene_wake_rows))
    )
    for scene_index in range(scene_count):
        if scene_pairs:
            wake_row, intent_row = scene_pairs[scene_index % len(scene_pairs)]
        else:
            wake_row = scene_wake_rows[scene_index % len(scene_wake_rows)]
            intent_row = scene_intent_rows[scene_index % len(scene_intent_rows)]
        wake_source_path = resolve_path(wake_row["wav"])
        command_source_path = resolve_path(intent_row["wav"])
        wake_speaker_id = row_speaker_id(wake_row, wake_source_path)
        command_speaker_id = row_speaker_id(intent_row, command_source_path)
        speaker_identity_match = bool(
            wake_speaker_id and command_speaker_id and wake_speaker_id == command_speaker_id
        )
        target_wake, _ = read_mono(resolve_path(wake_row["wav"]), sample_rate)
        target_command, _ = read_mono(resolve_path(intent_row["wav"]), sample_rate)
        passenger_wake, _ = read_mono(resolve_path(passenger_wake_row["wav"]), sample_rate)
        passenger_nonwake, _ = read_mono(resolve_path(passenger_nonwake_row["wav"]), sample_rate)
        target_wake = fit_start(target_wake, sample_rate)
        target_command = fit_start(target_command, int(round(proxy_command_seconds * sample_rate)))
        passenger_wake = fit_start(passenger_wake, sample_rate)
        passenger_nonwake = fit_start(passenger_nonwake, int(1.5 * sample_rate))
        enrollment_source_path = str(resolve_path(wake_row["wav"]))
        enrollment_bank_paths: list[str] = []
        enrollment_bank_waveforms: list[np.ndarray] = []
        if args.enrollment_source == "wake":
            enrollment_source = target_wake
            enrollment_bank_waveforms = [target_wake]
            enrollment_bank_paths = [enrollment_source_path]
        elif args.enrollment_source == "command":
            enrollment_source = target_command
            enrollment_source_path = str(resolve_path(intent_row["wav"]))
            enrollment_bank_waveforms = [target_command]
            enrollment_bank_paths = [enrollment_source_path]
        else:
            command_speaker_id = str(intent_row.get("speaker_id", ""))
            candidates = [
                row for row in all_intent_rows
                if str(row.get("speaker_id", "")) == command_speaker_id
                and str(row.get("wav", "")) != str(intent_row.get("wav", ""))
            ]
            if not candidates:
                raise RuntimeError(
                    f"No different-utterance enrollment found for command speaker {command_speaker_id!r}"
                )
            candidates.sort(key=lambda row: str(row.get("wav", "")))
            selected_candidates = candidates[: max(1, int(args.enrollment_bank_size))]
            enrollment_source_path = str(resolve_path(selected_candidates[0]["wav"]))
            enrollment_bank_paths = [str(resolve_path(row["wav"])) for row in selected_candidates]
            enrollment_bank_waveforms = [
                read_mono(resolve_path(row["wav"]), sample_rate)[0]
                for row in selected_candidates
            ]
            enrollment_source = enrollment_bank_waveforms[0]
        enrollment = fit_start(enrollment_source, int(round(stage1_seconds * sample_rate / 2.0)))
        enrollment_bank_waveforms = [
            fit_start(waveform, int(round(stage1_seconds * sample_rate / 2.0)))
            for waveform in (enrollment_bank_waveforms or [enrollment])
        ]
        with torch.inference_mode():
            bank_embeddings: list[torch.Tensor] = []
            for waveform in enrollment_bank_waveforms:
                embedding = pipeline.stage1.speaker_embedding(
                    torch.from_numpy(waveform).to(device).unsqueeze(0)
                )
                if embedding is not None:
                    bank_embeddings.append(embedding)
            reference_speaker_embedding = None
            if bank_embeddings:
                reference_speaker_embedding = F.normalize(
                    torch.stack(bank_embeddings, dim=0).mean(dim=0), dim=-1
                )

        for snr_index, cabin_snr_db in enumerate(args.cabin_snr_db):
            noise_path = None
            if args.clean or args.passenger_only or args.overlap:
                cabin_noise = np.zeros(
                    (int(round(14.0 * sample_rate)), int(cfg.get("num_mics", 4))),
                    dtype=np.float32,
                )
            else:
                noise_path = noise_files[(scene_index + snr_index) % len(noise_files)]
                raw_noise, _ = read_multichannel(noise_path, sample_rate)
                cabin_noise = prepare_channels(
                    raw_noise,
                    list(cfg.get("cabin_noise_channels", [0, 2, 4, 6])),
                    int(cfg.get("num_mics", 4)),
                )
            mixture, multichannel, target, speech, truth = build_scene(
                target_wake,
                target_command,
                passenger_wake,
                passenger_nonwake,
                cabin_noise,
                sample_rate,
                float(cabin_snr_db),
                float(args.passenger_snr_db),
                clean=bool(args.clean),
                passenger_only=bool(args.passenger_only),
                overlap=bool(args.overlap),
            )
            truth["expected_intent"] = str(intent_row["intent"])
            truth["target_wake_source"] = str(resolve_path(wake_row["wav"]))
            truth["target_command_source"] = str(resolve_path(intent_row["wav"]))
            truth["passenger_wake_source"] = str(resolve_path(passenger_wake_row["wav"]))
            truth["passenger_nonwake_source"] = str(resolve_path(passenger_nonwake_row["wav"]))
            truth["noise_source"] = None if noise_path is None else str(noise_path)
            truth["enrollment_source"] = args.enrollment_source
            truth["enrollment_source_wav"] = enrollment_source_path
            truth["enrollment_bank_size"] = len(enrollment_bank_paths)
            truth["enrollment_bank_wavs"] = enrollment_bank_paths
            # Enrollment and command can share an FSC speaker while the
            # separate Marvin recording still comes from GSC. Only matching
            # wake/command identities justify this flag.
            truth["wake_speaker_id"] = wake_speaker_id
            truth["command_speaker_id"] = command_speaker_id
            truth["speaker_consistent"] = speaker_identity_match
            truth["speaker_consistency_note"] = (
                "The public manifest speaker IDs match for wake and command, but the clips are still assembled "
                "synthetically and are not a real-driver continuous recording."
                if speaker_identity_match
                else "The Marvin wake clip and command clip have different or unverified speaker identities; "
                "command-speaker enrollment does not make this a same-driver evaluation."
            )
            truth["enrollment_consistency_note"] = (
                "A different FSC utterance from the command speaker is used for enrollment."
                if args.enrollment_source == "same_speaker_other"
                else "Enrollment reuses the command clip and is leakage-prone."
                if args.enrollment_source == "command"
                else "Diagnostic enrollment reuses the wake clip."
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
                reference_speaker_embedding=reference_speaker_embedding,
                bypass_stage1=bool(args.clean_bypass_stage1),
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
    speaker_identity_match_count = sum(
        bool(row.get("truth", {}).get("speaker_consistent", False)) for row in results
    )
    speaker_consistent = bool(results) and speaker_identity_match_count == len(results)
    if speaker_consistent:
        speaker_limit = (
            "Wake and command manifest identities match, but scenes are still assembled synthetically and "
            "are not a real-driver continuous recording."
        )
    elif args.enrollment_source == "same_speaker_other":
        speaker_limit = (
            "A different FSC utterance from the command speaker is used for enrollment, but the Marvin wake "
            "clip is not the same verified speaker; this is not a same-driver evaluation."
        )
    elif args.enrollment_source == "command":
        speaker_limit = (
            "Enrollment reuses the command clip and is leakage-prone; the Marvin wake clip is not a verified "
            "same-speaker recording."
        )
    elif args.clean:
        speaker_limit = (
            "This clean control has no passenger, cabin noise, or playback and uses cross-corpus wake/command "
            "speakers; it is not same-driver or real-cabin evidence."
        )
    else:
        speaker_limit = (
            "The target wake and command are cross-corpus clips with different or unverified speaker identities; "
            "this is a full-cascade engineering proxy, not same-driver evidence."
        )
    report = {
        "benchmark": (
            "continuous_full_pipeline_clean_control"
            if args.clean
            else "continuous_full_pipeline_overlap_proxy"
            if args.overlap
            else "continuous_full_pipeline_cross_corpus_proxy"
        ),
        "device": str(device),
        "config": str(resolve_path(args.config)),
        "checkpoints": cfg["checkpoints"],
        "sample_rate": sample_rate,
        "stage1_context_seconds": stage1_seconds,
        "effective_intent_input_seconds": float(cfg.get("intent_input_seconds", 2.0)),
        "proxy_command_seconds": proxy_command_seconds,
        "hop_seconds": float(args.hop_seconds),
        "effective_presence_threshold": float(cfg.get("presence_threshold", 0.5)),
        "effective_presence_activity_margin_threshold": cfg.get("presence_activity_margin_threshold"),
        "effective_presence_activity_margin_mode": cfg.get("presence_activity_margin_mode", "independent_top"),
        "effective_kws_threshold": float(cfg.get("kws_threshold", 0.5)),
        "effective_command_presence_threshold": float(cfg.get("command_presence_threshold", cfg.get("presence_threshold", 0.5))),
        "effective_command_delay_calls": int(cfg.get("command_delay_calls", 0)),
        "effective_command_timeout_calls": int(cfg.get("command_timeout_calls", 4)),
        "effective_speaker_gate_threshold": cfg.get("speaker_gate_threshold"),
        "effective_command_speaker_gate_threshold": cfg.get("command_speaker_gate_threshold"),
        "effective_speaker_fusion_mode": cfg.get("speaker_fusion_mode", "none"),
        "effective_speaker_fusion_center": cfg.get("speaker_fusion_center", 0.0),
        "effective_speaker_fusion_scale": cfg.get("speaker_fusion_scale", 0.10),
        "effective_speaker_fusion_alpha": cfg.get("speaker_fusion_alpha", 0.5),
        "effective_intent_confidence_threshold": cfg.get("intent_confidence_threshold"),
        "effective_intent_window_offset_seconds": float(cfg.get("intent_window_offset_seconds", 0.0)),
        "effective_command_window_strategy": str(cfg.get("command_window_strategy", "fixed")),
        "effective_command_buffer_calls": int(cfg.get("command_buffer_calls", 2)),
        "effective_command_confidence_commit_threshold": float(
            cfg.get("command_confidence_commit_threshold", 0.90)
        ),
        "effective_command_endpoint_kws_hop_seconds": float(cfg.get("command_endpoint_kws_hop_seconds", 0.25)),
        "effective_command_endpoint_margin_seconds": float(cfg.get("command_endpoint_margin_seconds", 0.50)),
        "event_association": "interval_overlap_v2",
        "effective_target_kws_fusion_mode": cfg.get("target_kws_fusion_mode") if cfg.get("checkpoints", {}).get("target_kws_secondary") else None,
        "enrollment_source": args.enrollment_source,
        "enrollment_bank_size": max(1, int(args.enrollment_bank_size)),
        "scene_sampling": str(args.scene_sampling),
        "scene_pairing": "speaker_id" if scene_pairs else "independent_rows",
        "scene_pair_source_rows": len(scene_pairs) if scene_pairs else None,
        "scene_wake_source_rows": len(scene_wake_rows),
        "scene_intent_source_rows": len(scene_intent_rows),
        "scene_wake_unique_rows_used": len({
            str(row.get("truth", {}).get("target_wake_source")) for row in results
        }),
        "scene_intent_unique_rows_used": len({
            str(row.get("truth", {}).get("target_command_source")) for row in results
        }),
        "clean": bool(args.clean),
        "passenger_only": bool(args.passenger_only),
        "overlap": bool(args.overlap),
        "clean_bypass_stage1": bool(args.clean_bypass_stage1),
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
        "speaker_consistent": speaker_consistent,
        "speaker_identity_match_count": speaker_identity_match_count,
        "production_valid": False,
        "limitation": speaker_limit
        + (
            " Stage 1 is bypassed by the explicit evaluation hook."
            if args.clean and args.clean_bypass_stage1
            else ""
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
