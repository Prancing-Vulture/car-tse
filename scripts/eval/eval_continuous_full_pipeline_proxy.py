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
import csv
import hashlib
import json
import random
import sys
import subprocess
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
from car_tse.train.wandb_logging import log_evaluation
from eval_continuous_kws import prepare_channels


def resolve_path(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


class _IdentityStage1(torch.nn.Module):
    """Evaluation-only Stage 1 bypass that forwards the mixture unchanged."""

    def forward(
        self,
        mixture: torch.Tensor,
        enrollment: torch.Tensor,
        multichannel: torch.Tensor | None = None,
        playback_ref: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None, None, None]:
        del enrollment, multichannel, playback_ref
        return mixture, None, None, None


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


def overlaps(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    """Whether two half-open time intervals contain any common audio."""

    return min(float(end_a), float(end_b)) > max(float(start_a), float(start_b))


def is_target_wake_event(
    event: dict[str, Any],
    windows: list[dict[str, Any]],
    target_start: float,
    target_end: float,
    required_hits: int,
) -> bool:
    positive_window_ids = [int(index) for index in event.get("wake_positive_support_window_ids", [])]
    # The runtime event has already passed its multi-hit confirmation rule.
    # Attribute it to the target when at least one positive crop in that
    # confirmation support contains labeled wake audio. Requiring every hit
    # crop to overlap the label makes short wake segments impossible to count
    # at coarse hops (e.g. a 1 s hop around a 1 s wake segment).
    if len(positive_window_ids) < max(1, int(required_hits)):
        return False
    return any(
        overlaps(
            float(windows[window_id]["kws_crop_start_seconds"]),
            float(windows[window_id]["kws_crop_end_seconds"]),
            target_start,
            target_end,
        )
        for window_id in positive_window_ids
    )


def is_target_command_event(
    event: dict[str, Any],
    target_wake_event_id: str | None,
    command_start: float,
    command_end: float,
) -> bool:
    return bool(
        target_wake_event_id is not None
        and event.get("associated_wake_event_id") == target_wake_event_id
        and float(event["decision_available_seconds"]) >= float(command_start)
        and float(event["decision_available_seconds"]) <= float(command_end) + 3.0
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path | str) -> str:
    """Return a portable path without leaking the runner's local username."""

    value = Path(path).resolve()
    try:
        return value.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return value.name


def source_id(path: Path | str) -> dict[str, str]:
    value = Path(path)
    return {
        "speaker_id": speaker_key(value),
        "source_class": value.parent.name,
        "recording": value.name,
    }


def provenance_record(label: str, path: Path) -> dict[str, Any]:
    path = path.resolve()
    record: dict[str, Any] = {"label": label, "artifact": display_path(path), "exists": path.exists()}
    if path.is_file():
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = file_sha256(path)
    return record


def build_provenance(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    scene_results: list[dict[str, Any]],
) -> dict[str, Any]:
    inputs = [
        ("pipeline_config", resolve_path(args.config)),
        ("paths_config", resolve_path(args.paths)),
        ("stage1_config", resolve_path(cfg["stage1_config"])),
        ("wake_manifest", resolve_path(args.wake_manifest)),
        ("intent_manifest", resolve_path(args.intent_manifest)),
        ("evaluator_script", Path(__file__).resolve()),
    ]
    checkpoints: list[tuple[str, Path]] = []
    checkpoint_root = resolve_path(Path(cfg["paths"]["checkpoint_root"]))
    for name, value in sorted(cfg.get("checkpoints", {}).items()):
        path = Path(value)
        checkpoints.append((name, path if path.is_absolute() else checkpoint_root / path))
    encoder_checkpoint = cfg.get("intent_encoder_checkpoint")
    if encoder_checkpoint:
        path = Path(encoder_checkpoint)
        checkpoints.append((
            "intent_encoder",
            path if path.is_absolute() else checkpoint_root / path,
        ))
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {
        "git_revision": revision,
        "git_worktree_dirty": dirty,
        "inputs": [provenance_record(name, path) for name, path in inputs],
        "checkpoints": [provenance_record(name, path) for name, path in checkpoints],
        "selected_noise_recording_ids": sorted({
            scene["truth"]["noise_source"] for scene in scene_results
        }),
    }


def find_independent_enrollment(
    rows: list[dict[str, Any]],
    speaker: str,
    target_path: Path,
    additional_recordings: list[Path] | None = None,
) -> dict[str, Any]:
    """Pick a second speech recording from the target speaker for enrollment."""

    target_resolved = target_path.resolve()
    candidates = [
        row for row in rows
        if not bool(row.get("noise_only", False))
        and speaker_key(resolve_path(row["wav"])) == speaker
        and resolve_path(row["wav"]).resolve() != target_resolved
    ]
    candidates.extend(
        {"wav": str(path), "label": int(path.parent.name == "marvin"), "noise_only": False}
        for path in (additional_recordings or [])
        if path.resolve() != target_resolved
    )
    # Prefer a second Marvin recording when present, then use another speech
    # command from the same speaker. Either choice avoids target-file leakage.
    candidates.sort(key=lambda row: (int(row.get("label", 0)) != 1, str(row["wav"])))
    if not candidates:
        raise RuntimeError(
            f"No independent enrollment recording exists for target speaker {speaker}"
        )
    return candidates[0]


def build_speaker_recording_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*_nohash_*.wav"):
        if "_background_noise_" not in path.parts:
            index[speaker_key(path)].append(path)
    for paths in index.values():
        paths.sort(key=lambda path: (path.parent.name != "marvin", path.as_posix()))
    return index


def find_disjoint_wake(
    rows: list[dict[str, Any]], excluded_speakers: set[str]
) -> dict[str, Any]:
    candidates = [
        row for row in rows
        if speaker_key(resolve_path(row["wav"])) not in excluded_speakers
    ]
    if not candidates:
        raise RuntimeError("No passenger wake recording is speaker-disjoint from all target scenes")
    return candidates[0]


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

    configured_label_map = cfg.get("intent_label_map")
    if isinstance(configured_label_map, dict):
        # A numeric-key mapping is already in the runtime's class-id -> label
        # direction. Existing metadata maps retain their legacy label -> id
        # direction and are inverted as before.
        if configured_label_map and all(str(key).isdigit() for key in configured_label_map):
            label_map = {str(key): str(value) for key, value in configured_label_map.items()}
        else:
            label_map = {str(value): str(key) for key, value in configured_label_map.items()}
    elif configured_label_map:
        map_path = resolve_path(configured_label_map)
        raw_map = json.loads(map_path.read_text(encoding="utf-8"))
        label_map = {str(value): key for key, value in raw_map.items()}
    else:
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
    stage23_fusion_checkpoint = cfg.get("checkpoints", {}).get("stage23_fusion")
    if stage23_fusion_checkpoint:
        stage23_fusion_checkpoint = resolve_path(
            Path(paths["checkpoint_root"]) / stage23_fusion_checkpoint
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
        intent_head_hidden=int(cfg.get("intent_head_hidden", 512)),
        intent_action_threshold=(
            float(cfg["intent_action_threshold"])
            if cfg.get("intent_action_threshold") is not None
            else None
        ),
        presence_threshold=float(cfg.get("presence_threshold", 0.5)),
        kws_threshold=float(cfg.get("kws_threshold", 0.5)),
        kws_secondary_checkpoint=secondary_kws_checkpoint,
        target_kws_checkpoint=target_kws_checkpoint,
        stage23_fusion_checkpoint=stage23_fusion_checkpoint,
        stage23_fusion_threshold=float(cfg.get("stage23_fusion_threshold", 0.5)),
        stage23_joint_enabled=bool(cfg.get("stage23_joint_enabled", False)),
        kws_fusion_mode=str(cfg.get("kws_fusion_mode", "mean")),
        kws_fusion_alpha=float(cfg.get("kws_fusion_alpha", 0.5)),
        wake_class_id=int(cfg.get("wake_class_id", 1)),
        kws_input_seconds=float(cfg.get("kws_input_seconds", 1.0)),
        kws_smoothing_window=int(cfg.get("kws_smoothing_window", 1)),
        kws_required_hits=int(cfg.get("kws_required_hits", 1)),
        kws_refractory_calls=int(cfg.get("kws_refractory_calls", 0)),
        intent_input_seconds=float(cfg.get("intent_input_seconds", 2.0)),
        command_timeout_calls=int(cfg.get("command_timeout_calls", 4)),
        command_timeout_seconds=(
            float(cfg["command_timeout_seconds"])
            if cfg.get("command_timeout_seconds") is not None
            else None
        ),
        command_end_silence_calls=int(cfg.get("command_end_silence_calls", 2)),
        command_min_speech_seconds=float(cfg.get("command_min_speech_seconds", 0.25)),
        command_capture_preroll_seconds=float(cfg.get("command_capture_preroll_seconds", 0.25)),
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


def score_primary_kws(pipeline: InCarPipeline, audio: np.ndarray, device: torch.device) -> float:
    samples = max(1, int(round(pipeline.kws_sample_rate * pipeline.kws_input_seconds)))
    wav = fit_start(audio, samples)
    tensor = torch.from_numpy(wav).unsqueeze(0).to(device)
    with torch.inference_mode():
        probability = torch.softmax(pipeline.kws(tensor), dim=-1)[..., pipeline.wake_class_id]
    return float(probability.mean().cpu())


def score_intent_track(
    pipeline: InCarPipeline,
    audio: np.ndarray,
    sample_rate: int,
    device: torch.device,
) -> dict[str, Any]:
    """Score a labeled two-second command crop outside runtime gating."""

    samples = max(1, int(round(float(pipeline.intent_input_seconds) * sample_rate)))
    waveform = fit_start(audio, samples)
    tensor = torch.from_numpy(waveform).unsqueeze(0).to(device)
    tensor = normalize_extracted_waveform(tensor)
    tensor = pipeline._prepare_intent_input(tensor)
    with torch.inference_mode():
        logits = pipeline.intent(tensor)
        probabilities = torch.softmax(logits, dim=-1)[0]
    class_id = int(probabilities.argmax().item())
    class_key = str(class_id)
    intent = pipeline.label_map.get(class_key, class_key) if pipeline.label_map else class_key
    return {
        "predicted_intent": intent,
        "predicted_class_id": class_id,
        "confidence": float(probabilities[class_id].cpu()),
    }


def describe_intent_capture(
    pipeline: InCarPipeline,
    sample_rate: int,
    target_command_start_seconds: float,
    target_command_end_seconds: float,
    capture_segments: list[tuple[int, int]],
    timeline_mapping_complete: bool,
) -> dict[str, Any] | None:
    """Map the exact concatenated Stage-4 prefix to source-time segments.

    Command presence rejection can skip source-time intervals while the stream
    clock keeps advancing. Therefore a capture buffer cannot be represented by
    ``last_audio_end - buffer_length``; its appended source segments must be
    tracked individually.
    """

    if not capture_segments:
        return None
    buffer_samples = sum(max(0, int(end) - int(start)) for start, end in capture_segments)
    requested_samples = max(1, int(round(sample_rate * float(pipeline.intent_input_seconds))))
    used_samples = min(buffer_samples, requested_samples)
    target_start_sample = int(round(target_command_start_seconds * sample_rate))
    target_end_sample = int(round(target_command_end_seconds * sample_rate))
    overlap_samples = 0
    remaining_used_samples = used_samples
    intent_offset_samples = 0
    used_segments: list[dict[str, float | int]] = []
    for start_sample, end_sample in capture_segments:
        segment_samples = max(0, int(end_sample) - int(start_sample))
        take_samples = min(segment_samples, remaining_used_samples)
        if take_samples <= 0:
            break
        used_end_sample = int(start_sample) + take_samples
        overlap_samples += max(
            0,
            min(used_end_sample, target_end_sample)
            - max(int(start_sample), target_start_sample),
        )
        used_segments.append({
            "intent_offset_start_seconds": intent_offset_samples / sample_rate,
            "intent_offset_end_seconds": (intent_offset_samples + take_samples) / sample_rate,
            "source_start_seconds": int(start_sample) / sample_rate,
            "source_end_seconds": used_end_sample / sample_rate,
            "samples": take_samples,
        })
        remaining_used_samples -= take_samples
        intent_offset_samples += take_samples
    target_duration_samples = max(1, target_end_sample - target_start_sample)
    return {
        "buffer_duration_seconds": buffer_samples / sample_rate,
        "intent_input_duration_seconds": used_samples / sample_rate,
        "timeline_mapping_complete": bool(timeline_mapping_complete and remaining_used_samples == 0),
        "intent_input_source_segments": used_segments,
        "target_command_overlap_seconds": (
            overlap_samples / sample_rate if timeline_mapping_complete and remaining_used_samples == 0 else None
        ),
        "target_command_timeline_overlap_fraction": (
            overlap_samples / target_duration_samples
            if timeline_mapping_complete and remaining_used_samples == 0
            else None
        ),
        "buffer_samples": buffer_samples,
        "intent_input_samples": used_samples,
    }


def evaluate_scene(
    pipeline: InCarPipeline,
    mixture: np.ndarray,
    multichannel: np.ndarray,
    enrollment: np.ndarray,
    target_signal: np.ndarray,
    truth: dict[str, Any],
    sample_rate: int,
    stage1_seconds: float,
    hop_seconds: float,
    device: torch.device,
    force_presence_reject_window: int | None = None,
    capture_stage1_tracks: bool = False,
) -> dict[str, Any]:
    window_samples = int(round(stage1_seconds * sample_rate))
    hop_samples = max(1, int(round(hop_seconds * sample_rate)))
    state = PipelineState()
    windows: list[dict[str, Any]] = []
    wake_events: list[dict[str, Any]] = []
    command_events: list[dict[str, Any]] = []
    recent_kws_rows: list[dict[str, Any]] = []
    active_wake_event_id: str | None = None
    duration = float(truth["duration_seconds"])
    intent_capture_for_call: dict[str, Any] | None = None
    original_infer_command = pipeline._infer_command
    capture_segments: list[tuple[int, int]] = []
    capture_mapping_complete = True
    stage4_inputs: list[np.ndarray] = []
    stage1_audio_tracks: dict[str, dict[str, Any]] = {}
    stage4_track_predictions: dict[str, Any] = {}
    wake_start_sample = int(round(float(truth["target_wake_start_seconds"]) * sample_rate))
    wake_end_sample = int(round(float(truth["target_wake_end_seconds"]) * sample_rate))
    command_start_sample = int(round(float(truth["target_command_start_seconds"]) * sample_rate))
    command_end_sample = int(round(float(truth["target_command_end_seconds"]) * sample_rate))
    # Choose one model context per event. Wake context starts at the wake onset
    # (matching the runtime's KWS crop); command context centers the full
    # labeled command in the Stage 1 window.
    wake_context_start = max(0, int(round(wake_start_sample / hop_samples)) * hop_samples)
    command_context_ideal = (command_start_sample + command_end_sample - window_samples) / 2.0
    command_context_start = max(0, int(round(command_context_ideal / hop_samples)) * hop_samples)
    active_window_index = -1
    original_presence_forward = pipeline.presence.forward

    def controlled_presence_forward(s1: torch.Tensor) -> torch.Tensor:
        if active_window_index == force_presence_reject_window:
            # Deterministically exercise the real runtime's reject path for
            # exactly one window; all other windows use the trained Stage 2.
            return torch.full_like(original_presence_forward(s1), -20.0)
        return original_presence_forward(s1)

    pipeline.presence.forward = controlled_presence_forward

    def capture_stage4_input(_module, inputs):
        if inputs and isinstance(inputs[0], torch.Tensor):
            stage4_inputs.append(inputs[0].detach().cpu().squeeze(0).numpy().astype(np.float32))

    stage4_input_hook = pipeline.intent.register_forward_pre_hook(capture_stage4_input)

    def capture_intent_input(
        command_state: PipelineState,
        presence_probability: float,
        speaker_similarity: float | None,
        s1: torch.Tensor,
    ):
        nonlocal intent_capture_for_call, capture_mapping_complete
        buffer_samples = (
            int(command_state.command_audio.shape[-1])
            if command_state.command_audio is not None
            else 0
        )
        capture_start = command_state.command_capture_start_sample
        if capture_start is not None and buffer_samples > 0:
            capture_segments[:] = [(int(capture_start), int(capture_start) + buffer_samples)]
        else:
            capture_segments.clear()
        intent_capture_for_call = describe_intent_capture(
            pipeline,
            sample_rate,
            float(truth["target_command_start_seconds"]),
            float(truth["target_command_end_seconds"]),
            capture_segments,
            capture_mapping_complete and sum(end - start for start, end in capture_segments) == buffer_samples,
        )
        result = original_infer_command(
            command_state,
            presence_probability,
            speaker_similarity,
            s1,
        )
        capture_segments.clear()
        capture_mapping_complete = True
        return result

    # Evaluation-only wrapper: trace the actual rolling-buffer span at the
    # moment Stage 4 runs. It preserves production model calls and outputs.
    pipeline._infer_command = capture_intent_input

    for start in range(0, max(1, len(mixture)), hop_samples):
        active_window_index = len(windows)
        intent_capture_for_call = None
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
                audio_start_sample=start,
            )
        if capture_stage1_tracks and start in {wake_context_start, command_context_start}:
            s1_raw = output.s1.detach().to(device="cpu").squeeze(0).numpy().astype(np.float32)
            s1_normalized = normalize_extracted_waveform(output.s1).detach().to(device="cpu").squeeze(0).numpy().astype(np.float32)
            context_name = "wake" if start == wake_context_start else "command"
            context: dict[str, Any] = {
                "source_window_start_sample": int(start),
                "source_window_end_sample": int(start + window_samples),
                "input_mono": mix_window.copy(),
                "input_multichannel": mc_window.T.copy(),
                "stage1_output_raw": s1_raw,
                "stage1_output_normalized": s1_normalized,
            }
            if context_name == "wake":
                event_start, event_end = wake_start_sample, wake_end_sample
            else:
                event_start, event_end = command_start_sample, command_end_sample
            crop_start = max(0, event_start - start)
            crop_end = min(window_samples, event_end - start)
            context["event_crop_start_sample"] = int(crop_start)
            context["event_crop_end_sample"] = int(max(crop_start, crop_end))
            context["event_audio_source_sample_start"] = int(start + crop_start)
            context["event_audio_source_sample_end"] = int(start + max(crop_start, crop_end))
            stage1_audio_tracks[context_name] = context
            if context_name == "command":
                offset_start, offset_end = crop_start, crop_end
                extracted_command = s1_normalized[offset_start:offset_end]
                raw_command = mix_window[offset_start:offset_end]
                clean_command = target_signal[start + offset_start:start + offset_end]
                stage4_track_predictions = {
                    "raw_mixture": score_intent_track(pipeline, raw_command, sample_rate, device),
                    "clean_target_reference": score_intent_track(pipeline, clean_command, sample_rate, device),
                    "stage1_normalized_output": score_intent_track(pipeline, extracted_command, sample_rate, device),
                    "expected_intent": str(truth["expected_intent"]),
                }
        buffer_samples = (
            int(state.command_audio.shape[-1])
            if state.command_audio is not None
            else 0
        )
        if output.wake_detected:
            capture_segments.clear()
            capture_mapping_complete = True
            if state.command_capture_start_sample is not None and buffer_samples > 0:
                capture_segments.append((
                    int(state.command_capture_start_sample),
                    int(state.command_capture_start_sample) + buffer_samples,
                ))
        elif state.command_audio is None or not state.awaiting_command:
            capture_segments.clear()
            capture_mapping_complete = True
        elif state.command_capture_start_sample is not None and buffer_samples > 0:
            capture_segments[:] = [(
                int(state.command_capture_start_sample),
                int(state.command_capture_start_sample) + buffer_samples,
            )]
        presence_probability = float(
            output.presence_probability
            if output.presence_probability is not None
            else pipeline.presence_probability(output.s1).mean().cpu()
        )
        wake_probability = output.wake_probability
        window_start = start / sample_rate
        decision_available = min(window_start + stage1_seconds, duration)
        kws_crop_start = window_start
        kws_crop_end = min(window_start + float(pipeline.kws_input_seconds), duration)
        row: dict[str, Any] = {
            "window_index": len(windows),
            "window_start_seconds": window_start,
            "window_end_seconds": min((start + window_samples) / sample_rate, duration),
            "decision_available_seconds": decision_available,
            "kws_crop_start_seconds": kws_crop_start,
            "kws_crop_end_seconds": kws_crop_end,
            "presence_probability": presence_probability,
            "speaker_similarity": output.speaker_similarity,
            "wake_probability": wake_probability,
            "wake_model_probability": output.wake_model_probability,
            "accepted": bool(output.accepted),
            "wake_detected": bool(output.wake_detected),
            "command_pending": bool(output.command_pending),
            "intent_ready": bool(output.intent_ready),
            "presence_reject_forced": bool(len(windows) == force_presence_reject_window),
            "intent": None,
            "intent_confidence": None,
            "action": None,
        }
        if overlaps(kws_crop_start, kws_crop_end, float(truth["target_wake_start_seconds"]), float(truth["target_wake_end_seconds"])):
            crop_start_sample = int(round(kws_crop_start * sample_rate))
            crop_end_sample = crop_start_sample + max(
                1, int(round(float(pipeline.kws_input_seconds) * sample_rate))
            )
            raw_crop = fit_start(mixture[crop_start_sample:crop_end_sample], crop_end_sample - crop_start_sample)
            clean_crop = fit_start(target_signal[crop_start_sample:crop_end_sample], crop_end_sample - crop_start_sample)
            extracted_crop = fit_start(
                normalize_extracted_waveform(output.s1)[..., : max(1, int(round(float(pipeline.kws_input_seconds) * sample_rate)))][0]
                .detach().cpu().numpy(),
                max(1, int(round(float(pipeline.kws_input_seconds) * sample_rate))),
            )
            row["kws_primary_raw_mixture_probability"] = score_primary_kws(pipeline, raw_crop, device)
            row["kws_primary_clean_target_probability"] = score_primary_kws(pipeline, clean_crop, device)
            row["kws_primary_spex_output_probability"] = score_primary_kws(pipeline, extracted_crop, device)
        if output.intent_logits is not None:
            probabilities = torch.softmax(output.intent_logits, dim=-1)[0]
            predicted_id = str(int(probabilities.argmax().item()))
            row["intent_id"] = predicted_id
            row["intent"] = pipeline.label_map.get(predicted_id, predicted_id) if pipeline.label_map else predicted_id
            row["intent_confidence"] = float(probabilities.max().cpu())
            row["action"] = output.action.action if output.action is not None else None
        windows.append(row)
        if wake_probability is None:
            # Stage 2 rejection clears the runtime's confirmation buffer.
            recent_kws_rows.clear()
        else:
            recent_kws_rows.append(row)
            recent_kws_rows = recent_kws_rows[-max(1, int(pipeline.kws_smoothing_window)):]
        if output.wake_detected:
            event_id = f"wake_{len(wake_events):04d}"
            support = list(recent_kws_rows)
            positive_support = [
                score_row for score_row in support
                if score_row["wake_probability"] is not None
                and float(score_row["wake_probability"]) >= float(pipeline.kws_threshold)
            ]
            row["wake_event_id"] = event_id
            row["wake_support_window_ids"] = [int(item["window_index"]) for item in support]
            row["wake_positive_support_window_ids"] = [
                int(item["window_index"]) for item in positive_support
            ]
            wake_events.append(row)
            active_wake_event_id = event_id
            recent_kws_rows.clear()
        if output.intent_ready:
            row["intent_event_id"] = f"intent_{len(command_events):04d}"
            row["associated_wake_event_id"] = active_wake_event_id
            if intent_capture_for_call is not None:
                row["intent_input_capture"] = intent_capture_for_call
            command_events.append(row)
            active_wake_event_id = None

    pipeline._infer_command = original_infer_command
    pipeline.presence.forward = original_presence_forward
    stage4_input_hook.remove()

    target_start = float(truth["target_wake_start_seconds"])
    target_end = float(truth["target_wake_end_seconds"])
    command_intent = str(truth["expected_intent"])
    required_hits = max(1, int(pipeline.kws_required_hits))
    target_wake_events = [
        event for event in wake_events
        if is_target_wake_event(event, windows, target_start, target_end, required_hits)
    ]
    false_events = [row for row in wake_events if row not in target_wake_events]
    target_wake_event = target_wake_events[0] if target_wake_events else None
    command_start = float(truth["target_command_start_seconds"])
    command_end = float(truth["target_command_end_seconds"])
    target_command_events = [
        event for event in command_events
        if is_target_command_event(
            event,
            target_wake_event.get("wake_event_id") if target_wake_event else None,
            command_start,
            command_end,
        )
    ]
    target_command_event = target_command_events[0] if target_command_events else None
    false_command_events = [row for row in command_events if row not in target_command_events]
    intent_correct = bool(target_command_event and target_command_event.get("intent") == command_intent)
    target_wake_detected = target_wake_event is not None
    latency = None
    if target_wake_event is not None:
        latency = float(target_wake_event["decision_available_seconds"] - target_start)
    action_latency = None
    if target_command_event is not None:
        action_latency = float(target_command_event["decision_available_seconds"] - target_start)
    return {
        "duration_seconds": duration,
        "windows": windows,
        "wake_events": wake_events,
        "command_events": command_events,
        "stage4_input_waveforms": stage4_inputs,
        "stage1_audio_tracks": stage1_audio_tracks,
        "stage4_track_predictions": stage4_track_predictions,
        "target_wake_detected": target_wake_detected,
        "target_intent_correct": intent_correct,
        "target_wake_event": target_wake_event,
        "target_command_event": target_command_event,
        "false_wake_events": len(false_events),
        "false_action_events": len(false_command_events),
        "target_wake_latency_seconds": latency,
        "target_action_latency_seconds": action_latency,
        "target_command_event_associated_with_target_wake": target_command_event is not None,
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
    parser.add_argument("--passenger-manifest", type=Path, default=None, help="Optional separate split for passenger wake/non-wake speech")
    parser.add_argument("--intent-manifest", type=Path, default=REPO_ROOT / "manifests/official/stage4_intent_fsc_test.jsonl")
    parser.add_argument("--noise-root", type=Path, default=REPO_ROOT / "data/raw/cavemove/extracted/cavemove_dataset")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "logs/continuous_full_pipeline_proxy.json")
    parser.add_argument("--artifact-dir", type=Path, default=REPO_ROOT / "reports/artifacts/continuous_full_pipeline_proxy")
    parser.add_argument("--scenes", type=int, default=4)
    parser.add_argument("--all-positive-wake-recordings", action="store_true", help="Use every positive row in the wake manifest instead of one row per speaker")
    parser.add_argument("--save-stage1-tracks", action="store_true", help="Save aligned pre/post-SpEx audio contexts for each target wake and command")
    parser.add_argument(
        "--bypass-stage1",
        action="store_true",
        help="Evaluation-only ablation: pass the noisy mixture directly to Stages 2-4 without source separation",
    )
    parser.add_argument("--cabin-snr-db", nargs="+", type=float, default=[-5.0, 0.0])
    parser.add_argument("--passenger-snr-db", type=float, default=0.0)
    parser.add_argument("--hop-seconds", type=float, default=1.0)
    parser.add_argument(
        "--force-presence-reject-window",
        type=int,
        default=None,
        help="For buffer diagnosis only: force Stage 2 to reject this zero-based window index.",
    )
    parser.add_argument("--presence-threshold", type=float, default=None)
    parser.add_argument("--command-presence-threshold", type=float, default=None)
    parser.add_argument("--kws-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--wandb", action="store_true", help="Log metrics to W&B (offline when no API key is set)")
    parser.add_argument("--wandb-project", default="car-tse")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default=None)
    parser.add_argument("--wandb-dir", type=Path, default=None)
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    cfg = load_stage_config(args.config, args.paths)
    if args.presence_threshold is not None:
        cfg["presence_threshold"] = float(args.presence_threshold)
    if args.command_presence_threshold is not None:
        cfg["command_presence_threshold"] = float(args.command_presence_threshold)
    if args.kws_threshold is not None:
        cfg["kws_threshold"] = float(args.kws_threshold)
    device = get_device()
    pipeline = load_pipeline(cfg, cfg["paths"], device)
    if args.bypass_stage1:
        pipeline.stage1 = _IdentityStage1().to(device)
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
    if args.all_positive_wake_recordings:
        target_wake_rows = wake_rows
    else:
        target_wake_rows = [rows[0] for rows in speaker_groups[: max(1, int(args.scenes))]]
    target_speaker_ids = {speaker_key(resolve_path(row["wav"])) for row in target_wake_rows}
    enrollment_rows = [
        row for row in load_jsonl(resolve_path(args.wake_manifest))
        if not bool(row.get("noise_only", False))
    ]
    gsc_root = Path(cfg["paths"]["corpora"]["speech_commands_v2"])
    recordings_by_speaker = build_speaker_recording_index(resolve_path(gsc_root))
    passenger_rows = (
        load_jsonl(resolve_path(args.passenger_manifest))
        if args.passenger_manifest is not None
        else load_jsonl(resolve_path(args.wake_manifest))
    )
    passenger_wake_rows = [
        row for row in passenger_rows
        if int(row.get("label", 0)) == 1 and not bool(row.get("noise_only", False))
    ]
    passenger_wake_row = find_disjoint_wake(passenger_wake_rows, target_speaker_ids)

    intent_rows = load_jsonl(resolve_path(args.intent_manifest))
    by_intent: dict[str, dict[str, Any]] = {}
    for row in intent_rows:
        by_intent.setdefault(str(row["intent"]), row)
    intent_rows = [by_intent[key] for key in sorted(by_intent)]
    if not intent_rows:
        raise RuntimeError("No intent examples found")

    negative_rows = [
        row for row in passenger_rows
        if int(row.get("label", 0)) == 0 and not bool(row.get("noise_only", False))
    ]
    if not negative_rows:
        raise RuntimeError("No non-wake passenger examples found")
    passenger_wake_speaker = speaker_key(resolve_path(passenger_wake_row["wav"]))
    passenger_nonwake_candidates = [
        row for row in negative_rows
        if speaker_key(resolve_path(row["wav"])) not in target_speaker_ids | {passenger_wake_speaker}
    ]
    if not passenger_nonwake_candidates:
        raise RuntimeError("No passenger non-wake recording is speaker-disjoint from target and passenger-wake speakers")
    passenger_nonwake_row = passenger_nonwake_candidates[0]
    noise_files = select_noise_files(resolve_path(args.noise_root), 20.0)
    if not noise_files:
        raise RuntimeError("No sufficiently long CAVEMOVE noise streams found")

    results: list[dict[str, Any]] = []
    saved_audio = False
    scenes_to_run = (
        len(target_wake_rows)
        if args.all_positive_wake_recordings
        else min(int(args.scenes), len(target_wake_rows))
    )
    for scene_index in range(scenes_to_run):
        wake_row = target_wake_rows[scene_index]
        target_speaker = speaker_key(resolve_path(wake_row["wav"]))
        enrollment_row = find_independent_enrollment(
            enrollment_rows,
            target_speaker,
            resolve_path(wake_row["wav"]),
            recordings_by_speaker.get(target_speaker, []),
        )
        intent_row = intent_rows[scene_index % len(intent_rows)]
        target_wake, _ = read_mono(resolve_path(wake_row["wav"]), sample_rate)
        target_command, _ = read_mono(resolve_path(intent_row["wav"]), sample_rate)
        passenger_wake, _ = read_mono(resolve_path(passenger_wake_row["wav"]), sample_rate)
        passenger_nonwake, _ = read_mono(resolve_path(passenger_nonwake_row["wav"]), sample_rate)
        enrollment_audio, _ = read_mono(resolve_path(enrollment_row["wav"]), sample_rate)
        target_wake = fit_start(target_wake, sample_rate)
        target_command = fit_start(target_command, 2 * sample_rate)
        passenger_wake = fit_start(passenger_wake, sample_rate)
        passenger_nonwake = fit_start(passenger_nonwake, int(1.5 * sample_rate))
        enrollment = fit_start(enrollment_audio, int(round(stage1_seconds * sample_rate / 2.0)))

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
            truth["target_wake_source"] = source_id(resolve_path(wake_row["wav"]))
            truth["target_enrollment_source"] = source_id(resolve_path(enrollment_row["wav"]))
            truth["target_enrollment_is_independent_recording"] = (
                resolve_path(enrollment_row["wav"]).resolve()
                != resolve_path(wake_row["wav"]).resolve()
            )
            truth["target_command_source"] = Path(intent_row["wav"]).name
            truth["target_command_source_speaker_id"] = intent_row.get("speaker_id")
            truth["passenger_wake_source"] = source_id(resolve_path(passenger_wake_row["wav"]))
            truth["passenger_nonwake_source"] = source_id(resolve_path(passenger_nonwake_row["wav"]))
            truth["noise_source"] = noise_path.relative_to(resolve_path(args.noise_root)).as_posix()
            truth["passenger_speakers_disjoint_from_all_targets"] = (
                speaker_key(resolve_path(passenger_wake_row["wav"])) not in target_speaker_ids
                and speaker_key(resolve_path(passenger_nonwake_row["wav"])) not in target_speaker_ids
            )
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
                target,
                truth,
                sample_rate,
                stage1_seconds,
                float(args.hop_seconds),
                device,
                force_presence_reject_window=args.force_presence_reject_window,
                capture_stage1_tracks=args.save_stage1_tracks,
            )
            wake_track_rows = [
                row for row in metrics["windows"]
                if row.get("kws_primary_spex_output_probability") is not None
            ]
            metrics["stage3_track_scores"] = {
                "scored_wake_overlapping_crops": len(wake_track_rows),
                "max_raw_mixture_probability": max(
                    (float(row["kws_primary_raw_mixture_probability"]) for row in wake_track_rows),
                    default=None,
                ),
                "max_clean_target_probability": max(
                    (float(row["kws_primary_clean_target_probability"]) for row in wake_track_rows),
                    default=None,
                ),
                "max_stage1_output_probability": max(
                    (float(row["kws_primary_spex_output_probability"]) for row in wake_track_rows),
                    default=None,
                ),
                "runtime_wake_detected": bool(metrics["target_wake_detected"]),
                "runtime_wake_threshold": float(pipeline.kws_threshold),
            }
            artifact_dir = resolve_path(args.artifact_dir)
            stage1_track_files: dict[str, dict[str, str]] = {}
            track_tensors = metrics.pop("stage1_audio_tracks", {})
            if args.save_stage1_tracks:
                track_dir = artifact_dir / "stage1_tracks" / f"point_{scene_index:04d}_snr_{float(cabin_snr_db):+05.1f}"
                for event_name, track in track_tensors.items():
                    event_paths: dict[str, str] = {}
                    event_start = int(track["event_crop_start_sample"])
                    event_end = int(track["event_crop_end_sample"])
                    event_source_start = int(track["event_audio_source_sample_start"])
                    event_source_end = int(track["event_audio_source_sample_end"])
                    for key, audio in (
                        ("input_mono_context", track["input_mono"]),
                        ("input_multichannel_context", track["input_multichannel"]),
                        ("spex_raw_context", track["stage1_output_raw"]),
                        ("spex_normalized_context", track["stage1_output_normalized"]),
                    ):
                        path = track_dir / f"{event_name}_{key}.wav"
                        save_audio(path, audio, sample_rate)
                        event_paths[key] = display_path(path)
                    clean_event = target[event_source_start:event_source_end]
                    clean_path = track_dir / f"{event_name}_clean_target_reference.wav"
                    save_audio(clean_path, clean_event, sample_rate)
                    event_paths["clean_target_reference"] = display_path(clean_path)
                    track["event_crop_start_seconds"] = event_start / sample_rate
                    track["event_crop_end_seconds"] = event_end / sample_rate
                    track["context_start_seconds"] = int(track["source_window_start_sample"]) / sample_rate
                    stage1_track_files[event_name] = {
                        "audio_files": event_paths,
                        "source_window_start_sample": int(track["source_window_start_sample"]),
                        "source_window_end_sample": int(track["source_window_end_sample"]),
                        "event_source_start_sample": event_source_start,
                        "event_source_end_sample": event_source_end,
                        "event_crop_start_sample_within_context": event_start,
                        "event_crop_end_sample_within_context": event_end,
                    }
            metrics["stage1_audio_tracks"] = stage1_track_files
            metrics["stage4_track_predictions"] = metrics.pop("stage4_track_predictions", {})
            saved_stage4_files = []
            for input_index, stage4_input in enumerate(metrics.pop("stage4_input_waveforms", [])):
                stage4_path = artifact_dir / (
                    f"stage4_input_scene_{scene_index:02d}_snr_{float(cabin_snr_db):+05.1f}_"
                    f"event_{input_index:02d}.wav"
                )
                save_audio(stage4_path, stage4_input, sample_rate)
                saved_stage4_files.append(display_path(stage4_path))
            metrics["stage4_input_audio_files"] = saved_stage4_files
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
    action_latencies = [float(row["target_action_latency_seconds"]) for row in results if row["target_action_latency_seconds"] is not None]
    output = resolve_path(args.output)
    report = {
        "benchmark": "continuous_full_pipeline_cross_corpus_proxy",
        "device": str(device),
        "config": display_path(args.config),
        "paths_config": display_path(args.paths),
        "checkpoints": cfg["checkpoints"],
        "provenance": build_provenance(args, cfg, results),
        "sample_rate": sample_rate,
        "stage1_context_seconds": stage1_seconds,
        "stage1_bypassed": bool(args.bypass_stage1),
        "hop_seconds": float(args.hop_seconds),
        "effective_presence_threshold": float(cfg.get("presence_threshold", 0.5)),
        "effective_command_presence_threshold": float(
            cfg.get("command_presence_threshold", cfg.get("presence_threshold", 0.5))
        ),
        "effective_kws_threshold": float(cfg.get("kws_threshold", 0.5)),
        "scenes": target_count,
        "wake_manifest_positive_rows": len(wake_rows),
        "all_positive_wake_recordings_used": bool(args.all_positive_wake_recordings),
        "stage1_audio_tracks_saved": bool(args.save_stage1_tracks),
        "command_manifest_unique_intents_used": len(intent_rows),
        "target_wake_denominator": target_count,
        "target_intent_denominator": target_count,
        "end_to_end_pass_denominator": target_count,
        "duration_seconds": total_duration,
        "duration_hours": total_duration / 3600.0,
        "target_wake_recall": wake_successes / max(target_count, 1),
        "target_intent_accuracy_overall": intent_successes / max(target_count, 1),
        "end_to_end_pass_count": sum(
            bool(row["target_wake_detected"]) and bool(row["target_intent_correct"])
            for row in results
        ),
        "end_to_end_pass_rate": sum(
            bool(row["target_wake_detected"]) and bool(row["target_intent_correct"])
            for row in results
        ) / max(target_count, 1),
        "safe_scene_pass_count": sum(
            bool(row["target_wake_detected"])
            and bool(row["target_intent_correct"])
            and int(row["false_action_events"]) == 0
            for row in results
        ),
        "safe_scene_pass_rate": sum(
            bool(row["target_wake_detected"])
            and bool(row["target_intent_correct"])
            and int(row["false_action_events"]) == 0
            for row in results
        ) / max(target_count, 1),
        "target_intent_accuracy_conditioned_on_wake": (
            (
                sum(bool(row["target_intent_correct"]) for row in results if row["target_wake_detected"])
                / wake_successes
                if wake_successes else None
            )
        ),
        "false_wake_events": false_events,
        "false_action_events": false_actions,
        "target_wake_latency_mean_seconds": float(np.mean(latencies)) if latencies else None,
        "target_wake_latency_p95_seconds": float(np.percentile(latencies, 95)) if latencies else None,
        "target_action_latency_p50_seconds": float(np.median(action_latencies)) if action_latencies else None,
        "target_action_latency_p95_seconds": float(np.percentile(action_latencies, 95)) if action_latencies else None,
        "speaker_consistent": False,
        "production_valid": False,
        "measurement_definitions": {
            "target_wake": (
                "A runtime-confirmed wake is attributed to the target when its positive confirmation "
                "support contains at least one above-threshold KWS crop overlapping the labeled wake. "
                "The runtime still enforces the configured multi-hit rule."
            ),
            "target_intent": (
                "An intent counts only when runtime emits it from command capture armed by that target wake "
                "and its decision is available from target-command start through three seconds after its end."
            ),
            "end_to_end_pass": "Target wake detected AND its associated command intent matches the label.",
            "safe_scene_pass": "End-to-end pass AND no other intent/action event occurs in the scene.",
            "decision_latency": (
                "Measured when the full Stage 1 context window is available (clamped to scene end), not at its center."
            ),
            "false_event_rates": "Not extrapolated per hour; this proxy is too short for a defensible rate estimate.",
        },
        "limitation": (
            "This is a continuous full-cascade engineering proxy. It uses the same runtime path and includes "
            "target wake, labelled command, passenger wake, passenger non-wake speech, and multichannel CAVEMOVE "
            "noise, but the target wake and command are cross-corpus clips rather than a natural same-speaker pair."
        ),
        "scenes_detail": results,
        "artifacts": {
            "directory": display_path(args.artifact_dir),
            "data_point_csv": display_path(output.with_suffix(".csv")),
            "files": [
                "cabin_input_mean.wav",
                "cabin_input_4ch.wav",
                "target_wake_plus_command.wav",
                "speech_with_passenger.wav",
                "enrollment.wav",
                "stage1_tracks/point_<index>_snr_<db>/*_input_*.wav",
                "stage1_tracks/point_<index>_snr_<db>/*_spex_*.wav",
                "stage1_tracks/point_<index>_snr_<db>/*_clean_target_reference.wav",
            ],
        },
    }
    write_metrics(output, report)
    csv_path = output.with_suffix(".csv")
    csv_columns = [
        "scene_index", "snr_db", "target_wake_speaker_id", "target_wake_recording",
        "enrollment_recording", "command_recording", "command_speaker_id", "expected_intent", "runtime_wake_detected",
        "stage3_raw_mixture_max_score", "stage3_clean_target_max_score", "stage3_stage1_output_max_score",
        "runtime_predicted_intent", "runtime_intent_correct", "stage4_raw_mixture_intent",
        "stage4_clean_target_intent", "stage4_stage1_output_intent", "stage1_wake_track_files",
        "stage1_command_track_files", "stage4_runtime_input_files",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns)
        writer.writeheader()
        for scene in results:
            truth = scene["truth"]
            scores = scene.get("stage3_track_scores", {})
            stage4_controls = scene.get("stage4_track_predictions", {})
            tracks = scene.get("stage1_audio_tracks", {})
            writer.writerow({
                "scene_index": scene.get("scene_index"),
                "snr_db": scene.get("snr_db"),
                "target_wake_speaker_id": truth.get("target_wake_source", {}).get("speaker_id"),
                "target_wake_recording": truth.get("target_wake_source", {}).get("recording"),
                "enrollment_recording": truth.get("target_enrollment_source", {}).get("recording"),
                "command_recording": truth.get("target_command_source"),
                "command_speaker_id": truth.get("target_command_source_speaker_id"),
                "expected_intent": scene.get("expected_intent"),
                "runtime_wake_detected": scene.get("target_wake_detected"),
                "stage3_raw_mixture_max_score": scores.get("max_raw_mixture_probability"),
                "stage3_clean_target_max_score": scores.get("max_clean_target_probability"),
                "stage3_stage1_output_max_score": scores.get("max_stage1_output_probability"),
                "runtime_predicted_intent": scene.get("predicted_intent"),
                "runtime_intent_correct": scene.get("target_intent_correct"),
                "stage4_raw_mixture_intent": stage4_controls.get("raw_mixture", {}).get("predicted_intent"),
                "stage4_clean_target_intent": stage4_controls.get("clean_target_reference", {}).get("predicted_intent"),
                "stage4_stage1_output_intent": stage4_controls.get("stage1_normalized_output", {}).get("predicted_intent"),
                "stage1_wake_track_files": json.dumps(tracks.get("wake", {}).get("audio_files", {}), sort_keys=True),
                "stage1_command_track_files": json.dumps(tracks.get("command", {}).get("audio_files", {}), sort_keys=True),
                "stage4_runtime_input_files": json.dumps(scene.get("stage4_input_audio_files", [])),
            })
    report["data_point_csv"] = display_path(csv_path)
    write_metrics(output, report)
    if args.wandb:
        window_rows = []
        for scene in results:
            truth = scene["truth"]
            for window in scene["windows"]:
                window_rows.append({
                    "scene_index": scene["scene_index"],
                    "snr_db": scene["snr_db"],
                    "target_speaker_id": truth["target_wake_source"]["speaker_id"],
                    "passenger_wake_speaker_id": truth["passenger_wake_source"]["speaker_id"],
                    "target_wake_start_seconds": truth["target_wake_start_seconds"],
                    "target_wake_end_seconds": truth["target_wake_end_seconds"],
                    "target_command_start_seconds": truth["target_command_start_seconds"],
                    "target_command_end_seconds": truth["target_command_end_seconds"],
                    "expected_intent": truth["expected_intent"],
                    **window,
                })
        log_evaluation(
            project=args.wandb_project,
            run_name=args.wandb_run_name or "continuous-pipeline-cross-corpus-proxy",
            job_type="pipeline-evaluation",
            config={
                "config": display_path(args.config),
                "paths_config": display_path(args.paths),
                "checkpoints": cfg["checkpoints"],
                "provenance": report["provenance"],
                "scenes": target_count,
                "sample_rate": sample_rate,
                "stage1_context_seconds": stage1_seconds,
                "hop_seconds": float(args.hop_seconds),
                "force_presence_reject_window": args.force_presence_reject_window,
                "cabin_snr_db": [float(value) for value in args.cabin_snr_db],
                "passenger_snr_db": float(args.passenger_snr_db),
                "seed": int(args.seed),
                "presence_threshold": report["effective_presence_threshold"],
                "command_presence_threshold": cfg.get("command_presence_threshold"),
                "kws_threshold": report["effective_kws_threshold"],
                "pipeline_settings": {
                    key: cfg[key]
                    for key in (
                        "presence_type",
                        "command_presence_threshold",
                        "kws_input_seconds",
                        "kws_smoothing_window",
                        "kws_required_hits",
                        "kws_refractory_calls",
                        "intent_type",
                        "intent_input_seconds",
                        "command_timeout_calls",
                        "command_timeout_seconds",
                        "command_end_silence_calls",
                        "command_min_speech_seconds",
                        "speaker_gate_threshold",
                        "speaker_gate_backend",
                    )
                    if key in cfg
                },
                "speaker_consistent": False,
                "production_valid": False,
            },
            metrics=report,
            output_path=output,
            mode=args.wandb_mode,
            directory=args.wandb_dir,
            scene_rows=results,
            window_rows=window_rows,
            report_artifact=True,
        )
    print(json.dumps({key: report[key] for key in (
        "device", "scenes", "duration_seconds", "target_wake_recall",
        "target_intent_accuracy_overall", "target_intent_accuracy_conditioned_on_wake",
        "end_to_end_pass_rate", "target_action_latency_p50_seconds", "target_action_latency_p95_seconds",
        "false_wake_events", "false_action_events", "target_wake_latency_mean_seconds",
        "target_wake_latency_p95_seconds", "speaker_consistent", "production_valid", "output",
    ) if key in report} | {"output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
