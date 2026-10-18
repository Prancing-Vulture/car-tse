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
from car_tse.models.kws_model import KWSModel, TargetSpeakerKWSModel
from car_tse.models.presence_head import (
    FramePresenceHead,
    PersonalActivityHead,
    PersonalFramePresenceHead,
    PresenceHead,
)
from car_tse.models.speaker_verifier import CompactSpeakerVerifier
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


def summarize_kws_threshold(
    presence_array: np.ndarray,
    score_array: np.ndarray,
    duration: float,
    presence_threshold: float,
    kws_threshold: float,
    smooth_window: int,
    required_hits: int,
    refractory_calls: int,
    hop_seconds: float,
    speaker_array: np.ndarray | None = None,
    speaker_gate_threshold: float | None = None,
    speaker_fusion_mode: str = "none",
    speaker_fusion_center: float = 0.0,
    speaker_fusion_scale: float = 0.10,
    speaker_fusion_alpha: float = 0.5,
    activity_margin_array: np.ndarray | None = None,
    activity_margin_threshold: float | None = None,
) -> dict[str, Any]:
    """Summarize one threshold using already-computed Stage 1/KWS scores."""

    presence_flags = presence_array >= float(presence_threshold)
    if speaker_array is None or speaker_gate_threshold is None:
        speaker_flags = np.ones_like(presence_flags, dtype=bool)
    else:
        speaker_flags = speaker_array >= float(speaker_gate_threshold)
    if activity_margin_array is None or activity_margin_threshold is None:
        activity_margin_flags = np.ones_like(presence_flags, dtype=bool)
    else:
        activity_margin_flags = activity_margin_array >= float(activity_margin_threshold)
    effective_score_array = np.asarray(score_array, dtype=np.float32)
    fusion_mode = str(speaker_fusion_mode).lower()
    if fusion_mode != "none":
        if speaker_array is None:
            raise ValueError("speaker fusion requires speaker similarity scores")
        scale = max(float(speaker_fusion_scale), 1.0e-4)
        speaker_probability = 1.0 / (
            1.0
            + np.exp(
                -(
                    np.asarray(speaker_array, dtype=np.float32)
                    - float(speaker_fusion_center)
                )
                / scale
            )
        )
        if fusion_mode == "min":
            effective_score_array = np.minimum(effective_score_array, speaker_probability)
        elif fusion_mode == "product":
            effective_score_array = effective_score_array * speaker_probability
        elif fusion_mode == "mean":
            alpha = min(max(float(speaker_fusion_alpha), 0.0), 1.0)
            effective_score_array = (
                alpha * effective_score_array
                + (1.0 - alpha) * speaker_probability
            )
        else:
            raise ValueError(
                "speaker_fusion_mode must be one of: none, min, product, mean"
            )
    raw_flags = presence_flags & activity_margin_flags & speaker_flags & (effective_score_array >= float(kws_threshold))
    raw_kws_flags = score_array >= float(kws_threshold)

    state = PipelineState()
    smooth_flags = np.zeros(len(score_array), dtype=bool)
    smooth_kws_only_flags = np.zeros(len(score_array), dtype=bool)
    for index, score in enumerate(effective_score_array.tolist()):
        if not presence_flags[index] or not activity_margin_flags[index] or not speaker_flags[index]:
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

    merge_gap_steps = max(1, int(np.floor(2.0 / hop_seconds)))
    raw_events = event_indices(raw_flags, merge_gap_steps)
    raw_kws_only_events = event_indices(raw_kws_flags, merge_gap_steps)
    # ``update_kws_state`` can remain true for several overlapping windows,
    # especially when the input contains a sustained wake-like phrase.  Count
    # one operating event for a contiguous/near-contiguous run rather than
    # treating every positive window as a separate false accept.  Keep the
    # positive-window counts separately for diagnostics and backwards
    # comparison with older logs.
    smooth_events = event_indices(smooth_flags, merge_gap_steps)
    smooth_kws_only_events = event_indices(smooth_kws_only_flags, merge_gap_steps)
    return {
        "presence_pass_rate": float(presence_flags.mean()),
        "activity_margin_pass_rate": float(activity_margin_flags.mean()),
        "speaker_gate_pass_rate": float(speaker_flags.mean()),
        "speaker_fusion_mode": fusion_mode,
        "speaker_fusion_center": float(speaker_fusion_center),
        "speaker_fusion_scale": float(speaker_fusion_scale),
        "speaker_fusion_alpha": float(speaker_fusion_alpha),
        "full_path_raw_positive_windows": int(raw_flags.sum()),
        "full_path_raw_event_rate_per_hour": len(raw_events) * 3600.0 / max(duration, 1e-6),
        "full_path_smoothed_positive_windows": int(smooth_flags.sum()),
        "full_path_smoothed_events": len(smooth_events),
        "full_path_smoothed_event_rate_per_hour": len(smooth_events) * 3600.0 / max(duration, 1e-6),
        "kws_only_raw_positive_windows": int(raw_kws_flags.sum()),
        "kws_only_raw_window_rate_per_hour": int(raw_kws_flags.sum()) * 3600.0 / max(duration, 1e-6),
        "kws_only_raw_event_rate_per_hour": len(raw_kws_only_events) * 3600.0 / max(duration, 1e-6),
        "kws_only_smoothed_positive_windows": int(smooth_kws_only_flags.sum()),
        "kws_only_smoothed_events": len(smooth_kws_only_events),
        "kws_only_smoothed_event_rate_per_hour": len(smooth_kws_only_events) * 3600.0 / max(duration, 1e-6),
    }


def evaluate_stream(
    stage1: Stage1System,
    presence: torch.nn.Module,
    kws: KWSModel,
    target_kws: TargetSpeakerKWSModel | None,
    target_kws_secondary: TargetSpeakerKWSModel | None,
    device: torch.device,
    multichannel: np.ndarray,
    enrollment: np.ndarray,
    stage1_seconds: float,
    hop_seconds: float,
    batch_size: int,
    presence_threshold: float,
    kws_threshold: float,
    speaker_gate_threshold: float | None,
    smooth_window: int,
    required_hits: int,
    refractory_calls: int,
    kws_input_seconds: float,
    presence_top_fraction: float = 0.20,
    kws_thresholds: list[float] | None = None,
    presence_thresholds: list[float] | None = None,
    speaker_gate_thresholds: list[float] | None = None,
    speaker_fusion_mode: str = "none",
    speaker_fusion_center: float = 0.0,
    speaker_fusion_scale: float = 0.10,
    speaker_fusion_alpha: float = 0.5,
    presence_activity_margin_threshold: float | None = None,
    presence_activity_margin_mode: str = "independent_top",
    compact_speaker_verifier: CompactSpeakerVerifier | None = None,
) -> dict[str, Any]:
    sample_rate = 16000
    window_samples = int(round(stage1_seconds * sample_rate))
    hop_samples = int(round(hop_seconds * sample_rate))
    windows, count = make_windows(multichannel, window_samples, hop_samples)
    mixture_windows, _ = make_windows(multichannel.mean(axis=1)[:, None], window_samples, hop_samples)
    presence_scores: list[float] = []
    activity_margin_scores: list[float] = []
    kws_scores: list[float] = []
    speaker_scores: list[float] = []
    compact_enrollment = None
    if compact_speaker_verifier is not None:
        enrollment_array = np.asarray(enrollment, dtype=np.float32).reshape(-1)
        if enrollment_array.shape[0] >= sample_rate:
            enrollment_array = enrollment_array[:sample_rate]
        else:
            enrollment_array = np.pad(enrollment_array, (0, sample_rate - enrollment_array.shape[0]))
        compact_enrollment = torch.from_numpy(enrollment_array.copy()).to(device)
    with torch.inference_mode():
        for start in range(0, count, max(1, int(batch_size))):
            batch_mc = torch.from_numpy(windows[start : start + batch_size]).to(device)
            batch_mix = torch.from_numpy(mixture_windows[start : start + batch_size, 0, :]).to(device)
            batch_enroll = torch.from_numpy(np.repeat(enrollment[None, :], batch_mc.shape[0], axis=0)).to(device)
            extracted, _, _, _ = stage1(batch_mix, batch_enroll, multichannel=batch_mc)
            if (
                speaker_gate_threshold is not None
                or speaker_gate_thresholds
                or str(speaker_fusion_mode).lower() != "none"
            ):
                if compact_speaker_verifier is not None:
                    compact_extracted = normalize_extracted_waveform(extracted)
                    if compact_extracted.shape[-1] >= sample_rate:
                        compact_extracted = compact_extracted[..., :sample_rate]
                    else:
                        compact_extracted = torch.nn.functional.pad(
                            compact_extracted, (0, sample_rate - compact_extracted.shape[-1])
                        )
                    compact_logits, _, _ = compact_speaker_verifier(
                        compact_enrollment.unsqueeze(0).repeat(batch_mc.shape[0], 1),
                        compact_extracted,
                    )
                    similarity = torch.sigmoid(compact_logits)
                else:
                    similarity = stage1.speaker_similarity(batch_enroll, extracted)
                if similarity is None:
                    raise RuntimeError("Speaker scoring requested but Stage 1 has no speaker embedding")
                speaker_scores.extend(similarity.cpu().tolist())
            if getattr(presence, "requires_enrollment", False):
                enrollment_embedding = stage1.speaker_embedding(batch_enroll)
                if enrollment_embedding is None:
                    raise RuntimeError("Personal presence requires Stage 1 speaker embeddings")
                pres_logits = presence(extracted, enrollment_embedding)
            else:
                pres_logits = presence(extracted)
            if getattr(presence, "is_multiclass", False):
                probabilities = presence.activity_probabilities(pres_logits)
                pres = presence.aggregate_probability(
                    probabilities[:, presence.target_class_index, :],
                    top_fraction=presence_top_fraction,
                )
                non_target = presence.aggregate_probability(
                    probabilities[:, 2, :],
                    top_fraction=presence_top_fraction,
                )
                activity_margin_scores.extend(
                    presence.aggregate_margin(
                        probabilities[:, presence.target_class_index, :],
                        probabilities[:, 2, :],
                        top_fraction=presence_top_fraction,
                        mode=presence_activity_margin_mode,
                    ).cpu().tolist()
                )
            elif getattr(presence, "is_framewise", False):
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
            if target_kws is not None:
                enrollment_embedding = stage1.speaker_embedding(batch_enroll)
                extracted_embedding = stage1.speaker_embedding(extracted)
                if enrollment_embedding is None or extracted_embedding is None:
                    raise RuntimeError("Target-speaker KWS requires Stage 1 speaker embeddings")
                scores = torch.softmax(
                    target_kws(conditioned, enrollment_embedding, extracted_embedding), dim=-1
                )[:, 1]
                if target_kws_secondary is not None:
                    secondary_scores = torch.softmax(
                        target_kws_secondary(conditioned, enrollment_embedding, extracted_embedding), dim=-1
                    )[:, 1]
                    scores = torch.minimum(scores, secondary_scores)
            else:
                scores = torch.softmax(kws(conditioned), dim=-1)[:, 1]
            presence_scores.extend(pres.cpu().tolist())
            kws_scores.extend(scores.cpu().tolist())

    presence_array = np.asarray(presence_scores, dtype=np.float32)
    activity_margin_array = (
        np.asarray(activity_margin_scores, dtype=np.float32)
        if activity_margin_scores
        else None
    )
    score_array = np.asarray(kws_scores, dtype=np.float32)
    speaker_array = (
        np.asarray(speaker_scores, dtype=np.float32)
        if (
            speaker_gate_threshold is not None
            or speaker_gate_thresholds
            or str(speaker_fusion_mode).lower() != "none"
        )
        else None
    )
    duration = len(multichannel) / sample_rate
    summary = summarize_kws_threshold(
        presence_array, score_array, duration, presence_threshold, kws_threshold,
        smooth_window, required_hits, refractory_calls, hop_seconds,
        speaker_array, speaker_gate_threshold,
        speaker_fusion_mode, speaker_fusion_center, speaker_fusion_scale,
        speaker_fusion_alpha,
        activity_margin_array, presence_activity_margin_threshold,
    )
    result = {
        "duration_seconds": duration,
        "windows": count,
        "max_presence_score": float(presence_array.max(initial=0.0)),
        "presence_activity_margin_threshold": (
            float(presence_activity_margin_threshold)
            if presence_activity_margin_threshold is not None
            else None
        ),
        "presence_activity_margin_mode": str(presence_activity_margin_mode),
        "max_presence_activity_margin": (
            float(activity_margin_array.max(initial=0.0))
            if activity_margin_array is not None
            else None
        ),
        "max_kws_score": float(score_array.max(initial=0.0)),
        "speaker_fusion_mode": str(speaker_fusion_mode).lower(),
        "speaker_fusion_center": float(speaker_fusion_center),
        "speaker_fusion_scale": float(speaker_fusion_scale),
        "speaker_fusion_alpha": float(speaker_fusion_alpha),
        "speaker_gate_threshold": float(speaker_gate_threshold) if speaker_gate_threshold is not None else None,
        "min_speaker_similarity": float(speaker_array.min(initial=0.0)) if speaker_array is not None else None,
        "max_speaker_similarity": float(speaker_array.max(initial=0.0)) if speaker_array is not None else None,
        **summary,
    }
    if kws_thresholds:
        result["threshold_sweep"] = [
            {"kws_threshold": float(threshold), **summarize_kws_threshold(
                presence_array, score_array, duration, presence_threshold, float(threshold),
                smooth_window, required_hits, refractory_calls, hop_seconds,
                speaker_array, speaker_gate_threshold,
                speaker_fusion_mode, speaker_fusion_center, speaker_fusion_scale,
                speaker_fusion_alpha,
                activity_margin_array, presence_activity_margin_threshold,
            )}
            for threshold in kws_thresholds
        ]
    if speaker_gate_thresholds:
        result["speaker_gate_threshold_sweep"] = [
            {
                "speaker_gate_threshold": float(threshold),
                **summarize_kws_threshold(
                    presence_array,
                    score_array,
                    duration,
                    presence_threshold,
                    kws_threshold,
                    smooth_window,
                    required_hits,
                    refractory_calls,
                    hop_seconds,
                    speaker_array,
                    float(threshold),
                    speaker_fusion_mode,
                    speaker_fusion_center,
                    speaker_fusion_scale,
                    speaker_fusion_alpha,
                    activity_margin_array,
                    presence_activity_margin_threshold,
                ),
            }
            for threshold in speaker_gate_thresholds
        ]
    if presence_thresholds:
        result["presence_threshold_sweep"] = [
            {
                "presence_threshold": float(threshold),
                **summarize_kws_threshold(
                    presence_array,
                    score_array,
                    duration,
                    float(threshold),
                    kws_threshold,
                    smooth_window,
                    required_hits,
                    refractory_calls,
                    hop_seconds,
                    speaker_array,
                    speaker_gate_threshold,
                    speaker_fusion_mode,
                    speaker_fusion_center,
                    speaker_fusion_scale,
                    speaker_fusion_alpha,
                    activity_margin_array,
                    presence_activity_margin_threshold,
                ),
            }
            for threshold in presence_thresholds
        ]
    return result


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
    parser.add_argument("--target-kws-config", type=Path, default=REPO_ROOT / "configs" / "stage3_target_kws_gsc_marvin.yaml")
    parser.add_argument("--target-kws-checkpoint", type=Path, default=None)
    parser.add_argument("--target-kws-secondary-checkpoint", type=Path, default=None)
    parser.add_argument("--presence-checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "stage2_presence_cavemove_cabin" / "best.pt")
    parser.add_argument(
        "--presence-type",
        choices=("scalar", "frame", "personal_frame", "personal_activity"),
        default="scalar",
    )
    parser.add_argument("--presence-speaker-embedding-dim", type=int, default=256)
    parser.add_argument("--presence-top-fraction", type=float, default=0.20)
    parser.add_argument("--enrollment-wav", type=Path, default=REPO_ROOT / "data" / "raw" / "librispeech" / "extracted" / "LibriSpeech" / "test-clean" / "1089" / "134686" / "1089-134686-0014.flac")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "logs" / "aishell5_continuous_kws_eval2.json")
    parser.add_argument("--presence-threshold", type=float, default=0.90)
    parser.add_argument("--presence-activity-margin-threshold", type=float, default=None)
    parser.add_argument(
        "--presence-activity-margin-mode",
        choices=("independent_top", "target_top", "framewise_top", "mean"),
        default="independent_top",
    )
    parser.add_argument("--speaker-gate-threshold", type=float, default=None)
    parser.add_argument("--speaker-gate-backend", choices=("spex", "compact"), default="spex")
    parser.add_argument("--compact-speaker-verifier-checkpoint", type=Path, default=None)
    parser.add_argument("--compact-speaker-verifier-embedding-dim", type=int, default=128)
    parser.add_argument("--compact-speaker-verifier-hidden", type=int, default=128)
    parser.add_argument("--kws-threshold", type=float, default=0.29)
    parser.add_argument("--kws-thresholds", nargs="+", type=float, default=None, help="Optional score thresholds to replay without recomputing Stage 1/KWS")
    parser.add_argument("--presence-thresholds", nargs="+", type=float, default=None, help="Optional presence thresholds to replay without recomputing Stage 1/KWS")
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--required-hits", type=int, default=2)
    parser.add_argument("--refractory-calls", type=int, default=2)
    parser.add_argument("--hop-seconds", type=float, default=1.0)
    parser.add_argument("--kws-input-seconds", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--channels", nargs="+", default=["DX01C01.wav", "DX02C01.wav", "DX03C01.wav", "DX04C01.wav"])
    args = parser.parse_args()

    stage1_cfg = load_stage_config(args.stage1_config, args.paths)
    kws_cfg = load_stage_config(args.kws_config, args.paths)
    target_kws_cfg = load_stage_config(args.target_kws_config, args.paths) if args.target_kws_checkpoint else None
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
    elif args.presence_type == "personal_frame":
        presence = PersonalFramePresenceHead(
            sample_rate=16000,
            n_mels=40,
            hidden=64,
            n_fft=512,
            hop=160,
            speaker_embedding_dim=int(args.presence_speaker_embedding_dim),
        ).to(device)
    elif args.presence_type == "personal_activity":
        presence = PersonalActivityHead(
            sample_rate=16000,
            n_mels=40,
            hidden=64,
            n_fft=512,
            hop=160,
            speaker_embedding_dim=int(args.presence_speaker_embedding_dim),
        ).to(device)
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
    target_kws = None
    target_kws_secondary = None
    target_kws_checkpoint = None
    target_kws_secondary_checkpoint = None
    if args.target_kws_checkpoint:
        target_kws_checkpoint = resolve_path(args.target_kws_checkpoint)
        target_model_cfg = target_kws_cfg["model"]
        target_kws = TargetSpeakerKWSModel(
            sample_rate=16000,
            n_mels=int(target_model_cfg["n_mels"]),
            channels=list(target_model_cfg["channels"]),
            num_classes=int(target_model_cfg["num_classes"]),
            speaker_embedding_dim=int(target_model_cfg.get("speaker_embedding_dim", 256)),
            hidden=int(target_model_cfg.get("hidden", 256)),
        ).to(device)
        load_checkpoint(target_kws_checkpoint, target_kws)
        target_kws.eval()
        if args.target_kws_secondary_checkpoint:
            target_kws_secondary_checkpoint = resolve_path(args.target_kws_secondary_checkpoint)
            target_kws_secondary = TargetSpeakerKWSModel(
                sample_rate=16000,
                n_mels=int(target_model_cfg["n_mels"]),
                channels=list(target_model_cfg["channels"]),
                num_classes=int(target_model_cfg["num_classes"]),
                speaker_embedding_dim=int(target_model_cfg.get("speaker_embedding_dim", 256)),
                hidden=int(target_model_cfg.get("hidden", 256)),
            ).to(device)
            load_checkpoint(target_kws_secondary_checkpoint, target_kws_secondary)
            target_kws_secondary.eval()

    compact_speaker_verifier = None
    if args.compact_speaker_verifier_checkpoint is not None:
        compact_speaker_verifier = CompactSpeakerVerifier(
            embedding_dim=int(args.compact_speaker_verifier_embedding_dim),
            hidden=int(args.compact_speaker_verifier_hidden),
            sample_rate=16000,
            n_mels=40,
            n_fft=400,
            hop=160,
        ).to(device)
        load_checkpoint(resolve_path(args.compact_speaker_verifier_checkpoint), compact_speaker_verifier)
        compact_speaker_verifier.eval()

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
        # A session can land exactly on the requested duration budget.  Do not
        # emit a synthetic zero-second row (which would create meaningless
        # 3.6e9 events/hour rates in the per-session diagnostics).
        if metadata["duration_seconds"] <= 0.0:
            break
        metrics = evaluate_stream(
            stage1, presence, kws, target_kws, target_kws_secondary, device, multichannel, enrollment,
            float(stage1_cfg.get("segment_seconds", 3.0)), float(args.hop_seconds), int(args.batch_size),
            float(args.presence_threshold), float(args.kws_threshold),
            float(args.speaker_gate_threshold) if args.speaker_gate_threshold is not None else None,
            int(args.smooth_window),
            int(args.required_hits), int(args.refractory_calls), float(args.kws_input_seconds),
            float(args.presence_top_fraction),
            [float(value) for value in args.kws_thresholds] if args.kws_thresholds else None,
            [float(value) for value in args.presence_thresholds] if args.presence_thresholds else None,
            presence_activity_margin_threshold=(
                float(args.presence_activity_margin_threshold)
                if args.presence_activity_margin_threshold is not None
                else None
            ),
            presence_activity_margin_mode=str(args.presence_activity_margin_mode),
            compact_speaker_verifier=compact_speaker_verifier,
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
    aggregate_threshold_sweep: list[dict[str, Any]] = []
    if args.kws_thresholds:
        total_duration_for_sweep = sum(float(row["duration_seconds"]) for row in session_results)
        for threshold in (float(value) for value in args.kws_thresholds):
            rows = [
                sweep_row
                for session in session_results
                for sweep_row in session.get("threshold_sweep", [])
                if float(sweep_row.get("kws_threshold", -1.0)) == threshold
            ]
            if not rows:
                continue
            aggregate_threshold_sweep.append(
                {
                    "kws_threshold": threshold,
                    "full_path_smoothed_events": int(
                        sum(int(row.get("full_path_smoothed_events", 0)) for row in rows)
                    ),
                    "full_path_smoothed_event_rate_per_hour": float(
                        sum(int(row.get("full_path_smoothed_events", 0)) for row in rows)
                        * 3600.0
                        / max(total_duration_for_sweep, 1.0e-6)
                    ),
                    "kws_only_smoothed_events": int(
                        sum(int(row.get("kws_only_smoothed_events", 0)) for row in rows)
                    ),
                    "kws_only_smoothed_event_rate_per_hour": float(
                        sum(int(row.get("kws_only_smoothed_events", 0)) for row in rows)
                        * 3600.0
                        / max(total_duration_for_sweep, 1.0e-6)
                    ),
                }
            )
    aggregate_presence_threshold_sweep: list[dict[str, Any]] = []
    if args.presence_thresholds:
        total_duration_for_sweep = sum(float(row["duration_seconds"]) for row in session_results)
        for threshold in (float(value) for value in args.presence_thresholds):
            rows = [
                sweep_row
                for session in session_results
                for sweep_row in session.get("presence_threshold_sweep", [])
                if abs(float(sweep_row.get("presence_threshold", -1.0)) - threshold) < 1.0e-8
            ]
            if not rows:
                continue
            full_events = sum(int(row.get("full_path_smoothed_events", 0)) for row in rows)
            kws_only_events = sum(int(row.get("kws_only_smoothed_events", 0)) for row in rows)
            aggregate_presence_threshold_sweep.append(
                {
                    "presence_threshold": threshold,
                    "full_path_smoothed_events": int(full_events),
                    "full_path_smoothed_event_rate_per_hour": float(
                        full_events * 3600.0 / max(total_duration_for_sweep, 1.0e-6)
                    ),
                    "kws_only_smoothed_events": int(kws_only_events),
                    "kws_only_smoothed_event_rate_per_hour": float(
                        kws_only_events * 3600.0 / max(total_duration_for_sweep, 1.0e-6)
                    ),
                }
            )
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
        "target_kws_checkpoint": str(target_kws_checkpoint) if target_kws_checkpoint else None,
        "target_kws_secondary_checkpoint": str(target_kws_secondary_checkpoint) if target_kws_secondary_checkpoint else None,
        "kws_mode": "target_speaker_ensemble_min" if target_kws_secondary_checkpoint else ("target_speaker" if target_kws_checkpoint else "generic"),
        "enrollment_wav": str(resolve_path(args.enrollment_wav)),
        "presence_threshold": float(args.presence_threshold),
        "presence_activity_margin_threshold": (
            float(args.presence_activity_margin_threshold)
            if args.presence_activity_margin_threshold is not None
            else None
        ),
        "presence_activity_margin_mode": str(args.presence_activity_margin_mode),
        "speaker_gate_threshold": float(args.speaker_gate_threshold) if args.speaker_gate_threshold is not None else None,
        "speaker_gate_backend": str(args.speaker_gate_backend),
        "compact_speaker_verifier_checkpoint": (
            str(resolve_path(args.compact_speaker_verifier_checkpoint))
            if args.compact_speaker_verifier_checkpoint is not None
            else None
        ),
        "kws_threshold": float(args.kws_threshold),
        "kws_thresholds": [float(value) for value in args.kws_thresholds] if args.kws_thresholds else None,
        "kws_input_seconds": float(args.kws_input_seconds),
        "hop_seconds": float(args.hop_seconds),
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
        "threshold_sweep": aggregate_threshold_sweep,
        "presence_threshold_sweep": aggregate_presence_threshold_sweep,
        "session_results": session_results,
        "note": "All recordings are target-absent relative to the fixed enrollment. This is a real continuous in-car speech/noise false-accept benchmark; it contains no wake-word positives.",
    }
    output = resolve_path(args.output)
    write_metrics(output, result)
    print({"output": str(output), "duration_hours": result["duration_hours"], "full_path_smoothed_event_rate_per_hour": result["full_path_smoothed_event_rate_per_hour"], "kws_only_smoothed_event_rate_per_hour": result["kws_only_smoothed_event_rate_per_hour"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
