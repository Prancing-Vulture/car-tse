#!/usr/bin/env python
"""Build an intent cache after Stage 1 on target/passenger mixtures.

The older overlap adaptation added passenger speech after a frozen Stage-1
waveform.  That is useful as a classifier stress test, but it does not expose
the extractor to the co-channel mixture.  This append-only builder puts the
passenger speech in the Stage-1 input first, then caches the actual extracted
target waveform used by the downstream intent head.

The passenger pool is explicitly supplied and should be train-only when the
result is used for a held-out evaluation.  This utility is for synthetic
engineering adaptation; it does not create real-driver or spatial-array
evidence.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.segment import segment_waveform
from car_tse.io.audio import read_mono
from car_tse.io.manifest import read_jsonl, write_jsonl
from car_tse.models.stage1_system import Stage1System
from car_tse.train.utils import get_device, load_checkpoint


def rms(wav: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(wav, dtype=np.float32) ** 2) + 1.0e-8))


def fit_clip(wav: np.ndarray, samples: int, start: int = 0) -> np.ndarray:
    out = np.zeros(int(samples), dtype=np.float32)
    values = np.asarray(wav, dtype=np.float32).reshape(-1)
    start = max(0, int(start))
    if start < len(out):
        end = min(len(out), start + len(values))
        out[start:end] = values[: end - start]
    return out


def add_at(destination: np.ndarray, source: np.ndarray, start: int) -> None:
    start = max(0, int(start))
    if start >= len(destination) or len(source) == 0:
        return
    end = min(len(destination), start + len(source))
    destination[start:end] += source[: end - start]


def speaker_id(path: str) -> str:
    return Path(path).parent.name


def load_speech_pool(manifest: Path, sample_rate: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rows = read_jsonl(manifest)
    wake = []
    nonwake = []
    for row in rows:
        if row.get("noise_only", False):
            continue
        wav, _ = read_mono(row["wav"], sample_rate)
        if int(row.get("label", 0)) == 1:
            wake.append(np.asarray(wav, dtype=np.float32))
        else:
            nonwake.append(np.asarray(wav, dtype=np.float32))
    if not wake or not nonwake:
        raise ValueError("passenger manifest must contain speech rows for labels 0 and 1")
    return wake, nonwake


def make_pair(
    target: np.ndarray,
    passenger_wake: np.ndarray,
    passenger_nonwake: np.ndarray,
    sample_rate: int,
    stage1_samples: int,
    passenger_snr_db: float,
    overlap: bool,
) -> np.ndarray:
    target_clip = fit_clip(target, stage1_samples)
    if not overlap:
        return target_clip
    passenger = np.zeros(stage1_samples, dtype=np.float32)
    add_at(passenger, fit_clip(passenger_wake, int(0.95 * sample_rate)), int(0.30 * sample_rate))
    add_at(passenger, fit_clip(passenger_nonwake, int(0.95 * sample_rate)), int(1.15 * sample_rate))
    target_rms = rms(target_clip)
    passenger_rms = rms(passenger)
    if target_rms > 1.0e-6 and passenger_rms > 1.0e-6:
        passenger *= target_rms * (10.0 ** (-float(passenger_snr_db) / 20.0)) / passenger_rms
    mixed = target_clip + passenger
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.95:
        mixed *= 0.95 / peak
    return mixed.astype(np.float32)


def build_split(
    split: str,
    rows: list[dict],
    output_dir: Path,
    manifest_out: Path,
    model: Stage1System,
    device: torch.device,
    sample_rate: int,
    stage1_samples: int,
    intent_samples: int,
    passenger_wake: list[np.ndarray],
    passenger_nonwake: list[np.ndarray],
    passenger_snr_db: float,
    base_limit: int,
    batch_size: int,
    seed: int,
) -> None:
    if base_limit > 0:
        rows = rows[: min(base_limit, len(rows))]
    selected: list[tuple[dict, bool, int]] = []
    for index, row in enumerate(rows):
        selected.append((row, False, index))
        selected.append((row, True, index))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}_joint_tse_overlap_wav_{sample_rate}hz.npy"
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(selected), intent_samples),
    )
    manifest_rows: list[dict] = []
    mix_batch: list[np.ndarray] = []
    enroll_batch: list[np.ndarray] = []
    multichannel_batch: list[np.ndarray] = []
    pending: list[tuple[dict, bool, int]] = []

    def flush() -> None:
        if not mix_batch:
            return
        with torch.inference_mode():
            mixture = torch.from_numpy(np.stack(mix_batch)).to(device)
            enrollment = torch.from_numpy(np.stack(enroll_batch)).to(device)
            multichannel = torch.from_numpy(np.stack(multichannel_batch).transpose(0, 2, 1)).to(device)
            extracted, _, _, _ = model(mixture, enrollment, multichannel=multichannel)
        values = extracted[:, :intent_samples].detach().cpu().numpy().astype(np.float32)
        base = len(manifest_rows)
        for local, waveform in enumerate(values):
            peak = float(np.max(np.abs(waveform)))
            if peak > 0.95:
                waveform = waveform * (0.95 / peak)
            output[base + local] = waveform
            source_row, overlap, source_index = pending[local]
            item = {
                "wav": source_row["wav"],
                "label": int(source_row["label"]),
                "intent": source_row.get("intent", ""),
                "condition": "joint_tse_cabin_passenger_overlap" if overlap else "joint_tse_cabin_clean",
                "source_row_index": int(source_index),
                "source_stage1_tse": True,
                "stage1_input_contains_passenger": bool(overlap),
                "passenger_snr_db": float(passenger_snr_db) if overlap else None,
            }
            manifest_rows.append(item)
        mix_batch.clear()
        enroll_batch.clear()
        multichannel_batch.clear()
        pending.clear()

    for output_index, (row, overlap, source_index) in enumerate(selected):
        target, _ = read_mono(row["wav"], sample_rate)
        # A same-speaker enrollment is sufficient for the intent-domain cache;
        # the full proxy still evaluates enrollment with a separate utterance.
        mixture = make_pair(
            target,
            passenger_wake[(output_index * 7 + seed) % len(passenger_wake)],
            passenger_nonwake[(output_index * 11 + seed) % len(passenger_nonwake)],
            sample_rate,
            stage1_samples,
            passenger_snr_db,
            overlap,
        )
        enrollment = fit_clip(target, stage1_samples)
        multichannel = np.repeat(mixture[:, None], 4, axis=1)
        mix_batch.append(mixture)
        enroll_batch.append(enrollment)
        multichannel_batch.append(multichannel)
        pending.append((row, overlap, source_index))
        if len(mix_batch) >= batch_size:
            flush()
        if (output_index + 1) % 512 == 0 or output_index + 1 == len(selected):
            print(f"{split}: prepared {output_index + 1}/{len(selected)}", flush=True)
    flush()
    output.flush()
    del output
    write_jsonl(manifest_out, manifest_rows)
    print(f"Wrote {split}: {len(manifest_rows)} -> {output_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-config", type=Path, default=REPO_ROOT / "configs/stage1_spex_dual_domain_rms010_ft1.yaml")
    parser.add_argument("--stage1-checkpoint", type=Path, default=REPO_ROOT / "checkpoints/stage1_spex_dual_domain_rms010_ft1/best.pt")
    parser.add_argument("--train-manifest", type=Path, default=REPO_ROOT / "manifests/train/stage4_cavemove_intent.jsonl")
    parser.add_argument("--val-manifest", type=Path, default=REPO_ROOT / "manifests/val/stage4_cavemove_intent.jsonl")
    parser.add_argument("--passenger-manifest", type=Path, default=REPO_ROOT / "manifests/train/stage3_kws_cabin_train.jsonl")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/cache/stage4_intent_joint_tse_overlap_2026-10-16_16k")
    parser.add_argument("--train-manifest-out", type=Path, default=REPO_ROOT / "manifests/train/stage4_intent_joint_tse_overlap_2026-10-16.jsonl")
    parser.add_argument("--val-manifest-out", type=Path, default=REPO_ROOT / "manifests/val/stage4_intent_joint_tse_overlap_2026-10-16.jsonl")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--train-limit", type=int, default=8192)
    parser.add_argument("--val-limit", type=int, default=2048)
    parser.add_argument("--passenger-snr-db", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()

    stage1_cfg = load_stage_config(args.stage1_config, REPO_ROOT / "configs/paths.yaml")
    device = get_device()
    sample_rate = int(stage1_cfg["sample_rate"])
    stage1_samples = int(sample_rate * stage1_cfg["segment_seconds"])
    intent_samples = int(sample_rate * 2.0)
    pretrained = Path(stage1_cfg["pretrained_checkpoint"])
    if not pretrained.is_absolute():
        pretrained = REPO_ROOT / pretrained
    model = Stage1System(
        spex_cfg=stage1_cfg["spex"],
        sample_rate=sample_rate,
        use_cabin_frontend=True,
        num_mics=int(stage1_cfg["num_mics"]),
        frontend_cfg=stage1_cfg.get("frontend"),
        pretrained_dir=str(pretrained),
        backend=stage1_cfg["spex_backend"],
    ).to(device)
    checkpoint = args.stage1_checkpoint if args.stage1_checkpoint.is_absolute() else REPO_ROOT / args.stage1_checkpoint
    load_checkpoint(checkpoint, model)
    model.eval()
    train_rows = read_jsonl(args.train_manifest)
    val_rows = read_jsonl(args.val_manifest)
    passenger_wake, passenger_nonwake = load_speech_pool(args.passenger_manifest, sample_rate)
    print(
        f"device={device} stage1={checkpoint} train_rows={len(train_rows)} val_rows={len(val_rows)} "
        f"passenger_wake={len(passenger_wake)} passenger_nonwake={len(passenger_nonwake)}",
        flush=True,
    )
    build_split(
        "train", train_rows, args.output_dir, args.train_manifest_out, model, device,
        sample_rate, stage1_samples, intent_samples, passenger_wake, passenger_nonwake,
        args.passenger_snr_db, args.train_limit, args.batch_size, args.seed,
    )
    build_split(
        "val", val_rows, args.output_dir, args.val_manifest_out, model, device,
        sample_rate, stage1_samples, intent_samples, passenger_wake, passenger_nonwake,
        args.passenger_snr_db, args.val_limit, args.batch_size, args.seed + 101,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
