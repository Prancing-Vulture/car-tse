#!/usr/bin/env python
"""Run hard-negative cabin speech through Stage 1 for production-path KWS tests."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.segment import segment_waveform
from car_tse.io.audio import read_mono, read_multichannel
from car_tse.io.manifest import read_jsonl, write_jsonl
from car_tse.models.stage1_system import Stage1System
from car_tse.train.utils import get_device, load_checkpoint


def noise_path(cache_dir: Path, source: str) -> Path:
    digest = hashlib.sha1(str(Path(source)).encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.npy"


def prepare_row(
    row: dict,
    enrollment_path: Path,
    noise_cache_dir: Path,
    sample_rate: int,
    segment_seconds: float,
    channels: list[int],
    index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_samples = int(sample_rate * segment_seconds)
    noise_only = bool(row.get("noise_only", False))
    if noise_only:
        mixture = np.zeros(num_samples, dtype=np.float32)
    else:
        speech, _ = read_mono(row["wav"], sample_rate)
        mixture = segment_waveform(np.asarray(speech, dtype=np.float32), num_samples, training=False)
    enrollment, _ = read_mono(str(enrollment_path), sample_rate)
    enrollment = segment_waveform(np.asarray(enrollment, dtype=np.float32), num_samples // 2, training=False)

    noise_source = row.get("cabin_noise_wav") or (row.get("wav") if noise_only else None)
    if not noise_source:
        raise ValueError("hard-negative row is missing cabin_noise_wav")
    cached = noise_path(noise_cache_dir, noise_source)
    if cached.exists():
        noise = np.load(cached, mmap_mode="r", allow_pickle=False)
    else:
        noise, _ = read_multichannel(noise_source, sample_rate)
    noise = np.asarray(noise, dtype=np.float32)
    if noise.ndim == 1:
        noise = noise[:, None]
    valid = [channel for channel in channels if channel < noise.shape[1]]
    if not valid:
        valid = list(range(min(len(channels), noise.shape[1])))
    noise = noise[:, valid]
    if noise.shape[1] < len(channels):
        noise = np.tile(noise, (1, int(np.ceil(len(channels) / noise.shape[1]))) )[:, : len(channels)]
    if len(noise) >= num_samples:
        start = (len(noise) - num_samples) // 2
        noise = noise[start : start + num_samples]
    else:
        noise = np.tile(noise, (int(np.ceil(num_samples / max(1, len(noise)))), 1))[:num_samples]
    snr_db = float(row.get("cabin_noise_snr_db", 0.0))
    speech_rms = float(np.sqrt(np.mean(mixture * mixture) + 1.0e-8))
    noise_rms = float(np.sqrt(np.mean(noise * noise) + 1.0e-8))
    if speech_rms > 1.0e-6 and noise_rms > 1.0e-6:
        noise = noise * (speech_rms * (10.0 ** (-snr_db / 20.0)) / noise_rms)
    multichannel = noise if noise_only else mixture[:, None] + noise
    return mixture, enrollment, multichannel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--stage1-config", type=Path, default=REPO_ROOT / "configs" / "stage1_spex_transfer_mc_cavemove.yaml")
    parser.add_argument("--stage1-checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "stage1_spex_transfer_mc_cavemove_cabin" / "best.pt")
    parser.add_argument("--hardneg-config", type=Path, default=REPO_ROOT / "configs" / "stage3_kws_cavemove_hardneg.yaml")
    parser.add_argument("--enrollment-wav", type=Path, required=True)
    parser.add_argument("--noise-cache-dir", type=Path, default=REPO_ROOT / "data" / "cache" / "cavemove_16k_mc")
    parser.add_argument("--output-cache-dir", type=Path, default=REPO_ROOT / "data" / "cache" / "stage3_kws_tse_hardneg_16k")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val", "test"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    stage1_cfg = load_stage_config(args.stage1_config, args.paths)
    hardneg_cfg = load_stage_config(args.hardneg_config, args.paths)
    paths = stage1_cfg["paths"]
    manifest_root = Path(paths["manifest_root"])
    noise_cache_dir = args.noise_cache_dir if args.noise_cache_dir.is_absolute() else REPO_ROOT / args.noise_cache_dir
    output_cache_dir = args.output_cache_dir if args.output_cache_dir.is_absolute() else REPO_ROOT / args.output_cache_dir
    enrollment = args.enrollment_wav if args.enrollment_wav.is_absolute() else REPO_ROOT / args.enrollment_wav
    checkpoint = args.stage1_checkpoint if args.stage1_checkpoint.is_absolute() else REPO_ROOT / args.stage1_checkpoint
    device = get_device()
    pretrained = Path(stage1_cfg["pretrained_checkpoint"])
    if not pretrained.is_absolute():
        pretrained = REPO_ROOT / pretrained
    stage1 = Stage1System(
        spex_cfg=stage1_cfg["spex"],
        sample_rate=stage1_cfg["sample_rate"],
        use_cabin_frontend=True,
        num_mics=stage1_cfg["num_mics"],
        frontend_cfg=stage1_cfg.get("frontend"),
        pretrained_dir=str(pretrained),
        backend=stage1_cfg["spex_backend"],
    ).to(device)
    load_checkpoint(checkpoint, stage1)
    stage1.eval()
    sample_rate = int(stage1_cfg["sample_rate"])
    segment_seconds = float(stage1_cfg["segment_seconds"])
    channels = [int(x) for x in stage1_cfg["cabin_noise_channels"]]
    output_cache_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        manifest_name = hardneg_cfg["manifest_train"] if split == "train" else hardneg_cfg["manifest_val"] if split == "val" else hardneg_cfg["manifest_eval"]
        rows = [
            row
            for row in read_jsonl(manifest_root / split / manifest_name)
            if int(row.get("label", 0)) == 0
        ]
        output_cache = output_cache_dir / f"{split}_cabin_wav_{sample_rate}hz.npy"
        output_manifest = manifest_root / split / f"stage3_kws_tse_hardneg_{split}.jsonl"
        expected = (len(rows), sample_rate)
        if output_cache.exists() and output_manifest.exists() and not args.force:
            try:
                existing = np.load(output_cache, mmap_mode="r", allow_pickle=False)
                if tuple(existing.shape) == expected:
                    print(f"Already cached {split}: {output_cache}", flush=True)
                    continue
            except (OSError, ValueError):
                pass
        output = np.lib.format.open_memmap(output_cache, mode="w+", dtype=np.float32, shape=expected)
        batch_mix: list[np.ndarray] = []
        batch_enroll: list[np.ndarray] = []
        batch_mc: list[np.ndarray] = []
        offset = 0
        for index, row in enumerate(rows):
            mixture, enroll, multichannel = prepare_row(
                row, enrollment, noise_cache_dir, sample_rate, segment_seconds, channels, index
            )
            batch_mix.append(mixture)
            batch_enroll.append(enroll)
            batch_mc.append(multichannel)
            if len(batch_mix) == args.batch_size or index + 1 == len(rows):
                with torch.inference_mode():
                    extracted, _, _, _ = stage1(
                        torch.from_numpy(np.stack(batch_mix)).to(device),
                        torch.from_numpy(np.stack(batch_enroll)).to(device),
                        multichannel=torch.from_numpy(np.stack(batch_mc).transpose(0, 2, 1)).to(device),
                    )
                values = extracted[:, :sample_rate].detach().cpu().numpy().astype(np.float32)
                for value in values:
                    peak = float(np.max(np.abs(value)))
                    output[offset] = value * (0.95 / peak) if peak > 0.95 else value
                    offset += 1
                batch_mix.clear()
                batch_enroll.clear()
                batch_mc.clear()
            if (index + 1) % 250 == 0 or index + 1 == len(rows):
                print(f"{split}: {index + 1}/{len(rows)}", flush=True)
        output.flush()
        del output
        out_rows = []
        for row in rows:
            out = dict(row)
            out["source_stage1_tse"] = True
            out["tse_enrollment_wav"] = str(enrollment)
            out_rows.append(out)
        write_jsonl(output_manifest, out_rows)
        print(f"Wrote {output_manifest} and {output_cache}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
