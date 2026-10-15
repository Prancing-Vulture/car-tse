#!/usr/bin/env python
"""Generate KWS examples from the current Stage 1 TSE output distribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.datasets import TripletExtractionDataset
from car_tse.data.segment import segment_waveform
from car_tse.io.audio import read_mono, read_multichannel
from car_tse.io.manifest import read_jsonl, write_jsonl
from car_tse.models.stage1_system import Stage1System
from car_tse.train.utils import get_device, load_checkpoint


def noise_cache_path(cache_dir: Path, source: str) -> Path:
    digest = hashlib.sha1(str(Path(source)).encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.npy"


def make_wake_example(
    row: dict,
    noise_cache_dir: Path,
    sample_rate: int,
    segment_seconds: float,
    channels: list[int],
    index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_samples = int(sample_rate * segment_seconds)
    speech, _ = read_mono(row["wav"], sample_rate)
    mixture = segment_waveform(speech, num_samples, training=False)
    enrollment = segment_waveform(speech, num_samples // 2, training=False)
    noise_path = row["cabin_noise_wav"]
    cached_path = noise_cache_path(noise_cache_dir, noise_path)
    if cached_path.exists():
        noise = np.load(cached_path, mmap_mode="r", allow_pickle=False)
    else:
        noise, _ = read_multichannel(noise_path, sample_rate)
    if noise.ndim == 1:
        noise = noise[:, None]
    valid = [channel for channel in channels if channel < noise.shape[1]]
    if not valid:
        valid = list(range(min(len(channels), noise.shape[1])))
    noise = np.asarray(noise[:, valid], dtype=np.float32)
    if noise.shape[1] < len(channels):
        noise = np.tile(noise, (1, int(np.ceil(len(channels) / noise.shape[1]))) )[:, : len(channels)]
    if len(noise) >= num_samples:
        rng = np.random.default_rng(2026 + index * 7919)
        start = int(rng.integers(0, len(noise) - num_samples + 1))
        noise = noise[start : start + num_samples]
    else:
        noise = np.tile(noise, (int(np.ceil(num_samples / max(1, len(noise)))), 1))[:num_samples]
    snr_db = float(row.get("cabin_noise_snr_db", 0.0))
    speech_rms = float(np.sqrt(np.mean(mixture * mixture) + 1e-8))
    noise_rms = float(np.sqrt(np.mean(noise * noise) + 1e-8))
    if speech_rms > 1e-6 and noise_rms > 1e-6:
        noise = noise * (speech_rms * (10.0 ** (-snr_db / 20.0)) / noise_rms)
    return mixture, enrollment, mixture[:, None] + noise


def write_tse_outputs(
    model: Stage1System,
    device: torch.device,
    output: np.memmap,
    rows: list[dict],
    labels: list[int],
    sample_rate: int,
    segment_seconds: float,
    channels: list[int],
    noise_cache_dir: Path,
    batch_size: int,
    stage1_negative_manifest: Path | None = None,
) -> list[dict]:
    num_samples = int(sample_rate * segment_seconds)
    kws_samples = sample_rate
    manifest_rows: list[dict] = []
    batch_mix: list[np.ndarray] = []
    batch_enroll: list[np.ndarray] = []
    batch_mc: list[np.ndarray] = []
    batch_labels: list[int] = []
    batch_sources: list[str] = []
    output_index = 0

    def flush() -> None:
        nonlocal output_index
        if not batch_mix:
            return
        with torch.inference_mode():
            mixture = torch.from_numpy(np.stack(batch_mix)).to(device)
            enrollment = torch.from_numpy(np.stack(batch_enroll)).to(device)
            multichannel = torch.from_numpy(np.stack(batch_mc).transpose(0, 2, 1)).to(device)
            extracted, _, _, _ = model(mixture, enrollment, multichannel=multichannel)
        extracted = extracted[:, :kws_samples].detach().cpu().numpy().astype(np.float32)
        for local_index, waveform in enumerate(extracted):
            peak = float(np.max(np.abs(waveform)))
            if peak > 0.95:
                waveform = waveform * (0.95 / peak)
            output[output_index] = waveform
            manifest_rows.append(
                {
                    "wav": batch_sources[local_index],
                    "label": int(batch_labels[local_index]),
                    "source_stage1_tse": True,
                    "source_stage1_negative_manifest": str(stage1_negative_manifest) if stage1_negative_manifest else None,
                }
            )
            output_index += 1
        batch_mix.clear()
        batch_enroll.clear()
        batch_mc.clear()
        batch_labels.clear()
        batch_sources.clear()

    for index, row in enumerate(rows):
        if labels[index] == 1:
            mixture, enrollment, multichannel = make_wake_example(
                row, noise_cache_dir, sample_rate, segment_seconds, channels, index
            )
        else:
            raise RuntimeError("negative rows are processed through the triplet loader")
        batch_mix.append(mixture)
        batch_enroll.append(enrollment)
        batch_mc.append(multichannel)
        batch_labels.append(labels[index])
        batch_sources.append(row["wav"])
        if len(batch_mix) == batch_size:
            flush()
    flush()
    return manifest_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--stage1-config", type=Path, default=REPO_ROOT / "configs" / "stage1_spex_transfer_mc_cavemove.yaml")
    parser.add_argument("--stage1-checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "stage1_spex_transfer_mc_cavemove_cabin" / "best.pt")
    parser.add_argument("--noise-cache-dir", type=Path, default=REPO_ROOT / "data" / "cache" / "cavemove_16k_mc")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "cache" / "stage3_kws_tse_16k")
    parser.add_argument("--output-prefix", default="stage3_kws_tse", help="Prefix for generated manifests")
    parser.add_argument("--train-positives", type=int, default=1000)
    parser.add_argument("--train-negatives", type=int, default=2000)
    parser.add_argument("--eval-positives", type=int, default=195)
    parser.add_argument("--eval-negatives", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    paths = load_stage_config(args.stage1_config, args.paths)["paths"]
    stage1_cfg = load_stage_config(args.stage1_config, args.paths)
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
    load_checkpoint(args.stage1_checkpoint, stage1)
    stage1.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    sample_rate = int(stage1_cfg["sample_rate"])
    segment_seconds = float(stage1_cfg["segment_seconds"])
    channels = [int(x) for x in stage1_cfg["cabin_noise_channels"]]
    manifest_root = Path(paths["manifest_root"])

    for split in ("train", "val", "test"):
        output_cache = args.output_dir / f"{split}_cabin_wav_{sample_rate}hz.npy"
        output_manifest = manifest_root / split / f"{args.output_prefix}_{split}.jsonl"
        if output_cache.exists() and output_manifest.exists() and not args.force:
            print(f"Already exists: {output_cache}")
            continue
        gsc_manifest = manifest_root / split / (
            "stage3_kws_cabin_train.jsonl" if split == "train" else "stage3_kws_cabin_val.jsonl" if split == "val" else "stage3_kws_cabin_test.jsonl"
        )
        gsc_rows = [row for row in read_jsonl(gsc_manifest) if int(row.get("label", 0)) == 1]
        rng.shuffle(gsc_rows)
        positive_limit = args.train_positives if split == "train" else args.eval_positives
        gsc_rows = gsc_rows[: min(positive_limit, len(gsc_rows))]
        negative_manifest = manifest_root / split / "stage1_cavemove_cabin.jsonl"
        negative_rows = read_jsonl(negative_manifest)
        rng.shuffle(negative_rows)
        negative_limit = args.train_negatives if split == "train" else args.eval_negatives
        negative_rows = negative_rows[: min(negative_limit, len(negative_rows))]
        total = len(gsc_rows) + len(negative_rows)
        output = np.lib.format.open_memmap(
            output_cache, mode="w+", dtype=np.float32, shape=(total, sample_rate)
        )
        manifest_rows: list[dict] = []
        positive_labels = [1] * len(gsc_rows)
        manifest_rows.extend(
            write_tse_outputs(
                stage1,
                device,
                output,
                gsc_rows,
                positive_labels,
                sample_rate,
                segment_seconds,
                channels,
                args.noise_cache_dir,
                args.batch_size,
            )
        )

        negative_ds = TripletExtractionDataset(
            negative_manifest,
            sample_rate,
            segment_seconds,
            training=False,
            cabin_noise=True,
            num_mics=stage1_cfg["num_mics"],
            cabin_noise_channels=channels,
            cabin_noise_snr_db=stage1_cfg["cabin_noise_snr_db"],
            cabin_noise_cache_size=64,
            cabin_noise_cache_dir=args.noise_cache_dir,
        )
        negative_indices = [negative_ds.rows.index(row) for row in negative_rows]
        negative_subset = torch.utils.data.Subset(negative_ds, negative_indices)
        loader = DataLoader(negative_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                mixture = batch["mixture"].to(device)
                enrollment = batch["enrollment"].to(device)
                multichannel = batch["multichannel"].to(device)
                extracted, _, _, _ = stage1(mixture, enrollment, multichannel=multichannel)
                extracted = extracted[:, :sample_rate].detach().cpu().numpy().astype(np.float32)
                for local_index, waveform in enumerate(extracted):
                    peak = float(np.max(np.abs(waveform)))
                    if peak > 0.95:
                        waveform = waveform * (0.95 / peak)
                    output[len(manifest_rows)] = waveform
                    manifest_rows.append(
                        {
                            "wav": negative_rows[batch_index * 4 + local_index]["mixture_wav"],
                            "label": 0,
                            "source_stage1_tse": True,
                            "source_stage1_negative_manifest": str(negative_manifest),
                        }
                    )
        output.flush()
        del output
        # The memmap and manifest are intentionally kept in the same
        # positive-then-negative order; WeightedRandomSampler removes any
        # ordering effect during training without copying the large array.
        write_jsonl(output_manifest, manifest_rows)
        print(f"{split}: {len(manifest_rows)} TSE-domain rows -> {output_cache}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
