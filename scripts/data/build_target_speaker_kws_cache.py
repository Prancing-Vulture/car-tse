#!/usr/bin/env python
"""Cache same-domain SpEx+ outputs and speaker embeddings for target-KWS training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.datasets import TripletExtractionDataset
from car_tse.io.manifest import read_jsonl, write_jsonl
from car_tse.models.stage1_system import Stage1System
from car_tse.train.utils import get_device, load_checkpoint
from car_tse.pipeline.in_car_pipeline import normalize_extracted_waveform


def collate(batch: list[dict]) -> dict:
    output: dict = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if all(isinstance(value, torch.Tensor) for value in values):
            output[key] = torch.stack(values)
        else:
            output[key] = values
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.yaml")
    parser.add_argument(
        "--stage1-config",
        type=Path,
        default=REPO_ROOT / "configs/stage1_spex_transfer_mc_cavemove.yaml",
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints/stage1_spex_transfer_mc_cavemove_cabin/best.pt",
    )
    parser.add_argument("--manifest-name", default="stage1_gsc_marvin_cabin.jsonl")
    parser.add_argument("--output-prefix", default="stage3_target_kws_gsc_marvin")
    parser.add_argument(
        "--output-cache-dir",
        type=Path,
        default=REPO_ROOT / "data/cache/stage3_target_kws_gsc_marvin_16k",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    stage1_cfg = load_stage_config(args.stage1_config, args.paths)
    paths = stage1_cfg["paths"]
    device = get_device()
    pretrained = Path(stage1_cfg["pretrained_checkpoint"])
    if not pretrained.is_absolute():
        pretrained = REPO_ROOT / pretrained
    checkpoint = args.stage1_checkpoint if args.stage1_checkpoint.is_absolute() else REPO_ROOT / args.stage1_checkpoint
    stage1 = Stage1System(
        spex_cfg=stage1_cfg["spex"],
        sample_rate=int(stage1_cfg["sample_rate"]),
        use_cabin_frontend=True,
        num_mics=int(stage1_cfg.get("num_mics", 4)),
        frontend_cfg=stage1_cfg.get("frontend"),
        pretrained_dir=str(pretrained),
        backend=stage1_cfg.get("spex_backend", "transfer_spexplus_16k"),
    ).to(device)
    load_checkpoint(checkpoint, stage1)
    stage1.eval()

    output_cache_dir = args.output_cache_dir if args.output_cache_dir.is_absolute() else REPO_ROOT / args.output_cache_dir
    output_cache_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = int(stage1_cfg["sample_rate"])
    segment_seconds = float(stage1_cfg.get("segment_seconds", 3.0))
    manifest_root = Path(paths["manifest_root"])
    noise_cache_dir = Path(paths["data_root"]) / ".." / "cache" / "cavemove_16k_mc"

    for split in ("train", "val", "test"):
        source_manifest = manifest_root / split / args.manifest_name
        if not source_manifest.exists():
            raise FileNotFoundError(source_manifest)
        rows = read_jsonl(source_manifest)
        wav_cache_path = output_cache_dir / f"{split}_wav_{sample_rate}hz.npy"
        enroll_cache_path = output_cache_dir / f"{split}_enrollment_embedding.npy"
        extracted_cache_path = output_cache_dir / f"{split}_extracted_embedding.npy"
        output_manifest = manifest_root / split / f"{args.output_prefix}_{split}.jsonl"
        dataset = TripletExtractionDataset(
            source_manifest,
            sample_rate,
            segment_seconds,
            training=False,
            cabin_noise=True,
            num_mics=int(stage1_cfg.get("num_mics", 4)),
            cabin_noise_channels=stage1_cfg.get("cabin_noise_channels", [0, 2, 4, 6]),
            cabin_noise_snr_db=stage1_cfg.get("cabin_noise_snr_db"),
            cabin_noise_cache_size=64,
            cabin_noise_cache_dir=noise_cache_dir,
        )
        wav_output = np.lib.format.open_memmap(
            wav_cache_path, mode="w+", dtype=np.float32, shape=(len(dataset), sample_rate)
        )
        enrollment_output = np.lib.format.open_memmap(
            enroll_cache_path, mode="w+", dtype=np.float32, shape=(len(dataset), 256)
        )
        extracted_output = np.lib.format.open_memmap(
            extracted_cache_path, mode="w+", dtype=np.float32, shape=(len(dataset), 256)
        )
        loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        )
        manifest_rows: list[dict] = []
        offset = 0
        with torch.inference_mode():
            for batch in loader:
                mixture = batch["mixture"].to(device)
                enrollment = batch["enrollment"].to(device)
                multichannel = batch.get("multichannel")
                if isinstance(multichannel, torch.Tensor):
                    multichannel = multichannel.to(device)
                extracted, _, _, _ = stage1(mixture, enrollment, multichannel=multichannel)
                # Match the runtime KWS crop: the 1-second window is centered
                # around the 0.75-second speech placement in these mixtures.
                crop_start = int(round(0.5 * sample_rate))
                wav = normalize_extracted_waveform(extracted)
                wav = wav[:, crop_start : crop_start + sample_rate]
                enrollment_embedding = stage1.speaker_embedding(enrollment)
                extracted_embedding = stage1.speaker_embedding(extracted)
                wav_output[offset : offset + len(wav)] = wav.detach().cpu().numpy().astype(np.float32)
                enrollment_output[offset : offset + len(wav)] = (
                    enrollment_embedding.detach().cpu().numpy().astype(np.float32)
                )
                extracted_output[offset : offset + len(wav)] = (
                    extracted_embedding.detach().cpu().numpy().astype(np.float32)
                )
                for index in range(len(wav)):
                    source = rows[offset + index]
                    manifest_rows.append(
                        {
                            "label": int(float(source.get("target_present", 0)) >= 0.5),
                            "source_manifest": str(source_manifest),
                            "source_id": source.get("id", ""),
                        }
                    )
                offset += len(wav)
        wav_output.flush()
        enrollment_output.flush()
        extracted_output.flush()
        del wav_output, enrollment_output, extracted_output
        write_jsonl(output_manifest, manifest_rows)
        positives = sum(int(row["label"]) for row in manifest_rows)
        print(
            f"{split}: {len(manifest_rows)} cached rows, positives={positives} -> {output_cache_dir}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
