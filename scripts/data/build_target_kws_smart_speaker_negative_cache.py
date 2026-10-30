#!/usr/bin/env python
"""Build target-KWS negative caches from the CC BY Smart Speaker corpus.

Each source clip is scripted as ``Hey Alexa, <command>`` and is therefore
negative for the current ``Marvin`` wake word.  The cache is deliberately
speaker-disjoint by source speaker.  The audio is clean single-channel speech
duplicated across the configured microphone interface; it is a hard negative
for continuous wake/command phrasing, not a substitute for real cabin data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.segment import segment_waveform
from car_tse.io.audio import read_mono
from car_tse.io.manifest import write_jsonl
from car_tse.models.stage1_system import Stage1System
from car_tse.pipeline.in_car_pipeline import normalize_extracted_waveform
from car_tse.train.utils import get_device, load_checkpoint


def resolve(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def pad_start(wav: np.ndarray, samples: int) -> np.ndarray:
    result = np.zeros(max(1, int(samples)), dtype=np.float32)
    result[: min(len(wav), len(result))] = np.asarray(wav, dtype=np.float32)[: len(result)]
    return result


def split_speakers(speakers: list[str]) -> dict[str, list[str]]:
    if len(speakers) < 3:
        raise RuntimeError("At least three source speakers are required for disjoint train/val/test splits")
    ordered = sorted(speakers)
    return {
        "train": ordered[:-2],
        "val": [ordered[-2]],
        "test": [ordered[-1]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"D:\datasets\smart_speaker_command_26417548\SmartSpeakerCommandDataset_RutowskiEtAl_2023"),
    )
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--stage1-config", type=Path, default=REPO_ROOT / "configs" / "stage1_spex_dual_domain_rms010_ft1.yaml")
    parser.add_argument("--stage1-checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "stage1_spex_dual_domain_rms010_ft1" / "best.pt")
    parser.add_argument("--enrollment-wav", type=Path, default=REPO_ROOT / "data" / "raw" / "librispeech" / "extracted" / "LibriSpeech" / "test-clean" / "1089" / "134686" / "1089-134686-0014.flac")
    parser.add_argument("--output-cache-dir", type=Path, default=REPO_ROOT / "data" / "cache" / "stage3_target_kws_smart_speaker_negative_16k")
    parser.add_argument("--output-prefix", default="stage3_target_kws_smart_speaker_negative")
    parser.add_argument("--max-files-per-split", type=int, default=0, help="0 uses every file in each speaker-disjoint split")
    parser.add_argument("--max-files-per-speaker", type=int, default=0, help="0 uses every file; otherwise keeps a balanced prefix per speaker")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    root = args.root.resolve()
    files = sorted(root.glob("Speaker*/*.wav"))
    if not files:
        raise FileNotFoundError(f"No Speaker*/.wav files found under {root}")
    by_speaker: dict[str, list[Path]] = {}
    for path in files:
        by_speaker.setdefault(path.parent.name, []).append(path)
    if int(args.max_files_per_speaker) > 0:
        by_speaker = {
            speaker: paths[: int(args.max_files_per_speaker)]
            for speaker, paths in by_speaker.items()
        }
    splits = split_speakers(list(by_speaker))

    cfg = load_stage_config(args.stage1_config, args.paths)
    device = get_device()
    stage1 = Stage1System(
        spex_cfg=cfg["spex"],
        sample_rate=int(cfg["sample_rate"]),
        use_cabin_frontend=True,
        num_mics=int(cfg.get("num_mics", 4)),
        frontend_cfg=cfg.get("frontend"),
        pretrained_dir=str(resolve(cfg["pretrained_checkpoint"])),
        backend=cfg.get("spex_backend", "transfer_spexplus_16k"),
    ).to(device)
    load_checkpoint(resolve(args.stage1_checkpoint), stage1)
    stage1.eval()

    enrollment, _ = read_mono(resolve(args.enrollment_wav), 16000)
    enrollment = segment_waveform(enrollment, 24000, training=False).astype(np.float32)
    enrollment_tensor = torch.from_numpy(enrollment)
    num_mics = int(cfg.get("num_mics", 4))
    window_samples = int(round(float(cfg.get("segment_seconds", 3.0)) * 16000))
    cache_root = resolve(args.output_cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_root = resolve(cfg["paths"]["manifest_root"])

    for split, speakers in splits.items():
        split_files = [path for speaker in speakers for path in by_speaker[speaker]]
        split_files = sorted(split_files)
        if int(args.max_files_per_split) > 0:
            split_files = split_files[: int(args.max_files_per_split)]
        if not split_files:
            raise RuntimeError(f"No files for split {split}: {speakers}")

        wav_cache = np.lib.format.open_memmap(
            cache_root / f"{split}_wav_16000hz.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(split_files), 16000),
        )
        enroll_cache = np.lib.format.open_memmap(
            cache_root / f"{split}_enrollment_embedding.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(split_files), 256),
        )
        extracted_cache = np.lib.format.open_memmap(
            cache_root / f"{split}_extracted_embedding.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(split_files), 256),
        )
        manifest_rows: list[dict] = []
        with torch.inference_mode():
            for start in range(0, len(split_files), max(1, int(args.batch_size))):
                chunk = split_files[start : start + int(args.batch_size)]
                mixtures: list[np.ndarray] = []
                enrollments: list[np.ndarray] = []
                for path in chunk:
                    wav, _ = read_mono(path, 16000)
                    window = pad_start(wav, window_samples)
                    multichannel = np.repeat(window[:, None], num_mics, axis=1).T
                    mixtures.append(multichannel.astype(np.float32))
                    enrollments.append(enrollment)
                multichannel_tensor = torch.from_numpy(np.stack(mixtures, axis=0)).to(device)
                mixture_tensor = multichannel_tensor.mean(dim=1)
                enrollment_batch = torch.from_numpy(np.stack(enrollments, axis=0)).to(device)
                extracted, _, _, _ = stage1(
                    mixture_tensor,
                    enrollment_batch,
                    multichannel=multichannel_tensor,
                )
                conditioned = normalize_extracted_waveform(extracted)[..., :16000]
                if conditioned.shape[-1] < 16000:
                    conditioned = torch.nn.functional.pad(conditioned, (0, 16000 - conditioned.shape[-1]))
                enrollment_embedding = stage1.speaker_embedding(enrollment_batch)
                extracted_embedding = stage1.speaker_embedding(extracted)
                wav_cache[start : start + len(chunk)] = conditioned.cpu().numpy().astype(np.float32)
                enroll_cache[start : start + len(chunk)] = enrollment_embedding.cpu().numpy().astype(np.float32)
                extracted_cache[start : start + len(chunk)] = extracted_embedding.cpu().numpy().astype(np.float32)
                for path in chunk:
                    manifest_rows.append(
                        {
                            "label": 0,
                            "source_manifest": "University of Rochester Smart Speaker Command Dataset",
                            "source_id": f"smart_speaker_{path.parent.name}_{path.stem}",
                            "source_domain": "clean_continuous_nonmarvin_command",
                            "source_license": "CC BY 4.0",
                            "source_url": "https://doi.org/10.60593/ur.d.26417548",
                            "speaker_id": path.parent.name,
                            "wake_phrase": "Hey Alexa",
                            "current_wake_phrase": "Marvin",
                            "audio_path": str(path),
                        }
                    )
        wav_cache.flush()
        enroll_cache.flush()
        extracted_cache.flush()
        del wav_cache, enroll_cache, extracted_cache
        manifest_path = manifest_root / split / f"{args.output_prefix}_{split}.jsonl"
        write_jsonl(manifest_path, manifest_rows)
        print(
            f"{split}: {len(manifest_rows)} files from speakers={speakers} -> {manifest_path}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
