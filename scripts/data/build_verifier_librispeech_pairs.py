#!/usr/bin/env python
"""Build LibriSpeech speaker-verifier pairs through the runtime Stage 1.

Each pair is ``(enrollment waveform, extracted waveform)``.  Positive pairs
enroll one utterance of a target speaker while the mixture contains a second
utterance of that same speaker; negative pairs enroll speaker A while the
mixture contains speaker B.  Every mixture adds a different-speaker
interference and a CAVEMOVE cabin-noise bed, and the extracted waveform is the
first second of the normalized Stage 1 output, exactly what the runtime
compact verifier sees.  Speaker lists come from
``manifests/meta/librispeech_speakers.json`` so train/val/test stay
speaker-disjoint.
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
import torch

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.segment import segment_waveform
from car_tse.io.audio import read_mono
from car_tse.pipeline.in_car_pipeline import normalize_extracted_waveform
from car_tse.train.utils import get_device, load_checkpoint
from car_tse.models.stage1_system import Stage1System


CORPUS_BY_SPLIT = {
    "train": "train-clean-100",
    "val": "dev-clean",
    "test": "test-clean",
}


def resolve(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def index_speakers(corpus_root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for speaker_dir in sorted(corpus_root.iterdir()):
        if not speaker_dir.is_dir():
            continue
        for flac in sorted(speaker_dir.rglob("*.flac")):
            index[speaker_dir.name].append(flac)
    return {speaker: files for speaker, files in index.items() if len(files) >= 2}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument(
        "--librispeech-root",
        type=Path,
        default=REPO_ROOT / "data/raw/librispeech/extracted/LibriSpeech",
    )
    parser.add_argument(
        "--noise-cache-dir",
        type=Path,
        default=REPO_ROOT / "data/cache/cavemove_16k_mc",
    )
    parser.add_argument(
        "--stage1-config",
        type=Path,
        default=REPO_ROOT / "configs/stage1_spex_dual_domain_rms010_ft1.yaml",
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints/stage1_spex_dual_domain_rms010_ft1/best.pt",
    )
    parser.add_argument("--positives", type=int, default=4096)
    parser.add_argument("--negatives", type=int, default=4096)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--context-seconds", type=float, default=3.0)
    parser.add_argument("--verifier-seconds", type=float, default=1.0)
    parser.add_argument("--enrollment-seconds", type=float, default=1.0)
    parser.add_argument("--interference-snr-min", type=float, default=-5.0)
    parser.add_argument("--interference-snr-max", type=float, default=10.0)
    parser.add_argument("--cabin-snr-min", type=float, default=-5.0)
    parser.add_argument("--cabin-snr-max", type=float, default=15.0)
    parser.add_argument("--extract-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument(
        "--output-cache-dir",
        type=Path,
        default=REPO_ROOT / "data/cache/verifier_librispeech_pairs_2026-10-12",
    )
    parser.add_argument("--manifest-root", type=Path, default=REPO_ROOT / "manifests")
    parser.add_argument("--manifest-template", default="verifier_librispeech_{split}.jsonl")
    parser.add_argument("--noise-channels", nargs="+", type=int, default=[0, 2, 4, 6])
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    np.random.seed(int(args.seed) % (2**32))
    sample_rate = int(args.sample_rate)
    context_samples = int(round(float(args.context_seconds) * sample_rate))
    verifier_samples = int(round(float(args.verifier_seconds) * sample_rate))
    enrollment_samples = int(round(float(args.enrollment_seconds) * sample_rate))

    corpus_root = resolve(args.librispeech_root) / CORPUS_BY_SPLIT[str(args.split)]
    speakers = index_speakers(corpus_root)
    if len(speakers) < 3:
        raise RuntimeError(f"Need at least three speakers with two utterances each, found {len(speakers)}")
    speaker_ids = sorted(speakers)
    total = int(args.positives) + int(args.negatives)
    if total <= 0:
        raise RuntimeError("No pairs requested")

    noise_files = sorted(resolve(args.noise_cache_dir).glob("*.npy"))
    if not noise_files:
        raise RuntimeError("No cached CAVEMOVE noise arrays found")

    cfg = load_stage_config(resolve(args.stage1_config), resolve(Path("configs/paths.yaml")))
    device = get_device() if args.device == "auto" else torch.device(args.device)
    stage1 = Stage1System(
        spex_cfg=cfg["spex"],
        sample_rate=sample_rate,
        use_cabin_frontend=True,
        num_mics=len(args.noise_channels),
        frontend_cfg=cfg.get("frontend"),
        pretrained_dir=str(resolve(cfg["pretrained_checkpoint"])),
        backend=cfg.get("spex_backend", "transfer_spexplus_16k"),
    ).to(device)
    stage1_checkpoint = resolve(args.stage1_checkpoint)
    load_checkpoint(stage1_checkpoint, stage1)
    stage1.eval()

    def load_utterance(path: Path) -> np.ndarray:
        wav, _ = read_mono(path, sample_rate)
        return np.asarray(wav, dtype=np.float32)

    workload = []
    for index in range(total):
        label = 1 if index < int(args.positives) else 0
        target_speaker = speaker_ids[index % len(speaker_ids)]
        target_files = speakers[target_speaker]
        enrollment_file = target_files[(index // len(speaker_ids)) % len(target_files)]
        target_file = target_files[((index // len(speaker_ids)) + 1) % len(target_files)]
        if label == 0:
            other = speaker_ids[(index * 7 + 3) % len(speaker_ids)]
            if other == target_speaker:
                other = speaker_ids[(speaker_ids.index(other) + 1) % len(speaker_ids)]
            target_file = speakers[other][index % len(speakers[other])]
        interference_speaker = speaker_ids[(index * 11 + 5) % len(speaker_ids)]
        interference_file = speakers[interference_speaker][index % len(speakers[interference_speaker])]
        workload.append(
            {
                "label": label,
                "target_speaker": target_speaker,
                "enrollment_file": enrollment_file,
                "target_file": target_file,
                "interference_file": interference_file,
                "noise_file": noise_files[index % len(noise_files)],
                "interference_snr": rng.uniform(
                    float(args.interference_snr_min), float(args.interference_snr_max)
                ),
                "cabin_snr": rng.uniform(float(args.cabin_snr_min), float(args.cabin_snr_max)),
                "noise_offset": rng.randrange(0, max(1, 400_000)),
            }
        )

    output_cache = resolve(args.output_cache_dir)
    output_cache.mkdir(parents=True, exist_ok=True)
    manifest_root = resolve(args.manifest_root)
    enrollment_waves = np.zeros((total, enrollment_samples), dtype=np.float32)
    extracted_waves = np.zeros((total, verifier_samples), dtype=np.float32)
    manifest_rows: list[dict[str, Any]] = []

    channels = [int(value) for value in args.noise_channels]
    batch_size = max(1, int(args.extract_batch_size))
    with torch.inference_mode():
        for start in range(0, total, batch_size):
            end = min(total, start + batch_size)
            mixtures = np.zeros((end - start, context_samples), dtype=np.float32)
            multichannel = np.zeros((end - start, len(channels), context_samples), dtype=np.float32)
            for offset, item in enumerate(workload[start:end]):
                target = segment_waveform(load_utterance(item["target_file"]), context_samples, training=False)
                interference = segment_waveform(
                    load_utterance(item["interference_file"]), context_samples, training=False
                )
                target_rms = float(np.sqrt(np.mean(target**2) + 1e-8))
                interference_rms = float(np.sqrt(np.mean(interference**2) + 1e-8))
                interference = interference * (
                    target_rms * (10.0 ** (-float(item["interference_snr"]) / 20.0))
                ) / max(interference_rms, 1e-6)
                speech = target + interference
                noise = np.load(item["noise_file"], mmap_mode="r", allow_pickle=False)
                noise = np.asarray(noise[:, channels], dtype=np.float32)
                noise_offset = int(item["noise_offset"]) % max(1, len(noise) - context_samples)
                noise_window = noise[noise_offset : noise_offset + context_samples]
                if len(noise_window) < context_samples:
                    repeats = int(np.ceil(context_samples / max(1, len(noise_window))))
                    noise_window = np.tile(noise_window, (repeats, 1))[:context_samples]
                noise_rms = float(np.sqrt(np.mean(noise_window**2) + 1e-8))
                noise_window = noise_window * (
                    target_rms * (10.0 ** (-float(item["cabin_snr"]) / 20.0)) / max(noise_rms, 1e-6)
                )
                mixtures[offset] = speech
                multichannel[offset] = (speech[:, None] + noise_window).T
                enrollment_waves[start + offset] = segment_waveform(
                    load_utterance(item["enrollment_file"]), enrollment_samples, training=False
                )
            extracted, _, _, _ = stage1(
                torch.from_numpy(mixtures).to(device),
                torch.from_numpy(enrollment_waves[start:end]).to(device),
                multichannel=torch.from_numpy(multichannel).to(device),
            )
            normalized = normalize_extracted_waveform(extracted).cpu().numpy()
            extracted_waves[start:end] = normalized[:, :verifier_samples]
            for offset, item in enumerate(workload[start:end]):
                manifest_rows.append(
                    {
                        "label": int(item["label"]),
                        "source": "librispeech_pair",
                        "split": str(args.split),
                        "target_speaker": str(item["target_speaker"]),
                        "enrollment_wav": str(item["enrollment_file"]),
                        "target_wav": str(item["target_file"]),
                    }
                )
            print(f"{args.split}: extracted {end}/{total}", flush=True)

    np.save(output_cache / f"{args.split}_enrollment_wav.npy", enrollment_waves)
    np.save(output_cache / f"{args.split}_wav_16000hz.npy", extracted_waves)
    manifest_dir = manifest_root / str(args.split)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / str(args.manifest_template).format(split=str(args.split))
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row) + "\n")
    summary = {
        "split": str(args.split),
        "pairs": total,
        "positives": int(args.positives),
        "negatives": int(args.negatives),
        "speakers": len(speaker_ids),
        "cache_dir": str(output_cache),
        "manifest": str(manifest_path),
    }
    (output_cache / f"build_summary_{args.split}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
