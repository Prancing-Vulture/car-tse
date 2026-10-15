#!/usr/bin/env python
"""Build speaker-disjoint LibriSpeech target-KWS negative caches.

LibriSpeech contains ordinary conversational/read speech and is therefore
negative for the current ``Marvin`` wake phrase.  This builder creates
speaker-disjoint 3-second Stage-1 windows at a fixed hop, runs the frozen
Stage-1 extractor, and stores the 1-second conditioned waveform plus the
enrollment/extracted embeddings required by ``train_stage3_target_kws.py``.

The output is a research hard-negative resource, not real-cabin evidence.
The source is kept as paths and the generated rows retain the LibriSpeech
provenance so the cache can be regenerated without copying the corpus.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
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


def pad_or_crop(wav: np.ndarray, start: int, samples: int) -> np.ndarray:
    result = np.zeros(max(1, int(samples)), dtype=np.float32)
    start = max(0, int(start))
    if start >= len(wav):
        return result
    end = min(len(wav), start + len(result))
    result[: end - start] = np.asarray(wav[start:end], dtype=np.float32)
    return result


def split_speakers(speakers: list[str], seed: int) -> dict[str, list[str]]:
    if len(speakers) < 6:
        raise RuntimeError("At least six LibriSpeech speakers are required for disjoint splits")
    ordered = list(sorted(speakers))
    random.Random(int(seed)).shuffle(ordered)
    n_test = max(1, round(len(ordered) * 0.15))
    n_val = max(1, round(len(ordered) * 0.15))
    return {
        "train": sorted(ordered[: len(ordered) - n_val - n_test]),
        "val": sorted(ordered[len(ordered) - n_val - n_test : len(ordered) - n_test]),
        "test": sorted(ordered[len(ordered) - n_test :]),
    }


def collect_candidates(
    root: Path,
    speakers: list[str],
    hop_samples: int,
    window_samples: int,
    seed: int,
) -> list[tuple[Path, str, int]]:
    candidates: list[tuple[Path, str, int]] = []
    speaker_set = set(speakers)
    files = sorted(path for path in root.rglob("*.flac") if path.parts and path.parts[-3] in speaker_set)
    if not files:
        # The expected structure is train-clean-100/<speaker>/<chapter>/*.flac.
        # Keep the fallback explicit for alternate extracted layouts.
        files = sorted(
            path
            for path in root.rglob("*.flac")
            if len(path.relative_to(root).parts) >= 3
            and path.relative_to(root).parts[0] in speaker_set
        )
    for path in files:
        relative = path.relative_to(root)
        speaker = str(relative.parts[0])
        # Read only the container header during candidate enumeration.  Full
        # decoding here made even a four-row smoke run scan the entire corpus;
        # audio is decoded once later for the selected windows.
        info = sf.info(str(path))
        length = max(1, int(round(float(info.frames) * 16000.0 / max(float(info.samplerate), 1.0))))
        starts = list(range(0, max(1, length), max(1, int(hop_samples))))
        # Avoid generating more than one padded tail per utterance.
        if length > window_samples and starts[-1] < length - window_samples:
            starts.append(length - window_samples)
        candidates.extend((path, speaker, int(start)) for start in starts)
    random.Random(int(seed)).shuffle(candidates)
    return candidates


def select_rows(
    candidates: list[tuple[Path, str, int]],
    target_hours: float,
    hop_seconds: float,
    max_rows: int,
) -> list[tuple[Path, str, int]]:
    desired = max(1, int(round(float(target_hours) * 3600.0 / max(float(hop_seconds), 1.0e-6))))
    if int(max_rows) > 0:
        desired = min(desired, int(max_rows))
    return candidates[: min(desired, len(candidates))]


def build_split(
    *,
    split: str,
    rows: list[tuple[Path, str, int]],
    stage1: Stage1System,
    enrollment: np.ndarray,
    stage1_cfg: dict[str, Any],
    device: torch.device,
    output_cache: Path,
    manifest_root: Path,
    prefix: str,
    batch_size: int,
) -> dict[str, Any]:
    if not rows:
        raise RuntimeError(f"No LibriSpeech rows selected for {split}")
    output_cache.mkdir(parents=True, exist_ok=True)
    wav_cache = np.lib.format.open_memmap(
        output_cache / f"{split}_wav_16000hz.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), 16000),
    )
    enroll_cache = np.lib.format.open_memmap(
        output_cache / f"{split}_enrollment_embedding.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), 256),
    )
    extracted_cache = np.lib.format.open_memmap(
        output_cache / f"{split}_extracted_embedding.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), 256),
    )
    window_samples = int(round(float(stage1_cfg.get("segment_seconds", 3.0)) * 16000))
    num_mics = int(stage1_cfg.get("num_mics", 4))
    enrollment_tensor = torch.from_numpy(enrollment.astype(np.float32, copy=False))
    manifest_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(rows), max(1, int(batch_size))):
            chunk = rows[start : start + int(batch_size)]
            mixtures: list[np.ndarray] = []
            for path, _, offset in chunk:
                wav, _ = read_mono(path, 16000)
                window = pad_or_crop(wav, offset, window_samples)
                mixtures.append(np.repeat(window[None, :], num_mics, axis=0).astype(np.float32))
            multichannel = torch.from_numpy(np.stack(mixtures, axis=0)).to(device)
            mixture = multichannel.mean(dim=1)
            enrollment_batch = enrollment_tensor.unsqueeze(0).repeat(len(chunk), 1).to(device)
            extracted, _, _, _ = stage1(mixture, enrollment_batch, multichannel=multichannel)
            conditioned = normalize_extracted_waveform(extracted)[..., :16000]
            if conditioned.shape[-1] < 16000:
                conditioned = torch.nn.functional.pad(conditioned, (0, 16000 - conditioned.shape[-1]))
            enrollment_embedding = stage1.speaker_embedding(enrollment_batch)
            extracted_embedding = stage1.speaker_embedding(extracted)
            wav_cache[start : start + len(chunk)] = conditioned.cpu().numpy().astype(np.float32)
            enroll_cache[start : start + len(chunk)] = enrollment_embedding.cpu().numpy().astype(np.float32)
            extracted_cache[start : start + len(chunk)] = extracted_embedding.cpu().numpy().astype(np.float32)
            for path, speaker, offset in chunk:
                manifest_rows.append(
                    {
                        "label": 0,
                        "source_manifest": "LibriSpeech train-clean-100",
                        "source_id": f"librispeech_{speaker}_{path.stem}_{int(offset)}",
                        "source_domain": "clean_read_speech_nonmarvin",
                        "source_license": "LibriSpeech license; verify before redistribution",
                        "source_url": "https://www.openslr.org/12",
                        "speaker_id": speaker,
                        "audio_path": str(path),
                        "window_start_seconds": float(offset) / 16000.0,
                        "current_wake_phrase": "Marvin",
                    }
                )
            if start == 0 or (start + len(chunk)) % (int(batch_size) * 20) == 0:
                print(f"{split}: {start + len(chunk)}/{len(rows)} windows", flush=True)
    wav_cache.flush()
    enroll_cache.flush()
    extracted_cache.flush()
    del wav_cache, enroll_cache, extracted_cache
    manifest_path = manifest_root / split / f"{prefix}_{split}.jsonl"
    write_jsonl(manifest_path, manifest_rows)
    return {
        "split": split,
        "rows": len(rows),
        "speakers": sorted({speaker for _, speaker, _ in rows}),
        "manifest": str(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--librispeech-root",
        type=Path,
        default=REPO_ROOT / "data/raw/librispeech/extracted/LibriSpeech/train-clean-100",
    )
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.yaml")
    parser.add_argument("--stage1-config", type=Path, default=REPO_ROOT / "configs/stage1_spex_dual_domain_rms010_ft1.yaml")
    parser.add_argument("--stage1-checkpoint", type=Path, default=REPO_ROOT / "checkpoints/stage1_spex_dual_domain_rms010_ft1/best.pt")
    parser.add_argument("--enrollment-wav", type=Path, default=REPO_ROOT / "data/raw/librispeech/extracted/LibriSpeech/test-clean/1089/134686/1089-134686-0014.flac")
    parser.add_argument("--output-cache-dir", type=Path, default=REPO_ROOT / "data/cache/stage3_target_kws_librispeech_negative_16k")
    parser.add_argument("--manifest-root", type=Path, default=REPO_ROOT / "manifests")
    parser.add_argument("--output-prefix", default="stage3_target_kws_librispeech_negative")
    parser.add_argument("--train-hours", type=float, default=10.0)
    parser.add_argument("--val-hours", type=float, default=2.0)
    parser.add_argument("--test-hours", type=float, default=10.0)
    parser.add_argument("--max-rows-per-split", type=int, default=0)
    parser.add_argument("--hop-seconds", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()

    root = resolve(args.librispeech_root)
    if not root.exists():
        raise FileNotFoundError(root)
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
    window_samples = int(round(float(cfg.get("segment_seconds", 3.0)) * 16000))
    hop_samples = max(1, int(round(float(args.hop_seconds) * 16000)))
    speakers = sorted({path.relative_to(root).parts[0] for path in root.rglob("*.flac")})
    split_map = split_speakers(speakers, int(args.seed))
    target_hours = {"train": float(args.train_hours), "val": float(args.val_hours), "test": float(args.test_hours)}
    summaries: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        candidates = collect_candidates(root, split_map[split], hop_samples, window_samples, int(args.seed) + len(split))
        rows = select_rows(candidates, target_hours[split], float(args.hop_seconds), int(args.max_rows_per_split))
        # Group selected rows by source path to avoid rereading the same FLAC
        # repeatedly while keeping a deterministic manifest order.
        rows.sort(key=lambda row: (str(row[0]), row[2]))
        summaries.append(
            build_split(
                split=split,
                rows=rows,
                stage1=stage1,
                enrollment=enrollment,
                stage1_cfg=cfg,
                device=device,
                output_cache=resolve(args.output_cache_dir),
                manifest_root=resolve(args.manifest_root),
                prefix=str(args.output_prefix),
                batch_size=int(args.batch_size),
            )
        )
    metadata = {
        "source": "LibriSpeech train-clean-100",
        "source_root": str(root),
        "source_url": "https://www.openslr.org/12",
        "source_license": "LibriSpeech license; verify before redistribution",
        "wake_phrase": "Marvin",
        "negative_only": True,
        "speaker_disjoint": True,
        "seed": int(args.seed),
        "hop_seconds": float(args.hop_seconds),
        "stage1_checkpoint": str(resolve(args.stage1_checkpoint)),
        "summaries": summaries,
    }
    output_meta = resolve(args.output_cache_dir) / "metadata.json"
    output_meta.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
