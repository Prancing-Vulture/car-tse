#!/usr/bin/env python
"""Cache fixed Stage 1 embeddings for fast presence-head experiments.

This avoids recomputing the frozen Stage 1 model for every presence-head epoch.
Positive rows are rendered as same-speaker ``marvin`` clips plus a short
CAVEMOVE noise segment, or as real AISHELL-5 far-field target-present segments
when ``mixture_multich_wavs`` and ``segment_start_seconds`` are present.
Negative rows read a short four-channel AISHELL-5 segment.  The cache stores
the raw three-second Stage 1 output because that is what the runtime presence
head consumes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.io.audio import read_mono
from car_tse.io.manifest import read_jsonl
from car_tse.models.stage1_system import Stage1System
from car_tse.train.utils import get_device, load_checkpoint


def resolve_path(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def fit_center(wave: np.ndarray, samples: int) -> np.ndarray:
    wave = np.asarray(wave, dtype=np.float32).reshape(-1)
    if len(wave) >= samples:
        start = (len(wave) - samples) // 2
        return wave[start : start + samples].copy()
    out = np.zeros(samples, dtype=np.float32)
    start = (samples - len(wave)) // 2
    out[start : start + len(wave)] = wave
    return out


def read_segment(path: Path, sample_rate: int, samples: int, start_frame: int = 0) -> np.ndarray:
    with sf.SoundFile(str(path), mode="r") as handle:
        if handle.samplerate != sample_rate:
            raise ValueError(f"Expected {sample_rate} Hz, got {handle.samplerate} Hz for {path}")
        total = len(handle)
        if total <= 0:
            return np.zeros(samples, dtype=np.float32)
        if total >= samples:
            start = min(max(0, int(start_frame)), total - samples)
            handle.seek(start)
            data = handle.read(samples, dtype="float32", always_2d=True)
        else:
            handle.seek(0)
            data = handle.read(total, dtype="float32", always_2d=True)
    out = np.zeros((samples, data.shape[1] if data.ndim == 2 else 1), dtype=np.float32)
    out[: len(data)] = data
    return out


def read_enrollment(path: Path, sample_rate: int, samples: int, start_frame: int | None = None) -> np.ndarray:
    if start_frame is None:
        audio, _ = read_mono(path, sample_rate)
        return fit_center(audio, samples)
    segment = read_segment(path, sample_rate, samples, start_frame)
    return fit_center(segment.mean(axis=1), samples)


def render_row(
    row: dict[str, Any],
    index: int,
    sample_rate: int,
    window_samples: int,
    enrollment_samples: int,
    noise_cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label = int(row.get("target_present", 0))
    enroll_start = row.get("enroll_start_seconds")
    enroll_start_frame = None if enroll_start is None else int(round(float(enroll_start) * sample_rate))
    enroll = read_enrollment(
        resolve_path(row["enroll_wav"]), sample_rate, enrollment_samples, start_frame=enroll_start_frame
    )
    if label:
        real_paths = row.get("mixture_multich_wavs")
        real_start = row.get("segment_start_seconds")
        if real_paths and real_start is not None:
            start_frame = int(round(float(real_start) * sample_rate))
            arrays = [
                read_segment(resolve_path(path), sample_rate, window_samples, start_frame)[:, 0]
                for path in list(real_paths)[:4]
            ]
            if not arrays:
                raise ValueError(f"Real target-present row has no channels: {row.get('id', '<unknown>')}")
            multichannel = np.stack(arrays, axis=1)
            if multichannel.shape[1] < 4:
                multichannel = np.tile(
                    multichannel, (1, int(np.ceil(4 / multichannel.shape[1])))
                )[:, :4]
            mixture = multichannel[:, :4].mean(axis=1)
            return mixture.astype(np.float32), enroll.astype(np.float32), multichannel[:, :4].T.astype(np.float32)
        target = fit_center(read_mono(resolve_path(row["mixture_wav"]), sample_rate)[0], window_samples)
        noise_path = resolve_path(row["cabin_noise_wav"])
        cache_key = str(noise_path)
        noise = noise_cache.get(cache_key)
        if noise is None:
            noise = read_segment(noise_path, sample_rate, window_samples, start_frame=(index * sample_rate) % max(1, sf.info(str(noise_path)).frames))
            noise_cache[cache_key] = noise
        channels = [int(x) for x in row.get("cabin_noise_channels", [0, 2, 4, 6])]
        valid = [x for x in channels if x < noise.shape[1]]
        if not valid:
            valid = list(range(noise.shape[1]))
        noise = noise[:, valid]
        if noise.shape[1] < 4:
            noise = np.tile(noise, (1, int(np.ceil(4 / max(1, noise.shape[1])))))[:, :4]
        else:
            noise = noise[:, :4]
        speech_rms = float(np.sqrt(np.mean(target * target) + 1e-8))
        noise_rms = float(np.sqrt(np.mean(noise * noise) + 1e-8))
        snr_db = float(row.get("cabin_noise_snr_db", 0.0))
        scale = speech_rms * (10.0 ** (-snr_db / 20.0)) / max(noise_rms, 1e-6)
        multichannel = target[:, None] + noise * scale
        mixture = multichannel.mean(axis=1)
    else:
        paths = [resolve_path(path) for path in row.get("mixture_multich_wavs", [row["mixture_wav"]])]
        arrays = []
        for channel_index, path in enumerate(paths[:4]):
            info = sf.info(str(path))
            max_start = max(0, int(info.frames) - window_samples)
            start = (index * sample_rate) % max(1, max_start + 1)
            arrays.append(read_segment(path, sample_rate, window_samples, start)[:, 0])
        if not arrays:
            arrays = [fit_center(read_mono(resolve_path(row["mixture_wav"]), sample_rate)[0], window_samples)]
        multichannel = np.stack(arrays, axis=1)
        if multichannel.shape[1] < 4:
            multichannel = np.tile(multichannel, (1, int(np.ceil(4 / multichannel.shape[1]))))[:, :4]
        mixture = multichannel[:, :4].mean(axis=1)
    return mixture.astype(np.float32), enroll.astype(np.float32), multichannel[:, :4].T.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.yaml")
    parser.add_argument("--stage1-config", type=Path, default=REPO_ROOT / "configs/stage1_spex_transfer_mc_cavemove.yaml")
    parser.add_argument("--stage1-checkpoint", type=Path, default=REPO_ROOT / "checkpoints/stage1_spex_transfer_mc_cavemove_cabin/best.pt")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help=".npy Stage 1 embedding cache")
    parser.add_argument("--labels-output", type=Path, required=True, help=".npy label cache")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-mics", type=int, default=4)
    args = parser.parse_args()

    stage1_cfg = load_stage_config(args.stage1_config, args.paths)
    sample_rate = int(stage1_cfg.get("sample_rate", 16000))
    window_samples = int(round(float(stage1_cfg.get("segment_seconds", 3.0)) * sample_rate))
    enrollment_samples = window_samples // 2
    rows = read_jsonl(resolve_path(args.manifest))
    if not rows:
        raise RuntimeError(f"Empty manifest: {args.manifest}")
    device = get_device()
    pretrained = resolve_path(stage1_cfg["pretrained_checkpoint"])
    stage1 = Stage1System(
        spex_cfg=stage1_cfg["spex"],
        sample_rate=sample_rate,
        use_cabin_frontend=True,
        num_mics=int(args.num_mics),
        frontend_cfg=stage1_cfg.get("frontend"),
        pretrained_dir=str(pretrained),
        backend=stage1_cfg.get("spex_backend", "transfer_spexplus_16k"),
    ).to(device)
    load_checkpoint(resolve_path(args.stage1_checkpoint), stage1)
    stage1.eval()
    output_cache = resolve_path(args.output)
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    cache = np.lib.format.open_memmap(output_cache, mode="w+", dtype=np.float32, shape=(len(rows), window_samples))
    labels = np.zeros(len(rows), dtype=np.int64)
    noise_cache: dict[str, np.ndarray] = {}
    batch_mix: list[np.ndarray] = []
    batch_enroll: list[np.ndarray] = []
    batch_multichannel: list[np.ndarray] = []
    batch_indices: list[int] = []

    def flush() -> None:
        if not batch_indices:
            return
        with torch.inference_mode():
            extracted, _, _, _ = stage1(
                torch.from_numpy(np.stack(batch_mix)).to(device),
                torch.from_numpy(np.stack(batch_enroll)).to(device),
                multichannel=torch.from_numpy(np.stack(batch_multichannel)).to(device),
            )
        values = extracted.detach().cpu().numpy().astype(np.float32)
        for local, row_index in enumerate(batch_indices):
            cache[row_index] = values[local]
        batch_mix.clear()
        batch_enroll.clear()
        batch_multichannel.clear()
        batch_indices.clear()

    for index, row in enumerate(rows):
        mixture, enroll, multichannel = render_row(row, index, sample_rate, window_samples, enrollment_samples, noise_cache)
        batch_mix.append(mixture)
        batch_enroll.append(enroll)
        batch_multichannel.append(multichannel)
        batch_indices.append(index)
        labels[index] = int(row.get("target_present", 0))
        if len(batch_indices) >= max(1, int(args.batch_size)):
            flush()
        if (index + 1) % 128 == 0:
            print(f"rendered {index + 1}/{len(rows)}", flush=True)
    flush()
    cache.flush()
    del cache
    output_labels = resolve_path(args.labels_output)
    output_labels.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_labels, labels)
    metadata = output_cache.with_suffix(".json")
    metadata.write_text(
        json.dumps(
            {
                "manifest": str(resolve_path(args.manifest)),
                "rows": len(rows),
                "sample_rate": sample_rate,
                "window_samples": window_samples,
                "device": str(device),
                "positive_rows": int(labels.sum()),
                "negative_rows": int((labels == 0).sum()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {args.output} and {args.labels_output}: {len(rows)} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
