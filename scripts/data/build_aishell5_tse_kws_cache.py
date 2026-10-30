#!/usr/bin/env python
"""Cache normalized Stage 1 outputs from continuous AISHELL-5 sessions.

The resulting rows are target-absent KWS negatives.  Sessions are supplied by
the caller so train/validation/test speakers can remain disjoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import soundfile as sf
import torch

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.segment import segment_waveform
from car_tse.io.audio import read_mono
from car_tse.io.manifest import write_jsonl
from car_tse.models.stage1_system import Stage1System
from car_tse.pipeline.in_car_pipeline import normalize_extracted_waveform
from car_tse.train.utils import get_device, load_checkpoint


def resolve_path(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def session_paths(root: Path, session_id: str, channels: list[str]) -> list[Path]:
    paths = [root / session_id / name for name in channels]
    if not all(path.exists() for path in paths):
        raise FileNotFoundError(f"Missing AISHELL-5 session {session_id}: {paths}")
    return paths


def window_starts(length: int, window_samples: int, hop_samples: int) -> list[int]:
    return list(range(0, max(1, length), hop_samples))


def load_multichannel(paths: list[Path], max_seconds: float) -> tuple[np.ndarray, float]:
    arrays = []
    for path in paths:
        wav, _ = read_mono(path, 16000)
        arrays.append(np.asarray(wav, dtype=np.float32))
    length = min(len(wav) for wav in arrays)
    limit = min(length, int(round(max_seconds * 16000)))
    return np.stack([wav[:limit] for wav in arrays], axis=1), limit / 16000.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--dataset-split", default="train", choices=("train", "Dev", "Eval1", "Eval2"))
    parser.add_argument("--output-split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--session-ids", nargs="+", required=True)
    parser.add_argument("--max-seconds", type=float, default=1200.0)
    parser.add_argument("--aishell-root", type=Path, default=REPO_ROOT / "data" / "raw" / "aishell5" / "extracted")
    parser.add_argument("--output-cache-dir", type=Path, default=REPO_ROOT / "data" / "cache" / "stage3_kws_aishell5_16k")
    parser.add_argument("--manifest-root", type=Path, default=REPO_ROOT / "manifests")
    parser.add_argument("--manifest-name", default="stage3_kws_aishell5_{split}.jsonl")
    parser.add_argument("--stage1-config", type=Path, default=REPO_ROOT / "configs" / "stage1_spex_transfer_mc_cavemove.yaml")
    parser.add_argument("--stage1-checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "stage1_spex_transfer_mc_cavemove_cabin" / "best.pt")
    parser.add_argument("--enrollment-wav", type=Path, default=REPO_ROOT / "data" / "raw" / "librispeech" / "extracted" / "LibriSpeech" / "test-clean" / "1089" / "134686" / "1089-134686-0014.flac")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hop-seconds", type=float, default=1.0)
    parser.add_argument("--channels", nargs="+", default=["DX01C01.wav", "DX02C01.wav", "DX03C01.wav", "DX04C01.wav"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    stage1_cfg = load_stage_config(args.stage1_config, args.paths)
    device = get_device()
    sample_rate = 16000
    stage1_seconds = float(stage1_cfg.get("segment_seconds", 3.0))
    window_samples = int(round(stage1_seconds * sample_rate))
    hop_samples = max(1, int(round(float(args.hop_seconds) * sample_rate)))
    out_dir = resolve_path(args.output_cache_dir)
    manifest_root = resolve_path(args.manifest_root)
    output_cache = out_dir / f"{args.output_split}_cabin_wav_{sample_rate}hz.npy"
    output_manifest = manifest_root / args.output_split / args.manifest_name.format(split=args.output_split)

    session_root = resolve_path(args.aishell_root) / args.dataset_split
    session_infos: list[tuple[str, list[Path], float, int]] = []
    remaining = float(args.max_seconds)
    total_rows = 0
    for session_id in args.session_ids:
        if remaining <= 0:
            break
        paths = session_paths(session_root, session_id, list(args.channels))
        info = sf.info(str(paths[0]))
        duration = min(float(info.duration), remaining)
        rows = len(window_starts(int(round(duration * sample_rate)), window_samples, hop_samples))
        session_infos.append((session_id, paths, duration, rows))
        total_rows += rows
        remaining -= duration
    if not session_infos:
        raise RuntimeError("No AISHELL-5 sessions selected")

    if output_cache.exists() and output_manifest.exists() and not args.force:
        try:
            cached = np.load(output_cache, mmap_mode="r", allow_pickle=False)
            if tuple(cached.shape) == (total_rows, sample_rate):
                print(f"Already cached {args.output_split}: {output_cache}", flush=True)
                return 0
        except (OSError, ValueError):
            pass

    pretrained = resolve_path(stage1_cfg["pretrained_checkpoint"])
    stage1 = Stage1System(
        spex_cfg=stage1_cfg["spex"], sample_rate=sample_rate, use_cabin_frontend=True,
        num_mics=len(args.channels), frontend_cfg=stage1_cfg.get("frontend"),
        pretrained_dir=str(pretrained), backend=stage1_cfg.get("spex_backend", "transfer_spexplus_16k"),
    ).to(device)
    load_checkpoint(resolve_path(args.stage1_checkpoint), stage1)
    stage1.eval()
    enrollment, _ = read_mono(resolve_path(args.enrollment_wav), sample_rate)
    enrollment = segment_waveform(enrollment, window_samples // 2, training=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(output_cache, mode="w+", dtype=np.float32, shape=(total_rows, sample_rate))
    manifest_rows: list[dict] = []
    offset = 0
    for session_id, paths, duration, row_count in session_infos:
        multichannel, actual_seconds = load_multichannel(paths, duration)
        starts = window_starts(len(multichannel), window_samples, hop_samples)
        for batch_start in range(0, len(starts), max(1, int(args.batch_size))):
            batch_starts = starts[batch_start : batch_start + args.batch_size]
            batch_mc = np.zeros((len(batch_starts), len(args.channels), window_samples), dtype=np.float32)
            for local, start in enumerate(batch_starts):
                end = min(start + window_samples, len(multichannel))
                batch_mc[local, :, : end - start] = multichannel[start:end].T
            batch_mix = batch_mc.mean(axis=1)
            batch_enroll = np.repeat(enrollment[None, :], len(batch_starts), axis=0)
            with torch.inference_mode():
                extracted, _, _, _ = stage1(
                    torch.from_numpy(batch_mix).to(device),
                    torch.from_numpy(batch_enroll).to(device),
                    multichannel=torch.from_numpy(batch_mc).to(device),
                )
                conditioned = normalize_extracted_waveform(extracted)[..., :sample_rate].cpu().numpy().astype(np.float32)
            for local, start in enumerate(batch_starts):
                output[offset] = conditioned[local]
                manifest_rows.append(
                    {
                        "wav": str(paths[0]),
                        "label": 0,
                        "split": args.output_split,
                        "source_aishell5": True,
                        "session_id": session_id,
                        "start_seconds": start / sample_rate,
                        "duration_seconds": actual_seconds,
                    }
                )
                offset += 1
        print(f"{args.output_split}/{session_id}: {len(starts)} rows", flush=True)
    output.flush()
    del output
    write_jsonl(output_manifest, manifest_rows)
    print(f"Wrote {output_cache} and {output_manifest}: {len(manifest_rows)} negatives", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
