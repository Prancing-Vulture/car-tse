#!/usr/bin/env python
"""Build target-speaker-KWS negative caches from real AISHELL-5 cabin speech.

The recordings are target-absent relative to the fixed enrollment voices used
by the wake model.  Train/Dev/Eval1 are used for fitting and selection; Eval2
is intentionally left untouched for the final continuous false-accept check.
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


def load_session(root: Path, session_id: str, channels: list[str], max_seconds: float) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for name in channels:
        wav, _ = read_mono(root / session_id / name, 16000)
        arrays.append(np.asarray(wav, dtype=np.float32))
    length = min(len(wav) for wav in arrays)
    length = min(length, int(round(max_seconds * 16000)))
    return np.stack([wav[:length] for wav in arrays], axis=1)


def window_rows(
    data: np.ndarray,
    session_id: str,
    window_samples: int,
    hop_samples: int,
    max_windows: int,
) -> list[tuple[str, float, np.ndarray]]:
    starts = list(range(0, max(1, len(data)), hop_samples))
    if len(starts) > max_windows:
        # Deterministic coverage over the session rather than taking only the
        # first few seconds of a long recording.
        positions = np.linspace(0, len(starts) - 1, max_windows, dtype=np.int64)
        starts = [starts[int(position)] for position in positions]
    rows: list[tuple[str, float, np.ndarray]] = []
    for start in starts:
        window = np.zeros((data.shape[1], window_samples), dtype=np.float32)
        end = min(start + window_samples, len(data))
        if end > start:
            window[:, : end - start] = data[start:end].T
        rows.append((session_id, start / 16000.0, window))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.yaml")
    parser.add_argument("--stage1-config", type=Path, default=REPO_ROOT / "configs/stage1_spex_dual_domain_rms010_ft1.yaml")
    parser.add_argument("--stage1-checkpoint", type=Path, default=REPO_ROOT / "checkpoints/stage1_spex_dual_domain_rms010_ft1/best.pt")
    parser.add_argument("--aishell-root", type=Path, default=REPO_ROOT / "data/raw/aishell5/extracted")
    parser.add_argument("--enrollment-wavs", nargs="+", type=Path, default=[
        REPO_ROOT / "data/raw/librispeech/extracted/LibriSpeech/test-clean/1089/134686/1089-134686-0014.flac",
    ])
    parser.add_argument("--output-cache-dir", type=Path, default=REPO_ROOT / "data/cache/stage3_target_kws_aishell5_negative_16k")
    parser.add_argument("--output-prefix", default="stage3_target_kws_aishell5_negative")
    parser.add_argument("--session-ids", nargs="+", default=None)
    parser.add_argument("--channels", nargs="+", default=["DX01C01.wav", "DX02C01.wav", "DX03C01.wav", "DX04C01.wav"])
    parser.add_argument("--max-seconds-per-session", type=float, default=120.0)
    parser.add_argument("--max-windows-per-split", type=int, default=1024)
    parser.add_argument("--max-windows-per-session", type=int, default=256)
    parser.add_argument("--hop-seconds", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    cfg = load_stage_config(args.stage1_config, args.paths)
    device = get_device()
    stage1_checkpoint = resolve(args.stage1_checkpoint)
    pretrained = resolve(cfg["pretrained_checkpoint"])
    stage1 = Stage1System(
        spex_cfg=cfg["spex"],
        sample_rate=int(cfg["sample_rate"]),
        use_cabin_frontend=True,
        num_mics=int(cfg.get("num_mics", 4)),
        frontend_cfg=cfg.get("frontend"),
        pretrained_dir=str(pretrained),
        backend=cfg.get("spex_backend", "transfer_spexplus_16k"),
    ).to(device)
    load_checkpoint(stage1_checkpoint, stage1)
    stage1.eval()

    enrollments: list[torch.Tensor] = []
    for path in args.enrollment_wavs:
        wav, _ = read_mono(resolve(path), 16000)
        enrollments.append(torch.from_numpy(segment_waveform(wav, 24000, training=False).astype(np.float32)))
    if not enrollments:
        raise RuntimeError("At least one enrollment wav is required")

    root = resolve(args.aishell_root)
    output_cache = resolve(args.output_cache_dir)
    output_cache.mkdir(parents=True, exist_ok=True)
    manifest_root = resolve(cfg["paths"]["manifest_root"])
    window_samples = int(round(float(cfg.get("segment_seconds", 3.0)) * 16000))
    hop_samples = max(1, int(round(args.hop_seconds * 16000)))
    split_map = {"train": "train", "val": "Dev", "test": "Eval1"}

    for output_split, source_split in split_map.items():
        split_root = root / source_split
        session_ids = list(args.session_ids) if args.session_ids else sorted(
            path.name for path in split_root.iterdir() if path.is_dir()
        )
        session_rows: list[tuple[str, float, np.ndarray]] = []
        for session_id in session_ids:
            paths = [split_root / session_id / name for name in args.channels]
            if not all(path.exists() for path in paths):
                continue
            data = load_session(root / source_split, session_id, args.channels, args.max_seconds_per_session)
            session_rows.extend(
                window_rows(data, session_id, window_samples, hop_samples, int(args.max_windows_per_session))
            )
            if len(session_rows) >= int(args.max_windows_per_split):
                break
        session_rows = session_rows[: int(args.max_windows_per_split)]
        if not session_rows:
            raise RuntimeError(f"No AISHELL-5 rows found for split {source_split}")

        wav_cache = np.lib.format.open_memmap(
            output_cache / f"{output_split}_wav_16000hz.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(session_rows), 16000),
        )
        enroll_cache = np.lib.format.open_memmap(
            output_cache / f"{output_split}_enrollment_embedding.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(session_rows), 256),
        )
        extracted_cache = np.lib.format.open_memmap(
            output_cache / f"{output_split}_extracted_embedding.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(session_rows), 256),
        )
        manifest_rows: list[dict] = []
        with torch.inference_mode():
            for start in range(0, len(session_rows), max(1, int(args.batch_size))):
                chunk = session_rows[start : start + int(args.batch_size)]
                multichannel = torch.from_numpy(np.stack([row[2] for row in chunk], axis=0)).to(device)
                mixture = multichannel.mean(dim=1)
                enrollment = torch.stack(
                    [enrollments[(start + index) % len(enrollments)] for index in range(len(chunk))]
                ).to(device)
                extracted, _, _, _ = stage1(mixture, enrollment, multichannel=multichannel)
                conditioned = normalize_extracted_waveform(extracted)
                conditioned = conditioned[..., :16000]
                if conditioned.shape[-1] < 16000:
                    conditioned = torch.nn.functional.pad(conditioned, (0, 16000 - conditioned.shape[-1]))
                enrollment_embedding = stage1.speaker_embedding(enrollment)
                extracted_embedding = stage1.speaker_embedding(extracted)
                wav_cache[start : start + len(chunk)] = conditioned.detach().cpu().numpy().astype(np.float32)
                enroll_cache[start : start + len(chunk)] = enrollment_embedding.detach().cpu().numpy().astype(np.float32)
                extracted_cache[start : start + len(chunk)] = extracted_embedding.detach().cpu().numpy().astype(np.float32)
                for index, (session_id, offset, _) in enumerate(chunk):
                    manifest_rows.append(
                        {
                            "label": 0,
                            "source_manifest": f"AISHELL-5/{source_split}",
                            "source_id": f"aishell5_{source_split}_{session_id}_{int(round(offset))}",
                            "source_domain": "aishell5_real_cabin_target_absent",
                            "session_id": session_id,
                            "window_start_seconds": offset,
                        }
                    )
        wav_cache.flush()
        enroll_cache.flush()
        extracted_cache.flush()
        del wav_cache, enroll_cache, extracted_cache
        manifest_path = manifest_root / output_split / f"{args.output_prefix}_{output_split}.jsonl"
        write_jsonl(manifest_path, manifest_rows)
        print(
            f"{output_split}: {len(manifest_rows)} target-absent AISHELL rows -> {manifest_path}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
