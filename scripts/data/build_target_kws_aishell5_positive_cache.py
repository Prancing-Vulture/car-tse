#!/usr/bin/env python
"""Build target-speaker-KWS positives by placing Marvin in real AISHELL cabins."""

from __future__ import annotations

import argparse
import hashlib
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


def read_session(root: Path, session_id: str, channels: list[str], max_seconds: float) -> np.ndarray:
    arrays = [np.asarray(read_mono(root / session_id / name, 16000)[0], dtype=np.float32) for name in channels]
    length = min(len(array) for array in arrays)
    length = min(length, int(round(max_seconds * 16000)))
    return np.stack([array[:length] for array in arrays], axis=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.yaml")
    parser.add_argument("--stage1-config", type=Path, default=REPO_ROOT / "configs/stage1_spex_dual_domain_rms010_ft1.yaml")
    parser.add_argument("--stage1-checkpoint", type=Path, default=REPO_ROOT / "checkpoints/stage1_spex_dual_domain_rms010_ft1/best.pt")
    parser.add_argument("--aishell-root", type=Path, default=REPO_ROOT / "data/raw/aishell5/extracted")
    parser.add_argument("--marvin-root", type=Path, default=REPO_ROOT / "data/raw/speech_commands_v2/extracted/marvin")
    parser.add_argument("--output-cache-dir", type=Path, default=REPO_ROOT / "data/cache/stage3_target_kws_aishell5_positive_16k")
    parser.add_argument("--output-prefix", default="stage3_target_kws_aishell5_positive")
    parser.add_argument("--channels", nargs="+", default=["DX01C01.wav", "DX02C01.wav", "DX03C01.wav", "DX04C01.wav"])
    parser.add_argument("--max-seconds-per-session", type=float, default=120.0)
    parser.add_argument("--max-windows-per-split", type=int, default=768)
    parser.add_argument("--max-windows-per-session", type=int, default=128)
    parser.add_argument("--hop-seconds", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()

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

    marvin_paths = sorted(resolve(args.marvin_root).glob("*.wav"))
    if not marvin_paths:
        raise RuntimeError("No Marvin clips found")
    by_speaker: dict[str, list[Path]] = {}
    for path in marvin_paths:
        by_speaker.setdefault(path.stem.split("_")[0], []).append(path)
    rng = np.random.default_rng(int(args.seed))
    speech_cache: dict[Path, np.ndarray] = {}
    def speech(path: Path) -> np.ndarray:
        if path not in speech_cache:
            speech_cache[path] = np.asarray(read_mono(path, 16000)[0], dtype=np.float32)
        return speech_cache[path]

    output_cache = resolve(args.output_cache_dir)
    output_cache.mkdir(parents=True, exist_ok=True)
    manifest_root = resolve(cfg["paths"]["manifest_root"])
    root = resolve(args.aishell_root)
    window_samples = int(round(float(cfg.get("segment_seconds", 3.0)) * 16000))
    hop_samples = max(1, int(round(args.hop_seconds * 16000)))
    split_map = {"train": "train", "val": "Dev", "test": "Eval1"}
    mic_gains = np.asarray([1.0, 0.90, 0.75, 0.60], dtype=np.float32)
    snr_values = np.asarray([-5.0, 0.0, 5.0, 10.0], dtype=np.float32)

    for output_split, source_split in split_map.items():
        rows: list[tuple[str, float, np.ndarray, Path, Path, float]] = []
        split_root = root / source_split
        for session_path in sorted(path for path in split_root.iterdir() if path.is_dir()):
            paths = [session_path / name for name in args.channels]
            if not all(path.exists() for path in paths):
                continue
            data = read_session(root / source_split, session_path.name, args.channels, args.max_seconds_per_session)
            starts = list(range(0, max(1, len(data)), hop_samples))
            if len(starts) > int(args.max_windows_per_session):
                positions = np.linspace(0, len(starts) - 1, int(args.max_windows_per_session), dtype=np.int64)
                starts = [starts[int(position)] for position in positions]
            for start in starts:
                window = np.zeros((data.shape[1], window_samples), dtype=np.float32)
                end = min(start + window_samples, len(data))
                if end > start:
                    window[:, : end - start] = data[start:end].T
                speaker_id = str(rng.choice(sorted(by_speaker)))
                wake_path = by_speaker[speaker_id][int(rng.integers(len(by_speaker[speaker_id])))]
                candidates = [path for path in by_speaker[speaker_id] if path != wake_path]
                enroll_path = candidates[int(rng.integers(len(candidates)))] if candidates else wake_path
                snr = float(rng.choice(snr_values))
                rows.append((session_path.name, start / 16000.0, window, wake_path, enroll_path, snr))
                if len(rows) >= int(args.max_windows_per_split):
                    break
            if len(rows) >= int(args.max_windows_per_split):
                break
        if not rows:
            raise RuntimeError(f"No AISHELL-5 rows found for split {source_split}")

        wav_cache = np.lib.format.open_memmap(output_cache / f"{output_split}_wav_16000hz.npy", mode="w+", dtype=np.float32, shape=(len(rows), 16000))
        enroll_cache = np.lib.format.open_memmap(output_cache / f"{output_split}_enrollment_embedding.npy", mode="w+", dtype=np.float32, shape=(len(rows), 256))
        extracted_cache = np.lib.format.open_memmap(output_cache / f"{output_split}_extracted_embedding.npy", mode="w+", dtype=np.float32, shape=(len(rows), 256))
        manifest_rows: list[dict] = []
        with torch.inference_mode():
            for batch_start in range(0, len(rows), max(1, int(args.batch_size))):
                chunk = rows[batch_start : batch_start + int(args.batch_size)]
                mixtures = []
                enrollments = []
                for _, _, cabin, wake_path, enroll_path, snr in chunk:
                    wake = speech(wake_path)
                    wake = np.pad(wake[:16000], (0, max(0, 16000 - len(wake[:16000]))))
                    wake_rms = float(np.sqrt(np.mean(wake * wake) + 1.0e-8))
                    cabin_rms = float(np.sqrt(np.mean(cabin * cabin) + 1.0e-8))
                    gain = cabin_rms / wake_rms * (10.0 ** (snr / 20.0))
                    placed = np.zeros(window_samples, dtype=np.float32)
                    start = int(round(0.20 * 16000))
                    placed[start : start + min(len(wake), window_samples - start)] = wake[: max(0, window_samples - start)]
                    mixture = cabin + mic_gains[:, None] * gain * placed[None, :]
                    mixtures.append(mixture.astype(np.float32))
                    enrollments.append(segment_waveform(speech(enroll_path), 24000, training=False).astype(np.float32))
                multichannel = torch.from_numpy(np.stack(mixtures, axis=0)).to(device)
                mixture = multichannel.mean(dim=1)
                enrollment = torch.from_numpy(np.stack(enrollments, axis=0)).to(device)
                extracted, _, _, _ = stage1(mixture, enrollment, multichannel=multichannel)
                conditioned = normalize_extracted_waveform(extracted)[..., :16000]
                if conditioned.shape[-1] < 16000:
                    conditioned = torch.nn.functional.pad(conditioned, (0, 16000 - conditioned.shape[-1]))
                enroll_embedding = stage1.speaker_embedding(enrollment)
                extracted_embedding = stage1.speaker_embedding(extracted)
                wav_cache[batch_start : batch_start + len(chunk)] = conditioned.cpu().numpy().astype(np.float32)
                enroll_cache[batch_start : batch_start + len(chunk)] = enroll_embedding.cpu().numpy().astype(np.float32)
                extracted_cache[batch_start : batch_start + len(chunk)] = extracted_embedding.cpu().numpy().astype(np.float32)
                for index, (session_id, offset, _, wake_path, enroll_path, snr) in enumerate(chunk):
                    digest = hashlib.sha1(str(wake_path).encode("utf-8")).hexdigest()[:10]
                    manifest_rows.append({
                        "label": 1,
                        "source_manifest": f"AISHELL-5/{source_split}",
                        "source_id": f"aishell5_positive_{source_split}_{session_id}_{int(round(offset))}_{digest}",
                        "source_domain": "aishell5_real_cabin_synthetic_target_present",
                        "session_id": session_id,
                        "window_start_seconds": offset,
                        "wake_source": str(wake_path),
                        "enrollment_source": str(enroll_path),
                        "target_snr_db": snr,
                    })
        wav_cache.flush()
        enroll_cache.flush()
        extracted_cache.flush()
        del wav_cache, enroll_cache, extracted_cache
        manifest_path = manifest_root / output_split / f"{args.output_prefix}_{output_split}.jsonl"
        write_jsonl(manifest_path, manifest_rows)
        print(f"{output_split}: {len(manifest_rows)} synthetic target-present AISHELL rows -> {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
