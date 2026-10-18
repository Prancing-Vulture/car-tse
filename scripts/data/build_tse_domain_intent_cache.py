#!/usr/bin/env python
"""Cache Stage 1 extracted audio for the intent classifier's input domain."""

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


def noise_cache_path(cache_dir: Path, source: str) -> Path:
    digest = hashlib.sha1(str(Path(source)).encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.npy"


def make_inputs(
    row: dict,
    noise_cache_dir: Path,
    sample_rate: int,
    segment_seconds: float,
    channels: list[int],
    index: int,
    training: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_samples = int(sample_rate * segment_seconds)
    speech, _ = read_mono(row["wav"], sample_rate)
    # Keep the enrollment and command on a deterministic, aligned crop.  The
    # intent model's input is the first two seconds of the extracted result.
    mixture = segment_waveform(speech, num_samples, training=training, rng=np.random.default_rng(2026 + index * 7919))
    enrollment = segment_waveform(
        speech,
        num_samples // 2,
        training=False,
        rng=np.random.default_rng(2027 + index * 7919),
    )

    source = row.get("cabin_noise_wav")
    if not source:
        raise ValueError(f"Missing cabin_noise_wav at row {index}")
    cached = noise_cache_path(noise_cache_dir, source)
    noise = np.load(cached, mmap_mode="r", allow_pickle=False) if cached.exists() else read_multichannel(source, sample_rate)[0]
    if noise.ndim == 1:
        noise = noise[:, None]
    requested = [int(x) for x in row.get("cabin_noise_channels", channels)]
    valid = [x for x in requested if 0 <= x < noise.shape[1]]
    if not valid:
        valid = list(range(min(len(channels), noise.shape[1])))
    noise = np.asarray(noise[:, valid], dtype=np.float32)
    if noise.shape[1] < len(channels):
        noise = np.tile(noise, (1, int(np.ceil(len(channels) / max(1, noise.shape[1])))))[:, : len(channels)]
    elif noise.shape[1] > len(channels):
        noise = noise[:, : len(channels)]

    if len(noise) >= num_samples:
        if training:
            rng = np.random.default_rng(3031 + index * 7919)
            start = int(rng.integers(0, len(noise) - num_samples + 1))
        else:
            start = (len(noise) - num_samples) // 2
        noise = np.asarray(noise[start : start + num_samples], dtype=np.float32)
    else:
        noise = np.tile(noise, (int(np.ceil(num_samples / max(1, len(noise)))), 1))[:num_samples]

    snr_db = float(row.get("cabin_noise_snr_db", 0.0))
    speech_rms = float(np.sqrt(np.mean(mixture * mixture) + 1e-8))
    noise_rms = float(np.sqrt(np.mean(noise * noise) + 1e-8))
    if speech_rms > 1e-6 and noise_rms > 1e-6:
        noise = noise * (speech_rms * (10.0 ** (-snr_db / 20.0)) / noise_rms)
    return mixture, enrollment, mixture[:, None] + noise


def cache_split(
    split: str,
    rows: list[dict],
    output_path: Path,
    output_manifest: Path,
    model: Stage1System,
    device: torch.device,
    sample_rate: int,
    stage1_seconds: float,
    intent_seconds: float,
    channels: list[int],
    noise_cache_dir: Path,
    batch_size: int,
    limit: int | None,
) -> None:
    if limit is not None:
        rows = rows[:limit]
    stage1_samples = int(sample_rate * stage1_seconds)
    intent_samples = int(sample_rate * intent_seconds)
    expected = (len(rows), intent_samples)
    if output_path.exists() and output_manifest.exists():
        try:
            cached = np.load(output_path, mmap_mode="r", allow_pickle=False)
            if tuple(cached.shape) == expected:
                print(f"Already cached {split}: {output_path}", flush=True)
                return
        except (OSError, ValueError):
            pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=expected)
    manifest_rows: list[dict] = []
    mix_batch: list[np.ndarray] = []
    enroll_batch: list[np.ndarray] = []
    mc_batch: list[np.ndarray] = []
    row_batch: list[dict] = []

    def flush() -> None:
        if not mix_batch:
            return
        with torch.inference_mode():
            mixture = torch.from_numpy(np.stack(mix_batch)).to(device)
            enrollment = torch.from_numpy(np.stack(enroll_batch)).to(device)
            multichannel = torch.from_numpy(np.stack(mc_batch).transpose(0, 2, 1)).to(device)
            extracted, _, _, _ = model(mixture, enrollment, multichannel=multichannel)
        extracted = extracted[:, :intent_samples].detach().cpu().numpy().astype(np.float32)
        base = len(manifest_rows)
        for local, waveform in enumerate(extracted):
            peak = float(np.max(np.abs(waveform)))
            if peak > 0.95:
                waveform = waveform * (0.95 / peak)
            output[base + local] = waveform
            source_row = row_batch[local]
            manifest_rows.append(
                {
                    "wav": source_row["wav"],
                    "label": int(source_row["label"]),
                    "intent": source_row.get("intent", ""),
                    "condition": "stage1_tse_cavemove_cabin",
                    "source_stage1_tse": True,
                    "cabin_noise_snr_db": source_row.get("cabin_noise_snr_db"),
                }
            )
        mix_batch.clear()
        enroll_batch.clear()
        mc_batch.clear()
        row_batch.clear()

    for index, row in enumerate(rows):
        mixture, enrollment, multichannel = make_inputs(
            row,
            noise_cache_dir,
            sample_rate,
            stage1_seconds,
            channels,
            index,
            training=split == "train",
        )
        # Stage 1 uses a 3-second context; the intent network receives a
        # fixed 2-second extracted segment from the same forward pass.
        if len(mixture) != stage1_samples:
            raise ValueError(f"Unexpected mixture shape at {split}:{index}: {mixture.shape}")
        mix_batch.append(mixture)
        enroll_batch.append(enrollment)
        mc_batch.append(multichannel)
        row_batch.append(row)
        if len(mix_batch) >= batch_size:
            flush()
        if (index + 1) % 500 == 0 or index + 1 == len(rows):
            print(f"{split}: prepared {index + 1}/{len(rows)}", flush=True)
    flush()
    output.flush()
    del output
    write_jsonl(output_manifest, manifest_rows)
    print(f"Wrote {split}: {len(manifest_rows)} -> {output_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "stage4_intent_cavemove.yaml")
    parser.add_argument("--stage1-config", type=Path, default=REPO_ROOT / "configs" / "stage1_spex_transfer_mc_cavemove.yaml")
    parser.add_argument("--stage1-checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "stage1_spex_transfer_mc_cavemove_cabin" / "best.pt")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--noise-cache-dir", type=Path, default=REPO_ROOT / "data" / "cache" / "cavemove_16k_mc")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "cache" / "stage4_intent_tse_domain_16k")
    parser.add_argument(
        "--train-manifest-out",
        type=Path,
        default=None,
        help="Optional output manifest for the generated train cache; defaults to the legacy filename.",
    )
    parser.add_argument(
        "--val-manifest-out",
        type=Path,
        default=None,
        help="Optional output manifest for the generated validation cache; defaults to the legacy filename.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_stage_config(args.config, args.paths)
    stage1_cfg = load_stage_config(args.stage1_config, args.paths)
    paths = cfg["paths"]
    device = get_device()
    sample_rate = int(stage1_cfg["sample_rate"])
    channels = [int(x) for x in stage1_cfg["cabin_noise_channels"]]
    pretrained = Path(stage1_cfg["pretrained_checkpoint"])
    if not pretrained.is_absolute():
        pretrained = REPO_ROOT / pretrained
    model = Stage1System(
        spex_cfg=stage1_cfg["spex"],
        sample_rate=sample_rate,
        use_cabin_frontend=True,
        num_mics=int(stage1_cfg["num_mics"]),
        frontend_cfg=stage1_cfg.get("frontend"),
        pretrained_dir=str(pretrained),
        backend=stage1_cfg["spex_backend"],
    ).to(device)
    load_checkpoint(args.stage1_checkpoint, model)
    model.eval()
    print(f"device={device} stage1_checkpoint={args.stage1_checkpoint}", flush=True)

    manifest_root = Path(paths["manifest_root"])
    noise_cache_dir = args.noise_cache_dir
    source_train = cfg.get("source_manifest_train", "stage4_cavemove_intent.jsonl")
    source_val = cfg.get("source_manifest_eval", "stage4_cavemove_intent.jsonl")
    train_rows = read_jsonl(manifest_root / "train" / source_train)
    val_rows = read_jsonl(manifest_root / "val" / source_val)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_manifest_out = args.train_manifest_out or (manifest_root / "train" / "stage4_intent_tse_domain_train.jsonl")
    val_manifest_out = args.val_manifest_out or (manifest_root / "val" / "stage4_intent_tse_domain_eval.jsonl")
    if not train_manifest_out.is_absolute():
        train_manifest_out = REPO_ROOT / train_manifest_out
    if not val_manifest_out.is_absolute():
        val_manifest_out = REPO_ROOT / val_manifest_out
    cache_split(
        "train", train_rows,
        args.output_dir / f"train_cabin_wav_{sample_rate}hz.npy",
        train_manifest_out,
        model, device, sample_rate, float(stage1_cfg["segment_seconds"]), float(cfg["segment_seconds"]),
        channels, noise_cache_dir, args.batch_size, args.train_limit,
    )
    cache_split(
        "val", val_rows,
        args.output_dir / f"val_cabin_wav_{sample_rate}hz.npy",
        val_manifest_out,
        model, device, sample_rate, float(stage1_cfg["segment_seconds"]), float(cfg["segment_seconds"]),
        channels, noise_cache_dir, args.batch_size, args.val_limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
