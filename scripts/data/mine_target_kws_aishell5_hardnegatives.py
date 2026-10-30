#!/usr/bin/env python
"""Mine high-scoring target-absent AISHELL windows for wake-model retraining."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.io.manifest import read_jsonl, write_jsonl
from car_tse.models.kws_model import TargetSpeakerKWSModel
from car_tse.train.utils import get_device, load_checkpoint


class CachedNegativeDataset(Dataset):
    def __init__(self, cache_dir: Path, split: str, sample_rate: int):
        self.wav = np.load(cache_dir / f"{split}_wav_{sample_rate}hz.npy", mmap_mode="r", allow_pickle=False)
        self.enrollment = np.load(cache_dir / f"{split}_enrollment_embedding.npy", mmap_mode="r", allow_pickle=False)
        self.extracted = np.load(cache_dir / f"{split}_extracted_embedding.npy", mmap_mode="r", allow_pickle=False)

    def __len__(self) -> int:
        return len(self.wav)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "wav": torch.from_numpy(np.array(self.wav[index], dtype=np.float32, copy=True)),
            "enrollment": torch.from_numpy(np.array(self.enrollment[index], dtype=np.float32, copy=True)),
            "extracted": torch.from_numpy(np.array(self.extracted[index], dtype=np.float32, copy=True)),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/stage3_target_kws_gsc_marvin_hardneg_aishell5.yaml")
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "checkpoints/stage3_target_kws_gsc_marvin_hardneg_aishell5_ft1/best.pt")
    parser.add_argument("--input-cache-dir", type=Path, default=REPO_ROOT / "data/cache/stage3_target_kws_aishell5_negative_16k")
    parser.add_argument("--input-prefix", default="stage3_target_kws_aishell5_negative")
    parser.add_argument("--output-cache-dir", type=Path, default=REPO_ROOT / "data/cache/stage3_target_kws_aishell5_mined_16k")
    parser.add_argument("--output-prefix", default="stage3_target_kws_aishell5_mined")
    parser.add_argument("--max-per-split", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    cfg = load_stage_config(args.config, REPO_ROOT / "configs/paths.yaml")
    input_cache = args.input_cache_dir if args.input_cache_dir.is_absolute() else REPO_ROOT / args.input_cache_dir
    output_cache = args.output_cache_dir if args.output_cache_dir.is_absolute() else REPO_ROOT / args.output_cache_dir
    output_cache.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else REPO_ROOT / args.checkpoint
    device = get_device()
    model = TargetSpeakerKWSModel(
        sample_rate=int(cfg["sample_rate"]),
        n_mels=int(cfg["model"].get("n_mels", 40)),
        channels=cfg["model"].get("channels", [32, 64, 128]),
        num_classes=int(cfg["model"].get("num_classes", 2)),
        speaker_embedding_dim=int(cfg["model"].get("speaker_embedding_dim", 256)),
        hidden=int(cfg["model"].get("hidden", 256)),
    ).to(device)
    load_checkpoint(checkpoint, model)
    model.eval()
    manifest_root = Path(cfg["paths"]["manifest_root"])
    if not manifest_root.is_absolute():
        manifest_root = REPO_ROOT / manifest_root

    for split in ("train", "val", "test"):
        dataset = CachedNegativeDataset(input_cache, split, int(cfg["sample_rate"]))
        loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False)
        scores: list[float] = []
        with torch.inference_mode():
            for batch in loader:
                logits = model(
                    batch["wav"].to(device),
                    batch["enrollment"].to(device),
                    batch["extracted"].to(device),
                )
                scores.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
        scores_array = np.asarray(scores, dtype=np.float32)
        limit = min(int(args.max_per_split), len(scores_array))
        selected = np.argsort(-scores_array, kind="stable")[:limit]
        wav = np.load(input_cache / f"{split}_wav_{int(cfg['sample_rate'])}hz.npy", mmap_mode="r", allow_pickle=False)
        enrollment = np.load(input_cache / f"{split}_enrollment_embedding.npy", mmap_mode="r", allow_pickle=False)
        extracted = np.load(input_cache / f"{split}_extracted_embedding.npy", mmap_mode="r", allow_pickle=False)
        output_wav = np.lib.format.open_memmap(
            output_cache / f"{split}_wav_{int(cfg['sample_rate'])}hz.npy",
            mode="w+", dtype=np.float32, shape=(limit,) + tuple(wav.shape[1:]),
        )
        output_enrollment = np.lib.format.open_memmap(
            output_cache / f"{split}_enrollment_embedding.npy",
            mode="w+", dtype=np.float32, shape=(limit,) + tuple(enrollment.shape[1:]),
        )
        output_extracted = np.lib.format.open_memmap(
            output_cache / f"{split}_extracted_embedding.npy",
            mode="w+", dtype=np.float32, shape=(limit,) + tuple(extracted.shape[1:]),
        )
        output_wav[:] = np.asarray(wav[selected], dtype=np.float32)
        output_enrollment[:] = np.asarray(enrollment[selected], dtype=np.float32)
        output_extracted[:] = np.asarray(extracted[selected], dtype=np.float32)
        output_wav.flush()
        output_enrollment.flush()
        output_extracted.flush()
        del output_wav, output_enrollment, output_extracted
        source_rows = read_jsonl(manifest_root / split / f"{args.input_prefix}_{split}.jsonl")
        rows = []
        for index in selected.tolist():
            row = dict(source_rows[int(index)])
            row["source_domain"] = "aishell5_real_cabin_target_absent_mined"
            row["mined_from_index"] = int(index)
            row["mined_score"] = float(scores_array[int(index)])
            rows.append(row)
        write_jsonl(manifest_root / split / f"{args.output_prefix}_{split}.jsonl", rows)
        print(
            f"{split}: selected {limit}/{len(scores_array)} negatives; "
            f"score range {float(scores_array[selected[-1]]):.4f}-{float(scores_array[selected[0]]):.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
