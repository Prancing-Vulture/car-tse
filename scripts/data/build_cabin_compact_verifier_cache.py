#!/usr/bin/env python
"""Build compact-verifier wave and WavLM embedding caches on cabin-aware data.

The ``stage3_target_kws_gsc_marvin_hardneg_aishell5`` manifests contain
synthetic CAVEMOVE/GSC-Marvin pairs plus real AISHELL-5 target-absent cabin
windows.  AISHELL rows reference a pseudo-manifest marker rather than a JSONL
file, so their enrollment falls back to the fixed scripted enrollment clip.
Outputs are consumed by ``scripts/train/train_compact_speaker_verifier_distill.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from car_tse.data.segment import segment_waveform  # noqa: E402
from car_tse.io.audio import read_mono  # noqa: E402
from scripts.eval.eval_pretrained_wavlm_speaker_verifier import model_embeddings  # noqa: E402


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_rows(manifest: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class SourceIndex:
    def __init__(self) -> None:
        self._manifests: dict[str, dict[str, dict[str, Any]]] = {}

    def get(self, manifest: str, source_id: str) -> dict[str, Any] | None:
        path = Path(manifest)
        if not path.exists():
            return None
        if manifest not in self._manifests:
            self._manifests[manifest] = {
                str(row.get("id", "")): row for row in load_rows(path)
            }
        row = self._manifests[manifest].get(source_id)
        if row is None:
            raise KeyError(f"Missing source row {source_id} in {manifest}")
        return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, default=REPO_ROOT / "manifests")
    parser.add_argument(
        "--manifest-template",
        default="stage3_target_kws_gsc_marvin_hardneg_aishell5_{split}.jsonl",
    )
    parser.add_argument(
        "--extracted-cache-dir",
        type=Path,
        default=REPO_ROOT / "data/cache/stage3_target_kws_gsc_marvin_hardneg_aishell5_16k",
    )
    parser.add_argument(
        "--fixed-enrollment-wav",
        type=Path,
        default=REPO_ROOT
        / "data/raw/librispeech/extracted/LibriSpeech/test-clean/1089/134686/1089-134686-0014.flac",
    )
    parser.add_argument(
        "--wave-output-dir",
        type=Path,
        default=REPO_ROOT / "data/cache/compact_speaker_verifier_cabin_aware_2026-10-12",
    )
    parser.add_argument(
        "--embedding-output-dir",
        type=Path,
        default=REPO_ROOT / "data/cache/wavlm_speaker_verifier_cabin_aware_2026-10-12",
    )
    parser.add_argument("--model-name", default="microsoft/wavlm-base-plus-sv")
    parser.add_argument("--model-cache-dir", type=Path, default=Path("D:/datasets/pretrained_models"))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )

    from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

    processor = AutoFeatureExtractor.from_pretrained(
        str(args.model_name), cache_dir=str(resolve(args.model_cache_dir))
    )
    model = AutoModelForAudioXVector.from_pretrained(
        str(args.model_name), cache_dir=str(resolve(args.model_cache_dir))
    ).to(device)
    model.eval()

    manifest_root = resolve(args.manifest_root)
    extracted_cache_dir = resolve(args.extracted_cache_dir)
    wave_output = resolve(args.wave_output_dir)
    embedding_output = resolve(args.embedding_output_dir)
    wave_output.mkdir(parents=True, exist_ok=True)
    embedding_output.mkdir(parents=True, exist_ok=True)
    num_samples = max(1, int(round(float(args.sample_rate) * float(args.seconds))))
    fixed_wav, _ = read_mono(resolve(args.fixed_enrollment_wav), int(args.sample_rate))
    fixed_enrollment = segment_waveform(fixed_wav, num_samples, training=False)

    index = SourceIndex()
    summary: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        manifest = manifest_root / split / str(args.manifest_template).format(split=split)
        rows = load_rows(manifest)
        extracted = np.load(
            extracted_cache_dir / f"{split}_wav_16000hz.npy", mmap_mode="r", allow_pickle=False
        )
        if len(rows) != len(extracted):
            raise ValueError(f"{split}: manifest/cache lengths differ")
        enrollment = np.zeros((len(rows), num_samples), dtype=np.float32)
        labels = np.zeros(len(rows), dtype=np.int64)
        fixed_rows = 0
        for position, row in enumerate(rows):
            source = index.get(str(row["source_manifest"]), str(row["source_id"]))
            if source is None:
                enrollment[position] = fixed_enrollment
                fixed_rows += 1
            else:
                wav, _ = read_mono(Path(source["enroll_wav"]), int(args.sample_rate))
                enrollment[position] = segment_waveform(wav, num_samples, training=False)
            labels[position] = int(row.get("label", 0))
        np.save(wave_output / f"{split}_enrollment_wav.npy", enrollment)

        enrollment_embeddings: list[torch.Tensor] = []
        extracted_embeddings: list[torch.Tensor] = []
        batch_size = max(1, int(args.embedding_batch_size))
        for start in range(0, len(rows), batch_size):
            end = min(len(rows), start + batch_size)
            enrollment_embeddings.append(
                model_embeddings(model, processor, list(enrollment[start:end]), device)
            )
            extracted_embeddings.append(
                model_embeddings(
                    model,
                    processor,
                    [np.asarray(extracted[i], dtype=np.float32) for i in range(start, end)],
                    device,
                )
            )
            print(f"{split}: embedded {end}/{len(rows)}", flush=True)
        enrollment_array = torch.cat(enrollment_embeddings, dim=0).numpy().astype(np.float32)
        extracted_array = torch.cat(extracted_embeddings, dim=0).numpy().astype(np.float32)
        np.save(embedding_output / f"{split}_enrollment_embedding.npy", enrollment_array)
        np.save(embedding_output / f"{split}_extracted_embedding.npy", extracted_array)
        summary[split] = {
            "records": int(len(rows)),
            "positives": int((labels == 1).sum()),
            "negatives": int((labels == 0).sum()),
            "fixed_enrollment_rows": int(fixed_rows),
            "embedding_dim": int(enrollment_array.shape[-1]),
        }
        print(f"{split}: cached {len(rows)} pairs -> {wave_output} / {embedding_output}", flush=True)

    (wave_output / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
