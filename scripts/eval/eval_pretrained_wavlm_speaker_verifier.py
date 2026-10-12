#!/usr/bin/env python
"""Evaluate a pretrained WavLM speaker-verification encoder on cabin mixtures.

This is an inference-only comparison against the frozen SpEx similarity.  It
does not modify the production pipeline or train a checkpoint.  The input
contract is the existing synthetic CAVEMOVE/GSC-Marvin target-present versus
target-absent cache, so the result is a representation diagnostic rather than
real-driver evidence.
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

from car_tse.data.segment import segment_waveform  # noqa: E402
from car_tse.io.audio import read_mono  # noqa: E402
from scripts.eval.calibrate_spex_speaker_verifier import (  # noqa: E402
    choose_threshold,
    roc_summary,
    threshold_metrics,
)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_source_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    manifests: dict[str, list[dict[str, Any]]] = {}
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        manifest_path = str(row.get("source_manifest", ""))
        if manifest_path and manifest_path not in manifests:
            path = Path(manifest_path)
            if not path.is_absolute():
                path = REPO_ROOT / path
            manifests[manifest_path] = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        source_id = str(row.get("source_id", ""))
        for source in manifests.get(manifest_path, []):
            if str(source.get("id", "")) == source_id:
                output[source_id] = source
                break
    missing = [str(row.get("source_id", "")) for row in rows if str(row.get("source_id", "")) not in output]
    if missing:
        raise KeyError(f"Could not resolve {len(missing)} source rows; first={missing[0]}")
    return output


def load_split(
    cache_root: Path,
    manifest_root: Path,
    manifest_template: str,
    split: str,
    max_rows: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    manifest_path = manifest_root / split / manifest_template.format(split=split)
    wav_path = cache_root / f"{split}_wav_16000hz.npy"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not wav_path.exists():
        raise FileNotFoundError(wav_path)
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    wav = np.load(wav_path, mmap_mode="r", allow_pickle=False)
    if len(rows) != len(wav):
        raise ValueError(f"{split}: manifest/cache length mismatch: {len(rows)} != {len(wav)}")
    if max_rows > 0 and len(rows) > max_rows:
        # Preserve the alternating positive/negative order in this cache so a
        # small diagnostic remains class-balanced and deterministic.
        indices = np.arange(max_rows, dtype=np.int64)
        rows = [rows[int(index)] for index in indices]
        wav = np.asarray(wav[indices], dtype=np.float32)
    labels = np.asarray([int(row.get("label", 0)) for row in rows], dtype=np.float32)
    return np.asarray(wav, dtype=np.float32), labels, rows


def model_embeddings(
    model: torch.nn.Module,
    processor: Any,
    waveforms: list[np.ndarray],
    device: torch.device,
) -> torch.Tensor:
    inputs = processor(
        waveforms,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    embeddings = getattr(outputs, "embeddings", None)
    if embeddings is None:
        embeddings = getattr(outputs, "pooler_output", None)
    if embeddings is None:
        raise RuntimeError(f"WavLM output has no embeddings/pooler_output: {type(outputs)!r}")
    return torch.nn.functional.normalize(embeddings.float(), dim=-1).cpu()


def load_enrollment_waveforms(rows: list[dict[str, Any]], source_rows: dict[str, dict[str, Any]]) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for row in rows:
        source = source_rows[str(row["source_id"])]
        wav, _ = read_mono(Path(source["enroll_wav"]), 16000)
        output.append(segment_waveform(wav, 16000, training=False))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        default="microsoft/wavlm-base-plus-sv",
        help="Transformers audio-XVector checkpoint; used only for this diagnostic",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=Path("D:/datasets/pretrained_models"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=REPO_ROOT / "data/cache/stage3_target_kws_gsc_marvin_ft2_stage1_16k",
    )
    parser.add_argument("--manifest-root", type=Path, default=REPO_ROOT / "manifests")
    parser.add_argument(
        "--manifest-template",
        default="stage3_target_kws_gsc_marvin_ft2_stage1_{split}.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "logs/wavlm_speaker_verifier_cabin_diagnostic_2026-10-11.json",
    )
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        default=REPO_ROOT / "data/cache/wavlm_speaker_verifier_gsc_marvin_2026-10-11",
        help="Directory for reusable WavLM enrollment/extracted embedding arrays",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-rows-per-split", type=int, default=0)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

    model_cache_dir = resolve_path(args.model_cache_dir)
    # The official model card uses a feature extractor rather than
    # AutoProcessor: this checkpoint has no tokenizer.
    processor = AutoFeatureExtractor.from_pretrained(args.model_name, cache_dir=model_cache_dir)
    model = AutoModelForAudioXVector.from_pretrained(args.model_name, cache_dir=model_cache_dir).to(device)
    model.eval()

    cache_root = resolve_path(args.cache_root)
    manifest_root = resolve_path(args.manifest_root)
    embedding_cache_dir = resolve_path(args.embedding_cache_dir)
    embedding_cache_dir.mkdir(parents=True, exist_ok=True)
    split_outputs: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        extracted_wav, labels, rows = load_split(
            cache_root,
            manifest_root,
            str(args.manifest_template),
            split,
            int(args.max_rows_per_split),
        )
        source_rows = load_source_rows(rows)
        enroll_wav = load_enrollment_waveforms(rows, source_rows)
        enrollment_embeddings: list[torch.Tensor] = []
        extracted_embeddings: list[torch.Tensor] = []
        batch_size = max(1, int(args.batch_size))
        for start in range(0, len(rows), batch_size):
            end = min(len(rows), start + batch_size)
            enrollment_embeddings.append(
                model_embeddings(model, processor, enroll_wav[start:end], device)
            )
            extracted_embeddings.append(
                model_embeddings(model, processor, [extracted_wav[i] for i in range(start, end)], device)
            )
            print(f"{split}: embedded {end}/{len(rows)}", flush=True)
        enrollment = torch.cat(enrollment_embeddings, dim=0).numpy()
        extracted = torch.cat(extracted_embeddings, dim=0).numpy()
        np.save(embedding_cache_dir / f"{split}_enrollment_embedding.npy", enrollment.astype(np.float32))
        np.save(embedding_cache_dir / f"{split}_extracted_embedding.npy", extracted.astype(np.float32))
        scores = np.sum(enrollment * extracted, axis=1).astype(np.float32)
        split_outputs[split] = {
            "scores": scores,
            "labels": labels,
            "records": len(rows),
            "source_ids": [str(row["source_id"]) for row in rows],
        }

    train_scores = split_outputs["train"]["scores"]
    train_labels = split_outputs["train"]["labels"]
    val_scores = split_outputs["val"]["scores"]
    val_labels = split_outputs["val"]["labels"]
    test_scores = split_outputs["test"]["scores"]
    test_labels = split_outputs["test"]["labels"]
    selected = choose_threshold(
        val_scores,
        val_labels,
        float(args.target_fpr),
    )
    test_at_selected = threshold_metrics(test_scores, test_labels, selected["threshold"])

    def summarize(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        return {
            "records": int(len(scores)),
            "positive_count": int((labels == 1).sum()),
            "negative_count": int((labels == 0).sum()),
            "roc": roc_summary(scores, labels),
            "positive_mean": float(scores[labels == 1].mean()),
            "negative_mean": float(scores[labels == 0].mean()),
            "positive_quantiles": np.quantile(scores[labels == 1], [0.01, 0.5, 0.99]).astype(float).tolist(),
            "negative_quantiles": np.quantile(scores[labels == 0], [0.01, 0.5, 0.99]).astype(float).tolist(),
            "threshold_sweep": [
                threshold_metrics(scores, labels, threshold)
                for threshold in thresholds
            ],
        }

    output = {
        "benchmark": "pretrained_wavlm_speaker_verifier_cabin_diagnostic",
        "model_name": str(args.model_name),
        "device": str(device),
        "cache_root": str(cache_root),
        "manifest_template": str(args.manifest_template),
        "embedding_cache_dir": str(embedding_cache_dir),
        "data_contract": {
            "positive_label": "target speaker present in synthetic cabin mixture",
            "negative_label": "target speaker absent; passenger/noise mixture",
            "real_driver_evidence": False,
            "real_cabin_evidence": False,
            "production_checkpoint_changed": False,
        },
        "validation_operating_point": {
            "target_fpr": float(args.target_fpr),
            **selected,
        },
        "test_at_validation_operating_point": test_at_selected,
        "splits": {
            split: summarize(value["scores"], value["labels"])
            for split, value in split_outputs.items()
        },
        "decision": (
            "diagnostic_only; promote only after permitted same-driver cabin tuning "
            "and a latency/resource audit"
        ),
    }
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "test_auc": output["splits"]["test"]["roc"]["auc"],
                "val_tpr": selected["tpr"],
                "val_fpr": selected["fpr"],
                "test_tpr": test_at_selected["tpr"],
                "test_fpr": test_at_selected["fpr"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
