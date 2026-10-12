#!/usr/bin/env python
"""Measure selective intent accuracy and confidence calibration.

This evaluates the frozen wav2vec intent head on a declared validation
manifest. It does not change weights or select a production threshold. The
result is useful for deciding whether an optional post-wake confidence gate
can suppress uncertain actions without hiding too many correct commands.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from transformers import AutoModel

REPO_ROOT_LOCAL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT_LOCAL / "src"))
sys.path.insert(0, str(REPO_ROOT_LOCAL / "scripts" / "train"))

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.datasets import IntentDataset
from car_tse.train.utils import get_device, load_checkpoint, write_metrics
from train_stage4_intent_wav2vec import IntentHead, normalize_waveform


def expected_calibration_error(confidence: torch.Tensor, correct: torch.Tensor, bins: int = 15) -> float:
    edges = torch.linspace(0.0, 1.0, bins + 1, device=confidence.device)
    ece = torch.zeros((), device=confidence.device)
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        mask = (confidence >= lower) & (confidence <= upper if index == bins - 1 else confidence < upper)
        if bool(mask.any()):
            ece = ece + mask.float().mean() * (confidence[mask].mean() - correct[mask].float().mean()).abs()
    return float(ece.item())


@torch.no_grad()
def collect_scores(
    head: IntentHead,
    encoder: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    head.eval()
    encoder.eval()
    all_probabilities: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for batch in loader:
        wav = normalize_waveform(batch["wav"].to(device, non_blocking=True))
        labels = batch["label"].to(device, non_blocking=True)
        features = encoder(wav).last_hidden_state.mean(dim=1)
        probabilities = torch.softmax(head(features), dim=-1)
        all_probabilities.append(probabilities.cpu())
        all_labels.append(labels.cpu())
    probabilities = torch.cat(all_probabilities, dim=0)
    labels = torch.cat(all_labels, dim=0)
    confidence, predictions = probabilities.max(dim=-1)
    correct = predictions.eq(labels)
    return probabilities, confidence, correct


@torch.no_grad()
def collect_feature_scores(
    head: IntentHead,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    head.eval()
    all_probabilities: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for features, labels in loader:
        probabilities = torch.softmax(head(features.to(device, non_blocking=True)), dim=-1)
        all_probabilities.append(probabilities.cpu())
        all_labels.append(labels.cpu())
    probabilities = torch.cat(all_probabilities, dim=0)
    labels = torch.cat(all_labels, dim=0)
    confidence, predictions = probabilities.max(dim=-1)
    correct = predictions.eq(labels)
    return probabilities, confidence, correct


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--encoder", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=None,
        help="Optional precomputed wav2vec feature directory; skips encoder inference.",
    )
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=None,
        help="JSONL manifest containing labels for --feature-dir, in feature-row order.",
    )
    parser.add_argument("--feature-file", default="val_features.npy")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.30, 0.40, 0.50, 0.53, 0.55, 0.60, 0.70, 0.80, 0.90],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    device = get_device()
    label_map_path = Path(paths["manifest_root"]) / "meta" / cfg["label_map"]
    num_classes = len(json.loads(label_map_path.read_text(encoding="utf-8")))
    if args.feature_dir is not None:
        feature_dir = args.feature_dir if args.feature_dir.is_absolute() else REPO_ROOT / args.feature_dir
        feature_path = feature_dir / args.feature_file
        manifest = args.feature_manifest
        if manifest is None:
            manifest = feature_dir / "stage4_intent_tse_domain_eval.jsonl"
        elif not manifest.is_absolute():
            manifest = REPO_ROOT / manifest
        labels = torch.tensor(
            [
                int(json.loads(line)["label"])
                for line in manifest.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ],
            dtype=torch.long,
        )
        features = torch.from_numpy(np.asarray(np.load(feature_path), dtype=np.float32))
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"feature/label count mismatch: {features.shape[0]} vs {labels.shape[0]}"
            )
        dataset = TensorDataset(features, labels)
        loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
        head = IntentHead(int(features.shape[-1]), 512, num_classes, 0.2).to(device)
        load_checkpoint(args.head_checkpoint, head)
        probabilities, confidence, correct = collect_feature_scores(head, loader, device)
        encoder_used = None
    else:
        manifest_root = Path(paths["manifest_root"])
        manifest_name = cfg["manifest_train"] if args.split == "train" else cfg["manifest_eval"]
        manifest = manifest_root / args.split / manifest_name
        cache_dir = Path(cfg["waveform_cache_dir"])
        if not cache_dir.is_absolute():
            cache_dir = REPO_ROOT / cache_dir
        cache_stem = "cabin_wav" if cfg.get("waveform_cache_includes_cabin_noise", False) else "wav"
        cache_path = cache_dir / f"{args.split}_{cache_stem}_{cfg['sample_rate']}hz.npy"
        dataset = IntentDataset(
            manifest,
            cfg["sample_rate"],
            cfg["segment_seconds"],
            training=False,
            waveform_cache_path=cache_path,
            waveform_cache_is_noisy=True,
            audio_cache_size=0,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        encoder = AutoModel.from_pretrained(args.encoder).to(device)
        encoder_state = torch.load(args.encoder_checkpoint, map_location="cpu", weights_only=False)
        encoder.load_state_dict(encoder_state["model_state_dict"], strict=True)
        head = IntentHead(int(encoder.config.hidden_size), 512, num_classes, 0.2).to(device)
        load_checkpoint(args.head_checkpoint, head)
        probabilities, confidence, correct = collect_scores(head, encoder, loader, device)
        labels = torch.tensor([int(row["label"]) for row in dataset.rows], dtype=torch.long)
        encoder_used = str(args.encoder_checkpoint.resolve())
    one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
    raw_accuracy = float(correct.float().mean().item())
    brier = float(((probabilities - one_hot) ** 2).sum(dim=-1).mean().item())
    thresholds: list[dict[str, float | int | None]] = []
    for threshold in sorted(set(float(value) for value in args.thresholds)):
        accepted = confidence >= threshold
        accepted_count = int(accepted.sum().item())
        thresholds.append(
            {
                "threshold": threshold,
                "accepted_examples": accepted_count,
                "coverage": float(accepted.float().mean().item()),
                "selective_accuracy": (
                    float(correct[accepted].float().mean().item()) if accepted_count else None
                ),
                "suppressed_examples": int((~accepted).sum().item()),
            }
        )

    report = {
        "benchmark": "intent_confidence_calibration",
        "config": str(args.config.resolve()),
        "split": args.split,
        "device": str(device),
        "examples": int(labels.numel()),
        "num_classes": num_classes,
        "head_checkpoint": str(args.head_checkpoint.resolve()),
        "encoder_checkpoint": str(args.encoder_checkpoint.resolve()) if encoder_used else None,
        "feature_dir": str(args.feature_dir.resolve()) if args.feature_dir is not None else None,
        "raw_accuracy": raw_accuracy,
        "ece_15_bins": expected_calibration_error(confidence, correct),
        "brier_score": brier,
        "mean_confidence": float(confidence.mean().item()),
        "thresholds": thresholds,
        "note": "Validation-only calibration study; no threshold is promoted without tuning/test driver data.",
    }
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    write_metrics(output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
