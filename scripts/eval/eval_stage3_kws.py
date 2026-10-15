#!/usr/bin/env python
"""Evaluate binary wake-word detection on a requested split."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.datasets import KWSDataset
from car_tse.models.kws_model import KWSModel
from car_tse.train.utils import get_device, load_checkpoint, write_metrics
from car_tse.train.wandb_logging import log_evaluation


def file_fingerprint(path: Path) -> dict[str, int | str | bool]:
    record: dict[str, int | str | bool] = {"file": path.name, "exists": path.is_file()}
    if not path.is_file():
        return record
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    record["size_bytes"] = path.stat().st_size
    record["sha256"] = digest.hexdigest()
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "stage3_kws_cavemove.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint override")
    parser.add_argument("--output", type=Path, default=None, help="Optional metrics output override")
    parser.add_argument("--threshold", type=float, default=None, help="Wake probability cutoff; defaults to the configured pipeline cutoff")
    parser.add_argument("--wandb", action="store_true", help="Log split metrics and report artifact to W&B")
    parser.add_argument("--wandb-project", default="car-tse")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--wandb-dir", type=Path, default=None)
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional W&B run name; defaults to a checkpoint-and-split-specific name",
    )
    args = parser.parse_args()
    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    device = get_device()
    manifest_name = cfg["manifest_val"] if args.split == "val" else cfg["manifest_eval"]
    manifest = args.manifest or (Path(paths["manifest_root"]) / args.split / manifest_name)
    if not manifest.is_absolute():
        manifest = Path(paths["manifest_root"]) / args.split / manifest
    cache_dir = Path(cfg.get("waveform_cache_dir", REPO_ROOT / "data" / "cache" / "stage3_kws_16k"))
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    cache_stem = "cabin_wav" if cfg.get("waveform_cache_includes_cabin_noise", False) else "wav"
    ds = KWSDataset(
        manifest,
        cfg["sample_rate"],
        cfg["segment_seconds"],
        training=False,
        cabin_noise=bool(cfg.get("cabin_noise", False)),
        cabin_noise_snr_db=cfg.get("cabin_noise_snr_db"),
        cabin_noise_cache_size=int(cfg.get("cabin_noise_cache_size", 32)),
        audio_cache_size=int(cfg.get("audio_cache_size", 2048)),
        waveform_cache_path=cache_dir / f"{args.split}_{cache_stem}_{cfg['sample_rate']}hz.npy",
        waveform_cache_is_noisy=bool(cfg.get("waveform_cache_includes_cabin_noise", False)),
        cabin_noise_cache_dir=(
            (Path(cfg["cabin_noise_cache_dir"]) if Path(cfg["cabin_noise_cache_dir"]).is_absolute() else REPO_ROOT / cfg["cabin_noise_cache_dir"])
            if cfg.get("cabin_noise_cache_dir")
            else None
        ),
    )
    if cfg.get("waveform_cache_includes_cabin_noise", False) and ds.waveform_cache is None:
        raise RuntimeError("Expected the cached frozen-Stage1 test waveforms, but the cache is missing or mis-shaped")
    workers = int(cfg["train"].get("num_workers", 0))
    loader = DataLoader(
        ds,
        batch_size=cfg["train"].get("val_batch_size", 256),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )
    model = KWSModel(
        cfg["sample_rate"],
        cfg["model"]["n_mels"],
        cfg["model"]["channels"],
        cfg["model"]["num_classes"],
    ).to(device)
    checkpoint = args.checkpoint or (paths["checkpoint_root"] / cfg["name"] / "best.pt")
    if not checkpoint.is_absolute():
        checkpoint = REPO_ROOT / checkpoint
    load_checkpoint(checkpoint, model)
    model.eval()
    threshold = float(
        args.threshold if args.threshold is not None else cfg.get("validation_threshold", 0.5)
    )
    correct = total = tp = fp = fn = 0
    tn = 0
    score_rows: list[tuple[int, float]] = []
    with torch.inference_mode():
        for batch in loader:
            probabilities = torch.softmax(model(batch["wav"].to(device, non_blocking=True)), dim=-1)[..., 1]
            labels = batch["label"].to(device, non_blocking=True)
            score_rows.extend(
                (int(label), float(score))
                for label, score in zip(labels.cpu().tolist(), probabilities.cpu().tolist())
            )
            predictions = probabilities >= threshold
            correct += int((predictions == labels).sum())
            total += labels.numel()
            tp += int((predictions & (labels == 1)).sum())
            fp += int((predictions & (labels == 0)).sum())
            fn += int(((~predictions) & (labels == 1)).sum())
            tn += int(((~predictions) & (labels == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    truth = [label for label, _ in score_rows]
    scores = [score for _, score in score_rows]
    fpr_points, tpr_points, roc_thresholds = roc_curve(truth, scores, drop_intermediate=False)
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(truth, scores)
    result = {
        "split": args.split,
        "wake_word": cfg.get("wake_word", "marvin"),
        "accuracy": correct / max(total, 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-8),
        "examples": total,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "decision_threshold": threshold,
        "evaluation_manifest": manifest.name,
        "checkpoint": checkpoint.name,
        "waveform_cache_loaded": ds.waveform_cache is not None,
        "roc_auc": float(roc_auc_score(truth, scores)),
        "average_precision": float(average_precision_score(truth, scores)),
        "roc_curve": {
            "false_positive_rate": fpr_points.tolist(),
            "true_positive_rate": tpr_points.tolist(),
            "thresholds": roc_thresholds.tolist(),
        },
        "pr_curve": {
            "precision": pr_precision.tolist(),
            "recall": pr_recall.tolist(),
            "thresholds": pr_thresholds.tolist(),
        },
    }
    output = args.output or (paths["log_root"] / cfg["name"] / f"metrics_{args.split}.json")
    if not output.is_absolute():
        output = REPO_ROOT / output
    write_metrics(output, result)
    if args.wandb:
        log_evaluation(
            project=args.wandb_project,
            run_name=args.wandb_run_name or f"{checkpoint.parent.name}-{args.split}-evaluation",
            job_type=f"stage3-kws-{args.split}-evaluation",
            config={
                "stage": "stage3_kws",
                "split": args.split,
                "decision_threshold": threshold,
                "checkpoint": file_fingerprint(checkpoint),
                "manifest": file_fingerprint(manifest),
                "waveform_cache": file_fingerprint(
                    cache_dir / f"{args.split}_{cache_stem}_{cfg['sample_rate']}hz.npy"
                ),
                "waveform_cache_includes_cabin_noise": bool(cfg.get("waveform_cache_includes_cabin_noise", False)),
                "waveform_cache_crop_start_seconds": float(cfg.get("waveform_cache_crop_start_seconds", 0.0)),
                "config": args.config.name,
                "wake_word": cfg.get("wake_word", "marvin"),
            },
            metrics=result,
            output_path=output,
            mode=args.wandb_mode,
            directory=args.wandb_dir,
            confusion_matrix=[[tn, fp], [fn, tp]],
            class_names=["non_wake_or_absent", "target_wake"],
            curves={
                "stage3_roc": [
                    {"x_key": "false_positive_rate", "y_key": "true_positive_rate", "false_positive_rate": float(x), "true_positive_rate": float(y)}
                    for x, y in zip(fpr_points, tpr_points)
                ],
                "stage3_precision_recall": [
                    {"x_key": "recall", "y_key": "precision", "recall": float(x), "precision": float(y)}
                    for x, y in zip(pr_recall, pr_precision)
                ],
            },
            report_artifact=True,
        )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
