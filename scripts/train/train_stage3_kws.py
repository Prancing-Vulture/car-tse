#!/usr/bin/env python
"""Train a fast binary wake-word detector on cabin-augmented speech."""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.datasets import KWSDataset
from car_tse.models.kws_model import KWSModel
from car_tse.train.utils import get_device, load_checkpoint, save_checkpoint, write_metrics


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


def loader_kwargs(cfg: dict, device: torch.device) -> dict:
    workers = int(cfg["train"].get("num_workers", 0))
    return {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
        "prefetch_factor": 2 if workers > 0 else None,
    }


@torch.no_grad()
def evaluate(
    model: KWSModel,
    loader: DataLoader,
    device: torch.device,
    decision_threshold: float = 0.5,
) -> dict[str, float]:
    model.eval()
    correct = total = tp = fp = fn = 0
    for batch in loader:
        wav = batch["wav"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        wake_probability = torch.softmax(model(wav), dim=-1)[..., 1]
        predictions = (wake_probability >= float(decision_threshold)).long()
        correct += int((predictions == labels).sum())
        total += labels.numel()
        tp += int(((predictions == 1) & (labels == 1)).sum())
        fp += int(((predictions == 1) & (labels == 0)).sum())
        fn += int(((predictions == 0) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "accuracy": correct / max(total, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "examples": float(total),
        "true_positive": float(tp),
        "false_positive": float(fp),
        "false_negative": float(fn),
        "decision_threshold": float(decision_threshold),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "stage3_kws_cavemove.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Optional checkpoint to fine-tune")
    parser.add_argument("--run-name", default=None, help="Optional isolated output run name")
    parser.add_argument("--wandb", action="store_true", help="Log epoch metrics to Weights & Biases")
    parser.add_argument("--wandb-project", default="car-tse")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-dir", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    seed = int(cfg["train"].get("seed", 20261014))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = get_device()
    cache_dir = Path(cfg.get("waveform_cache_dir", REPO_ROOT / "data" / "cache" / "stage3_kws_16k"))
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    cache_stem = "cabin_wav" if cfg.get("waveform_cache_includes_cabin_noise", False) else "wav"
    train_cache_path = cache_dir / f"train_{cache_stem}_{cfg['sample_rate']}hz.npy"
    val_cache_path = cache_dir / f"val_{cache_stem}_{cfg['sample_rate']}hz.npy"
    dataset_kwargs = {
        "cabin_noise": bool(cfg.get("cabin_noise", False)),
        "cabin_noise_snr_db": cfg.get("cabin_noise_snr_db"),
        "cabin_noise_cache_size": int(cfg.get("cabin_noise_cache_size", 32)),
        "audio_cache_size": int(cfg.get("audio_cache_size", 2048)),
        "cabin_noise_cache_dir": (
            (Path(cfg["cabin_noise_cache_dir"]) if Path(cfg["cabin_noise_cache_dir"]).is_absolute() else REPO_ROOT / cfg["cabin_noise_cache_dir"])
            if cfg.get("cabin_noise_cache_dir")
            else None
        ),
    }
    train_manifest = Path(paths["manifest_root"]) / "train" / cfg["manifest_train"]
    val_manifest = Path(paths["manifest_root"]) / "val" / cfg["manifest_val"]
    train_ds = KWSDataset(
        train_manifest,
        cfg["sample_rate"],
        cfg["segment_seconds"],
        training=True,
        waveform_cache_path=train_cache_path,
        waveform_cache_is_noisy=bool(cfg.get("waveform_cache_includes_cabin_noise", False)),
        **dataset_kwargs,
    )
    val_ds = KWSDataset(
        val_manifest,
        cfg["sample_rate"],
        cfg["segment_seconds"],
        training=False,
        waveform_cache_path=val_cache_path,
        waveform_cache_is_noisy=bool(cfg.get("waveform_cache_includes_cabin_noise", False)),
        **dataset_kwargs,
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        print(f"Missing KWS data: train={train_manifest} val={val_manifest}")
        return 1
    if cfg.get("waveform_cache_includes_cabin_noise", False) and (
        train_ds.waveform_cache is None or val_ds.waveform_cache is None
    ):
        raise RuntimeError(
            "This run is configured to train on cached noisy Stage-1 outputs, but a cache is missing "
            "or has the wrong shape. Check train/validation cache files before starting training."
        )
    common_loader = loader_kwargs(cfg, device)
    labels = torch.tensor([int(row["label"]) for row in train_ds.rows], dtype=torch.long)
    counts = torch.bincount(labels, minlength=cfg["model"]["num_classes"]).float()
    if cfg["model"]["num_classes"] != 2 or torch.any(labels < 0) or torch.any(labels > 1):
        raise ValueError("Stage 3 sampling expects binary labels encoded as 0=negative and 1=wake")
    negative_to_positive_ratio = cfg["train"].get("negative_to_positive_sampling_ratio")
    if negative_to_positive_ratio is not None:
        negative_to_positive_ratio = float(negative_to_positive_ratio)
        if negative_to_positive_ratio <= 0:
            raise ValueError("negative_to_positive_sampling_ratio must be positive")
        class_sampling_probabilities = torch.tensor(
            [
                negative_to_positive_ratio / (1.0 + negative_to_positive_ratio),
                1.0 / (1.0 + negative_to_positive_ratio),
            ],
            dtype=torch.float32,
        )
        per_example_weights = class_sampling_probabilities[labels] / counts[labels].clamp_min(1.0)
        sampler = WeightedRandomSampler(
            weights=per_example_weights,
            num_samples=len(per_example_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        sampler_strategy = f"negative_to_positive_{negative_to_positive_ratio:g}_to_1"
    elif bool(cfg["train"].get("balanced_sampler", True)):
        per_example_weights = 1.0 / counts[labels].clamp_min(1.0)
        sampler = WeightedRandomSampler(
            weights=per_example_weights,
            num_samples=len(per_example_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        sampler_strategy = "balanced_1_to_1"
    else:
        sampler = None
        sampler_strategy = "natural_manifest_distribution"
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        sampler=sampler,
        shuffle=sampler is None,
        generator=torch.Generator().manual_seed(seed),
        **common_loader,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"].get("val_batch_size", cfg["train"]["batch_size"]),
        shuffle=False,
        **common_loader,
    )
    model = KWSModel(
        cfg["sample_rate"],
        cfg["model"]["n_mels"],
        cfg["model"]["channels"],
        cfg["model"]["num_classes"],
    ).to(device)
    if args.init_checkpoint is not None:
        checkpoint = args.init_checkpoint if args.init_checkpoint.is_absolute() else REPO_ROOT / args.init_checkpoint
        load_checkpoint(checkpoint, model)
        print(f"initialized_checkpoint={checkpoint}", flush=True)
    decision_threshold = float(cfg.get("validation_threshold", 0.5))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["train"]["max_epochs"])
    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    run_name = args.run_name or cfg["name"]
    print(f"training_sampler_strategy={sampler_strategy}", flush=True)
    out_dir = paths["checkpoint_root"] / run_name
    wandb_run = None
    if args.wandb:
        import wandb

        init_checkpoint_path = (
            (args.init_checkpoint if args.init_checkpoint.is_absolute() else REPO_ROOT / args.init_checkpoint)
            if args.init_checkpoint is not None
            else None
        )

        if args.wandb_dir is not None:
            args.wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            job_type="train-stage3-kws",
            mode=args.wandb_mode,
            dir=str(args.wandb_dir) if args.wandb_dir is not None else None,
            config={
                "stage": "stage3_kws",
                "config": args.config.name,
                "config_sha256": file_fingerprint(args.config if args.config.is_absolute() else REPO_ROOT / args.config).get("sha256"),
                "init_checkpoint": file_fingerprint(init_checkpoint_path) if init_checkpoint_path else None,
                "train_manifest": file_fingerprint(train_manifest),
                "validation_manifest": file_fingerprint(val_manifest),
                "train_waveform_cache": file_fingerprint(train_cache_path),
                "validation_waveform_cache": file_fingerprint(val_cache_path),
                "input_domain": (
                    "cached_frozen_stage1_extracted_with_cabin_noise"
                    if train_ds.waveform_cache is not None
                    else "raw_waveform_with_optional_on_the_fly_cabin_noise"
                ),
                "max_epochs": int(cfg["train"]["max_epochs"]),
                "validation_threshold": decision_threshold,
                "learning_rate": float(cfg["train"]["lr"]),
                "batch_size": int(cfg["train"]["batch_size"]),
                "train_examples": len(train_ds),
                "train_positive_examples": int(labels.sum().item()),
                "train_negative_examples": int(len(labels) - labels.sum().item()),
                "training_sampler_strategy": sampler_strategy,
                "negative_to_positive_sampling_ratio": negative_to_positive_ratio,
                "validation_examples": len(val_ds),
                "cabin_noise_augmentation_on_the_fly": bool(cfg.get("cabin_noise", False)),
                "waveform_cache_includes_cabin_noise": bool(cfg.get("waveform_cache_includes_cabin_noise", False)),
                "waveform_cache_crop_start_seconds": float(cfg.get("waveform_cache_crop_start_seconds", 0.0)),
                "cabin_noise_snr_db": cfg.get("cabin_noise_snr_db"),
                "seed": int(cfg["train"].get("seed", 0)),
                "device": str(device),
            },
        )
    # A fine-tune candidate must never lose its warm-start if training does not
    # improve validation F1. This also makes the saved candidate directly
    # comparable with the checkpoint it started from.
    best_f1 = float("-inf")
    if args.init_checkpoint is not None:
        initial_val = evaluate(model, val_loader, device, decision_threshold)
        best_f1 = initial_val["f1"]
        initial_metrics = {"epoch": 0, "validation": initial_val, "initialized_from": str(args.init_checkpoint)}
        save_checkpoint(out_dir / "best.pt", model, 0, initial_metrics)
        print(
            f"warm_start_val_acc={initial_val['accuracy']:.3f} "
            f"warm_start_val_f1={initial_val['f1']:.3f} "
            f"warm_start_val_recall={initial_val['recall']:.3f}",
            flush=True,
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": 0,
                    "validation/accuracy": initial_val["accuracy"],
                    "validation/precision": initial_val["precision"],
                    "validation/recall": initial_val["recall"],
                    "validation/f1": initial_val["f1"],
                    "validation/false_positive": initial_val["false_positive"],
                    "validation/false_negative": initial_val["false_negative"],
                    "validation/decision_threshold": decision_threshold,
                },
                step=0,
                commit=True,
            )
    bad_epochs = 0
    patience = int(cfg["train"].get("patience", 4))
    for epoch in range(1, cfg["train"]["max_epochs"] + 1):
        model.train()
        correct = total = 0
        sampled_positives = sampled_negatives = 0
        accumulated_train_loss = 0.0
        for batch in train_loader:
            wav = batch["wav"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(wav)
            loss = F.cross_entropy(logits.float(), target)
            accumulated_train_loss += float(loss.detach()) * target.numel()
            sampled_positives += int((target == 1).sum())
            sampled_negatives += int((target == 0).sum())
            opt.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                opt.step()
            correct += int((logits.argmax(-1) == target).sum())
            total += target.numel()
        scheduler.step()
        val = evaluate(model, val_loader, device, decision_threshold)
        train_acc = correct / max(total, 1)
        train_loss = accumulated_train_loss / max(total, 1)
        metrics = {
            "epoch": epoch,
            "train_accuracy": train_acc,
            "train_loss": train_loss,
            "sampled_positive_examples": sampled_positives,
            "sampled_negative_examples": sampled_negatives,
            "sampled_negative_to_positive_ratio": sampled_negatives / max(sampled_positives, 1),
            "val": val,
            "lr": scheduler.get_last_lr()[0],
            "best_f1": best_f1,
        }
        write_metrics(paths["log_root"] / run_name / "metrics_last.json", metrics)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/accuracy": train_acc,
                    "train/loss": train_loss,
                    "train/sampled_positive_examples": sampled_positives,
                    "train/sampled_negative_examples": sampled_negatives,
                    "train/sampled_negative_to_positive_ratio": metrics["sampled_negative_to_positive_ratio"],
                    "validation/accuracy": val["accuracy"],
                    "validation/precision": val["precision"],
                    "validation/recall": val["recall"],
                    "validation/f1": val["f1"],
                    "validation/false_positive": val["false_positive"],
                    "validation/false_negative": val["false_negative"],
                    "validation/decision_threshold": decision_threshold,
                    "learning_rate": scheduler.get_last_lr()[0],
                },
                step=epoch,
                commit=True,
            )
        print(
            f"epoch {epoch} train_acc={train_acc:.3f} train_loss={train_loss:.4f} "
            f"sampled_neg_pos={metrics['sampled_negative_to_positive_ratio']:.2f} "
            f"val_acc={val['accuracy']:.3f} val_f1={val['f1']:.3f} "
            f"val_recall={val['recall']:.3f}",
            flush=True,
        )
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            bad_epochs = 0
            save_checkpoint(out_dir / "best.pt", model, epoch, metrics)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping after {epoch} epochs", flush=True)
                break
    write_metrics(paths["log_root"] / run_name / "train_last.json", {"best_f1": best_f1})
    if wandb_run is not None:
        wandb_run.summary["best_validation_f1"] = best_f1
        wandb_run.summary["validation_decision_threshold"] = decision_threshold
        best_checkpoint = out_dir / "best.pt"
        wandb_run.summary["best_checkpoint"] = best_checkpoint.name
        if best_checkpoint.is_file():
            artifact = wandb.Artifact(
                name=f"{run_name}-best",
                type="model",
                metadata={"best_validation_f1": best_f1, "stage": "stage3_kws"},
            )
            artifact.add_file(str(best_checkpoint), name=best_checkpoint.name)
            try:
                logged_artifact = wandb_run.log_artifact(artifact)
                logged_artifact.wait()
                wandb_run.summary["best_checkpoint_artifact"] = (
                    f"{logged_artifact.name}:{logged_artifact.version}"
                )
            except Exception as exc:
                # Artifact upload is optional; never leave a completed training
                # run in a failed state when its metrics/checkpoint are saved.
                wandb_run.summary["checkpoint_artifact_upload_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
        wandb_run.finish()
    print(f"Done. Best checkpoint: {out_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
