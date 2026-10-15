#!/usr/bin/env python
"""Train a fast binary wake-word detector on cabin-augmented speech."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.datasets import KWSDataset
from car_tse.models.kws_model import KWSModel
from car_tse.train.utils import append_jsonl, get_device, load_checkpoint, save_checkpoint, write_metrics


def loader_kwargs(cfg: dict, device: torch.device) -> dict:
    workers = int(cfg["train"].get("num_workers", 0))
    return {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
        "prefetch_factor": 2 if workers > 0 else None,
    }


@torch.no_grad()
def evaluate(model: KWSModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    correct = total = tp = fp = fn = 0
    for batch in loader:
        wav = batch["wav"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        predictions = model(wav).argmax(-1)
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
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "stage3_kws_cavemove.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Optional checkpoint to fine-tune")
    parser.add_argument("--run-name", default=None, help="Optional isolated output run name")
    args = parser.parse_args()
    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    device = get_device()
    cache_dir = Path(cfg.get("waveform_cache_dir", REPO_ROOT / "data" / "cache" / "stage3_kws_16k"))
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    cache_stem = "cabin_wav" if cfg.get("waveform_cache_includes_cabin_noise", False) else "wav"
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
        waveform_cache_path=cache_dir / f"train_{cache_stem}_{cfg['sample_rate']}hz.npy",
        waveform_cache_is_noisy=bool(cfg.get("waveform_cache_includes_cabin_noise", False)),
        **dataset_kwargs,
    )
    val_ds = KWSDataset(
        val_manifest,
        cfg["sample_rate"],
        cfg["segment_seconds"],
        training=False,
        waveform_cache_path=cache_dir / f"val_{cache_stem}_{cfg['sample_rate']}hz.npy",
        waveform_cache_is_noisy=bool(cfg.get("waveform_cache_includes_cabin_noise", False)),
        **dataset_kwargs,
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        print(f"Missing KWS data: train={train_manifest} val={val_manifest}")
        return 1
    common_loader = loader_kwargs(cfg, device)
    labels = torch.tensor([int(row["label"]) for row in train_ds.rows], dtype=torch.long)
    counts = torch.bincount(labels, minlength=cfg["model"]["num_classes"]).float()
    sample_weights = 1.0 / counts[labels].clamp_min(1.0)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        sampler=sampler,
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
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["train"]["max_epochs"])
    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    run_name = args.run_name or cfg["name"]
    out_dir = paths["checkpoint_root"] / run_name
    run_id = uuid.uuid4().hex
    history_path = paths["log_root"] / run_name / f"epoch_metrics_{run_id}.jsonl"
    write_metrics(
        paths["log_root"] / run_name / f"run_metadata_{run_id}.json",
        {
            "run_id": run_id,
            "run_name": run_name,
            "device": str(device),
            "config": str(args.config),
            "train_manifest": str(train_manifest),
            "val_manifest": str(val_manifest),
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        },
    )
    best_f1 = float("-inf")
    bad_epochs = 0
    patience = int(cfg["train"].get("patience", 4))
    for epoch in range(1, cfg["train"]["max_epochs"] + 1):
        model.train()
        correct = total = 0
        for batch in train_loader:
            wav = batch["wav"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(wav)
            loss = F.cross_entropy(logits.float(), target)
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
        val = evaluate(model, val_loader, device)
        train_acc = correct / max(total, 1)
        metrics = {
            "epoch": epoch,
            "run_id": run_id,
            "train_accuracy": train_acc,
            "val": val,
            "lr": scheduler.get_last_lr()[0],
            "best_f1": best_f1,
        }
        write_metrics(paths["log_root"] / run_name / "metrics_last.json", metrics)
        append_jsonl(history_path, metrics)
        print(
            f"epoch {epoch} train_acc={train_acc:.3f} "
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
    write_metrics(
        paths["log_root"] / run_name / "train_last.json",
        {"best_f1": best_f1, "run_id": run_id, "history": str(history_path)},
    )
    print(f"Done. Best checkpoint: {out_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
