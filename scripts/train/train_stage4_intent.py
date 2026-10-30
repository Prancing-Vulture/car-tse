#!/usr/bin/env python
"""Train the cabin-robust direct speech-to-intent classifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.datasets import IntentDataset
from car_tse.models.intent_classifier import IntentClassifier
from car_tse.train.utils import get_device, load_checkpoint, save_checkpoint, write_metrics


def loader_kwargs(cfg: dict, device: torch.device) -> dict:
    workers = int(cfg["train"].get("num_workers", 0))
    result = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        result["prefetch_factor"] = int(cfg["train"].get("prefetch_factor", 2))
    return result


def dataset_kwargs(cfg: dict) -> dict:
    noise_cache_dir = Path(cfg.get("cabin_noise_cache_dir", "")) if cfg.get("cabin_noise_cache_dir") else None
    if noise_cache_dir is not None and not noise_cache_dir.is_absolute():
        noise_cache_dir = REPO_ROOT / noise_cache_dir
    return {
        "cabin_noise": bool(cfg.get("cabin_noise", False)),
        "cabin_noise_snr_db": cfg.get("cabin_noise_snr_db"),
        "cabin_noise_cache_size": int(cfg.get("cabin_noise_cache_size", 32)),
        "audio_cache_size": int(cfg.get("audio_cache_size", 2048)),
        "cabin_noise_cache_dir": noise_cache_dir,
        "waveform_cache_is_noisy": bool(cfg.get("waveform_cache_includes_cabin_noise", False)),
    }


@torch.no_grad()
def evaluate(
    model: IntentClassifier,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[str, float]:
    model.eval()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    total = 0
    top5 = 0
    for batch in loader:
        wav = batch["wav"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(wav)
        predictions = logits.argmax(-1)
        confusion.index_put_((labels.cpu(), predictions.cpu()), torch.ones_like(labels.cpu()), accumulate=True)
        total += labels.numel()
        top5 += int(
            (logits.topk(min(5, num_classes), dim=-1).indices == labels[:, None])
            .any(dim=1)
            .sum()
        )

    tp = confusion.diag().float()
    support = confusion.sum(dim=1).float()
    predicted = confusion.sum(dim=0).float()
    precision = tp / predicted.clamp_min(1.0)
    recall = tp / support.clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
    return {
        "accuracy": float(tp.sum().item() / max(total, 1)),
        "top5_accuracy": float(top5 / max(total, 1)),
        "macro_precision": float(precision.mean().item()),
        "macro_recall": float(recall.mean().item()),
        "macro_f1": float(f1.mean().item()),
        "examples": float(total),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "stage4_intent_cavemove.yaml",
    )
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    args = parser.parse_args()

    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    device = get_device()
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    label_map_path = Path(paths["manifest_root"]) / "meta" / cfg["label_map"]
    num_classes = (
        len(json.loads(label_map_path.read_text(encoding="utf-8")))
        if label_map_path.exists()
        else 31
    )
    data_kwargs = dataset_kwargs(cfg)
    cache_dir = Path(cfg.get("waveform_cache_dir", REPO_ROOT / "data" / "cache" / "stage4_intent_16k"))
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    train_manifest = Path(paths["manifest_root"]) / "train" / cfg["manifest_train"]
    val_manifest = Path(paths["manifest_root"]) / "val" / cfg["manifest_eval"]
    cache_stem = "cabin_wav" if cfg.get("waveform_cache_includes_cabin_noise", False) else "wav"
    train_ds = IntentDataset(
        train_manifest,
        cfg["sample_rate"],
        cfg["segment_seconds"],
        training=True,
        waveform_cache_path=cache_dir / f"train_{cache_stem}_{cfg['sample_rate']}hz.npy",
        **data_kwargs,
    )
    val_ds = IntentDataset(
        val_manifest,
        cfg["sample_rate"],
        cfg["segment_seconds"],
        training=False,
        waveform_cache_path=cache_dir / f"val_{cache_stem}_{cfg['sample_rate']}hz.npy",
        **data_kwargs,
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        print(f"Missing intent data: train={train_manifest} val={val_manifest}")
        return 1

    common_loader = loader_kwargs(cfg, device)
    labels = torch.tensor([int(row["label"]) for row in train_ds.rows], dtype=torch.long)
    counts = torch.bincount(labels, minlength=num_classes).float()
    use_balanced_sampler = bool(cfg["train"].get("balanced_sampler", True))
    sampler = None
    shuffle = True
    if use_balanced_sampler:
        sample_weights = 1.0 / counts[labels].clamp_min(1.0)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=shuffle,
        sampler=sampler,
        **common_loader,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"].get("val_batch_size", cfg["train"]["batch_size"])),
        shuffle=False,
        **common_loader,
    )
    model = IntentClassifier(
        cfg["sample_rate"],
        cfg["model"]["n_mels"],
        cfg["model"]["hidden"],
        cfg["model"]["num_layers"],
        num_classes,
        bool(cfg["model"].get("normalize_input", False)),
        bidirectional=bool(cfg["model"].get("bidirectional", True)),
    ).to(device)
    pretrained_checkpoint = cfg.get("pretrained_checkpoint")
    if pretrained_checkpoint:
        pretrained_checkpoint = Path(pretrained_checkpoint)
        if not pretrained_checkpoint.is_absolute():
            pretrained_checkpoint = REPO_ROOT / pretrained_checkpoint
        if pretrained_checkpoint.exists():
            loaded = load_checkpoint(pretrained_checkpoint, model)
            print(
                f"initialized_from={pretrained_checkpoint} "
                f"epoch={loaded.get('epoch', 'unknown')}",
                flush=True,
            )
        else:
            raise FileNotFoundError(f"Configured pretrained checkpoint not found: {pretrained_checkpoint}")
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=int(cfg["train"]["max_epochs"])
    )
    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    out_dir = Path(paths["checkpoint_root"]) / cfg["name"]
    log_dir = Path(paths["log_root"]) / cfg["name"]
    best_macro_f1 = float("-inf")
    bad_epochs = 0
    patience = int(cfg["train"].get("patience", 3))

    print(
        f"device={device} train={len(train_ds)} val={len(val_ds)} classes={num_classes} "
        f"balanced_sampler={use_balanced_sampler} amp={use_amp}",
        flush=True,
    )
    for epoch in range(1, int(cfg["train"]["max_epochs"]) + 1):
        model.train()
        train_correct = train_total = 0
        train_loss = 0.0
        for batch in train_loader:
            wav = batch["wav"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(wav)
                loss = F.cross_entropy(logits.float(), target)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite intent loss at epoch={epoch}; "
                    f"amp={use_amp} batch_size={target.numel()}"
                )
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
                opt.step()
            train_loss += float(loss.detach()) * target.numel()
            train_correct += int((logits.argmax(-1) == target).sum())
            train_total += target.numel()

        scheduler.step()
        val = evaluate(model, val_loader, device, num_classes)
        metrics = {
            "epoch": epoch,
            "device": str(device),
            "train_loss": train_loss / max(train_total, 1),
            "train_accuracy": train_correct / max(train_total, 1),
            "val": val,
            "lr": scheduler.get_last_lr()[0],
            "best_macro_f1": best_macro_f1,
            "balanced_sampler": use_balanced_sampler,
            "cabin_noise": bool(cfg.get("cabin_noise", False)),
        }
        write_metrics(log_dir / "metrics_last.json", metrics)
        print(
            f"epoch {epoch} train_loss={metrics['train_loss']:.4f} "
            f"train_acc={metrics['train_accuracy']:.3f} "
            f"val_acc={val['accuracy']:.3f} val_top5={val['top5_accuracy']:.3f} "
            f"val_macro_f1={val['macro_f1']:.3f}",
            flush=True,
        )
        if val["macro_f1"] > best_macro_f1:
            best_macro_f1 = val["macro_f1"]
            bad_epochs = 0
            metrics["best_macro_f1"] = best_macro_f1
            save_checkpoint(out_dir / "best.pt", model, epoch, metrics)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping after {epoch} epochs", flush=True)
                break

    write_metrics(
        log_dir / "train_last.json",
        {"best_macro_f1": best_macro_f1, "epochs_completed": epoch},
    )
    print(f"Done. Best checkpoint: {out_dir / 'best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
