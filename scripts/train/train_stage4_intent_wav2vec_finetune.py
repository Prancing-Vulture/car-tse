#!/usr/bin/env python
"""Fine-tune the upper wav2vec 2.0 layers for cabin-domain intent recognition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoModel

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.datasets import IntentDataset
from car_tse.train.utils import get_device, load_checkpoint, save_checkpoint, write_metrics
from train_stage4_intent_wav2vec import IntentHead, evaluate, normalize_waveform


def set_trainable_layers(encoder: nn.Module, unfreeze_layers: int) -> None:
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    encoder.feature_projection.requires_grad_(True)
    encoder.encoder.layer_norm.requires_grad_(True)
    layers = encoder.encoder.layers
    for layer in layers[-unfreeze_layers:]:
        layer.requires_grad_(True)


def loader_args(device: torch.device, batch_size: int) -> dict:
    return {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "stage4_intent_tse_domain.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--encoder", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--head-checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "stage4_intent_wav2vec_tse_domain" / "best.pt")
    parser.add_argument("--encoder-checkpoint", type=Path, default=None)
    parser.add_argument("--run-name", default="stage4_intent_wav2vec_ft4_tse_domain")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--encoder-lr", type=float, default=1.0e-5)
    parser.add_argument("--head-lr", type=float, default=3.0e-4)
    parser.add_argument("--unfreeze-layers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    device = get_device()
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    label_map_path = Path(paths["manifest_root"]) / "meta" / cfg["label_map"]
    num_classes = len(json.loads(label_map_path.read_text(encoding="utf-8")))
    manifest_root = Path(paths["manifest_root"])
    cache_dir = Path(cfg["waveform_cache_dir"])
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    common_kwargs = {
        "cabin_noise": False,
        "cabin_noise_cache_size": 0,
        "audio_cache_size": 0,
        "waveform_cache_is_noisy": True,
    }
    train_ds = IntentDataset(
        manifest_root / "train" / cfg["manifest_train"],
        cfg["sample_rate"],
        cfg["segment_seconds"],
        training=True,
        waveform_cache_path=cache_dir / f"train_cabin_wav_{cfg['sample_rate']}hz.npy",
        **common_kwargs,
    )
    val_ds = IntentDataset(
        manifest_root / "val" / cfg["manifest_eval"],
        cfg["sample_rate"],
        cfg["segment_seconds"],
        training=False,
        waveform_cache_path=cache_dir / f"val_cabin_wav_{cfg['sample_rate']}hz.npy",
        **common_kwargs,
    )
    encoder = AutoModel.from_pretrained(args.encoder).to(device)
    if args.encoder_checkpoint is not None:
        encoder_checkpoint = args.encoder_checkpoint
        if not encoder_checkpoint.is_absolute():
            encoder_checkpoint = REPO_ROOT / encoder_checkpoint
        state = torch.load(encoder_checkpoint, map_location="cpu", weights_only=False)
        encoder.load_state_dict(state["model_state_dict"], strict=True)
        print(f"initialized_encoder={encoder_checkpoint} epoch={state.get('epoch', 'unknown')}", flush=True)
    set_trainable_layers(encoder, args.unfreeze_layers)
    encoder.config.use_cache = False
    hidden = int(encoder.config.hidden_size)
    head = IntentHead(hidden, 512, num_classes, 0.2).to(device)
    if args.head_checkpoint.exists():
        loaded = load_checkpoint(args.head_checkpoint, head)
        print(f"initialized_head={args.head_checkpoint} epoch={loaded.get('epoch', 'unknown')}", flush=True)

    labels = torch.tensor([int(row["label"]) for row in train_ds.rows], dtype=torch.long)
    counts = torch.bincount(labels, minlength=num_classes).float()
    sampler = WeightedRandomSampler(1.0 / counts[labels].clamp_min(1.0), num_samples=len(labels), replacement=True)
    train_loader = DataLoader(train_ds, sampler=sampler, **loader_args(device, args.batch_size))
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args(device, args.batch_size))
    encoder_params = [p for p in encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.encoder_lr},
            {"params": head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=1.0e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    out_dir = Path(paths["checkpoint_root"]) / args.run_name
    log_dir = Path(paths["log_root"]) / args.run_name
    best_f1 = float("-inf")
    bad_epochs = 0
    print(
        f"device={device} encoder={args.encoder} train={len(train_ds)} val={len(val_ds)} "
        f"unfreeze_layers={args.unfreeze_layers} amp={use_amp}",
        flush=True,
    )
    for epoch in range(1, args.max_epochs + 1):
        encoder.train()
        head.train()
        train_loss = train_correct = train_total = 0
        for batch in train_loader:
            wav = normalize_waveform(batch["wav"].to(device, non_blocking=True))
            target = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                states = encoder(wav).last_hidden_state
                features = states.mean(dim=1)
                logits = head(features)
                loss = F.cross_entropy(logits.float(), target, label_smoothing=0.03)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(encoder_params + list(head.parameters()), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder_params + list(head.parameters()), 5.0)
                optimizer.step()
            train_loss += float(loss.detach()) * target.numel()
            train_correct += int((logits.argmax(-1) == target).sum())
            train_total += target.numel()
        scheduler.step()
        encoder.eval()
        val = evaluate(head, val_loader, device, num_classes) if False else None
        # Evaluate with the same encoder+pooling path used for training.
        confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
        total = top5 = 0
        with torch.no_grad():
            for batch in val_loader:
                wav = normalize_waveform(batch["wav"].to(device, non_blocking=True))
                target = batch["label"].to(device, non_blocking=True)
                logits = head(encoder(wav).last_hidden_state.mean(dim=1))
                pred = logits.argmax(-1)
                confusion.index_put_((target.cpu(), pred.cpu()), torch.ones_like(target.cpu()), accumulate=True)
                total += target.numel()
                top5 += int((logits.topk(min(5, num_classes), dim=-1).indices == target[:, None]).any(dim=1).sum())
        tp = confusion.diag().float()
        support = confusion.sum(dim=1).float()
        predicted = confusion.sum(dim=0).float()
        precision = tp / predicted.clamp_min(1.0)
        recall = tp / support.clamp_min(1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
        val = {
            "accuracy": float(tp.sum().item() / max(total, 1)),
            "top5_accuracy": float(top5 / max(total, 1)),
            "macro_precision": float(precision.mean().item()),
            "macro_recall": float(recall.mean().item()),
            "macro_f1": float(f1.mean().item()),
            "examples": float(total),
        }
        metrics = {
            "epoch": epoch,
            "device": str(device),
            "encoder": args.encoder,
            "unfreeze_layers": args.unfreeze_layers,
            "train_loss": train_loss / max(train_total, 1),
            "train_accuracy": train_correct / max(train_total, 1),
            "val": val,
            "lr": [group["lr"] for group in optimizer.param_groups],
        }
        write_metrics(log_dir / "metrics_last.json", metrics)
        print(
            f"epoch {epoch} train_loss={metrics['train_loss']:.4f} train_acc={metrics['train_accuracy']:.3f} "
            f"val_acc={val['accuracy']:.3f} val_top5={val['top5_accuracy']:.3f} val_macro_f1={val['macro_f1']:.3f}",
            flush=True,
        )
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            bad_epochs = 0
            metrics["best_macro_f1"] = best_f1
            save_checkpoint(out_dir / "best.pt", head, epoch, metrics)
            torch.save({"model_state_dict": encoder.state_dict(), "epoch": epoch}, out_dir / "encoder_best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs", flush=True)
                break
    write_metrics(log_dir / "train_last.json", {"best_macro_f1": best_f1, "epochs_completed": epoch})
    print(f"Done. Best head checkpoint: {out_dir / 'best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
