#!/usr/bin/env python
"""Train an intent head on pretrained wav2vec 2.0 speech representations."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModel

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.datasets import IntentDataset
from car_tse.train.utils import append_jsonl, get_device, save_checkpoint, write_metrics


class FeatureDataset(Dataset):
    def __init__(self, feature_path: Path, rows: list[dict]):
        self.features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        self.labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
        if self.features.shape[0] != len(self.labels):
            raise ValueError(f"Feature/manifest length mismatch: {self.features.shape[0]} vs {len(self.labels)}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "features": torch.from_numpy(np.asarray(self.features[idx], dtype=np.float32)),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


class IntentHead(nn.Module):
    def __init__(self, input_dim: int, hidden: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def normalize_waveform(wav: torch.Tensor) -> torch.Tensor:
    wav = wav - wav.mean(dim=1, keepdim=True)
    std = wav.std(dim=1, keepdim=True).clamp_min(1.0e-5)
    return wav / std


@torch.inference_mode()
def cache_features(
    encoder: nn.Module,
    dataset: IntentDataset,
    output_path: Path,
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> int:
    if output_path.exists():
        try:
            cached = np.load(output_path, mmap_mode="r", allow_pickle=False)
            hidden = int(encoder.config.hidden_size)
            if tuple(cached.shape) == (len(dataset), hidden):
                print(f"Using cached features: {output_path}", flush=True)
                return hidden
        except (OSError, ValueError):
            pass

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    hidden = int(encoder.config.hidden_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(dataset), hidden),
    )
    encoder.eval()
    offset = 0
    for index, batch in enumerate(loader):
        wav = normalize_waveform(batch["wav"].to(device, non_blocking=True))
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            states = encoder(wav).last_hidden_state
            features = states.mean(dim=1)
        values = features.float().cpu().numpy()
        output[offset : offset + len(values)] = values
        offset += len(values)
        if (index + 1) % 50 == 0 or offset == len(dataset):
            print(f"features: {offset}/{len(dataset)}", flush=True)
    output.flush()
    del output
    return hidden


@torch.no_grad()
def evaluate(model: IntentHead, loader: DataLoader, device: torch.device, num_classes: int) -> dict[str, float]:
    model.eval()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    total = top5 = 0
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(features)
        pred = logits.argmax(-1)
        confusion.index_put_((labels.cpu(), pred.cpu()), torch.ones_like(labels.cpu()), accumulate=True)
        total += labels.numel()
        top5 += int((logits.topk(min(5, num_classes), dim=-1).indices == labels[:, None]).any(dim=1).sum())
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
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "stage4_intent_tse_domain.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--encoder", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--encoder-checkpoint", type=Path, default=None, help="Optional encoder state used by the frozen feature extractor.")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--feature-batch-size", type=int, default=24)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--head-hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--feature-dir", type=Path, default=REPO_ROOT / "data" / "cache" / "stage4_intent_wav2vec_tse_domain")
    parser.add_argument("--run-name", default="stage4_intent_wav2vec_tse_domain")
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Optional existing intent-head checkpoint to continue from.")
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
    train_manifest = manifest_root / "train" / cfg["manifest_train"]
    val_manifest = manifest_root / "val" / cfg["manifest_eval"]
    data_kwargs = {
        "cabin_noise": False,
        "cabin_noise_cache_size": 0,
        "audio_cache_size": 0,
        "waveform_cache_is_noisy": True,
    }
    train_audio = IntentDataset(
        train_manifest,
        cfg["sample_rate"],
        cfg["segment_seconds"],
        training=False,
        waveform_cache_path=cache_dir
        / cfg.get("waveform_cache_train", f"train_cabin_wav_{cfg['sample_rate']}hz.npy"),
        **data_kwargs,
    )
    val_audio = IntentDataset(
        val_manifest,
        cfg["sample_rate"],
        cfg["segment_seconds"],
        training=False,
        waveform_cache_path=cache_dir
        / cfg.get("waveform_cache_eval", f"val_cabin_wav_{cfg['sample_rate']}hz.npy"),
        **data_kwargs,
    )
    if len(train_audio) == 0 or len(val_audio) == 0:
        raise RuntimeError(f"Missing TSE-domain intent data: train={len(train_audio)} val={len(val_audio)}")

    encoder = AutoModel.from_pretrained(args.encoder).to(device)
    if args.encoder_checkpoint is not None:
        encoder_checkpoint = args.encoder_checkpoint
        if not encoder_checkpoint.is_absolute():
            encoder_checkpoint = REPO_ROOT / encoder_checkpoint
        encoder_state = torch.load(encoder_checkpoint, map_location="cpu", weights_only=False)
        encoder.load_state_dict(encoder_state["model_state_dict"], strict=True)
        print(f"initialized_encoder={encoder_checkpoint} epoch={encoder_state.get('epoch', 'unknown')}", flush=True)
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    encoder.eval()
    feature_dir = args.feature_dir
    if not feature_dir.is_absolute():
        feature_dir = REPO_ROOT / feature_dir
    train_features = feature_dir / "train_features.npy"
    val_features = feature_dir / "val_features.npy"
    hidden = cache_features(encoder, train_audio, train_features, device, args.feature_batch_size, args.amp and device.type == "cuda")
    cache_features(encoder, val_audio, val_features, device, args.feature_batch_size, args.amp and device.type == "cuda")

    train_ds = FeatureDataset(train_features, train_audio.rows)
    val_ds = FeatureDataset(val_features, val_audio.rows)
    labels = torch.from_numpy(train_ds.labels)
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = 1.0 / counts[labels].clamp_min(1.0)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    model = IntentHead(hidden, args.head_hidden, num_classes, args.dropout).to(device)
    if args.init_checkpoint is not None:
        init_checkpoint = args.init_checkpoint
        if not init_checkpoint.is_absolute():
            init_checkpoint = REPO_ROOT / init_checkpoint
        loaded = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        state = loaded.get("model_state_dict", loaded)
        model.load_state_dict(state, strict=True)
        print(f"initialized_head={init_checkpoint} epoch={loaded.get('epoch', 'unknown')}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_epochs)
    out_dir = Path(paths["checkpoint_root"]) / args.run_name
    log_dir = Path(paths["log_root"]) / args.run_name
    run_id = uuid.uuid4().hex
    history_path = log_dir / f"epoch_metrics_{run_id}.jsonl"
    write_metrics(
        log_dir / f"run_metadata_{run_id}.json",
        {
            "run_id": run_id,
            "run_name": args.run_name,
            "device": str(device),
            "config": str(args.config),
            "train_manifest": str(train_manifest),
            "val_manifest": str(val_manifest),
            "feature_dir": str(feature_dir),
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
            "encoder_checkpoint": str(args.encoder_checkpoint) if args.encoder_checkpoint else None,
        },
    )
    best_f1 = float("-inf")
    bad_epochs = 0
    print(f"device={device} encoder={args.encoder} hidden={hidden} train={len(train_ds)} val={len(val_ds)} classes={num_classes}", flush=True)
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_loss = train_correct = train_total = 0
        for batch in train_loader:
            features = batch["features"].to(device, non_blocking=True)
            labels_batch = batch["label"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(features)
            loss = F.cross_entropy(logits, labels_batch, label_smoothing=0.03)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_loss += float(loss.detach()) * labels_batch.numel()
            train_correct += int((logits.argmax(-1) == labels_batch).sum())
            train_total += labels_batch.numel()
        scheduler.step()
        val = evaluate(model, val_loader, device, num_classes)
        metrics = {
            "epoch": epoch,
            "run_id": run_id,
            "device": str(device),
            "encoder": args.encoder,
            "train_loss": train_loss / max(train_total, 1),
            "train_accuracy": train_correct / max(train_total, 1),
            "val": val,
            "lr": scheduler.get_last_lr()[0],
            "feature_cache": str(feature_dir),
        }
        write_metrics(log_dir / "metrics_last.json", metrics)
        append_jsonl(history_path, metrics)
        print(
            f"epoch {epoch} train_loss={metrics['train_loss']:.4f} train_acc={metrics['train_accuracy']:.3f} "
            f"val_acc={val['accuracy']:.3f} val_top5={val['top5_accuracy']:.3f} val_macro_f1={val['macro_f1']:.3f}",
            flush=True,
        )
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            bad_epochs = 0
            metrics["best_macro_f1"] = best_f1
            save_checkpoint(out_dir / "best.pt", model, epoch, metrics)
            # Keep the exact frozen encoder state next to the head so the
            # candidate can be loaded without relying on an unrelated run.
            torch.save({"model_state_dict": encoder.state_dict(), "epoch": epoch}, out_dir / "encoder_best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs", flush=True)
                break
    write_metrics(
        log_dir / "train_last.json",
        {
            "best_macro_f1": best_f1,
            "epochs_completed": epoch,
            "run_id": run_id,
            "history": str(history_path),
        },
    )
    print(f"Done. Best checkpoint: {out_dir / 'best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
