#!/usr/bin/env python
"""Train/evaluate a direct-audio Speech Commands classifier.

This is a separate, reproducible comparison track for Hussein et al.'s
Speech-Commands-style command-recognition experiment.  It is deliberately
not wired into the in-car pipeline: the benchmark has no wake word, TSE,
speaker authorization, passenger speech, or vehicle-action state machine.

The headline result uses the official Speech Commands validation/testing
lists.  A small random 70/30 split can be requested for protocol inspection,
but must not replace the official split in the paper because it permits
speaker/session leakage.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, get_cosine_schedule_with_warmup

from car_tse.io.audio import read_mono
from car_tse.train.utils import get_device, write_metrics


def read_split(path: Path) -> set[str]:
    return {line.strip().replace("/", "\\") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def build_rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    validation = read_split(root / "validation_list.txt")
    testing = read_split(root / "testing_list.txt")
    labels = sorted(
        item.name
        for item in root.iterdir()
        if item.is_dir() and item.name != "_background_noise_"
    )
    label_map = {name: index for index, name in enumerate(labels)}
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for label in labels:
        for wav_path in sorted((root / label).glob("*.wav")):
            rel = wav_path.relative_to(root).as_posix().replace("/", "\\")
            row = {"wav": str(wav_path), "label": label_map[label], "class": label, "relative": rel}
            if rel in validation:
                val_rows.append(row)
            elif rel in testing:
                test_rows.append(row)
            else:
                train_rows.append(row)
    return train_rows, val_rows, test_rows, labels


class SpeechCommandsDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], sample_rate: int, max_seconds: float, training: bool):
        self.rows = rows
        self.sample_rate = int(sample_rate)
        self.max_samples = max(1, int(round(float(max_seconds) * self.sample_rate)))
        self.training = bool(training)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        wav, sample_rate = read_mono(row["wav"], self.sample_rate)
        if sample_rate != self.sample_rate:
            raise ValueError(f"Unexpected sample rate {sample_rate} for {row['wav']}")
        wav = np.asarray(wav, dtype=np.float32)
        if wav.size == 0:
            wav = np.zeros(1, dtype=np.float32)
        if wav.shape[0] > self.max_samples:
            if self.training:
                start = random.randint(0, wav.shape[0] - self.max_samples)
            else:
                start = (wav.shape[0] - self.max_samples) // 2
            wav = wav[start : start + self.max_samples]
        wav = wav - float(wav.mean())
        wav = wav / max(float(wav.std()), 1.0e-5)
        return {"wav": torch.from_numpy(wav), "label": int(row["label"]), "class": row["class"]}


class Collate:
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        wavs = [item["wav"] for item in batch]
        lengths = torch.tensor([wav.shape[0] for wav in wavs], dtype=torch.long)
        input_values = pad_sequence(wavs, batch_first=True, padding_value=0.0)
        time = torch.arange(input_values.shape[1], dtype=torch.long).unsqueeze(0)
        attention_mask = (time < lengths.unsqueeze(1)).to(torch.long)
        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        }


class MaskedAttentivePool(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.score = nn.Linear(hidden, 1)

    def forward(self, states: torch.Tensor, feature_mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(self.norm(states)).squeeze(-1)
        scores = scores.masked_fill(~feature_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        attentive = torch.sum(states * weights.unsqueeze(-1), dim=1)
        masked = states * feature_mask.unsqueeze(-1)
        mean = masked.sum(dim=1) / feature_mask.sum(dim=1, keepdim=True).clamp_min(1).to(states.dtype)
        return torch.cat([attentive, mean], dim=-1)


class SpeechCommandsModel(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int, hidden: int, dropout: float):
        super().__init__()
        self.encoder = encoder
        encoder_hidden = int(encoder.config.hidden_size)
        self.pool = MaskedAttentivePool(encoder_hidden)
        self.head = nn.Sequential(
            nn.LayerNorm(encoder_hidden * 2),
            nn.Linear(encoder_hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def _feature_mask(self, attention_mask: torch.Tensor, frames: int) -> torch.Tensor:
        lengths = attention_mask.sum(dim=-1)
        get_lengths = getattr(self.encoder, "_get_feat_extract_output_lengths", None)
        if get_lengths is not None:
            feature_lengths = get_lengths(lengths)
        else:
            feature_lengths = torch.div(lengths + 319, 320, rounding_mode="floor")
        feature_lengths = feature_lengths.to(attention_mask.device).clamp(min=1, max=frames)
        positions = torch.arange(frames, device=attention_mask.device).unsqueeze(0)
        return positions < feature_lengths.unsqueeze(1)

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        states = self.encoder(input_values=input_values, attention_mask=attention_mask).last_hidden_state
        return self.head(self.pool(states, self._feature_mask(attention_mask, states.shape[1])))


def set_trainable_layers(encoder: nn.Module, unfreeze_layers: int) -> None:
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    if hasattr(encoder, "feature_projection"):
        encoder.feature_projection.requires_grad_(True)
    layers = getattr(getattr(encoder, "encoder", None), "layers", None)
    if layers is None:
        raise RuntimeError("Expected a Wav2Vec2/WavLM-style encoder")
    for layer in layers[-int(unfreeze_layers) :]:
        layer.requires_grad_(True)


def freeze_feature_extractor_eval(encoder: nn.Module) -> None:
    feature_extractor = getattr(encoder, "feature_extractor", None)
    if feature_extractor is not None:
        feature_extractor.eval()


def metrics_from_confusion(confusion: torch.Tensor, top5: int, total: int) -> dict[str, float]:
    true_positive = confusion.diag().float()
    support = confusion.sum(dim=1).float()
    predicted = confusion.sum(dim=0).float()
    precision = true_positive / predicted.clamp_min(1.0)
    recall = true_positive / support.clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1.0e-8)
    return {
        "accuracy": float(true_positive.sum().item() / max(total, 1)),
        "top5_accuracy": float(top5 / max(total, 1)),
        "macro_precision": float(precision.mean().item()),
        "macro_recall": float(recall.mean().item()),
        "macro_f1": float(f1.mean().item()),
        "examples": int(total),
    }


@torch.no_grad()
def evaluate(model: SpeechCommandsModel, loader: DataLoader, device: torch.device, num_classes: int, amp: bool) -> tuple[dict[str, float], torch.Tensor]:
    model.eval()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    total = top5 = 0
    for batch in loader:
        values = batch["input_values"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(values, mask)
        pred = logits.argmax(dim=-1)
        confusion.index_put_((target.cpu(), pred.cpu()), torch.ones_like(target.cpu()), accumulate=True)
        top5 += int((logits.topk(min(5, num_classes), dim=-1).indices == target[:, None]).any(dim=1).sum())
        total += int(target.numel())
    return metrics_from_confusion(confusion, top5, total), confusion


def plot_history(history: list[dict[str, Any]], path: Path, run_name: str) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o")
    axes[0].set_title("Training loss")
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], marker="o", label="train")
    axes[1].plot(epochs, [row["val"]["accuracy"] for row in history], marker="o", label="validation")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[2].plot(epochs, [row["val"]["macro_f1"] for row in history], marker="o", label="macro-F1")
    axes[2].plot(epochs, [row["val"]["top5_accuracy"] for row in history], marker="o", label="top-5")
    axes[2].set_title("Validation metrics")
    axes[2].legend()
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    fig.suptitle(run_name)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/speech_commands_v2/extracted"))
    parser.add_argument("--encoder", default="microsoft/wavlm-base-plus")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-seconds", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--unfreeze-layers", type=int, default=4)
    parser.add_argument("--encoder-lr", type=float, default=1.0e-5)
    parser.add_argument("--head-lr", type=float, default=3.0e-4)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--run-name", default="speech_commands_wavlm_hussein_track_2026-10-18")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("D:/datasets/speech_commands/artifacts/checkpoints"))
    parser.add_argument("--log-root", type=Path, default=Path("D:/datasets/speech_commands/artifacts/logs"))
    parser.add_argument("--figure-root", type=Path, default=Path("D:/Prawin_New/tse/car-tse-main/reports/figures"))
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=20261018)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    use_amp = bool(args.amp and device.type == "cuda")

    root = args.data_root.resolve()
    train_rows, val_rows, test_rows, labels = build_rows(root)
    train_dataset = SpeechCommandsDataset(train_rows, args.sample_rate, args.max_seconds, True)
    val_dataset = SpeechCommandsDataset(val_rows, args.sample_rate, args.max_seconds, False)
    test_dataset = SpeechCommandsDataset(test_rows, args.sample_rate, args.max_seconds, False)
    collate = Collate()
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "collate_fn": collate,
    }
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, **loader_kwargs)

    encoder = AutoModel.from_pretrained(args.encoder).to(device)
    encoder.config.apply_spec_augment = False
    encoder.config.mask_time_prob = 0.0
    encoder.config.mask_feature_prob = 0.0
    encoder.config.use_cache = False
    set_trainable_layers(encoder, args.unfreeze_layers)
    model = SpeechCommandsModel(encoder, len(labels), args.hidden, args.dropout).to(device)

    counts = torch.bincount(torch.tensor([int(row["label"]) for row in train_rows]), minlength=len(labels)).float().clamp_min(1.0)
    weights = (counts.mean() / counts).pow(0.25)
    weights = (weights / weights.mean()).clamp(0.5, 3.0).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.02)
    encoder_params = [p for p in encoder.parameters() if p.requires_grad]
    head_params = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [{"params": encoder_params, "lr": args.encoder_lr}, {"params": head_params, "lr": args.head_lr}],
        weight_decay=1.0e-2,
    )
    update_steps = math.ceil(len(train_loader) / max(args.grad_accumulation, 1)) * args.max_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, int(update_steps * 0.08)), update_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    checkpoint_dir = args.checkpoint_root.resolve() / args.run_name
    log_dir = args.log_root.resolve() / args.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    write_metrics(log_dir / f"run_metadata_{run_id}.json", {
        "run_id": run_id,
        "run_name": args.run_name,
        "device": str(device),
        "encoder": args.encoder,
        "data_root": str(root),
        "train_examples": len(train_rows),
        "validation_examples": len(val_rows),
        "test_examples": len(test_rows),
        "classes": labels,
        "official_split": True,
        "max_seconds": args.max_seconds,
        "unfreeze_layers": args.unfreeze_layers,
        "seed": args.seed,
    })
    (log_dir / "label_map.json").write_text(json.dumps({name: i for i, name in enumerate(labels)}, indent=2), encoding="utf-8")
    print(f"device={device} train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)} classes={len(labels)} amp={use_amp}", flush=True)

    best_f1 = float("-inf")
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    metrics_path = log_dir / f"epoch_metrics_{run_id}.jsonl"
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        freeze_feature_extractor_eval(encoder)
        total_loss = correct = total = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            values = batch["input_values"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(values, mask)
                loss = loss_fn(logits.float(), target)
                scaled = loss / max(args.grad_accumulation, 1)
            if use_amp:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            should_step = step % max(args.grad_accumulation, 1) == 0 or step == len(train_loader)
            if should_step:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach()) * target.numel()
            correct += int((logits.argmax(-1) == target).sum())
            total += int(target.numel())
        val, _ = evaluate(model, val_loader, device, len(labels), use_amp)
        row = {
            "epoch": epoch,
            "run_id": run_id,
            "train_loss": total_loss / max(total, 1),
            "train_accuracy": correct / max(total, 1),
            "val": val,
            "lr": [group["lr"] for group in optimizer.param_groups],
        }
        history.append(row)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(f"epoch={epoch} train_loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.4f} val_acc={val['accuracy']:.4f} val_f1={val['macro_f1']:.4f} val_top5={val['top5_accuracy']:.4f}", flush=True)
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "labels": labels, "args": vars(args), "epoch": epoch, "val": val}, checkpoint_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early_stop=1 best_epoch={best_epoch}", flush=True)
                break

    best = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_metrics, confusion = evaluate(model, test_loader, device, len(labels), use_amp)
    (log_dir / "metrics_official_test.json").write_text(json.dumps({"best_epoch": best_epoch, "test": test_metrics}, indent=2), encoding="utf-8")
    np.save(log_dir / "confusion_official_test.npy", confusion.numpy())
    (log_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    plot_history(history, args.figure_root.resolve() / args.run_name, args.run_name)
    print(json.dumps({"best_epoch": best_epoch, "test": test_metrics, "checkpoint": str(checkpoint_dir / "best.pt")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
