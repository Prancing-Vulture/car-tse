#!/usr/bin/env python
"""Fine-tune an acoustic SLURP action model with masked variable-length pooling.

This is the stronger direct-audio SLURP track.  It is intentionally separate
from the in-car target-speaker pipeline and from the frozen-feature lower
bound.  It predicts the stable 46-class SLURP ``action`` field and uses the
scenario as an auxiliary task.  The encoder is fine-tuned only in its upper
layers by default, which keeps the run practical on a single GPU.
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
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, get_cosine_schedule_with_warmup

from car_tse.io.audio import read_mono
from car_tse.train.utils import append_jsonl, get_device, write_metrics


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class SlurpAudioDataset(Dataset):
    """Read complete SLURP recordings, capped only for pathological long files."""

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
        if wav.size == 0:
            wav = np.zeros(1, dtype=np.float32)
        if wav.shape[0] > self.max_samples:
            # SLURP commands normally fit well below this cap.  A deterministic
            # center crop avoids changing the validation/test protocol.
            start = (wav.shape[0] - self.max_samples) // 2
            wav = wav[start : start + self.max_samples]
        wav = np.asarray(wav, dtype=np.float32)
        wav = wav - float(wav.mean())
        wav = wav / max(float(wav.std()), 1.0e-5)
        return {
            "wav": torch.from_numpy(wav),
            "label": int(row["label"]),
            "scenario": row["scenario"],
            "microphone": row.get("microphone", "unknown"),
        }


class SlurpCollate:
    def __init__(self, scenario_map: dict[str, int]):
        self.scenario_map = scenario_map

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
            "scenario": torch.tensor([self.scenario_map[item["scenario"]] for item in batch], dtype=torch.long),
            "microphone": [item["microphone"] for item in batch],
        }


def make_collate(scenario_map: dict[str, int]):
    return SlurpCollate(scenario_map)


class MaskedAttentivePool(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.score = nn.Linear(hidden, 1)

    def forward(self, states: torch.Tensor, feature_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(states)
        scores = self.score(normalized).squeeze(-1)
        scores = scores.masked_fill(~feature_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        attentive = torch.sum(states * weights.unsqueeze(-1), dim=1)
        masked_states = states * feature_mask.unsqueeze(-1)
        mean = masked_states.sum(dim=1) / feature_mask.sum(dim=1, keepdim=True).clamp_min(1).to(states.dtype)
        return torch.cat([attentive, mean], dim=-1)


class SlurpActionModel(nn.Module):
    def __init__(self, encoder: nn.Module, num_actions: int, num_scenarios: int, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.encoder = encoder
        encoder_hidden = int(encoder.config.hidden_size)
        self.pool = MaskedAttentivePool(encoder_hidden)
        self.shared = nn.Sequential(
            nn.LayerNorm(encoder_hidden * 2),
            nn.Linear(encoder_hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.action_head = nn.Linear(hidden, num_actions)
        self.scenario_head = nn.Linear(hidden, num_scenarios)

    def _feature_mask(self, attention_mask: torch.Tensor, frames: int) -> torch.Tensor:
        lengths = attention_mask.sum(dim=-1)
        get_lengths = getattr(self.encoder, "_get_feat_extract_output_lengths", None)
        if get_lengths is not None:
            feature_lengths = get_lengths(lengths)
        else:
            # Wav2Vec2/WavLM use a fixed convolutional reduction of roughly 320.
            feature_lengths = torch.div(lengths + 319, 320, rounding_mode="floor")
        feature_lengths = feature_lengths.to(device=attention_mask.device).clamp(min=1, max=frames)
        positions = torch.arange(frames, device=attention_mask.device).unsqueeze(0)
        return positions < feature_lengths.unsqueeze(1)

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        states = self.encoder(input_values=input_values, attention_mask=attention_mask).last_hidden_state
        feature_mask = self._feature_mask(attention_mask, states.shape[1])
        shared = self.shared(self.pool(states, feature_mask))
        return self.action_head(shared), self.scenario_head(shared)


def set_trainable_layers(encoder: nn.Module, unfreeze_layers: int) -> None:
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    if hasattr(encoder, "feature_projection"):
        encoder.feature_projection.requires_grad_(True)
    transformer = getattr(encoder, "encoder", None)
    layers = getattr(transformer, "layers", None)
    if layers is None:
        raise RuntimeError("Expected a Wav2Vec2/WavLM-style encoder with encoder.layers")
    for layer in layers[-int(unfreeze_layers) :]:
        layer.requires_grad_(True)


def freeze_feature_extractor_eval(encoder: nn.Module) -> None:
    feature_extractor = getattr(encoder, "feature_extractor", None)
    if feature_extractor is not None:
        feature_extractor.eval()


def metric_from_confusion(confusion: torch.Tensor, top5: int, total: int) -> dict[str, float]:
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
        "examples": float(total),
    }


@torch.no_grad()
def evaluate(
    model: SlurpActionModel,
    loader: DataLoader,
    device: torch.device,
    num_actions: int,
    amp: bool,
) -> dict[str, float]:
    model.eval()
    confusion = torch.zeros((num_actions, num_actions), dtype=torch.long)
    total = top5 = 0
    for batch in loader:
        input_values = batch["input_values"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits, _ = model(input_values, attention_mask)
        pred = logits.argmax(dim=-1)
        confusion.index_put_((target.cpu(), pred.cpu()), torch.ones_like(target.cpu()), accumulate=True)
        top5 += int((logits.topk(min(5, num_actions), dim=-1).indices == target[:, None]).any(dim=1).sum())
        total += int(target.numel())
    return metric_from_confusion(confusion, top5, total)


@torch.no_grad()
def evaluate_microphones(
    model: SlurpActionModel,
    rows: list[dict[str, Any]],
    dataset: SlurpAudioDataset,
    scenario_map: dict[str, int],
    device: torch.device,
    num_actions: int,
    batch_size: int,
    amp: bool,
    num_workers: int = 0,
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    collate = make_collate(scenario_map)
    for microphone in ("all", "distant", "close_talk"):
        indices = list(range(len(rows))) if microphone == "all" else [i for i, row in enumerate(rows) if row.get("microphone") == microphone]
        if not indices:
            continue
        subset = torch.utils.data.Subset(dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=num_workers > 0,
            collate_fn=collate,
        )
        output[microphone] = evaluate(model, loader, device, num_actions, amp)
    return output


def plot_history(history: list[dict[str, Any]], output_dir: Path, run_name: str) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="train loss")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], marker="o", label="train")
    axes[1].plot(epochs, [row["val"]["accuracy"] for row in history], marker="o", label="validation")
    axes[1].set_title("Action accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    axes[2].plot(epochs, [row["val"]["macro_f1"] for row in history], marker="o", label="macro-F1")
    axes[2].plot(epochs, [row["val"]["top5_accuracy"] for row in history], marker="o", label="top-5")
    axes[2].set_title("Validation metrics")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    fig.suptitle(run_name)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{run_name}.png", dpi=180)
    fig.savefig(output_dir / f"{run_name}.pdf")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, default=Path("D:/datasets/slurp/manifests"))
    parser.add_argument("--train-manifest", default="slurp_train.jsonl")
    parser.add_argument("--devel-manifest", default="slurp_devel.jsonl")
    parser.add_argument("--test-manifest", default="slurp_test.jsonl")
    parser.add_argument("--encoder", default="microsoft/wavlm-base-plus")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-seconds", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--grad-accumulation", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--unfreeze-layers", type=int, default=6)
    parser.add_argument("--encoder-lr", type=float, default=1.0e-5)
    parser.add_argument("--head-lr", type=float, default=3.0e-4)
    parser.add_argument("--scenario-loss-weight", type=float, default=0.15)
    parser.add_argument("--class-weight-power", type=float, default=0.25)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--run-name", default="slurp_action_wavlm_attn_ft_2026-10-18")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("D:/datasets/slurp/artifacts/checkpoints"))
    parser.add_argument("--log-root", type=Path, default=Path("D:/datasets/slurp/artifacts/logs"))
    parser.add_argument("--figure-root", type=Path, default=Path("D:/Prawin_New/tse/car-tse-main/reports/figures"))
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
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

    manifest_root = args.manifest_root.resolve()
    train_rows = read_jsonl(manifest_root / args.train_manifest)
    devel_rows = read_jsonl(manifest_root / args.devel_manifest)
    test_rows = read_jsonl(manifest_root / args.test_manifest)
    all_rows = train_rows + devel_rows + test_rows
    num_actions = max(int(row["label"]) for row in all_rows) + 1
    scenario_names = sorted({str(row["scenario"]) for row in all_rows})
    scenario_map = {name: index for index, name in enumerate(scenario_names)}
    num_scenarios = len(scenario_map)

    train_dataset = SlurpAudioDataset(train_rows, args.sample_rate, args.max_seconds, training=True)
    devel_dataset = SlurpAudioDataset(devel_rows, args.sample_rate, args.max_seconds, training=False)
    test_dataset = SlurpAudioDataset(test_rows, args.sample_rate, args.max_seconds, training=False)
    collate = make_collate(scenario_map)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collate,
    )
    devel_loader = DataLoader(
        devel_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collate,
    )

    encoder = AutoModel.from_pretrained(args.encoder).to(device)
    encoder.config.apply_spec_augment = False
    encoder.config.mask_time_prob = 0.0
    encoder.config.mask_feature_prob = 0.0
    encoder.config.use_cache = False
    if args.gradient_checkpointing:
        encoder.gradient_checkpointing_enable()
    set_trainable_layers(encoder, args.unfreeze_layers)
    model = SlurpActionModel(encoder, num_actions, num_scenarios, args.hidden, args.dropout).to(device)

    labels = torch.tensor([int(row["label"]) for row in train_rows], dtype=torch.long)
    counts = torch.bincount(labels, minlength=num_actions).float().clamp_min(1.0)
    class_weights = (counts.mean() / counts).pow(float(args.class_weight_power))
    class_weights = (class_weights / class_weights.mean()).clamp(0.5, 3.0).to(device)
    action_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.02)
    scenario_loss = nn.CrossEntropyLoss(label_smoothing=0.02)
    encoder_params = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    head_params = [parameter for name, parameter in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.encoder_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=1.0e-2,
    )
    update_steps = math.ceil(len(train_loader) / max(args.grad_accumulation, 1)) * args.max_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=max(1, int(update_steps * 0.08)), num_training_steps=update_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    checkpoint_dir = args.checkpoint_root.resolve() / args.run_name
    log_dir = args.log_root.resolve() / args.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    history_path = log_dir / f"epoch_metrics_{run_id}.jsonl"
    write_metrics(
        log_dir / f"run_metadata_{run_id}.json",
        {
            "run_id": run_id,
            "run_name": args.run_name,
            "device": str(device),
            "encoder": args.encoder,
            "manifest_root": str(manifest_root),
            "train_manifest": str(manifest_root / args.train_manifest),
            "devel_manifest": str(manifest_root / args.devel_manifest),
            "test_manifest": str(manifest_root / args.test_manifest),
            "max_seconds": args.max_seconds,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "grad_accumulation": args.grad_accumulation,
            "num_workers": args.num_workers,
            "unfreeze_layers": args.unfreeze_layers,
            "scenario_loss_weight": args.scenario_loss_weight,
            "class_weight_power": args.class_weight_power,
            "num_actions": num_actions,
            "num_scenarios": num_scenarios,
            "scenario_map": scenario_map,
            "seed": args.seed,
        },
    )
    (log_dir / "label_maps.json").write_text(json.dumps({"scenario": scenario_map}, indent=2), encoding="utf-8")

    print(
        f"device={device} encoder={args.encoder} train={len(train_dataset)} devel={len(devel_dataset)} "
        f"test={len(test_dataset)} actions={num_actions} scenarios={num_scenarios} "
        f"max_seconds={args.max_seconds} unfreeze_layers={args.unfreeze_layers} amp={use_amp}",
        flush=True,
    )
    best_f1 = float("-inf")
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        freeze_feature_extractor_eval(encoder)
        train_loss = train_correct = train_total = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            input_values = batch["input_values"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            target_scenario = batch["scenario"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                action_logits, scenario_logits = model(input_values, attention_mask)
                loss = action_loss(action_logits.float(), target) + args.scenario_loss_weight * scenario_loss(scenario_logits.float(), target_scenario)
                scaled_loss = loss / max(args.grad_accumulation, 1)
            if use_amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
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
            train_loss += float(loss.detach()) * target.numel()
            train_correct += int((action_logits.argmax(-1) == target).sum())
            train_total += int(target.numel())
        val = evaluate(model, devel_loader, device, num_actions, use_amp)
        metrics = {
            "epoch": epoch,
            "run_id": run_id,
            "device": str(device),
            "train_loss": train_loss / max(train_total, 1),
            "train_accuracy": train_correct / max(train_total, 1),
            "val": val,
            "lr": [group["lr"] for group in optimizer.param_groups],
        }
        history.append(metrics)
        append_jsonl(history_path, metrics)
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
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                    "encoder": args.encoder,
                    "num_actions": num_actions,
                    "num_scenarios": num_scenarios,
                    "scenario_map": scenario_map,
                    "max_seconds": args.max_seconds,
                },
                checkpoint_dir / "best.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs", flush=True)
                break

    best_state = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_state["model_state_dict"], strict=True)
    test_by_microphone = evaluate_microphones(
        model,
        test_rows,
        test_dataset,
        scenario_map,
        device,
        num_actions,
        args.eval_batch_size,
        use_amp,
    )
    test_metrics = {
        "run_id": run_id,
        "best_epoch": int(best_state["epoch"]),
        "best_devel_macro_f1": float(best_f1),
        "test_by_microphone": test_by_microphone,
        "encoder": args.encoder,
        "max_seconds": args.max_seconds,
        "unfreeze_layers": args.unfreeze_layers,
        "test_manifest": str(manifest_root / args.test_manifest),
        "checkpoint": str(checkpoint_dir / "best.pt"),
    }
    write_metrics(log_dir / "metrics_test.json", test_metrics)
    write_metrics(log_dir / "train_last.json", {"best_macro_f1": best_f1, "epochs_completed": len(history), "run_id": run_id, "history": str(history_path)})
    plot_history(history, args.figure_root.resolve(), args.run_name)
    print(json.dumps(test_metrics, indent=2), flush=True)
    print(f"Done. Best checkpoint: {checkpoint_dir / 'best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
