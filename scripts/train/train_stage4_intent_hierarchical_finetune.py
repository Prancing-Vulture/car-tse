#!/usr/bin/env python
"""Fine-tune the flat intent head with an auxiliary factorized loss.

The exported model is still the normal 31-way IntentHead.  During training,
small auxiliary heads predict the intent's action/object/location factors;
their loss is used only as a representation regularizer and is not needed at
runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModel

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.datasets import IntentDataset
from car_tse.train.utils import append_jsonl, get_device, load_checkpoint, save_checkpoint, write_metrics
from train_stage4_intent_wav2vec import IntentHead, normalize_waveform
from train_stage4_intent_wav2vec_finetune import loader_args, set_trainable_layers


def factors(intent: str) -> tuple[str, str, str]:
    parts = [part.strip() for part in str(intent).split("|")]
    return tuple((parts + ["__none__", "__none__"])[:3])  # type: ignore[return-value]


class FactorizedDataset(Dataset):
    def __init__(self, base: IntentDataset, maps: tuple[dict[str, int], dict[str, int], dict[str, int]]):
        self.base = base
        self.rows = base.rows
        self.maps = maps

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base[index]
        action, object_name, location = factors(self.rows[index].get("intent", ""))
        item["factor_action"] = self.maps[0][action]
        item["factor_object"] = self.maps[1][object_name]
        item["factor_location"] = self.maps[2][location]
        return item


def metric_from_confusion(confusion: torch.Tensor, total: int, top5: int) -> dict[str, float]:
    tp = confusion.diag().float()
    support = confusion.sum(dim=1).float()
    predicted = confusion.sum(dim=0).float()
    precision = tp / predicted.clamp_min(1.0)
    recall = tp / support.clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1.0e-8)
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
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/stage4_intent_runtime_jitter_clean_overlap_2026-10-17.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.yaml")
    parser.add_argument("--encoder", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--run-name", default="stage4_intent_hierarchical_ft1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--encoder-lr", type=float, default=5.0e-6)
    parser.add_argument("--head-lr", type=float, default=1.0e-4)
    parser.add_argument("--unfreeze-layers", type=int, default=4)
    parser.add_argument("--aux-weight", type=float, default=0.15)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    device = get_device()
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    manifest_root = Path(paths["manifest_root"])
    label_map_path = manifest_root / "meta" / cfg["label_map"]
    num_classes = len(json.loads(label_map_path.read_text(encoding="utf-8")))
    cache_dir = Path(cfg["waveform_cache_dir"])
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    common_kwargs = {
        "cabin_noise": False,
        "cabin_noise_cache_size": 0,
        "audio_cache_size": 0,
        "waveform_cache_is_noisy": True,
    }
    train_base = IntentDataset(
        manifest_root / "train" / cfg["manifest_train"], cfg["sample_rate"], cfg["segment_seconds"],
        training=True, waveform_cache_path=cache_dir / cfg["waveform_cache_train"], **common_kwargs,
    )
    val_base = IntentDataset(
        manifest_root / "val" / cfg["manifest_eval"], cfg["sample_rate"], cfg["segment_seconds"],
        training=False, waveform_cache_path=cache_dir / cfg["waveform_cache_eval"], **common_kwargs,
    )
    all_intents = [str(row.get("intent", "")) for row in train_base.rows + val_base.rows]
    maps: tuple[dict[str, int], dict[str, int], dict[str, int]] = tuple(
        {name: index for index, name in enumerate(sorted({parts[i] for parts in map(factors, all_intents)}))}
        for i in range(3)
    )  # type: ignore[assignment]
    train_ds = FactorizedDataset(train_base, maps)
    val_ds = FactorizedDataset(val_base, maps)

    encoder = AutoModel.from_pretrained(args.encoder).to(device)
    encoder_checkpoint = args.encoder_checkpoint if args.encoder_checkpoint.is_absolute() else REPO_ROOT / args.encoder_checkpoint
    state = torch.load(encoder_checkpoint, map_location="cpu", weights_only=False)
    encoder.load_state_dict(state["model_state_dict"], strict=True)
    set_trainable_layers(encoder, args.unfreeze_layers)
    encoder.config.apply_spec_augment = False
    encoder.config.mask_time_prob = 0.0
    encoder.config.mask_feature_prob = 0.0
    encoder.config.use_cache = False
    hidden = int(encoder.config.hidden_size)
    head = IntentHead(hidden, 512, num_classes, 0.2).to(device)
    loaded = load_checkpoint(args.head_checkpoint, head)
    auxiliary = nn.ModuleList([nn.Linear(hidden, len(mapping)) for mapping in maps]).to(device)

    labels = torch.tensor([int(row["label"]) for row in train_base.rows], dtype=torch.long)
    counts = torch.bincount(labels, minlength=num_classes).float()
    sampler = WeightedRandomSampler(1.0 / counts[labels].clamp_min(1.0), len(labels), replacement=True)
    train_loader = DataLoader(train_ds, sampler=sampler, **loader_args(device, args.batch_size))
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args(device, args.batch_size))
    encoder_params = [p for p in encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": encoder_params, "lr": args.encoder_lr}, {"params": head.parameters(), "lr": args.head_lr},
         {"params": auxiliary.parameters(), "lr": args.head_lr}], weight_decay=1.0e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    out_dir = Path(paths["checkpoint_root"]) / args.run_name
    log_dir = Path(paths["log_root"]) / args.run_name
    run_id = uuid.uuid4().hex
    history_path = log_dir / f"epoch_metrics_{run_id}.jsonl"
    write_metrics(log_dir / f"run_metadata_{run_id}.json", {
        "run_id": run_id, "run_name": args.run_name, "device": str(device), "config": str(args.config),
        "init_head_checkpoint": str(args.head_checkpoint), "encoder_checkpoint": str(args.encoder_checkpoint),
        "unfreeze_layers": args.unfreeze_layers, "aux_weight": args.aux_weight,
        "factor_vocab_sizes": [len(mapping) for mapping in maps],
    })
    best_f1 = float("-inf")
    bad_epochs = 0
    print(f"device={device} train={len(train_ds)} val={len(val_ds)} aux_weight={args.aux_weight} unfreeze_layers={args.unfreeze_layers}", flush=True)
    for epoch in range(1, args.max_epochs + 1):
        encoder.train(); head.train(); auxiliary.train()
        train_loss = train_correct = train_total = 0
        for batch in train_loader:
            wav = normalize_waveform(batch["wav"].to(device, non_blocking=True))
            target = batch["label"].to(device, non_blocking=True)
            factor_targets = [batch[f"factor_{name}"].to(device, non_blocking=True) for name in ("action", "object", "location")]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                features = encoder(wav).last_hidden_state.mean(dim=1)
                logits = head(features)
                loss = F.cross_entropy(logits.float(), target, label_smoothing=0.03)
                aux_loss = sum(F.cross_entropy(module(features).float(), factor_target) for module, factor_target in zip(auxiliary, factor_targets))
                loss = loss + float(args.aux_weight) * aux_loss
            if use_amp:
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(encoder_params + list(head.parameters()) + list(auxiliary.parameters()), 5.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); torch.nn.utils.clip_grad_norm_(encoder_params + list(head.parameters()) + list(auxiliary.parameters()), 5.0); optimizer.step()
            train_loss += float(loss.detach()) * target.numel()
            train_correct += int((logits.argmax(-1) == target).sum()); train_total += target.numel()
        scheduler.step(); encoder.eval(); head.eval(); auxiliary.eval()
        confusion = torch.zeros((num_classes, num_classes), dtype=torch.long); total = top5 = 0
        with torch.no_grad():
            for batch in val_loader:
                wav = normalize_waveform(batch["wav"].to(device, non_blocking=True)); target = batch["label"].to(device, non_blocking=True)
                logits = head(encoder(wav).last_hidden_state.mean(dim=1)); pred = logits.argmax(-1)
                confusion.index_put_((target.cpu(), pred.cpu()), torch.ones_like(target.cpu()), accumulate=True)
                total += target.numel(); top5 += int((logits.topk(min(5, num_classes), dim=-1).indices == target[:, None]).any(dim=1).sum())
        val = metric_from_confusion(confusion, total, top5)
        metrics = {"epoch": epoch, "run_id": run_id, "device": str(device), "train_loss": train_loss / max(train_total, 1), "train_accuracy": train_correct / max(train_total, 1), "val": val, "lr": [group["lr"] for group in optimizer.param_groups]}
        write_metrics(log_dir / "metrics_last.json", metrics); append_jsonl(history_path, metrics)
        print(f"epoch {epoch} train_loss={metrics['train_loss']:.4f} train_acc={metrics['train_accuracy']:.3f} val_acc={val['accuracy']:.3f} val_top5={val['top5_accuracy']:.3f} val_macro_f1={val['macro_f1']:.3f}", flush=True)
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]; bad_epochs = 0; metrics["best_macro_f1"] = best_f1
            save_checkpoint(out_dir / "best.pt", head, epoch, metrics)
            torch.save({"model_state_dict": encoder.state_dict(), "epoch": epoch}, out_dir / "encoder_best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs", flush=True); break
    write_metrics(log_dir / "train_last.json", {"best_macro_f1": best_f1, "epochs_completed": epoch, "run_id": run_id, "history": str(history_path)})
    print(f"Done. Best head checkpoint: {out_dir / 'best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
