#!/usr/bin/env python
"""Train the target-presence gate on the current CAVEMOVE Stage 1 output."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from torch.utils.data import DataLoader, Subset

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.datasets import PresenceDataset
from car_tse.losses.presence import presence_loss
from car_tse.models.presence_head import PresenceHead
from car_tse.models.stage1_system import Stage1System
from car_tse.train.utils import get_device, load_checkpoint, save_checkpoint, write_metrics


def collate(batch):
    out = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            out[key] = torch.stack([item[key] for item in batch])
        elif isinstance(batch[0][key], (int, float)):
            out[key] = torch.tensor([item[key] for item in batch])
        else:
            out[key] = [item[key] for item in batch]
    return out


def split_indices(rows: list[dict], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    train: list[int] = []
    val: list[int] = []
    for label in (0, 1):
        indices = [i for i, row in enumerate(rows) if int(row.get("target_present", 1)) == label]
        rng.shuffle(indices)
        cut = max(1, int(round(len(indices) * val_fraction))) if indices else 0
        val.extend(indices[:cut])
        train.extend(indices[cut:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


@torch.no_grad()
def evaluate(head, stage1, loader, device, use_amp, threshold=0.5):
    head.eval()
    total = correct = tp = fp = fn = 0
    for batch in loader:
        mix = batch["mixture"].to(device, non_blocking=True)
        enroll = batch["enrollment"].to(device, non_blocking=True)
        labels = batch["target_present"].to(device, non_blocking=True).float()
        mc = batch.get("multichannel")
        if isinstance(mc, torch.Tensor):
            mc = mc.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            s1, _, _, _ = stage1(mix, enroll, multichannel=mc)
        logits = head(s1.float())
        pred = torch.sigmoid(logits) >= threshold
        truth = labels >= 0.5
        correct += int((pred == truth).sum())
        total += labels.numel()
        tp += int((pred & truth).sum())
        fp += int((pred & ~truth).sum())
        fn += int((~pred & truth).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "accuracy": correct / max(total, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "examples": total,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "stage2_presence_cavemove.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    args = parser.parse_args()
    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    device = get_device()
    stage1_cfg_path = Path(cfg["stage1_config"])
    if not stage1_cfg_path.is_absolute():
        stage1_cfg_path = REPO_ROOT / stage1_cfg_path
    stage1_cfg = load_stage_config(stage1_cfg_path, args.paths)
    stage1 = Stage1System(
        spex_cfg=stage1_cfg["spex"],
        sample_rate=stage1_cfg["sample_rate"],
        use_cabin_frontend=stage1_cfg.get("use_cabin_frontend", True),
        num_mics=stage1_cfg.get("num_mics", 4),
        frontend_cfg=stage1_cfg.get("frontend"),
        pretrained_dir=str(stage1_cfg["pretrained_checkpoint"]),
        backend=stage1_cfg.get("spex_backend", "transfer_spexplus_16k"),
    ).to(device)
    load_checkpoint(paths["checkpoint_root"] / cfg["spex_checkpoint"], stage1)
    stage1.eval()
    for parameter in stage1.parameters():
        parameter.requires_grad = False

    manifest = Path(paths["manifest_root"]) / "train" / cfg["manifest"]
    dataset_kwargs = {
        "cabin_noise": bool(cfg.get("cabin_noise", True)),
        "num_mics": int(cfg.get("num_mics", 4)),
        "cabin_noise_channels": cfg.get("cabin_noise_channels", [0, 2, 4, 6]),
        "cabin_noise_snr_db": cfg.get("cabin_noise_snr_db"),
        "cabin_noise_cache_size": int(cfg.get("cabin_noise_cache_size", 32)),
    }
    dataset = PresenceDataset(
        manifest,
        cfg["sample_rate"],
        cfg["segment_seconds"],
        training=True,
        **dataset_kwargs,
    )
    train_indices, val_indices = split_indices(
        dataset.rows,
        float(cfg["train"].get("val_fraction", 0.2)),
        int(cfg["train"].get("seed", 2026)),
    )
    train_ds = Subset(dataset, train_indices)
    val_base = PresenceDataset(
        manifest,
        cfg["sample_rate"],
        cfg["segment_seconds"],
        training=False,
        **dataset_kwargs,
    )
    val_ds = Subset(val_base, val_indices)
    workers = int(cfg["train"].get("num_workers", 0))
    loader_args = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
        "prefetch_factor": 2 if workers > 0 else None,
    }
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        collate_fn=collate,
        **loader_args,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"].get("val_batch_size", cfg["train"]["batch_size"]),
        shuffle=False,
        collate_fn=collate,
        **loader_args,
    )
    head = PresenceHead(cfg["model"]["hidden"]).to(device)
    init_checkpoint = cfg.get("init_checkpoint")
    if init_checkpoint:
        init_path = Path(init_checkpoint)
        if not init_path.is_absolute():
            init_path = paths["checkpoint_root"] / init_path
        load_checkpoint(init_path, head)
        print(f"Initialized presence head from {init_path}", flush=True)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg["train"]["lr"])
    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    out_dir = paths["checkpoint_root"] / cfg["name"]
    best_f1 = float("-inf")
    for epoch in range(1, cfg["train"]["max_epochs"] + 1):
        head.train()
        stats_sum = {"bce": 0.0, "absent_energy": 0.0}
        batches = 0
        for batch in train_loader:
            mix = batch["mixture"].to(device, non_blocking=True)
            enroll = batch["enrollment"].to(device, non_blocking=True)
            labels = batch["target_present"].to(device, non_blocking=True).float()
            mc = batch.get("multichannel")
            if isinstance(mc, torch.Tensor):
                mc = mc.to(device, non_blocking=True)
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                s1, _, _, _ = stage1(mix, enroll, multichannel=mc)
            logits = head(s1.float())
            loss, stats = presence_loss(
                logits,
                labels,
                s1.float(),
                cfg["train"].get("absent_output_penalty", 1.0),
            )
            opt.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(head.parameters(), cfg["train"]["grad_clip"])
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), cfg["train"]["grad_clip"])
                opt.step()
            for key in stats_sum:
                stats_sum[key] += stats[key]
            batches += 1
        val = evaluate(head, stage1, val_loader, device, use_amp)
        metrics = {
            "epoch": epoch,
            "train_bce": stats_sum["bce"] / max(batches, 1),
            "train_absent_energy": stats_sum["absent_energy"] / max(batches, 1),
            "val": val,
            "train_examples": len(train_ds),
            "val_examples": len(val_ds),
            "best_f1": best_f1,
        }
        write_metrics(paths["log_root"] / cfg["name"] / "metrics_last.json", metrics)
        print(f"epoch {epoch} val_acc={val['accuracy']:.3f} val_f1={val['f1']:.3f} val_recall={val['recall']:.3f}", flush=True)
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            save_checkpoint(out_dir / "best.pt", head, epoch, metrics)
    write_metrics(paths["log_root"] / cfg["name"] / "train_last.json", {"best_f1": best_f1})
    print(f"Done. Best checkpoint: {out_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
