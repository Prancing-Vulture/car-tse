#!/usr/bin/env python
"""Train Stage 1 SpEx+ (single- or multi-channel)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from torch.utils.data import DataLoader

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.datasets import TripletExtractionDataset
from car_tse.losses.stage1 import stage1_loss
from car_tse.models.stage1_system import Stage1System
from car_tse.train.utils import get_device, save_checkpoint, schedule_for_epoch, write_metrics


def collate(batch):
    out = {}
    for k in batch[0]:
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch])
        else:
            out[k] = [b[k] for b in batch]
    return out


def apply_schedule(model: Stage1System, block: dict) -> None:
    model.set_encoder_frozen(block.get("freeze_encoder", False))
    if block.get("freeze_spex"):
        model.set_spex_frozen(True)
    elif "freeze_spex" in block:
        model.set_spex_frozen(False)


def build_optimizer(model: Stage1System, block: dict) -> torch.optim.Optimizer:
    groups = []
    if block.get("lr_encoder", 0) > 0:
        enc_params = []
        for m in model.spex.encoder_params():
            enc_params += list(m.parameters())
        groups.append({"params": [p for p in enc_params if p.requires_grad], "lr": block["lr_encoder"]})
    rest = [p for n, p in model.named_parameters() if p.requires_grad and not any(id(p) == id(ep) for ep in (groups[0]["params"] if groups else []))]
    if block.get("lr_rest", 1e-4) > 0:
        groups.append({"params": rest, "lr": block.get("lr_rest", 1e-4)})
    if model.frontend is not None and block.get("lr_frontend"):
        groups.append({"params": model.frontend.parameters(), "lr": block["lr_frontend"]})
    if not groups:
        groups = [{"params": model.parameters(), "lr": 1e-4}]
    return torch.optim.AdamW(groups)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "stage1_spex_sc.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    device = get_device()
    manifest = Path(paths["manifest_root"]) / args.split / cfg["train"]["manifest"]
    ds = TripletExtractionDataset(manifest, cfg["sample_rate"], cfg["segment_seconds"], training=True)
    if len(ds) == 0:
        print(f"No data in {manifest}; run data prep first.")
        return 1
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate)
    pretrained = paths["spex_pretrained_dir"]
    init_ckpt = cfg["train"].get("init_checkpoint")
    if init_ckpt:
        pretrained = paths["checkpoint_root"] / init_ckpt
    model = Stage1System(
        spex_cfg=cfg["spex"],
        sample_rate=cfg["sample_rate"],
        use_cabin_frontend=cfg.get("use_cabin_frontend", False),
        num_mics=cfg.get("num_mics", 4),
        frontend_cfg=cfg.get("frontend"),
        pretrained_dir=str(pretrained),
    ).to(device)
    out_dir = paths["checkpoint_root"] / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    best_si = float("-inf")
    for epoch in range(1, cfg["train"]["max_epochs"] + 1):
        block = schedule_for_epoch(cfg["train"]["schedule"], epoch)
        apply_schedule(model, block)
        opt = build_optimizer(model, block)
        model.train()
        stats_acc = {}
        n = 0
        for batch in loader:
            mix = batch["mixture"].to(device)
            enroll = batch["enrollment"].to(device)
            target = batch["target"].to(device)
            mc = batch.get("multichannel")
            pb = batch.get("playback_ref")
            if isinstance(mc, list):
                mc = None
            if isinstance(pb, list):
                pb = None
            if mc is not None:
                mc = mc.to(device)
            if pb is not None:
                pb = pb.to(device)
            s1, s2, s3, logits = model(mix, enroll, multichannel=mc, playback_ref=pb)
            labels = None
            if "speaker_label" in batch:
                labels = torch.tensor(batch["speaker_label"], device=device, dtype=torch.long)
            loss, st = stage1_loss(s1, s2, s3, target, logits, labels, **cfg["loss"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step()
            for k, v in st.items():
                stats_acc[k] = stats_acc.get(k, 0) + v
            n += 1
        mean_si = stats_acc.get("si_sdr_s1", 0) / max(n, 1)
        print(f"epoch {epoch} si_sdr_s1={mean_si:.2f}")
        if mean_si > best_si:
            best_si = mean_si
            save_checkpoint(out_dir / "best.pt", model, epoch, {"si_sdr_s1": mean_si})
    write_metrics(paths["log_root"] / cfg["name"] / "train_last.json", {"best_si_sdr_s1": best_si})
    print(f"Done. Best checkpoint: {out_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
