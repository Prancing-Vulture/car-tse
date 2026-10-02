#!/usr/bin/env python
"""Train target-presence head."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from torch.utils.data import DataLoader

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.datasets import PresenceDataset
from car_tse.losses.presence import presence_loss
from car_tse.models.presence_head import PresenceHead
from car_tse.models.stage1_system import Stage1System
from car_tse.train.utils import get_device, load_checkpoint, save_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "stage2_presence.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    args = parser.parse_args()
    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    device = get_device()
    stage1_cfg = load_stage_config(REPO_ROOT / "configs" / "stage1_spex_mc.yaml", args.paths)
    stage1 = Stage1System(
        spex_cfg=stage1_cfg["spex"],
        sample_rate=stage1_cfg["sample_rate"],
        use_cabin_frontend=stage1_cfg.get("use_cabin_frontend", False),
        num_mics=stage1_cfg.get("num_mics", 4),
        frontend_cfg=stage1_cfg.get("frontend"),
        pretrained_dir=str(paths["spex_pretrained_dir"]),
    ).to(device)
    ckpt = paths["checkpoint_root"] / cfg["spex_checkpoint"]
    if ckpt.exists():
        load_checkpoint(ckpt, stage1)
    for p in stage1.parameters():
        p.requires_grad = False
    head = PresenceHead(cfg["model"]["hidden"]).to(device)
    manifest = Path(paths["manifest_root"]) / "train" / cfg["manifest"]
    ds = PresenceDataset(manifest, cfg["sample_rate"], cfg["segment_seconds"])
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg["train"]["lr"])
    out_dir = paths["checkpoint_root"] / cfg["name"]
    for epoch in range(1, cfg["train"]["max_epochs"] + 1):
        head.train()
        loss_sum = 0.0
        for batch in loader:
            mix = batch["mixture"].to(device)
            enroll = batch["enrollment"].to(device)
            labels = torch.tensor(batch["target_present"], device=device)
            with torch.no_grad():
                s1, _, _, _ = stage1(mix, enroll)
            logits = head(s1)
            loss, _ = presence_loss(logits, labels, s1, cfg["train"]["absent_output_penalty"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), cfg["train"]["grad_clip"])
            opt.step()
            loss_sum += float(loss)
        print(f"epoch {epoch} loss={loss_sum / len(loader):.4f}")
    save_checkpoint(out_dir / "best.pt", head, cfg["train"]["max_epochs"], {})
    bundle = {"stage1": stage1.state_dict(), "presence": head.state_dict()}
    torch.save(bundle, out_dir / "bundle.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
