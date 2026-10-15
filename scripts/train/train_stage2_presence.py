#!/usr/bin/env python
"""Train target-presence head."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from torch.utils.data import DataLoader, default_collate

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.datasets import PresenceDataset
from car_tse.losses.presence import presence_loss
from car_tse.models.presence_head import PresenceHead
from car_tse.models.stage1_system import Stage1System
from car_tse.train.utils import get_device, load_checkpoint, save_checkpoint


def collate_presence(batch: list[dict], expected_mics: int) -> dict:
    normalized = []
    for item in batch:
        row = dict(item)
        multichannel = row.get("multichannel")
        if multichannel is None:
            multichannel = row["mixture"].unsqueeze(0)
        elif multichannel.dim() == 1:
            multichannel = multichannel.unsqueeze(0)
        if multichannel.shape[0] < expected_mics:
            repeats = (expected_mics + multichannel.shape[0] - 1) // multichannel.shape[0]
            multichannel = multichannel.repeat(repeats, 1)[:expected_mics]
        elif multichannel.shape[0] > expected_mics:
            multichannel = multichannel[:expected_mics]
        row["multichannel"] = multichannel
        normalized.append(row)
    return default_collate(normalized)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "stage2_presence.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Optional presence head checkpoint to fine-tune")
    parser.add_argument("--run-name", default=None, help="Optional isolated output run name")
    args = parser.parse_args()
    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    device = get_device()
    stage1_cfg_path = Path(cfg.get("stage1_config", REPO_ROOT / "configs" / "stage1_spex_mc.yaml"))
    if not stage1_cfg_path.is_absolute():
        stage1_cfg_path = REPO_ROOT / stage1_cfg_path
    stage1_cfg = load_stage_config(stage1_cfg_path, args.paths)
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
    if args.init_checkpoint is not None:
        init_checkpoint = args.init_checkpoint if args.init_checkpoint.is_absolute() else REPO_ROOT / args.init_checkpoint
        load_checkpoint(init_checkpoint, head)
        print(f"initialized_presence={init_checkpoint}", flush=True)
    manifest = Path(paths["manifest_root"]) / "train" / cfg["manifest"]
    ds = PresenceDataset(manifest, cfg["sample_rate"], cfg["segment_seconds"])
    expected_mics = int(getattr(stage1.frontend, "num_mics", cfg.get("num_mics", 4)))
    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        collate_fn=lambda batch: collate_presence(batch, expected_mics),
    )
    opt = torch.optim.AdamW(head.parameters(), lr=cfg["train"]["lr"])
    run_name = args.run_name or cfg["name"]
    out_dir = paths["checkpoint_root"] / run_name
    for epoch in range(1, cfg["train"]["max_epochs"] + 1):
        head.train()
        loss_sum = 0.0
        for batch in loader:
            mix = batch["mixture"].to(device)
            enroll = batch["enrollment"].to(device)
            labels = batch["target_present"].to(device=device, dtype=torch.float32)
            multichannel = batch.get("multichannel")
            playback_ref = batch.get("playback_ref")
            if multichannel is not None:
                multichannel = multichannel.to(device)
                expected_mics = int(getattr(stage1.frontend, "num_mics", multichannel.shape[1]))
                if multichannel.shape[1] < expected_mics:
                    repeats = (expected_mics + multichannel.shape[1] - 1) // multichannel.shape[1]
                    multichannel = multichannel.repeat(1, repeats, 1)[:, :expected_mics]
                elif multichannel.shape[1] > expected_mics:
                    multichannel = multichannel[:, :expected_mics]
            if playback_ref is not None:
                playback_ref = playback_ref.to(device)
            with torch.no_grad():
                s1, _, _, _ = stage1(mix, enroll, multichannel=multichannel, playback_ref=playback_ref)
            logits = head(s1)
            loss, _ = presence_loss(logits, labels, s1, cfg["train"]["absent_output_penalty"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), cfg["train"]["grad_clip"])
            opt.step()
            loss_sum += float(loss.detach())
        print(f"epoch {epoch} loss={loss_sum / len(loader):.4f}")
    save_checkpoint(out_dir / "best.pt", head, cfg["train"]["max_epochs"], {})
    bundle = {"stage1": stage1.state_dict(), "presence": head.state_dict()}
    torch.save(bundle, out_dir / "bundle.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
