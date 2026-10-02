#!/usr/bin/env python
"""Train wake-word / KWS on accepted extracted speech (manifest wavs)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.datasets import KWSDataset
from car_tse.models.kws_model import KWSModel
from car_tse.train.utils import get_device, save_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "stage3_kws.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    args = parser.parse_args()
    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    device = get_device()
    manifest = Path(paths["manifest_root"]) / "train" / cfg["manifest_train"]
    ds = KWSDataset(manifest, cfg["sample_rate"], cfg["segment_seconds"])
    if len(ds) == 0:
        print(f"No KWS data at {manifest}")
        return 1
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    model = KWSModel(
        cfg["sample_rate"],
        cfg["model"]["n_mels"],
        cfg["model"]["channels"],
        cfg["model"]["num_classes"],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    out_dir = paths["checkpoint_root"] / cfg["name"]
    for epoch in range(1, cfg["train"]["max_epochs"] + 1):
        model.train()
        correct = total = 0
        for batch in loader:
            wav = batch["wav"].to(device)
            labels = batch["label"].to(device)
            logits = model(wav)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step()
            correct += (logits.argmax(-1) == labels).sum().item()
            total += labels.numel()
        print(f"epoch {epoch} acc={correct / max(total, 1):.3f}")
    save_checkpoint(out_dir / "best.pt", model, cfg["train"]["max_epochs"], {})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
