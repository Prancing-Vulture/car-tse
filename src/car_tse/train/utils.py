"""Shared training utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(path: Path, model: torch.nn.Module, epoch: int, extra: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, model: torch.nn.Module) -> dict[str, Any]:
    cpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(cpt["model_state_dict"], strict=False)
    return cpt


def write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    # Preserve every epoch: metrics_last.json is intentionally a convenient
    # snapshot, but replacing it each epoch otherwise destroys learning curves.
    if path.name == "metrics_last.json" and "epoch" in metrics:
        history_path = path.with_name("metrics_history.jsonl")
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, separators=(",", ":")) + "\n")


def schedule_for_epoch(schedule: list[dict], epoch: int) -> dict[str, Any]:
    for block in schedule:
        lo, hi = block["epochs"]
        if lo <= epoch <= hi:
            return block
    return schedule[-1] if schedule else {}
