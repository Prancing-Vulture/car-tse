"""Shared helpers for data preparation scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from car_tse.config import REPO_ROOT, resolve_paths


def prep_argparser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    return p


def manifest_dir(paths_cfg: dict, split: str) -> Path:
    root = Path(paths_cfg["manifest_root"])
    d = root / split
    d.mkdir(parents=True, exist_ok=True)
    return d


def corpus_path(paths_cfg: dict, name: str) -> Path:
    return Path(paths_cfg["corpus_paths"][name])
