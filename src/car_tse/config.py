"""Load merged YAML configs and paths."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path | str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_paths(paths_file: Path | str | None = None) -> dict[str, Any]:
    paths_file = Path(paths_file or REPO_ROOT / "configs" / "paths.yaml")
    if not paths_file.exists():
        paths_file = REPO_ROOT / "configs" / "paths.example.yaml"
    cfg = load_yaml(paths_file)
    root = Path(cfg.get("data_root", "."))
    corpora = cfg.get("corpora", {})
    resolved = copy.deepcopy(cfg)
    resolved["corpus_paths"] = {
        k: (root / v) if not Path(v).is_absolute() else Path(v)
        for k, v in corpora.items()
    }
    resolved["manifest_root"] = Path(cfg.get("manifest_root", REPO_ROOT / "manifests"))
    resolved["checkpoint_root"] = Path(cfg.get("checkpoint_root", REPO_ROOT / "checkpoints"))
    resolved["log_root"] = Path(cfg.get("log_root", REPO_ROOT / "logs"))
    resolved["spex_pretrained_dir"] = Path(
        cfg.get("spex_pretrained_dir", REPO_ROOT / "checkpoints" / "spex_pretrained")
    )
    return resolved


def load_stage_config(stage_yaml: Path | str, paths_file: Path | str | None = None) -> dict[str, Any]:
    stage = load_yaml(stage_yaml)
    paths = resolve_paths(paths_file)
    return {"paths": paths, **stage}
