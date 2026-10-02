#!/usr/bin/env python
"""Baseline: single-channel SpEx+ without cabin frontend."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import REPO_ROOT


def main() -> int:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval" / "eval_stage1_spex.py"),
        "--config",
        str(REPO_ROOT / "configs" / "stage1_spex_sc.yaml"),
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
