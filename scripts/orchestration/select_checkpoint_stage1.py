#!/usr/bin/env python
"""Select Stage1 checkpoint by val SI-SDR with absent-output energy constraint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import REPO_ROOT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "logs" / "stage1_spex_mc")
    parser.add_argument("--max-absent-energy", type=float, default=0.05)
    args = parser.parse_args()
    metrics_files = sorted(args.log_dir.glob("metrics_*.json"))
    if not metrics_files:
        print("No metrics found; using best.pt in checkpoint dir.")
        return 0
    best = None
    best_score = float("-inf")
    for mf in metrics_files:
        data = json.loads(mf.read_text(encoding="utf-8"))
        si = data.get("si_sdr", float("-inf"))
        absent = data.get("absent_energy", 0.0)
        if absent <= args.max_absent_energy and si > best_score:
            best_score = si
            best = mf
    print(f"Selected metrics file: {best} (si_sdr={best_score})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
