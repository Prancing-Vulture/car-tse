#!/usr/bin/env python
"""Combine the main Tier-1 run with the matched local-pretraining control."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_DIR = REPO_ROOT / "reports/evaluation/tier1_ablation_2026-10-19/tse"
CONTROL_DIR = REPO_ROOT / "reports/evaluation/tier1_ablation_2026-10-19/tse_local_pretrained_control"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    main_path = MAIN_DIR / "per_example_metrics.jsonl"
    control_path = CONTROL_DIR / "per_example_metrics.jsonl"
    main_rows = read_jsonl(main_path)
    control_rows = read_jsonl(control_path)
    if len(main_rows) != len(control_rows):
        raise ValueError(f"row-count mismatch: main={len(main_rows)}, control={len(control_rows)}")

    for index, (main, control) in enumerate(zip(main_rows, control_rows)):
        if str(main["row_id"]) != str(control["row_id"]):
            raise ValueError(f"row-id mismatch at index {index}: {main['row_id']} vs {control['row_id']}")
        main["variants"]["spx_pretrained_local_init_no_cabin_ft"] = control["variants"][
            "spx_pretrained_local_init_no_cabin_ft"
        ]

    complete_jsonl = MAIN_DIR / "per_example_metrics_complete.jsonl"
    with complete_jsonl.open("w", encoding="utf-8") as handle:
        for row in main_rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")

    keys = ("variant", "cabin_noise_snr_db", "gender_pair", "passenger_sir_bin")
    metrics = ("si_sdr_db", "si_sdri_vs_delay_sum_db", "pesq_wb", "stoi")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in main_rows:
        for name, values in row["variants"].items():
            key = (
                name,
                float(row["cabin_noise_snr_db"]),
                str(row["gender_pair"]),
                str(row["passenger_sir_bin"]),
            )
            groups[key].append(values)
    complete_csv = MAIN_DIR / "summary_by_snr_gender_sir_complete.csv"
    columns = list(keys) + ["n"] + [f"{metric}_{stat}" for metric in metrics for stat in ("mean", "std")]
    with complete_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for key, records in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
            summary: dict[str, Any] = dict(zip(keys, key))
            summary["n"] = len(records)
            for metric in metrics:
                values = [float(record[metric]) for record in records if record.get(metric) is not None]
                summary[f"{metric}_mean"] = float(np.mean(values)) if values else ""
                summary[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else ""
            writer.writerow(summary)

    manifest = {
        "experiment": "complete Tier-1 TSE ablation results with local pretrained SpEx+ control",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "main_per_example_sha256": sha256_file(main_path),
        "local_control_per_example_sha256": sha256_file(control_path),
        "row_count": len(main_rows),
        "row_alignment": "all row IDs match and the delay-sum reference was checked against the original run before computing paired controls",
        "outputs": {
            complete_jsonl.name: sha256_file(complete_jsonl),
            complete_csv.name: sha256_file(complete_csv),
        },
    }
    manifest_path = MAIN_DIR / "complete_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(main_rows), "per_example": str(complete_jsonl), "summary": str(complete_csv), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
