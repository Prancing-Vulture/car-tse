#!/usr/bin/env python
"""Evaluate the exact local SpEx+ initialization without cabin fine-tuning."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from pesq import pesq
from pystoi import stoi
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from car_tse.data.datasets import TripletExtractionDataset
from car_tse.metrics.si_sdr import si_sdr_per_example
from car_tse.models.stage1_system import Stage1System


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_metric(function: Any, target: np.ndarray, estimate: np.ndarray) -> float | None:
    try:
        value = float(function(target, estimate))
        return value if np.isfinite(value) else None
    except Exception:
        return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_summary(rows: list[dict[str, Any]], output: Path) -> None:
    metric_names = ("si_sdr_db", "si_sdri_vs_delay_sum_db", "pesq_wb", "stoi")
    groups: dict[tuple[str, float, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for variant, metrics in row["variants"].items():
            groups[(
                variant,
                float(row["cabin_noise_snr_db"]),
                str(row["gender_pair"]),
                str(row["passenger_sir_bin"]),
            )].append(metrics)
    columns = [
        "variant", "cabin_noise_snr_db", "gender_pair", "passenger_sir_bin", "n",
        *[f"{metric}_{stat}" for metric in metric_names for stat in ("mean", "std")],
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for key, values in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
            summary: dict[str, Any] = dict(zip(columns[:4], key))
            summary["n"] = len(values)
            for metric in metric_names:
                finite = [float(value[metric]) for value in values if value.get(metric) is not None]
                summary[f"{metric}_mean"] = float(np.mean(finite)) if finite else ""
                summary[f"{metric}_std"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else ""
            writer.writerow(summary)


def write_paired_differences(rows: list[dict[str, Any]], output: Path) -> None:
    comparisons = {
        "selected_spx_mean_mono_minus_local_init": (
            "spx_ft_mean_delay_sum", "spx_pretrained_local_init_no_cabin_ft"
        ),
        "selected_learned_frontend_4ch_minus_local_init": (
            "learned_frontend_4ch_test_time_ablation", "spx_pretrained_local_init_no_cabin_ft"
        ),
    }
    grouped: dict[tuple[str, float, str, str], list[float]] = defaultdict(list)
    for row in rows:
        for comparison, (left, right) in comparisons.items():
            left_metric = row["variants"].get(left, {}).get("si_sdri_vs_delay_sum_db")
            right_metric = row["variants"].get(right, {}).get("si_sdri_vs_delay_sum_db")
            if left_metric is None or right_metric is None:
                continue
            key = (
                comparison,
                float(row["cabin_noise_snr_db"]),
                str(row["gender_pair"]),
                str(row["passenger_sir_bin"]),
            )
            grouped[key].append(float(left_metric) - float(right_metric))
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["comparison", "cabin_noise_snr_db", "gender_pair", "passenger_sir_bin", "n", "si_sdri_difference_mean_db", "si_sdri_difference_std_db"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for key, values in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
            writer.writerow({
                **dict(zip(columns[:4], key)),
                "n": len(values),
                "si_sdri_difference_mean_db": float(np.mean(values)),
                "si_sdri_difference_std_db": float(np.std(values, ddof=1)) if len(values) > 1 else "",
            })


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "manifests/test/stage1_cavemove_cabin.jsonl")
    parser.add_argument("--stage-config", type=Path, default=REPO_ROOT / "configs/stage1_spex_fsc_passenger_cochannel_ft1_2026-10-16.yaml")
    parser.add_argument("--base-spex", type=Path, default=REPO_ROOT / "reports/evaluation/tier1_ablation_2026-10-19/tse/base_spex_safe_state_dict.pt")
    parser.add_argument("--main-results", type=Path, default=REPO_ROOT / "reports/evaluation/tier1_ablation_2026-10-19/tse/per_example_metrics.jsonl")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "reports/evaluation/tier1_ablation_2026-10-19/tse_local_pretrained_control")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.stage_config.read_text(encoding="utf-8"))
    sample_rate = int(cfg["sample_rate"])
    dataset = TripletExtractionDataset(
        args.manifest,
        sample_rate=sample_rate,
        segment_seconds=float(cfg["segment_seconds"]),
        training=False,
        cabin_noise=True,
        num_mics=4,
        cabin_noise_channels=[0, 2, 4, 6],
        cabin_noise_cache_size=32,
    )
    if args.limit:
        dataset.rows = dataset.rows[: args.limit]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    main_rows = read_jsonl(args.main_results)
    if len(main_rows) != len(dataset):
        raise ValueError(
            f"Main run has {len(main_rows)} rows but dataset has {len(dataset)}; "
            "use the same fixed test manifest and row limit."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    model = Stage1System(
        spex_cfg=cfg["spex"],
        sample_rate=sample_rate,
        use_cabin_frontend=False,
        num_mics=4,
        frontend_cfg=cfg.get("frontend"),
        pretrained_dir=str(args.base_spex.resolve()),
        backend=cfg.get("spex_backend", "transfer_spexplus_16k"),
    ).to(device).eval()

    output_rows: list[dict[str, Any]] = []
    index = 0
    for batch in loader:
        enrollment = batch["enrollment"].to(device).float()
        target = batch["target"].to(device).float()
        multichannel = batch["multichannel"].to(device).float()
        delay_sum = multichannel.mean(dim=1)
        estimated = model.spex(delay_sum, enrollment)[0]
        base_scores = si_sdr_per_example(estimated, target)
        delay_scores = si_sdr_per_example(delay_sum, target)
        estimated_np = estimated.detach().float().cpu().numpy()
        target_np = target.detach().float().cpu().numpy()
        for batch_index in range(len(target_np)):
            source = main_rows[index]
            dataset_row = dataset.rows[index]
            row_id = str(dataset_row.get("id", str(index)))
            if str(source["row_id"]) != row_id:
                raise ValueError(
                    f"Dataset order mismatch at row {index}: {row_id!r} != {source['row_id']!r}"
                )
            estimate_i = estimated_np[batch_index]
            target_i = target_np[batch_index]
            base_si_sdr = float(base_scores[batch_index].cpu())
            delay_si_sdr = float(delay_scores[batch_index].cpu())
            main_delay_si_sdr = float(source["delay_sum_input_si_sdr_db"])
            if not math.isclose(delay_si_sdr, main_delay_si_sdr, rel_tol=0.0, abs_tol=1.0e-4):
                raise ValueError(
                    f"Input crop mismatch at row {index}: current delay-sum SI-SDR "
                    f"{main_delay_si_sdr:.6f} dB, supplemental {delay_si_sdr:.6f} dB"
                )
            variants = {
                name: dict(source["variants"][name])
                for name in (
                    "spx_ft_single_ch0",
                    "spx_ft_mean_delay_sum",
                    "learned_frontend_4ch_test_time_ablation",
                    "spx_pretrained_no_cabin_ft_wsj0_2mix",
                )
            }
            variants["spx_pretrained_local_init_no_cabin_ft"] = {
                "si_sdr_db": base_si_sdr,
                "si_sdri_vs_delay_sum_db": base_si_sdr - delay_si_sdr,
                "pesq_wb": safe_metric(lambda ref, est: pesq(sample_rate, ref, est, "wb"), target_i, estimate_i),
                "stoi": safe_metric(lambda ref, est: stoi(ref, est, sample_rate, extended=False), target_i, estimate_i),
            }
            output_rows.append({
                "row_id": row_id,
                "target_speaker_id": source["target_speaker_id"],
                "target_gender": source["target_gender"],
                "interferer_gender": source["interferer_gender"],
                "gender_pair": source["gender_pair"],
                "cabin_noise_snr_db": source["cabin_noise_snr_db"],
                "passenger_sir_db_estimated": source["passenger_sir_db_estimated"],
                "passenger_sir_bin": source["passenger_sir_bin"],
                "variants": variants,
            })
            index += 1
        print(f"evaluated {index}/{len(dataset)} rows", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_example_path = args.output_dir / "per_example_metrics.jsonl"
    with per_example_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    summary_path = args.output_dir / "summary_by_snr_gender_sir.csv"
    write_summary(output_rows, summary_path)
    paired_path = args.output_dir / "paired_sidr_gains.csv"
    write_paired_differences(output_rows, paired_path)

    current_payload = torch.load(
        REPO_ROOT / "checkpoints/stage1_spex_fsc_passenger_cochannel_ft1_2026-10-16/best.pt",
        map_location="cpu",
        weights_only=True,
    )["model_state_dict"]
    base_payload = torch.load(args.base_spex, map_location="cpu", weights_only=True)["model_state_dict"]
    differing_keys = [
        key for key, value in base_payload.items()
        if not torch.equal(value, current_payload[f"spex.model.{key}"])
    ]
    manifest = {
        "experiment": "matched local SpEx+ initialization versus selected cabin frontend checkpoint",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "pytorch": torch.__version__,
        "test_manifest": str(args.manifest.resolve()),
        "test_manifest_sha256": sha256_file(args.manifest),
        "main_results": str(args.main_results.resolve()),
        "main_results_sha256": sha256_file(args.main_results),
        "base_spex_checkpoint": str(args.base_spex.resolve()),
        "base_spex_checkpoint_sha256": sha256_file(args.base_spex),
        "selected_stage1_checkpoint_sha256": sha256_file(
            REPO_ROOT / "checkpoints/stage1_spex_fsc_passenger_cochannel_ft1_2026-10-16/best.pt"
        ),
        "base_spex_tensor_count": len(base_payload),
        "base_spex_tensor_values_different_from_selected_checkpoint": len(differing_keys),
        "different_state_keys": differing_keys,
        "baseline_definition": "The local transfer-SpEx+ initialization checkpoint is evaluated on the delay-sum mean of the four cabin channels with the same enrollment as the selected checkpoint; it has no learned cabin front end and no cabin fine-tuning.",
        "paired_comparisons": "The summary and paired gain file compare against the already-completed main run, which used the identical fixed test rows and delay-sum SI-SDRi reference.",
        "files": [per_example_path.name, summary_path.name, paired_path.name],
    }
    manifest_path = args.output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "rows": len(output_rows), "changed_spex_tensors": len(differing_keys)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
