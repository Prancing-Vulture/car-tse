#!/usr/bin/env python
"""Compare a bounded dual-domain Stage-1 checkpoint with the accepted baseline.

The comparison uses the same deterministic validation rows for both models and
reports SI-SDR improvement separately for AISHELL-5 and CAVEMOVE. It is an
inference-only regression check; it never changes checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.datasets import TripletExtractionDataset
from car_tse.metrics.si_sdr import si_sdr_per_example
from car_tse.models.stage1_system import Stage1System
from car_tse.train.utils import get_device, load_checkpoint, write_metrics


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    keys = set().union(*(item.keys() for item in batch))
    for key in keys:
        values = [item.get(key) for item in batch]
        if all(isinstance(value, torch.Tensor) for value in values):
            output[key] = torch.stack(values)
        else:
            output[key] = values
    return output


def build_model(cfg: dict[str, Any], checkpoint: Path, device: torch.device) -> Stage1System:
    pretrained = Path(cfg["pretrained_checkpoint"])
    if not pretrained.is_absolute():
        pretrained = REPO_ROOT / pretrained
    model = Stage1System(
        spex_cfg=cfg["spex"],
        sample_rate=int(cfg["sample_rate"]),
        use_cabin_frontend=bool(cfg.get("use_cabin_frontend", False)),
        num_mics=int(cfg.get("num_mics", 4)),
        frontend_cfg=cfg.get("frontend"),
        pretrained_dir=str(pretrained),
        backend=cfg.get("spex_backend", "transfer_spexplus_16k"),
    ).to(device)
    load_checkpoint(checkpoint, model)
    model.eval()
    return model


def evaluate_model(
    model: Stage1System,
    loader: DataLoader,
    rows: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    by_domain: dict[str, list[float]] = defaultdict(list)
    by_domain_input: dict[str, list[float]] = defaultdict(list)
    all_scores: list[float] = []
    all_input: list[float] = []
    row_index = 0
    with torch.inference_mode():
        for batch in loader:
            mix = batch["mixture"].to(device)
            enrollment = batch["enrollment"].to(device)
            target = batch["target"].to(device)
            mc = batch.get("multichannel")
            if isinstance(mc, torch.Tensor):
                mc = mc.to(device)
            output, _, _, _ = model(mix, enrollment, multichannel=mc)
            baseline = model.frontend(mc) if model.frontend is not None and mc is not None else mix
            output_scores = si_sdr_per_example(output, target).detach().cpu().numpy()
            input_scores = si_sdr_per_example(baseline, target).detach().cpu().numpy()
            for offset, (out_score, input_score) in enumerate(zip(output_scores, input_scores)):
                domain = str(rows[row_index + offset].get("domain", "unknown"))
                by_domain[domain].append(float(out_score - input_score))
                by_domain_input[domain].append(float(input_score))
                all_scores.append(float(out_score - input_score))
                all_input.append(float(input_score))
            row_index += len(output_scores)

    def summarize(values: list[float], inputs: list[float]) -> dict[str, Any]:
        return {
            "examples": len(values),
            "mean_si_sdri_db": float(np.mean(values)) if values else None,
            "median_si_sdri_db": float(np.median(values)) if values else None,
            "p10_si_sdri_db": float(np.percentile(values, 10)) if values else None,
            "p90_si_sdri_db": float(np.percentile(values, 90)) if values else None,
            "negative_si_sdri_examples": int(sum(value < 0 for value in values)),
            "mean_input_si_sdr_db": float(np.mean(inputs)) if inputs else None,
        }

    return {
        "by_domain": {
            domain: summarize(values, by_domain_input[domain])
            for domain, values in sorted(by_domain.items())
        },
        "overall": summarize(all_scores, all_input),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.yaml")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "manifests/val/stage1_dual_domain_rms010.jsonl")
    parser.add_argument("--baseline-config", type=Path, default=REPO_ROOT / "configs/stage1_spex_transfer_mc_cavemove.yaml")
    parser.add_argument("--candidate-config", type=Path, default=REPO_ROOT / "configs/stage1_spex_dual_domain_rms010_ft1.yaml")
    parser.add_argument("--baseline-checkpoint", type=Path, default=REPO_ROOT / "checkpoints/stage1_spex_transfer_mc_cavemove_cabin/best.pt")
    parser.add_argument("--candidate-checkpoint", type=Path, default=REPO_ROOT / "checkpoints/stage1_spex_dual_domain_rms010_ft1/best.pt")
    parser.add_argument("--max-per-domain", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "logs/stage1_dual_domain_regression_2026-10-10.json")
    args = parser.parse_args()

    baseline_cfg = load_stage_config(args.baseline_config, args.paths)
    candidate_cfg = load_stage_config(args.candidate_config, args.paths)
    device = get_device()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row.get("domain", "unknown"))].append(index)
    selected_indices: list[int] = []
    for domain in sorted(grouped):
        selected_indices.extend(grouped[domain][: max(0, int(args.max_per_domain))])
    selected_indices.sort()
    selected_rows = [rows[index] for index in selected_indices]

    dataset_kwargs = {
        "cabin_noise": True,
        "num_mics": 4,
        "cabin_noise_channels": [0, 2, 4, 6],
        "cabin_noise_snr_db": [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0],
        "cabin_noise_cache_size": 32,
        "cabin_noise_cache_dir": REPO_ROOT / "data/cache/cavemove_16k_mc",
    }
    # The dataset reads rows from disk; assign the selected rows directly so
    # both models receive identical deterministic crops and noise placement.
    dataset = TripletExtractionDataset(
        args.manifest,
        sample_rate=int(candidate_cfg["sample_rate"]),
        segment_seconds=float(candidate_cfg["segment_seconds"]),
        training=False,
        **dataset_kwargs,
    )
    dataset.rows = selected_rows
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, collate_fn=collate)
    baseline = build_model(baseline_cfg, args.baseline_checkpoint, device)
    candidate = build_model(candidate_cfg, args.candidate_checkpoint, device)
    baseline_result = evaluate_model(baseline, loader, selected_rows, device)
    candidate_result = evaluate_model(candidate, loader, selected_rows, device)
    report = {
        "benchmark": "stage1_dual_domain_regression",
        "device": str(device),
        "manifest": str(args.manifest),
        "examples_by_domain": {domain: len([row for row in selected_rows if str(row.get("domain")) == domain]) for domain in sorted(grouped)},
        "baseline": {
            "config": str(args.baseline_config),
            "checkpoint": str(args.baseline_checkpoint),
            "metrics": baseline_result,
        },
        "candidate": {
            "config": str(args.candidate_config),
            "checkpoint": str(args.candidate_checkpoint),
            "metrics": candidate_result,
        },
        "delta_candidate_minus_baseline": {
            domain: (
                float(candidate_result["by_domain"][domain]["mean_si_sdri_db"])
                - float(baseline_result["by_domain"][domain]["mean_si_sdri_db"])
            )
            for domain in candidate_result["by_domain"]
            if candidate_result["by_domain"][domain]["mean_si_sdri_db"] is not None
            and baseline_result["by_domain"].get(domain, {}).get("mean_si_sdri_db") is not None
        },
        "promotion_rule": "candidate must not regress either domain; this report is diagnostic only",
    }
    write_metrics(args.output, report)
    print(json.dumps({"output": str(args.output), "baseline": baseline_result, "candidate": candidate_result, "delta": report["delta_candidate_minus_baseline"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
