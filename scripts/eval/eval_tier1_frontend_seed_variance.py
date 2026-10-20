#!/usr/bin/env python
"""Evaluate seeded current-frontend fine-tunes on the fixed Stage-1 test set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from car_tse.data.datasets import TripletExtractionDataset
from car_tse.metrics.si_sdr import si_sdr_per_example
from car_tse.models.stage1_system import Stage1System
from run_tier1_tse_ablation import gender_pair, sha256_file


def seed_from_checkpoint(path: Path) -> str:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    seed = payload.get("extra", {}).get("seed")
    if seed is None:
        raise ValueError(f"checkpoint does not record its training seed: {path}")
    return str(seed)


def make_model(cfg: dict[str, Any], base_spex: Path, checkpoint: Path, device: torch.device) -> Stage1System:
    model = Stage1System(
        spex_cfg=cfg["spex"],
        sample_rate=int(cfg["sample_rate"]),
        use_cabin_frontend=True,
        num_mics=4,
        frontend_cfg=cfg["frontend"],
        pretrained_dir=str(base_spex),
        backend="transfer_spexplus_16k",
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=REPO / "manifests" / "test" / "stage1_cavemove_cabin.jsonl")
    parser.add_argument("--stage-config", type=Path, default=REPO / "configs" / "stage1_spex_fsc_passenger_cochannel_ft1_2026-10-16.yaml")
    parser.add_argument("--base-spex", type=Path, default=Path(r"E:\car-tse-data\models\car_tse_spexplus_from_stage1.pt"))
    parser.add_argument("--speakers-file", type=Path, default=Path(r"E:\bss-audio-data\speech\LibriSpeech\LibriSpeech\SPEAKERS.TXT"))
    parser.add_argument("--seed-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO / "reports" / "evaluation" / "tier1_ablation_2026-10-19" / "tse_seed_variance")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    cfg = yaml.safe_load(args.stage_config.read_text(encoding="utf-8"))
    dataset = TripletExtractionDataset(
        args.manifest,
        sample_rate=int(cfg["sample_rate"]),
        segment_seconds=float(cfg["segment_seconds"]),
        training=False,
        cabin_noise=True,
        num_mics=4,
        cabin_noise_channels=[0, 2, 4, 6],
        cabin_noise_cache_size=32,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    genders = {}
    for line in args.speakers_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.lstrip().startswith(";") or "|" not in line:
            continue
        fields = [x.strip() for x in line.split("|")]
        if len(fields) >= 2 and fields[0].isdigit() and fields[1] in {"M", "F"}:
            genders[fields[0]] = fields[1]

    seed_models: list[tuple[str, Path, Stage1System]] = []
    for checkpoint in args.seed_checkpoint:
        seed = seed_from_checkpoint(checkpoint)
        model = make_model(cfg, args.base_spex, checkpoint, device)
        seed_models.append((seed, checkpoint, model))

    per_example: list[dict[str, Any]] = []
    offset = 0
    @torch.inference_mode()
    def infer_batch(batch: dict[str, Any], rows: list[dict[str, Any]]) -> None:
        nonlocal per_example
        mix = batch["mixture"].to(device).float()
        target = batch["target"].to(device).float()
        multichannel = batch["multichannel"].to(device).float()
        batch_size, _, _ = multichannel.shape
        delay_sum = multichannel.mean(dim=1)
        base_sisdr = si_sdr_per_example(delay_sum, target)
        outputs: dict[str, torch.Tensor] = {}
        for seed, _, model in seed_models:
            for channels in (1, 2, 4):
                masked = torch.zeros_like(multichannel)
                masked[:, :channels] = multichannel[:, :channels]
                outputs[f"{channels}ch_seed{seed}"] = model(
                    mix, batch["enrollment"].to(device).float(), multichannel=masked
                )[0]
        scores = {name: si_sdr_per_example(value, target) for name, value in outputs.items()}
        for i, row in enumerate(rows):
            target_gender, interferer_gender, pair = gender_pair(row, genders)
            speech_sir = 10.0 * math.log10(
                float(target[i].square().mean().cpu())
                / max(float((mix[i] - target[i]).square().mean().cpu()), 1.0e-12)
            )
            sir_bin = "low_le_minus3_db" if speech_sir <= -3 else ("mid_minus3_to3_db" if speech_sir <= 3 else "high_gt3_db")
            for name in outputs:
                per_example.append({
                    "row_id": row.get("id", ""),
                    "variant": name,
                    "seed": name.split("seed", 1)[1],
                    "active_channels": int(name.split("ch_", 1)[0]),
                    "cabin_noise_snr_db": float(row.get("cabin_noise_snr_db", row.get("meta", {}).get("cabin_noise_snr_db", float("nan")))),
                    "target_gender": target_gender,
                    "interferer_gender": interferer_gender,
                    "gender_pair": pair,
                    "passenger_sir_db_estimated": speech_sir,
                    "passenger_sir_bin": sir_bin,
                    "si_sdr_db": float(scores[name][i].cpu()),
                    "si_sdri_vs_delay_sum_db": float(scores[name][i].cpu() - base_sisdr[i].cpu()),
                })
        del outputs, scores
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for batch in loader:
        count = batch["mixture"].shape[0]
        infer_batch(batch, dataset.rows[offset : offset + count])
        offset += count
        print(f"evaluated {offset}/{len(dataset)} rows", flush=True)

    per_example_path = args.output_dir / "per_example_seed_metrics.jsonl"
    with per_example_path.open("w", encoding="utf-8") as f:
        for row in per_example:
            f.write(json.dumps(row, allow_nan=False) + "\n")

    strata = ("active_channels", "cabin_noise_snr_db", "gender_pair", "passenger_sir_bin")
    metrics = ("si_sdr_db", "si_sdri_vs_delay_sum_db")
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    seeds_per_stratum: dict[tuple, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in per_example:
        key = tuple(row[k] for k in strata)
        groups[key + (row["variant"],)].append(row)
        seeds_per_stratum[key][row["seed"]].append(row)

    summary_path = args.output_dir / "summary_per_seed_by_snr_gender_sir.csv"
    columns = ["variant", *strata, "n", *[f"{m}_{s}" for m in metrics for s in ("mean", "std")]]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for key, rows in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
            out = {"variant": key[-1], **dict(zip(strata, key[:-1])), "n": len(rows)}
            for metric in metrics:
                values = [float(r[metric]) for r in rows]
                out[f"{metric}_mean"] = float(np.mean(values))
                out[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else ""
            writer.writerow(out)

    seed_summary_path = args.output_dir / "summary_mean_std_over_seeds_by_snr_gender_sir.csv"
    seed_columns = [*strata, "n_seeds", "n_examples_per_seed", *[f"{m}_{s}_over_seed_means" for m in metrics for s in ("mean", "std")]]
    with seed_summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=seed_columns)
        writer.writeheader()
        for key, by_seed in sorted(seeds_per_stratum.items(), key=lambda item: tuple(str(x) for x in item[0])):
            out = dict(zip(strata, key))
            out["n_seeds"] = len(by_seed)
            per_seed_metric_means: dict[str, list[float]] = {metric: [] for metric in metrics}
            counts = []
            for rows in by_seed.values():
                counts.append(len(rows))
                for metric in metrics:
                    per_seed_metric_means[metric].append(float(np.mean([r[metric] for r in rows])))
            out["n_examples_per_seed"] = int(min(counts)) if counts else 0
            for metric in metrics:
                values = per_seed_metric_means[metric]
                out[f"{metric}_mean_over_seed_means"] = float(np.mean(values)) if values else ""
                out[f"{metric}_std_over_seed_means"] = float(np.std(values, ddof=1)) if len(values) > 1 else ""
            writer.writerow(out)

    metadata = {
        "experiment": "seed variance for learned 4-channel frontend, tested with 1/2/4 channel ablations",
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "device": str(device),
        "pytorch": torch.__version__,
        "test_manifest": str(args.manifest),
        "test_manifest_sha256": sha256_file(args.manifest),
        "stage1_config": str(args.stage_config),
        "stage1_checkpoint_sha256": sha256_file(REPO / "checkpoints" / "stage1_spex_fsc_passenger_cochannel_ft1_2026-10-16" / "best.pt"),
        "base_spex_checkpoint_sha256": sha256_file(args.base_spex),
        "seeds": [{"seed": seed, "checkpoint": str(path), "checkpoint_sha256": sha256_file(path)} for seed, path, _ in seed_models],
        "channel_count_method": "test-time ablation by keeping the first N channels and zeroing the rest; seed models remain four-channel trained",
        "si_sdri_reference": "delay-sum/mean of the four cabin channels",
        "variance_definition": "The companion summary reports mean and sample standard deviation across the seed-specific per-stratum SI-SDRi means; per-example standard deviation is separately reported in summary_per_seed_by_snr_gender_sir.csv.",
        "files": [per_example_path.name, summary_path.name, seed_summary_path.name],
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
