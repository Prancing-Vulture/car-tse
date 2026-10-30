#!/usr/bin/env python
"""Evaluate Stage 1 SI-SDR and SI-SDR improvement on a held-out manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from torch.utils.data import DataLoader

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.datasets import TripletExtractionDataset
from car_tse.metrics.si_sdr import si_sdr_per_example
from car_tse.models.stage1_system import Stage1System
from car_tse.train.utils import get_device, load_checkpoint, write_metrics
from car_tse.train.wandb_logging import log_evaluation


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def optional_tensor(batch: dict, key: str, device: torch.device) -> torch.Tensor | None:
    value = batch.get(key)
    if isinstance(value, list):
        return None
    return value.to(device) if isinstance(value, torch.Tensor) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "stage1_spex_sc.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="car-tse")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--wandb-dir", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    device = get_device()
    manifest = Path(paths["manifest_root"]) / args.split / cfg["train"]["manifest"]
    dataset_kwargs = {
        "cabin_noise": bool(cfg.get("cabin_noise", False)),
        "num_mics": int(cfg.get("num_mics", 4)),
        "cabin_noise_channels": cfg.get("cabin_noise_channels"),
        "cabin_noise_snr_db": cfg.get("cabin_noise_snr_db"),
        "cabin_noise_cache_size": int(cfg.get("cabin_noise_cache_size", 32)),
    }
    dataset = TripletExtractionDataset(
        manifest,
        cfg["sample_rate"],
        cfg["segment_seconds"],
        training=False,
        **dataset_kwargs,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    pretrained = Path(cfg.get("pretrained_checkpoint", paths["spex_pretrained_dir"]))
    if not pretrained.is_absolute():
        pretrained = REPO_ROOT / pretrained
    model = Stage1System(
        spex_cfg=cfg["spex"],
        sample_rate=cfg["sample_rate"],
        use_cabin_frontend=cfg.get("use_cabin_frontend", False),
        num_mics=cfg.get("num_mics", 4),
        frontend_cfg=cfg.get("frontend"),
        pretrained_dir=str(pretrained),
        backend=cfg.get("spex_backend", "vendor_8k"),
    ).to(device)
    checkpoint = args.checkpoint or (paths["checkpoint_root"] / cfg["name"] / "best.pt")
    if not checkpoint.is_absolute():
        checkpoint = REPO_ROOT / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {checkpoint}")
    load_checkpoint(checkpoint, model)
    model.eval()

    output_scores: list[float] = []
    input_scores: list[float] = []
    absent_energy: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            mix = batch["mixture"].to(device)
            enroll = batch["enrollment"].to(device)
            target = batch["target"].to(device)
            multichannel = optional_tensor(batch, "multichannel", device)
            playback_ref = optional_tensor(batch, "playback_ref", device)
            model_input = mix
            if model.frontend is not None:
                if multichannel is None:
                    continue
                model_input = model.frontend(multichannel, playback_ref)
            extracted, _, _, _ = model(
                mix, enroll, multichannel=multichannel, playback_ref=playback_ref
            )
            present_values = batch.get("target_present")
            if isinstance(present_values, list):
                present = torch.tensor(present_values, device=device, dtype=torch.bool)
            elif isinstance(present_values, torch.Tensor):
                present = present_values.to(device).bool()
            else:
                present = torch.ones(extracted.shape[0], device=device, dtype=torch.bool)
            if present.any():
                output_scores.extend(
                    float(x)
                    for x in si_sdr_per_example(extracted[present], target[present]).cpu()
                )
                input_scores.extend(
                    float(x)
                    for x in si_sdr_per_example(model_input[present], target[present]).cpu()
                )
            if (~present).any():
                absent_energy.append(float(extracted[~present].float().square().mean().cpu()))

    if not output_scores:
        raise RuntimeError(f"No present-target examples were evaluated from {manifest}")
    output = torch.tensor(output_scores, dtype=torch.float32)
    baseline = torch.tensor(input_scores, dtype=torch.float32)
    improvement = output - baseline
    metrics = {
        "si_sdr_s1": float(output.mean()),
        "input_si_sdr": float(baseline.mean()),
        "si_sdri": float(improvement.mean()),
        "si_sdri_median": float(improvement.median()),
        "si_sdri_p10": float(torch.quantile(improvement, 0.10)),
        "si_sdri_p90": float(torch.quantile(improvement, 0.90)),
        "si_sdri_negative": float((improvement < 0).sum()),
        "examples": int(output.numel()),
        "absent_output_energy": float(sum(absent_energy) / max(len(absent_energy), 1)),
    }
    output_path = paths["log_root"] / cfg["name"] / f"metrics_{args.split}.json"
    write_metrics(
        output_path,
        {
            "stage": "stage1_spex",
            "split": args.split,
            "checkpoint": str(checkpoint),
            "manifest_sha256": sha256_file(manifest),
            "metrics": metrics,
            "per_example_si_sdr_db": output_scores,
            "per_example_input_si_sdr_db": input_scores,
            "per_example_si_sdri_db": [float(x) for x in improvement],
        },
    )
    if args.wandb:
        log_evaluation(
            project=args.wandb_project,
            run_name=args.run_name or f"{cfg['name']}-{args.split}-evaluation",
            job_type=f"stage1-spex-{args.split}-evaluation",
            config={
                "stage": "stage1_spex",
                "split": args.split,
                "seed": int(cfg["train"].get("seed", 20261011)),
                "config": args.config.name,
                "config_sha256": sha256_file(args.config if args.config.is_absolute() else REPO_ROOT / args.config),
                "checkpoint_sha256": sha256_file(checkpoint),
                "manifest_sha256": sha256_file(manifest),
            },
            metrics=metrics,
            output_path=output_path,
            mode=args.wandb_mode,
            directory=args.wandb_dir,
            distributions={"si_sdri_db": [float(x) for x in improvement]},
        )
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
