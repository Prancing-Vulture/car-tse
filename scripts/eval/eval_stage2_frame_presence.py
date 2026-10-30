#!/usr/bin/env python
"""Evaluate a trained frame-presence head on a fixed held-out manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))

import torch
from torch.utils.data import DataLoader

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.datasets import PresenceDataset
from car_tse.models.presence_head import FramePresenceHead
from car_tse.models.stage1_system import Stage1System
from car_tse.train.utils import get_device, load_checkpoint
from train_stage2_frame_presence import collate, evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/stage2_frame_presence_gsc_cavemove_pilot.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.yaml")
    parser.add_argument("--manifest", default="stage2_presence_gsc_cavemove.jsonl")
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "checkpoints/stage2_frame_presence_gsc_cavemove_pilot/best.pt")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--locked-threshold", type=float, default=0.40, help="Threshold selected on validation, fixed before test evaluation")
    parser.add_argument("--run-name", default=None, help="Optional W&B run name")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "reports/evaluation/stage2_frame_presence_test.json")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="car-tse")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    device = get_device()
    stage1_cfg_path = Path(cfg["stage1_config"])
    if not stage1_cfg_path.is_absolute():
        stage1_cfg_path = REPO_ROOT / stage1_cfg_path
    stage1_cfg = load_stage_config(stage1_cfg_path, args.paths)
    pretrained = Path(stage1_cfg["pretrained_checkpoint"])
    if not pretrained.is_absolute():
        pretrained = REPO_ROOT / pretrained
    stage1 = Stage1System(
        spex_cfg=stage1_cfg["spex"],
        sample_rate=int(stage1_cfg["sample_rate"]),
        use_cabin_frontend=bool(stage1_cfg.get("use_cabin_frontend", True)),
        num_mics=int(stage1_cfg.get("num_mics", 4)),
        frontend_cfg=stage1_cfg.get("frontend"),
        pretrained_dir=str(pretrained),
        backend=stage1_cfg.get("spex_backend", "transfer_spexplus_16k"),
    ).to(device)
    stage1_path = Path(paths["checkpoint_root"]) / cfg["spex_checkpoint"]
    load_checkpoint(stage1_path, stage1)
    stage1.eval()
    for parameter in stage1.parameters():
        parameter.requires_grad_(False)

    model_cfg = cfg.get("model", {})
    head = FramePresenceHead(
        sample_rate=int(cfg["sample_rate"]),
        n_mels=int(model_cfg.get("n_mels", 40)),
        hidden=int(model_cfg.get("hidden", 64)),
        n_fft=int(model_cfg.get("n_fft", 512)),
        hop=int(model_cfg.get("hop", 160)),
    ).to(device)
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else Path(paths["checkpoint_root"]) / args.checkpoint
    load_checkpoint(checkpoint, head)

    manifest_path = Path(paths["manifest_root"]) / args.split / args.manifest
    dataset = PresenceDataset(
        manifest_path,
        int(cfg["sample_rate"]),
        float(cfg["segment_seconds"]),
        training=False,
        cabin_noise=bool(cfg.get("cabin_noise", True)),
        num_mics=int(cfg.get("num_mics", 4)),
        cabin_noise_channels=cfg.get("cabin_noise_channels", [0, 2, 4, 6]),
        cabin_noise_snr_db=cfg.get("cabin_noise_snr_db"),
        cabin_noise_cache_size=int(cfg.get("cabin_noise_cache_size", 32)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"].get("val_batch_size", 16)),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )
    model_cfg = cfg.get("model", {})
    result = evaluate(
        head,
        stage1,
        loader,
        device,
        bool(cfg["train"].get("amp", True)) and device.type == "cuda",
        int(model_cfg.get("n_fft", 512)),
        int(model_cfg.get("hop", 160)),
        float(model_cfg.get("target_threshold_db_below_peak", 30.0)),
        float(model_cfg.get("top_fraction", 0.20)),
    )
    fixed = next(
        (item for item in result["chunk_threshold_sweep"] if abs(float(item["threshold"]) - args.locked_threshold) < 1e-8),
        None,
    )
    if fixed is None:
        raise ValueError(f"Locked threshold {args.locked_threshold} is not in the evaluator's threshold sweep")
    output = {
        "split": args.split,
        "manifest": str(manifest_path),
        "checkpoint": str(checkpoint),
        "examples": len(dataset),
        "locked_validation_threshold": args.locked_threshold,
        "chunk_metrics_at_locked_threshold": fixed,
        "chunk_metrics_at_0_50": result["chunk_at_0.5"],
        "frame_metrics_at_0_50": result["frame_at_0.5"],
        "frame_threshold_sweep": result["frame_threshold_sweep"],
        "chunk_threshold_sweep": result["chunk_threshold_sweep"],
        "protocol_note": "The locked threshold is selected on validation; test metrics are reported at that fixed threshold. Included test sweeps are descriptive only and are not used for model selection.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")

    if args.wandb:
        import wandb

        if args.wandb_dir is not None:
            args.wandb_dir.mkdir(parents=True, exist_ok=True)
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name or f"stage2-presence-{args.split}-heldout-eval",
            job_type="eval-stage2-frame-presence",
            mode=args.wandb_mode,
            dir=str(args.wandb_dir) if args.wandb_dir else None,
            config={
                "stage": "stage2_frame_presence",
                "split": args.split,
                "manifest": str(manifest_path),
                "checkpoint": str(checkpoint),
                "examples": len(dataset),
                "locked_validation_threshold": args.locked_threshold,
                "speaker_disjoint_from_train": True,
                "note": output["protocol_note"],
            },
        )
        run.log({f"test/chunk_{key}": value for key, value in fixed.items() if key != "threshold"})
        run.log({"test/chunk_threshold": args.locked_threshold, "test/threshold_0_50_f1": result["chunk_at_0.5"]["f1"]})
        for scope, points in (("frame", result["frame_threshold_sweep"]), ("chunk", result["chunk_threshold_sweep"])):
            roc_table = wandb.Table(columns=["false_accept_rate", "recall"])
            pr_table = wandb.Table(columns=["recall", "precision"])
            for point in points:
                roc_table.add_data(float(point["false_accept_rate"]), float(point["recall"]))
                pr_table.add_data(float(point["recall"]), float(point["precision"]))
            run.log({
                f"test/{scope}_roc_data": roc_table,
                f"test/{scope}_roc_curve": wandb.plot.line(
                    roc_table, "false_accept_rate", "recall", title=f"{scope.title()} ROC"
                ),
                f"test/{scope}_pr_data": pr_table,
                f"test/{scope}_pr_curve": wandb.plot.line(
                    pr_table, "recall", "precision", title=f"{scope.title()} Precision-Recall"
                ),
            })
        run.summary["test_checkpoint"] = str(checkpoint)
        run.summary["test_manifest"] = str(manifest_path)
        run.summary["test_metrics_path"] = str(args.output)
        run.finish()

    print(json.dumps(output, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
