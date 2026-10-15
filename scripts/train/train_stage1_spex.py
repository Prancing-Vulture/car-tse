#!/usr/bin/env python
"""Train Stage 1 SpEx+ (single- or multi-channel)."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.datasets import TripletExtractionDataset
from car_tse.losses.stage1 import stage1_loss
from car_tse.metrics.si_sdr import si_sdr, si_sdr_per_example
from car_tse.models.stage1_system import Stage1System
from car_tse.train.utils import get_device, load_checkpoint, save_checkpoint, schedule_for_epoch, write_metrics


def collate(batch):
    out = {}
    for k in batch[0]:
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch])
        else:
            out[k] = [b[k] for b in batch]
    return out


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_schedule(model: Stage1System, block: dict) -> None:
    model.set_encoder_frozen(block.get("freeze_encoder", False))
    if block.get("freeze_spex"):
        model.set_spex_frozen(True)
    elif "freeze_spex" in block:
        model.set_spex_frozen(False)


def build_optimizer(model: Stage1System, block: dict) -> torch.optim.Optimizer:
    groups = []
    encoder_ids: set[int] = set()
    frontend_ids: set[int] = set()
    if block.get("lr_encoder", 0) > 0:
        enc_params = []
        for m in model.spex.encoder_params():
            enc_params += list(m.parameters())
        enc_params = [p for p in enc_params if p.requires_grad]
        encoder_ids = {id(p) for p in enc_params}
        groups.append({"params": enc_params, "lr": block["lr_encoder"]})
    if model.frontend is not None and block.get("lr_frontend"):
        frontend_params = [p for p in model.frontend.parameters() if p.requires_grad]
        frontend_ids = {id(p) for p in frontend_params}
        groups.append({"params": frontend_params, "lr": block["lr_frontend"]})
    rest = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in encoder_ids and id(p) not in frontend_ids
    ]
    if block.get("lr_rest", 1e-4) > 0:
        groups.append({"params": rest, "lr": block.get("lr_rest", 1e-4)})
    if not groups:
        groups = [{"params": model.parameters(), "lr": 1e-4}]
    return torch.optim.AdamW(groups)


def _move_optional(batch: dict, key: str, device: torch.device) -> torch.Tensor | None:
    value = batch.get(key)
    if isinstance(value, list):
        return None
    return value.to(device) if isinstance(value, torch.Tensor) else None


@torch.no_grad()
def evaluate(model: Stage1System, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    output_scores: list[float] = []
    input_scores: list[float] = []
    absent_energy: list[float] = []
    for batch in loader:
        mix = batch["mixture"].to(device)
        enroll = batch["enrollment"].to(device)
        target = batch["target"].to(device)
        mc = _move_optional(batch, "multichannel", device)
        pb = _move_optional(batch, "playback_ref", device)
        model_input = mix
        if model.frontend is not None:
            if mc is None:
                continue
            model_input = model.frontend(mc, pb)
        s1, _, _, _ = model(mix, enroll, multichannel=mc, playback_ref=pb)
        present_values = batch.get("target_present")
        if isinstance(present_values, list):
            present_mask = torch.tensor(present_values, device=device, dtype=torch.bool)
        elif isinstance(present_values, torch.Tensor):
            present_mask = present_values.to(device).bool()
        else:
            present_mask = torch.ones(s1.shape[0], device=device, dtype=torch.bool)
        if present_mask.any():
            output_scores.extend(float(x) for x in si_sdr_per_example(s1[present_mask], target[present_mask]).detach().cpu())
            input_scores.extend(float(x) for x in si_sdr_per_example(model_input[present_mask], target[present_mask]).detach().cpu())
        if (~present_mask).any():
            absent_energy.append(float(s1[~present_mask].float().square().mean().detach().cpu()))
    if not output_scores:
        return {"si_sdr_s1": float("nan"), "input_si_sdr": float("nan"), "si_sdri": float("nan")}
    output = torch.tensor(output_scores)
    baseline = torch.tensor(input_scores)
    improvement = output - baseline
    return {
        "si_sdr_s1": float(output.mean()),
        "input_si_sdr": float(baseline.mean()),
        "si_sdri": float(improvement.mean()),
        "si_sdri_median": float(improvement.median()),
        "si_sdri_p10": float(torch.quantile(improvement, 0.10)),
        "si_sdri_p90": float(torch.quantile(improvement, 0.90)),
        "si_sdri_negative": float((improvement < 0).sum()),
        "examples": float(output.numel()),
        "absent_output_energy": float(sum(absent_energy) / max(len(absent_energy), 1)) if absent_energy else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "stage1_spex_sc.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="car-tse")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-dir", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    seed = int(args.seed if args.seed is not None else cfg["train"].get("seed", 20261014))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = get_device()
    manifest = Path(paths["manifest_root"]) / args.split / cfg["train"]["manifest"]
    dataset_kwargs = {
        "cabin_noise": bool(cfg.get("cabin_noise", False)),
        "num_mics": int(cfg.get("num_mics", 4)),
        "cabin_noise_channels": cfg.get("cabin_noise_channels"),
        "cabin_noise_snr_db": cfg.get("cabin_noise_snr_db"),
        "cabin_noise_cache_size": int(cfg.get("cabin_noise_cache_size", 32)),
    }
    ds = TripletExtractionDataset(
        manifest,
        cfg["sample_rate"],
        cfg["segment_seconds"],
        training=True,
        **dataset_kwargs,
    )
    if len(ds) == 0:
        print(f"No data in {manifest}; run data prep first.")
        return 1
    train_limit = cfg["train"].get("train_limit")
    train_data = Subset(ds, range(min(int(train_limit), len(ds)))) if train_limit else ds
    val_manifest_value = Path(cfg["train"].get("val_manifest", manifest.name))
    if val_manifest_value.is_absolute():
        val_manifest = val_manifest_value
    else:
        val_split = cfg["train"].get("val_split", "val")
        val_manifest = Path(paths["manifest_root"]) / val_split / val_manifest_value
    val_loader = None
    if val_manifest.exists():
        val_ds = TripletExtractionDataset(
            val_manifest,
            cfg["sample_rate"],
            cfg["segment_seconds"],
            training=False,
            **dataset_kwargs,
        )
        val_limit = cfg["train"].get("val_limit")
        val_data = Subset(val_ds, range(min(int(val_limit), len(val_ds)))) if val_limit else val_ds
        val_workers = int(cfg["train"].get("num_workers", 0))
        val_loader = DataLoader(
            val_data,
            batch_size=cfg["train"].get("val_batch_size", 2),
            collate_fn=collate,
            num_workers=val_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=val_workers > 0,
            prefetch_factor=2 if val_workers > 0 else None,
        )
    pretrained = cfg.get("pretrained_checkpoint", paths["spex_pretrained_dir"])
    pretrained = Path(pretrained)
    if not pretrained.is_absolute():
        pretrained = REPO_ROOT / pretrained
    init_ckpt = cfg["train"].get("init_checkpoint")
    init_checkpoint_payload = None
    start_epoch = 1
    best_si = float("-inf")
    model = Stage1System(
        spex_cfg=cfg["spex"],
        sample_rate=cfg["sample_rate"],
        use_cabin_frontend=cfg.get("use_cabin_frontend", False),
        num_mics=cfg.get("num_mics", 4),
        frontend_cfg=cfg.get("frontend"),
        pretrained_dir=str(pretrained),
        backend=cfg.get("spex_backend", "vendor_8k"),
    ).to(device)
    if init_ckpt:
        init_path = paths["checkpoint_root"] / init_ckpt
        if init_path.exists():
            init_checkpoint_payload = load_checkpoint(init_path, model)
            start_epoch = max(1, int(init_checkpoint_payload.get("epoch", 0)) + 1)
            best_si = float(
                init_checkpoint_payload.get("extra", {}).get("selection_metric", float("-inf"))
            )
    run_name = args.run_name or str(cfg["name"])
    out_dir = paths["checkpoint_root"] / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    optimizer = None
    previous_block = None
    train_workers = int(cfg["train"].get("num_workers", 0))
    train_loader = DataLoader(
        train_data,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        collate_fn=collate,
        num_workers=train_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=train_workers > 0,
        prefetch_factor=2 if train_workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
    wandb_run = None
    if args.wandb:
        import wandb

        if args.wandb_dir is not None:
            args.wandb_dir.mkdir(parents=True, exist_ok=True)
        config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
        manifest_path = Path(paths["manifest_root"]) / args.split / cfg["train"]["manifest"]
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            job_type="train-stage1-spex",
            mode=args.wandb_mode,
            dir=str(args.wandb_dir) if args.wandb_dir is not None else None,
            config={
                "stage": "stage1_spex",
                "config": config_path.name,
                "config_sha256": sha256_file(config_path),
                "pretrained_checkpoint": str(pretrained),
                "pretrained_checkpoint_sha256": sha256_file(pretrained) if pretrained.is_file() else None,
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "seed": seed,
                "sample_rate": int(cfg["sample_rate"]),
                "segment_seconds": float(cfg["segment_seconds"]),
                "batch_size": int(cfg["train"]["batch_size"]),
                "max_epochs": int(cfg["train"]["max_epochs"]),
                "start_epoch": start_epoch,
                "device": str(device),
                "torch_version": torch.__version__,
            },
        )
        wandb_run.define_metric("epoch")
        wandb_run.define_metric("train/*", step_metric="epoch")
        wandb_run.define_metric("validation/*", step_metric="epoch")
        wandb_run.define_metric("learning_rate/*", step_metric="epoch")
        resume_history = cfg["train"].get("resume_metrics_history")
        if start_epoch > 1 and resume_history:
            resume_history_path = Path(resume_history)
            if not resume_history_path.is_absolute():
                resume_history_path = REPO_ROOT / resume_history_path
            prior_rows = [
                json.loads(line)
                for line in resume_history_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if prior_rows:
                prior = prior_rows[-1]
                prior_train = prior.get("train", {})
                prior_val = prior.get("val", {})
                prior_epoch = int(prior["epoch"])
                resume_row = {
                    "epoch": prior_epoch,
                    "train/loss": prior.get("train_loss"),
                    "train/si_sdr_s1": prior_train.get("si_sdr_s1"),
                    "validation/si_sdri": prior_val.get("si_sdri"),
                    "best_validation/si_sdri": prior.get("best_metric"),
                }
                resume_row.update({f"train/{key}": value for key, value in prior_train.items()})
                resume_row.update({f"validation/{key}": value for key, value in prior_val.items()})
                wandb_run.log(resume_row, step=prior_epoch, commit=True)
    for epoch in range(start_epoch, cfg["train"]["max_epochs"] + 1):
        block = schedule_for_epoch(cfg["train"]["schedule"], epoch)
        apply_schedule(model, block)
        if optimizer is None or block != previous_block:
            optimizer = build_optimizer(model, block)
            previous_block = block.copy()
        model.train()
        stats_acc = {}
        loss_sum = 0.0
        n = 0
        for batch in train_loader:
            mix = batch["mixture"].to(device)
            enroll = batch["enrollment"].to(device)
            target = batch["target"].to(device)
            target_present = None
            if "target_present" in batch:
                target_present = torch.tensor(batch["target_present"], device=device, dtype=torch.float32)
            mc = batch.get("multichannel")
            pb = batch.get("playback_ref")
            if isinstance(mc, list):
                mc = None
            if isinstance(pb, list):
                pb = None
            if mc is not None:
                mc = mc.to(device)
            if pb is not None:
                pb = pb.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                s1, s2, s3, logits = model(mix, enroll, multichannel=mc, playback_ref=pb)
            labels = None
            if "speaker_label" in batch:
                labels = torch.tensor(batch["speaker_label"], device=device, dtype=torch.long)
            # Keep the SI-SDR reduction in float32 under AMP; this avoids
            # small-epsilon underflow while retaining half-precision model
            # execution for the expensive convolutions.
            loss, st = stage1_loss(
                s1.float(),
                s2.float(),
                s3.float(),
                target.float(),
                logits,
                labels,
                target_present=target_present,
                absent_output_penalty=float(cfg["loss"].get("absent_output_penalty", 0.0)),
                **{key: value for key, value in cfg["loss"].items() if key != "absent_output_penalty"},
            )
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                optimizer.step()
            for k, v in st.items():
                stats_acc[k] = stats_acc.get(k, 0) + v
            loss_sum += float(loss.detach())
            n += 1
        mean_si = stats_acc.get("si_sdr_s1", 0) / max(n, 1)
        val_stats = evaluate(model, val_loader, device) if val_loader is not None else {}
        metric = val_stats.get("si_sdri", mean_si)
        print(
            f"epoch {epoch} train_si_sdr={mean_si:.2f} "
            + (f"val_si_sdr={val_stats['si_sdr_s1']:.2f} val_si_sdri={metric:.2f}" if val_stats else "")
        , flush=True)
        if metric > best_si:
            best_si = metric
            save_checkpoint(
                out_dir / "best.pt",
                model,
                epoch,
                {"train_si_sdr_s1": mean_si, "val": val_stats, "selection_metric": metric, "seed": seed},
            )
        train_stats = {key: value / max(n, 1) for key, value in stats_acc.items()}
        metrics = {
            "epoch": epoch,
            "seed": seed,
            "train_loss": loss_sum / max(n, 1),
            "train": train_stats,
            "val": val_stats,
            "best_metric": best_si,
        }
        write_metrics(
            paths["log_root"] / run_name / "metrics_last.json",
            metrics,
        )
        if wandb_run is not None:
            log_row = {
                "epoch": epoch,
                "train/loss": loss_sum / max(n, 1),
                "train/si_sdr_s1": mean_si,
                "validation/si_sdri": metric,
                "best_validation/si_sdri": best_si,
            }
            log_row.update({f"train/{key}": value for key, value in train_stats.items()})
            log_row.update({f"validation/{key}": value for key, value in val_stats.items()})
            log_row.update({f"learning_rate/group_{index}": group["lr"] for index, group in enumerate(optimizer.param_groups)})
            wandb_run.log(log_row, step=epoch, commit=True)
    write_metrics(paths["log_root"] / run_name / "train_last.json", {"best_metric": best_si, "seed": seed})
    if wandb_run is not None:
        wandb_run.summary["best_validation_si_sdri"] = best_si
        wandb_run.summary["seed"] = seed
        wandb_run.summary["best_checkpoint"] = str(out_dir / "best.pt")
        wandb_run.finish()
    print(f"Done. Best checkpoint: {out_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
