#!/usr/bin/env python
"""Train the enrollment-conditioned target-speaker wake-word detector."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.io.manifest import read_jsonl
from car_tse.models.kws_model import TargetSpeakerKWSModel
from car_tse.train.utils import append_jsonl, get_device, load_checkpoint, save_checkpoint, write_metrics


class CachedTargetSpeakerDataset(Dataset):
    def __init__(self, manifest: Path, cache_dir: Path, split: str, sample_rate: int):
        self.rows = read_jsonl(manifest)
        self.wav = np.load(cache_dir / f"{split}_wav_{sample_rate}hz.npy", mmap_mode="r", allow_pickle=False)
        self.enrollment = np.load(
            cache_dir / f"{split}_enrollment_embedding.npy", mmap_mode="r", allow_pickle=False
        )
        self.extracted = np.load(
            cache_dir / f"{split}_extracted_embedding.npy", mmap_mode="r", allow_pickle=False
        )
        if not (len(self.rows) == len(self.wav) == len(self.enrollment) == len(self.extracted)):
            raise ValueError("Target-KWS cache and manifest lengths do not match")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "wav": torch.from_numpy(np.array(self.wav[index], dtype=np.float32, copy=True)),
            "enrollment_embedding": torch.from_numpy(
                np.array(self.enrollment[index], dtype=np.float32, copy=True)
            ),
            "extracted_embedding": torch.from_numpy(
                np.array(self.extracted[index], dtype=np.float32, copy=True)
            ),
            "label": torch.tensor(int(self.rows[index]["label"]), dtype=torch.long),
        }


@torch.no_grad()
def evaluate(model: TargetSpeakerKWSModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    correct = total = tp = fp = fn = 0
    for batch in loader:
        wav = batch["wav"].to(device, non_blocking=True)
        enrollment = batch["enrollment_embedding"].to(device, non_blocking=True)
        extracted = batch["extracted_embedding"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        predictions = model(wav, enrollment, extracted).argmax(-1)
        correct += int((predictions == labels).sum())
        total += labels.numel()
        tp += int(((predictions == 1) & (labels == 1)).sum())
        fp += int(((predictions == 1) & (labels == 0)).sum())
        fn += int(((predictions == 0) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "accuracy": correct / max(total, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "examples": float(total),
        "true_positive": float(tp),
        "false_positive": float(fp),
        "false_negative": float(fn),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/stage3_target_kws_gsc_marvin.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.yaml")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    device = get_device()
    cache_dir = Path(cfg["cache_dir"])
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    manifest_root = Path(paths["manifest_root"])
    train_manifest = manifest_root / "train" / cfg["manifest_train"]
    val_manifest = manifest_root / "val" / cfg["manifest_val"]
    train_ds = CachedTargetSpeakerDataset(train_manifest, cache_dir, "train", int(cfg["sample_rate"]))
    val_ds = CachedTargetSpeakerDataset(val_manifest, cache_dir, "val", int(cfg["sample_rate"]))
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Target-KWS cache is empty")

    labels = torch.tensor([int(row["label"]) for row in train_ds.rows], dtype=torch.long)
    counts = torch.bincount(labels, minlength=int(cfg["model"]["num_classes"])).float()
    use_balanced_sampler = bool(cfg["train"].get("balanced_sampler", True))
    sampler = None
    if use_balanced_sampler:
        sampler = WeightedRandomSampler(
            weights=(1.0 / counts[labels].clamp_min(1.0)),
            num_samples=len(labels),
            replacement=True,
        )
    workers = int(cfg["train"].get("num_workers", 0))
    loader_kwargs = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
        "prefetch_factor": 2 if workers > 0 else None,
    }
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        sampler=sampler,
        shuffle=sampler is None,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"].get("val_batch_size", cfg["train"]["batch_size"])),
        shuffle=False,
        **loader_kwargs,
    )
    model = TargetSpeakerKWSModel(
        sample_rate=int(cfg["sample_rate"]),
        n_mels=int(cfg["model"].get("n_mels", 40)),
        channels=cfg["model"].get("channels", [32, 64, 128]),
        num_classes=int(cfg["model"].get("num_classes", 2)),
        speaker_embedding_dim=int(cfg["model"].get("speaker_embedding_dim", 256)),
        hidden=int(cfg["model"].get("hidden", 256)),
    ).to(device)
    init_checkpoint = args.init_checkpoint or cfg.get("init_checkpoint")
    if init_checkpoint:
        init_path = init_checkpoint if Path(init_checkpoint).is_absolute() else REPO_ROOT / init_checkpoint
        if bool(cfg.get("init_full_model", False)):
            # Target-speaker checkpoints contain both audio.* and the
            # enrollment/extracted relation classifier.  Loading them into
            # model.audio would silently match nothing because of the prefix.
            load_checkpoint(init_path, model)
            print(f"initialized_full_target_kws_checkpoint={init_path}", flush=True)
        else:
            # Warm-start only the acoustic branch from the ordinary KWS model.
            load_checkpoint(init_path, model.audio)
            print(f"initialized_audio_checkpoint={init_path}", flush=True)

    teacher = None
    distill_alpha = float(cfg.get("distill_alpha", 0.0))
    distill_temperature = max(1.0, float(cfg.get("distill_temperature", 2.0)))
    distill_checkpoint = cfg.get("distill_checkpoint")
    if distill_alpha > 0.0 and distill_checkpoint:
        teacher = TargetSpeakerKWSModel(
            sample_rate=int(cfg["sample_rate"]),
            n_mels=int(cfg["model"].get("n_mels", 40)),
            channels=cfg["model"].get("channels", [32, 64, 128]),
            num_classes=int(cfg["model"].get("num_classes", 2)),
            speaker_embedding_dim=int(cfg["model"].get("speaker_embedding_dim", 256)),
            hidden=int(cfg["model"].get("hidden", 256)),
        ).to(device)
        teacher_path = distill_checkpoint if Path(distill_checkpoint).is_absolute() else REPO_ROOT / distill_checkpoint
        load_checkpoint(teacher_path, teacher)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False
        print(
            f"distillation_teacher={teacher_path} alpha={distill_alpha:.3f} "
            f"temperature={distill_temperature:.2f}",
            flush=True,
        )

    if bool(cfg.get("freeze_audio", False)):
        for parameter in model.audio.parameters():
            parameter.requires_grad = False
        print("frozen_audio_branch=true", flush=True)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable target-KWS parameters remain")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=float(cfg["train"]["lr"]))
    configured_class_weights = cfg["train"].get("class_weights")
    class_weights = None
    if configured_class_weights is not None:
        class_weights = torch.tensor(
            [float(value) for value in configured_class_weights],
            dtype=torch.float32,
            device=device,
        )
        if class_weights.numel() != int(cfg["model"].get("num_classes", 2)):
            raise ValueError("train.class_weights must contain one value per class")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg["train"]["max_epochs"])
    )
    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    run_name = args.run_name or cfg["name"]
    out_dir = Path(paths["checkpoint_root"]) / run_name
    run_id = uuid.uuid4().hex
    history_path = Path(paths["log_root"]) / run_name / f"epoch_metrics_{run_id}.jsonl"
    write_metrics(
        Path(paths["log_root"]) / run_name / f"run_metadata_{run_id}.json",
        {
            "run_id": run_id,
            "run_name": run_name,
            "device": str(device),
            "config": str(args.config),
            "train_manifest": str(train_manifest),
            "val_manifest": str(val_manifest),
            "cache_dir": str(cache_dir),
            "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
            "balanced_sampler": use_balanced_sampler,
            "class_weights": None if class_weights is None else class_weights.detach().cpu().tolist(),
        },
    )
    best_f1 = float("-inf")
    bad_epochs = 0
    patience = int(cfg["train"].get("patience", 3))
    for epoch in range(1, int(cfg["train"]["max_epochs"]) + 1):
        model.train()
        correct = total = 0
        for batch in train_loader:
            wav = batch["wav"].to(device, non_blocking=True)
            enrollment = batch["enrollment_embedding"].to(device, non_blocking=True)
            extracted = batch["extracted_embedding"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(wav, enrollment, extracted)
            classification_loss = F.cross_entropy(logits.float(), labels, weight=class_weights)
            loss = classification_loss
            distill_loss = torch.zeros((), device=device)
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(wav, enrollment, extracted)
                student_log_probs = F.log_softmax(logits.float() / distill_temperature, dim=-1)
                teacher_probs = F.softmax(teacher_logits.float() / distill_temperature, dim=-1)
                distill_loss = F.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction="batchmean",
                ) * (distill_temperature**2)
                loss = (1.0 - distill_alpha) * classification_loss + distill_alpha * distill_loss
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
                optimizer.step()
            correct += int((logits.argmax(-1) == labels).sum())
            total += labels.numel()
        scheduler.step()
        val = evaluate(model, val_loader, device)
        metrics = {
            "epoch": epoch,
            "run_id": run_id,
            "train_accuracy": correct / max(total, 1),
            "val": val,
            "lr": scheduler.get_last_lr()[0],
            "best_f1": best_f1,
            "distill_alpha": distill_alpha,
            "distill_temperature": distill_temperature,
        }
        write_metrics(Path(paths["log_root"]) / run_name / "metrics_last.json", metrics)
        append_jsonl(history_path, metrics)
        print(
            f"epoch {epoch} train_acc={metrics['train_accuracy']:.3f} "
            f"val_acc={val['accuracy']:.3f} val_f1={val['f1']:.3f} "
            f"val_recall={val['recall']:.3f}",
            flush=True,
        )
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            bad_epochs = 0
            save_checkpoint(out_dir / "best.pt", model, epoch, metrics)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping after {epoch} epochs", flush=True)
                break
    write_metrics(
        Path(paths["log_root"]) / run_name / "train_last.json",
        {"best_f1": best_f1, "run_id": run_id, "history": str(history_path)},
    )
    print(f"Done. Best checkpoint: {out_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
