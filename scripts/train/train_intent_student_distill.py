#!/usr/bin/env python
"""Train a compact intent student with labels plus frozen Wav2Vec teacher logits.

The teacher logits are computed from the already cached Wav2Vec features, so
this experiment does not load the 95M-parameter encoder during every student
step.  The resulting checkpoint is still a candidate until the ordinary
intent benchmarks and the full pipeline proxy accept it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.datasets import IntentDataset
from car_tse.models.intent_classifier import IntentClassifier
from car_tse.models.wav2vec_intent import Wav2VecIntentHead
from car_tse.train.utils import get_device, load_checkpoint, save_checkpoint, write_metrics


class StudentDataset(Dataset):
    def __init__(self, audio: IntentDataset, teacher_logits_path: Path):
        self.audio = audio
        self.teacher_logits = np.load(teacher_logits_path, mmap_mode="r", allow_pickle=False)
        if self.teacher_logits.shape[0] != len(audio):
            raise ValueError(
                f"teacher/audio length mismatch: {self.teacher_logits.shape[0]} vs {len(audio)}"
            )

    def __len__(self) -> int:
        return len(self.audio)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.audio[index]
        item["teacher_logits"] = torch.from_numpy(
            np.asarray(self.teacher_logits[index], dtype=np.float32)
        )
        item["label"] = torch.tensor(int(self.audio.rows[index]["label"]), dtype=torch.long)
        return item


def make_teacher_logits(
    feature_path: Path,
    output_path: Path,
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
    num_classes: int,
) -> None:
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    if output_path.exists():
        try:
            cached = np.load(output_path, mmap_mode="r", allow_pickle=False)
            if tuple(cached.shape) == (len(features), num_classes):
                print(f"Using cached teacher logits: {output_path}", flush=True)
                return
        except (OSError, ValueError):
            pass
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    head = Wav2VecIntentHead(input_dim=int(features.shape[1]), hidden=512, num_classes=num_classes)
    head.load_state_dict(payload.get("model_state_dict", payload), strict=True)
    head.to(device).eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(features), num_classes),
    )
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(
                np.asarray(features[start : start + batch_size], dtype=np.float32)
            ).to(device, non_blocking=True)
            output[start : start + len(batch)] = head(batch).float().cpu().numpy()
            if start == 0 or start + len(batch) == len(features) or (start // batch_size + 1) % 20 == 0:
                print(f"teacher_logits: {min(start + len(batch), len(features))}/{len(features)}", flush=True)
    output.flush()
    del output


@torch.no_grad()
def evaluate(model: IntentClassifier, loader: DataLoader, device: torch.device, num_classes: int) -> dict[str, float]:
    model.eval()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    total = top5 = agreement = 0
    for batch in loader:
        wav = batch["wav"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        teacher = batch["teacher_logits"].to(device, non_blocking=True)
        logits = model(wav)
        predictions = logits.argmax(-1)
        teacher_predictions = teacher.argmax(-1)
        confusion.index_put_((labels.cpu(), predictions.cpu()), torch.ones_like(labels.cpu()), accumulate=True)
        total += labels.numel()
        top5 += int((logits.topk(min(5, num_classes), dim=-1).indices == labels[:, None]).any(dim=1).sum())
        agreement += int((predictions == teacher_predictions).sum())
    tp = confusion.diag().float()
    support = confusion.sum(dim=1).float()
    predicted = confusion.sum(dim=0).float()
    precision = tp / predicted.clamp_min(1.0)
    recall = tp / support.clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
    return {
        "accuracy": float(tp.sum().item() / max(total, 1)),
        "top5_accuracy": float(top5 / max(total, 1)),
        "macro_precision": float(precision.mean().item()),
        "macro_recall": float(recall.mean().item()),
        "macro_f1": float(f1.mean().item()),
        "teacher_agreement": float(agreement / max(total, 1)),
        "examples": float(total),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/stage4_intent_tse_domain_norm_student.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.yaml")
    parser.add_argument("--teacher-features", type=Path, default=REPO_ROOT / "data/cache/stage4_intent_wav2vec_tse_domain")
    parser.add_argument("--teacher-head-checkpoint", type=Path, default=REPO_ROOT / "checkpoints/stage4_intent_wav2vec_ft8_resume_tse_domain/best.pt")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--run-name", default="stage4_intent_tse_domain_distill_t3_a05")
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--kd-weight", type=float, default=0.5)
    parser.add_argument("--teacher-batch-size", type=int, default=1024)
    args = parser.parse_args()

    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    device = get_device()
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    label_map_path = Path(paths["manifest_root"]) / "meta" / cfg["label_map"]
    num_classes = len(json.loads(label_map_path.read_text(encoding="utf-8")))
    manifest_root = Path(paths["manifest_root"])
    cache_dir = Path(cfg["waveform_cache_dir"])
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    data_kwargs = {
        "cabin_noise": False,
        "cabin_noise_cache_size": 0,
        "audio_cache_size": 0,
        "waveform_cache_is_noisy": True,
    }
    train_audio = IntentDataset(
        manifest_root / "train" / cfg["manifest_train"],
        cfg["sample_rate"],
        cfg["segment_seconds"],
        training=False,
        waveform_cache_path=cache_dir / f"train_cabin_wav_{cfg['sample_rate']}hz.npy",
        **data_kwargs,
    )
    val_audio = IntentDataset(
        manifest_root / "val" / cfg["manifest_eval"],
        cfg["sample_rate"],
        cfg["segment_seconds"],
        training=False,
        waveform_cache_path=cache_dir / f"val_cabin_wav_{cfg['sample_rate']}hz.npy",
        **data_kwargs,
    )
    if len(train_audio) == 0 or len(val_audio) == 0:
        raise RuntimeError(f"Missing intent data: train={len(train_audio)} val={len(val_audio)}")

    teacher_dir = args.teacher_features if args.teacher_features.is_absolute() else REPO_ROOT / args.teacher_features
    train_teacher = teacher_dir / "train_teacher_logits_tse_domain.npy"
    val_teacher = teacher_dir / "val_teacher_logits_tse_domain.npy"
    teacher_head = args.teacher_head_checkpoint if args.teacher_head_checkpoint.is_absolute() else REPO_ROOT / args.teacher_head_checkpoint
    make_teacher_logits(teacher_dir / "train_features.npy", train_teacher, teacher_head, device, args.teacher_batch_size, num_classes)
    make_teacher_logits(teacher_dir / "val_features.npy", val_teacher, teacher_head, device, args.teacher_batch_size, num_classes)

    train_ds = StudentDataset(train_audio, train_teacher)
    val_ds = StudentDataset(val_audio, val_teacher)
    labels = torch.tensor([int(row["label"]) for row in train_audio.rows], dtype=torch.long)
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = 1.0 / counts[labels].clamp_min(1.0)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    loader_kwargs = {"num_workers": 0, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_ds, batch_size=int(cfg["train"]["batch_size"]), sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["train"].get("val_batch_size", cfg["train"]["batch_size"])), shuffle=False, **loader_kwargs)

    model = IntentClassifier(
        cfg["sample_rate"],
        cfg["model"]["n_mels"],
        cfg["model"]["hidden"],
        cfg["model"]["num_layers"],
        num_classes,
        bool(cfg["model"].get("normalize_input", False)),
    ).to(device)
    if args.init_checkpoint:
        init_checkpoint = args.init_checkpoint if args.init_checkpoint.is_absolute() else REPO_ROOT / args.init_checkpoint
        load_checkpoint(init_checkpoint, model)
        print(f"initialized_student={init_checkpoint}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=1.0e-4)
    max_epochs = int(cfg["train"]["max_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    temperature = max(float(args.temperature), 1.0)
    kd_weight = min(max(float(args.kd_weight), 0.0), 1.0)
    out_dir = Path(paths["checkpoint_root"]) / args.run_name
    log_dir = Path(paths["log_root"]) / args.run_name
    best_f1 = float("-inf")
    bad_epochs = 0
    patience = int(cfg["train"].get("patience", 3))
    print(f"device={device} train={len(train_ds)} val={len(val_ds)} classes={num_classes} T={temperature} kd_weight={kd_weight}", flush=True)
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = train_correct = train_total = 0
        for batch in train_loader:
            wav = batch["wav"].to(device, non_blocking=True)
            labels_batch = batch["label"].to(device, non_blocking=True)
            teacher = batch["teacher_logits"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(wav)
            ce = F.cross_entropy(logits.float(), labels_batch, label_smoothing=0.03)
            kd = F.kl_div(
                F.log_softmax(logits.float() / temperature, dim=-1),
                F.softmax(teacher.float() / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature * temperature)
            loss = (1.0 - kd_weight) * ce + kd_weight * kd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
            optimizer.step()
            train_loss += float(loss.detach()) * labels_batch.numel()
            train_correct += int((logits.argmax(-1) == labels_batch).sum())
            train_total += labels_batch.numel()
        scheduler.step()
        val = evaluate(model, val_loader, device, num_classes)
        metrics = {
            "epoch": epoch,
            "device": str(device),
            "train_loss": train_loss / max(train_total, 1),
            "train_accuracy": train_correct / max(train_total, 1),
            "val": val,
            "lr": scheduler.get_last_lr()[0],
            "temperature": temperature,
            "kd_weight": kd_weight,
            "teacher_head": str(teacher_head),
        }
        write_metrics(log_dir / "metrics_last.json", metrics)
        print(
            f"epoch {epoch} train_loss={metrics['train_loss']:.4f} train_acc={metrics['train_accuracy']:.3f} "
            f"val_acc={val['accuracy']:.3f} val_top5={val['top5_accuracy']:.3f} "
            f"val_macro_f1={val['macro_f1']:.3f} teacher_agree={val['teacher_agreement']:.3f}",
            flush=True,
        )
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            bad_epochs = 0
            metrics["best_macro_f1"] = best_f1
            save_checkpoint(out_dir / "best.pt", model, epoch, metrics)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping after {epoch} epochs", flush=True)
                break
    write_metrics(log_dir / "train_last.json", {"best_macro_f1": best_f1, "epochs_completed": epoch})
    print(f"Done. Best checkpoint: {out_dir / 'best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
