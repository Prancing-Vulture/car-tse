#!/usr/bin/env python3
"""Train and evaluate a frozen-wav2vec2 SLURP action baseline.

This is intentionally separate from the FSC/car pipeline.  It uses the
official SLURP action inventory, trains only a 46-class action head on the
official train split, selects on development, and reports the held-out test
score for all, distant, and close-talk recordings.  The same script can be
pointed at the derived speaker-disjoint manifests for the stress protocol.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModel

from car_tse.config import REPO_ROOT
from car_tse.data.datasets import IntentDataset
from car_tse.train.utils import append_jsonl, get_device, save_checkpoint, write_metrics
from train_stage4_intent_wav2vec import IntentHead, cache_features, evaluate


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class FeatureDataset(Dataset):
    def __init__(self, feature_path: Path, rows: list[dict], indices: list[int] | None = None):
        self.features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        self.indices = list(range(len(rows))) if indices is None else list(indices)
        self.labels = np.asarray([int(rows[index]["label"]) for index in self.indices], dtype=np.int64)
        if self.features.shape[0] != len(rows):
            raise ValueError(f"Feature/manifest length mismatch: {self.features.shape[0]} vs {len(rows)}")
        if any(label < 0 for label in self.labels):
            raise ValueError("SLURP manifest contains an action outside the real-train label space")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_index = self.indices[index]
        return {
            "features": torch.from_numpy(np.asarray(self.features[source_index], dtype=np.float32)),
            "label": torch.tensor(int(self.labels[index]), dtype=torch.long),
        }


def loader(dataset: Dataset, device: torch.device, batch_size: int, shuffle: bool = False, sampler=None) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    if sampler is not None:
        kwargs["sampler"] = sampler
    else:
        kwargs["shuffle"] = shuffle
    return DataLoader(dataset, **kwargs)


def subset_indices(rows: list[dict], microphone: str) -> list[int]:
    if microphone == "all":
        return list(range(len(rows)))
    return [index for index, row in enumerate(rows) if row.get("microphone") == microphone]


def evaluate_groups(
    model: torch.nn.Module,
    feature_path: Path,
    rows: list[dict],
    device: torch.device,
    batch_size: int,
    num_classes: int,
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for microphone in ("all", "distant", "close_talk"):
        indices = subset_indices(rows, microphone)
        if not indices:
            continue
        dataset = FeatureDataset(feature_path, rows, indices)
        output[microphone] = evaluate(model, loader(dataset, device, batch_size), device, num_classes)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, default=Path("D:/datasets/slurp/manifests"))
    parser.add_argument("--train-manifest", default="slurp_train.jsonl")
    parser.add_argument("--devel-manifest", default="slurp_devel.jsonl")
    parser.add_argument("--test-manifest", default="slurp_test.jsonl")
    parser.add_argument("--encoder", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--segment-seconds", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=16)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--head-hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--feature-dir", type=Path, default=Path("D:/datasets/slurp/features_wav2vec4s"))
    parser.add_argument("--run-name", default="slurp_action_wav2vec_official_all_2026-10-18")
    parser.add_argument("--checkpoint-root", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument("--log-root", type=Path, default=REPO_ROOT / "logs")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=20261018)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    manifest_root = args.manifest_root.resolve()
    train_path = manifest_root / args.train_manifest
    devel_path = manifest_root / args.devel_manifest
    test_path = manifest_root / args.test_manifest
    train_rows = read_jsonl(train_path)
    devel_rows = read_jsonl(devel_path)
    test_rows = read_jsonl(test_path)
    if not train_rows or not devel_rows or not test_rows:
        raise RuntimeError("train, devel, and test manifests must be non-empty")
    if any(int(row.get("label", -1)) < 0 for row in train_rows + devel_rows + test_rows):
        raise RuntimeError("all official evaluation rows must be in the real-train action label map")
    num_classes = max(int(row["label"]) for row in train_rows + devel_rows + test_rows) + 1

    feature_dir = args.feature_dir.resolve()
    feature_dir.mkdir(parents=True, exist_ok=True)
    dataset_kwargs = {
        "cabin_noise": False,
        "cabin_noise_cache_size": 0,
        "audio_cache_size": 64,
        "waveform_cache_is_noisy": True,
    }
    train_audio = IntentDataset(train_path, 16000, args.segment_seconds, training=False, **dataset_kwargs)
    devel_audio = IntentDataset(devel_path, 16000, args.segment_seconds, training=False, **dataset_kwargs)
    test_audio = IntentDataset(test_path, 16000, args.segment_seconds, training=False, **dataset_kwargs)
    if len(train_audio) != len(train_rows) or len(devel_audio) != len(devel_rows) or len(test_audio) != len(test_rows):
        raise ValueError("manifest and audio dataset lengths differ")

    encoder = AutoModel.from_pretrained(args.encoder).to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    encoder.eval()
    amp = bool(args.amp and device.type == "cuda")
    feature_paths = {
        "train": feature_dir / "train_features.npy",
        "devel": feature_dir / "devel_features.npy",
        "test": feature_dir / "test_features.npy",
    }
    cache_features(encoder, train_audio, feature_paths["train"], device, args.feature_batch_size, amp)
    cache_features(encoder, devel_audio, feature_paths["devel"], device, args.feature_batch_size, amp)
    cache_features(encoder, test_audio, feature_paths["test"], device, args.feature_batch_size, amp)
    hidden = int(encoder.config.hidden_size)

    train_dataset = FeatureDataset(feature_paths["train"], train_rows)
    devel_dataset = FeatureDataset(feature_paths["devel"], devel_rows)
    labels = torch.from_numpy(train_dataset.labels)
    counts = torch.bincount(labels, minlength=num_classes).float()
    sampler = WeightedRandomSampler(1.0 / counts[labels].clamp_min(1.0), num_samples=len(labels), replacement=True)
    train_loader = loader(train_dataset, device, args.batch_size, sampler=sampler)
    devel_loader = loader(devel_dataset, device, args.batch_size)

    model = IntentHead(hidden, args.head_hidden, num_classes, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    checkpoint_dir = (args.checkpoint_root / args.run_name).resolve()
    log_dir = (args.log_root / args.run_name).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    history_path = log_dir / f"epoch_metrics_{run_id}.jsonl"
    write_metrics(
        log_dir / f"run_metadata_{run_id}.json",
        {
            "run_id": run_id,
            "run_name": args.run_name,
            "device": str(device),
            "encoder": args.encoder,
            "manifest_root": str(manifest_root),
            "train_manifest": str(train_path),
            "devel_manifest": str(devel_path),
            "test_manifest": str(test_path),
            "segment_seconds": args.segment_seconds,
            "num_classes": num_classes,
            "feature_dir": str(feature_dir),
            "seed": args.seed,
        },
    )

    best_f1 = float("-inf")
    bad_epochs = 0
    print(
        f"device={device} train={len(train_dataset)} devel={len(devel_dataset)} "
        f"test={len(test_rows)} classes={num_classes} segment={args.segment_seconds}s",
        flush=True,
    )
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        total_loss = total_correct = total_examples = 0
        for batch in train_loader:
            features = batch["features"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = F.cross_entropy(logits, target, label_smoothing=0.03)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * target.numel()
            total_correct += int((logits.argmax(-1) == target).sum())
            total_examples += target.numel()
        scheduler.step()
        devel_metrics = evaluate(model, devel_loader, device, num_classes)
        metrics = {
            "epoch": epoch,
            "run_id": run_id,
            "device": str(device),
            "train_loss": total_loss / max(total_examples, 1),
            "train_accuracy": total_correct / max(total_examples, 1),
            "val": devel_metrics,
            "lr": scheduler.get_last_lr()[0],
        }
        append_jsonl(history_path, metrics)
        write_metrics(log_dir / "metrics_last.json", metrics)
        print(
            f"epoch {epoch} train_loss={metrics['train_loss']:.4f} "
            f"train_acc={metrics['train_accuracy']:.3f} val_acc={devel_metrics['accuracy']:.3f} "
            f"val_macro_f1={devel_metrics['macro_f1']:.3f}",
            flush=True,
        )
        if devel_metrics["macro_f1"] > best_f1:
            best_f1 = devel_metrics["macro_f1"]
            bad_epochs = 0
            metrics["best_macro_f1"] = best_f1
            save_checkpoint(checkpoint_dir / "best.pt", model, epoch, metrics)
            torch.save({"model_state_dict": encoder.state_dict(), "epoch": epoch}, checkpoint_dir / "encoder_best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs", flush=True)
                break

    best_state = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_state["model_state_dict"], strict=True)
    test_metrics = evaluate_groups(model, feature_paths["test"], test_rows, device, args.batch_size, num_classes)
    write_metrics(
        log_dir / "metrics_test.json",
        {
            "run_id": run_id,
            "best_epoch": best_state.get("epoch"),
            "best_devel_macro_f1": best_f1,
            "test_by_microphone": test_metrics,
            "test_manifest": str(test_path),
            "feature_dir": str(feature_dir),
        },
    )
    write_metrics(
        log_dir / "train_last.json",
        {"run_id": run_id, "best_macro_f1": best_f1, "history": str(history_path)},
    )
    print(json.dumps({"best_devel_macro_f1": best_f1, "test_by_microphone": test_metrics}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
