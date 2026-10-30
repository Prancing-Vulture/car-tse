#!/usr/bin/env python
"""Train a wav2vec intent head on clean and held-out-safe overlap features.

This is deliberately a head-only adaptation.  The wav2vec encoder stays frozen,
the clean feature cache is reused, and only the overlap waveform cache is
encoded once.  Validation is reported separately for clean FSC examples and
for the train-split passenger-mixture validation examples so a domain tradeoff
cannot be hidden by a single pooled number.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModel

from car_tse.config import REPO_ROOT
from car_tse.data.datasets import IntentDataset
from car_tse.train.utils import append_jsonl, get_device, save_checkpoint, write_metrics
from train_stage4_intent_wav2vec import IntentHead, cache_features, evaluate


class FeatureDataset(Dataset):
    def __init__(self, feature_path: Path, rows: list[dict]):
        self.features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        self.labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
        if self.features.shape[0] != len(self.labels):
            raise ValueError(
                f"Feature/manifest length mismatch for {feature_path}: "
                f"{self.features.shape[0]} vs {len(self.labels)}"
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "features": torch.from_numpy(np.asarray(self.features[idx], dtype=np.float32)),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metric_average(clean: dict[str, float], overlap: dict[str, float]) -> float:
    return 0.5 * (float(clean["macro_f1"]) + float(overlap["macro_f1"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--encoder-checkpoint", type=Path, default=None)
    parser.add_argument("--manifest-root", type=Path, default=REPO_ROOT / "manifests")
    parser.add_argument("--clean-train-manifest", default="train/stage4_intent_train.jsonl")
    parser.add_argument("--clean-val-manifest", default="val/stage4_intent_eval.jsonl")
    parser.add_argument(
        "--clean-feature-dir",
        type=Path,
        default=REPO_ROOT / "data/cache/stage4_intent_wav2vec_dual_domain_head_v1",
    )
    parser.add_argument(
        "--clean-wave-cache-dir",
        type=Path,
        default=REPO_ROOT / "data/cache/stage4_intent_16k",
    )
    parser.add_argument("--reencode-clean", action="store_true")
    parser.add_argument(
        "--overlap-train-manifest",
        default="train/stage4_intent_overlap_trainsplit_0db_2026-10-16.jsonl",
    )
    parser.add_argument(
        "--overlap-val-manifest",
        default="val/stage4_intent_overlap_trainsplit_0db_2026-10-16.jsonl",
    )
    parser.add_argument(
        "--overlap-wave-cache-dir",
        type=Path,
        default=REPO_ROOT / "data/cache/stage4_intent_overlap_trainsplit_0db_2026-10-16_16k",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=REPO_ROOT / "data/cache/stage4_intent_wav2vec_mixed_clean_overlap_2026-10-16",
    )
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "checkpoints/stage4_intent_wav2vec_mixed_clean_overlap_2026-10-16/best.pt")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "logs/stage4_intent_wav2vec_mixed_clean_overlap_2026-10-16.json")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--domain-balanced", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=16)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    manifest_root = args.manifest_root.resolve()
    clean_train_rows = read_jsonl(manifest_root / args.clean_train_manifest)
    clean_val_rows = read_jsonl(manifest_root / args.clean_val_manifest)
    overlap_train_rows = read_jsonl(manifest_root / args.overlap_train_manifest)
    overlap_val_rows = read_jsonl(manifest_root / args.overlap_val_manifest)
    if not clean_train_rows or not clean_val_rows or not overlap_train_rows or not overlap_val_rows:
        raise RuntimeError("all four clean/overlap manifests must be non-empty")

    args.clean_feature_dir = args.clean_feature_dir.resolve()
    args.clean_wave_cache_dir = args.clean_wave_cache_dir.resolve()
    args.overlap_wave_cache_dir = args.overlap_wave_cache_dir.resolve()
    args.feature_dir = args.feature_dir.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    args.feature_dir.mkdir(parents=True, exist_ok=True)

    overlap_common = {
        "cabin_noise": False,
        "cabin_noise_cache_size": 0,
        "audio_cache_size": 0,
        "waveform_cache_is_noisy": True,
    }
    overlap_train_audio = IntentDataset(
        manifest_root / args.overlap_train_manifest,
        16000,
        2.0,
        training=False,
        waveform_cache_path=args.overlap_wave_cache_dir / "train_overlap_wav_16000hz.npy",
        **overlap_common,
    )
    overlap_val_audio = IntentDataset(
        manifest_root / args.overlap_val_manifest,
        16000,
        2.0,
        training=False,
        waveform_cache_path=args.overlap_wave_cache_dir / "val_overlap_wav_16000hz.npy",
        **overlap_common,
    )
    if len(overlap_train_audio) != len(overlap_train_rows) or len(overlap_val_audio) != len(overlap_val_rows):
        raise ValueError("overlap waveform cache and manifests are not aligned")

    encoder = AutoModel.from_pretrained(args.encoder).to(device)
    if args.encoder_checkpoint is not None:
        encoder_checkpoint = args.encoder_checkpoint.resolve()
        loaded_encoder = torch.load(encoder_checkpoint, map_location="cpu", weights_only=False)
        encoder.load_state_dict(loaded_encoder["model_state_dict"], strict=True)
        print(
            f"initialized_encoder={encoder_checkpoint} epoch={loaded_encoder.get('epoch', 'unknown')}",
            flush=True,
        )
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    encoder.eval()
    hidden = int(encoder.config.hidden_size)
    if args.reencode_clean:
        clean_common = {
            "cabin_noise": False,
            "cabin_noise_cache_size": 0,
            "audio_cache_size": 0,
            "waveform_cache_is_noisy": True,
        }
        clean_train_audio = IntentDataset(
            manifest_root / args.clean_train_manifest,
            16000,
            2.0,
            training=False,
            waveform_cache_path=args.clean_wave_cache_dir / "train_wav_16000hz.npy",
            **clean_common,
        )
        clean_val_audio = IntentDataset(
            manifest_root / args.clean_val_manifest,
            16000,
            2.0,
            training=False,
            waveform_cache_path=args.clean_wave_cache_dir / "val_wav_16000hz.npy",
            **clean_common,
        )
        if len(clean_train_audio) != len(clean_train_rows) or len(clean_val_audio) != len(clean_val_rows):
            raise ValueError("clean waveform cache and manifests are not aligned")
        clean_train_features = args.feature_dir / "clean_train_features.npy"
        clean_val_features = args.feature_dir / "clean_val_features.npy"
        cache_features(
            encoder,
            clean_train_audio,
            clean_train_features,
            device,
            args.feature_batch_size,
            args.amp and device.type == "cuda",
        )
        cache_features(
            encoder,
            clean_val_audio,
            clean_val_features,
            device,
            args.feature_batch_size,
            args.amp and device.type == "cuda",
        )
    else:
        clean_train_features = args.clean_feature_dir / "train_features.npy"
        clean_val_features = args.clean_feature_dir / "val_features.npy"
        if not clean_train_features.exists() or not clean_val_features.exists():
            raise FileNotFoundError(
                "expected the existing clean feature cache at "
                f"{args.clean_feature_dir}; pass --reencode-clean to build a compatible cache"
            )
    overlap_train_features = args.feature_dir / "overlap_train_features.npy"
    overlap_val_features = args.feature_dir / "overlap_val_features.npy"
    cache_features(
        encoder,
        overlap_train_audio,
        overlap_train_features,
        device,
        args.feature_batch_size,
        args.amp and device.type == "cuda",
    )
    cache_features(
        encoder,
        overlap_val_audio,
        overlap_val_features,
        device,
        args.feature_batch_size,
        args.amp and device.type == "cuda",
    )

    clean_train = FeatureDataset(clean_train_features, clean_train_rows)
    clean_val = FeatureDataset(clean_val_features, clean_val_rows)
    overlap_train = FeatureDataset(overlap_train_features, overlap_train_rows)
    overlap_val = FeatureDataset(overlap_val_features, overlap_val_rows)
    train_ds = ConcatDataset([clean_train, overlap_train])
    labels = torch.cat([torch.from_numpy(clean_train.labels), torch.from_numpy(overlap_train.labels)])
    num_classes = int(max(labels.max().item(), max(int(row["label"]) for row in clean_val_rows), max(int(row["label"]) for row in overlap_val_rows)) + 1)
    counts = torch.bincount(labels, minlength=num_classes).float()
    if args.domain_balanced:
        # Sample each source domain equally.  A class-balanced sampler alone
        # can still let the larger clean corpus dominate the representation.
        domain_weights = torch.cat(
            [
                torch.full((len(clean_train),), 1.0 / max(len(clean_train), 1)),
                torch.full((len(overlap_train),), 1.0 / max(len(overlap_train), 1)),
            ]
        )
        sampler = WeightedRandomSampler(domain_weights, num_samples=len(labels), replacement=True)
    else:
        sampler = WeightedRandomSampler(
            1.0 / counts[labels].clamp_min(1.0),
            num_samples=len(labels),
            replacement=True,
        )
    loader_kwargs = {"num_workers": 0, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, **loader_kwargs)
    clean_val_loader = DataLoader(clean_val, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    overlap_val_loader = DataLoader(overlap_val, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = IntentHead(hidden, args.hidden, num_classes, args.dropout).to(device)
    if args.init_checkpoint is not None:
        init_checkpoint = args.init_checkpoint.resolve()
        loaded = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(loaded.get("model_state_dict", loaded), strict=True)
        print(f"initialized_head={init_checkpoint} epoch={loaded.get('epoch', 'unknown')}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    out_dir = args.checkpoint.parent
    log_dir = args.output.parent / args.output.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    history_path = log_dir / f"epoch_metrics_{run_id}.jsonl"
    metadata = {
        "run_id": run_id,
        "run_name": args.output.stem,
        "device": str(device),
        "encoder": args.encoder,
        "clean_train_manifest": str(manifest_root / args.clean_train_manifest),
        "clean_val_manifest": str(manifest_root / args.clean_val_manifest),
        "overlap_train_manifest": str(manifest_root / args.overlap_train_manifest),
        "overlap_val_manifest": str(manifest_root / args.overlap_val_manifest),
        "clean_feature_dir": str(args.clean_feature_dir),
        "clean_wave_cache_dir": str(args.clean_wave_cache_dir),
        "reencode_clean": bool(args.reencode_clean),
        "overlap_feature_dir": str(args.feature_dir),
        "train_examples": len(train_ds),
        "clean_val_examples": len(clean_val),
        "overlap_val_examples": len(overlap_val),
        "seed": args.seed,
        "init_checkpoint": str(args.init_checkpoint.resolve()) if args.init_checkpoint else None,
        "encoder_checkpoint": str(args.encoder_checkpoint.resolve()) if args.encoder_checkpoint else None,
        "domain_balanced": bool(args.domain_balanced),
    }
    write_metrics(log_dir / f"run_metadata_{run_id}.json", metadata)
    best_score = float("-inf")
    bad_epochs = 0
    print(
        f"device={device} hidden={hidden} train={len(train_ds)} clean_val={len(clean_val)} "
        f"overlap_val={len(overlap_val)} classes={num_classes}",
        flush=True,
    )
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_loss = train_correct = train_total = 0
        for batch in train_loader:
            features = batch["features"].to(device, non_blocking=True)
            labels_batch = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = F.cross_entropy(logits, labels_batch, label_smoothing=0.03)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += float(loss.detach()) * labels_batch.numel()
            train_correct += int((logits.argmax(-1) == labels_batch).sum())
            train_total += labels_batch.numel()
        scheduler.step()
        clean_metrics = evaluate(model, clean_val_loader, device, num_classes)
        overlap_metrics = evaluate(model, overlap_val_loader, device, num_classes)
        score = metric_average(clean_metrics, overlap_metrics)
        metrics = {
            "epoch": epoch,
            "run_id": run_id,
            "device": str(device),
            "encoder": args.encoder,
            "train_loss": train_loss / max(train_total, 1),
            "train_accuracy": train_correct / max(train_total, 1),
            "val": {
                "accuracy": 0.5 * (clean_metrics["accuracy"] + overlap_metrics["accuracy"]),
                "top5_accuracy": 0.5 * (clean_metrics["top5_accuracy"] + overlap_metrics["top5_accuracy"]),
                "macro_f1": score,
                "examples": clean_metrics["examples"] + overlap_metrics["examples"],
            },
            "val_clean": clean_metrics,
            "val_overlap": overlap_metrics,
            "balanced_selection_score": score,
            "lr": scheduler.get_last_lr()[0],
            "feature_dir": str(args.feature_dir),
        }
        write_metrics(log_dir / "metrics_last.json", metrics)
        append_jsonl(history_path, metrics)
        print(
            f"epoch {epoch} train_loss={metrics['train_loss']:.4f} train_acc={metrics['train_accuracy']:.3f} "
            f"clean_acc={clean_metrics['accuracy']:.3f} overlap_acc={overlap_metrics['accuracy']:.3f} "
            f"balanced_f1={score:.3f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            bad_epochs = 0
            metrics["best_balanced_selection_score"] = best_score
            save_checkpoint(out_dir / "best.pt", model, epoch, metrics)
            torch.save({"model_state_dict": encoder.state_dict(), "epoch": epoch}, out_dir / "encoder_best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs", flush=True)
                break
    write_metrics(
        log_dir / "train_last.json",
        {
            "best_balanced_selection_score": best_score,
            "epochs_completed": epoch,
            "run_id": run_id,
            "history": str(history_path),
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Done. Best checkpoint: {out_dir / 'best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
