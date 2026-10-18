#!/usr/bin/env python
"""Train a small speaker-verifier head on frozen WavLM pair embeddings.

This is a research candidate, not a production replacement.  WavLM remains
frozen and is used only to generate the cached embeddings; the trainable head
learns a relation between enrollment and extracted embeddings.  Train,
validation, and test arrays are kept separate, and the test threshold is never
used for model selection.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from car_tse.train.utils import save_checkpoint  # noqa: E402
from scripts.eval.calibrate_spex_speaker_verifier import (  # noqa: E402
    choose_threshold,
    roc_summary,
    threshold_metrics,
)


class SpeakerVerifierHead(nn.Module):
    """Compact relation classifier over frozen speaker embeddings."""

    def __init__(self, embedding_dim: int, hidden: int = 128) -> None:
        super().__init__()
        relation_dim = int(embedding_dim) * 4 + 1
        self.embedding_dim = int(embedding_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(relation_dim),
            nn.Linear(relation_dim, int(hidden)),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, enrollment: torch.Tensor, extracted: torch.Tensor) -> torch.Tensor:
        enrollment = torch.nn.functional.normalize(enrollment.float(), dim=-1)
        extracted = torch.nn.functional.normalize(extracted.float(), dim=-1)
        relation = torch.cat(
            [
                enrollment,
                extracted,
                torch.abs(enrollment - extracted),
                enrollment * extracted,
                torch.nn.functional.cosine_similarity(
                    enrollment, extracted, dim=-1
                ).unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.net(relation).squeeze(-1)


def load_split(cache_dir: Path, manifest_root: Path, template: str, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    enrollment = np.load(cache_dir / f"{split}_enrollment_embedding.npy", mmap_mode="r")
    extracted = np.load(cache_dir / f"{split}_extracted_embedding.npy", mmap_mode="r")
    manifest = manifest_root / split / template.format(split=split)
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != len(enrollment) or len(rows) != len(extracted):
        raise ValueError(f"{split}: embedding/manifest lengths do not match")
    labels = np.asarray([int(row.get("label", 0)) for row in rows], dtype=np.float32)
    return np.asarray(enrollment, dtype=np.float32), np.asarray(extracted, dtype=np.float32), labels


@torch.no_grad()
def predict(model: nn.Module, enrollment: torch.Tensor, extracted: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    for start in range(0, len(enrollment), 512):
        logits = model(
            enrollment[start : start + 512].to(device),
            extracted[start : start + 512].to(device),
        )
        values.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(values) if values else np.zeros(0, dtype=np.float32)


def split_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "records": int(len(labels)),
        "positive_count": int((labels == 1).sum()),
        "negative_count": int((labels == 0).sum()),
        "roc": roc_summary(scores, labels),
    }
    if threshold is not None:
        result["at_threshold"] = threshold_metrics(scores, labels, threshold)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "data/cache/wavlm_speaker_verifier_gsc_marvin_2026-10-11",
    )
    parser.add_argument("--manifest-root", type=Path, default=REPO_ROOT / "manifests")
    parser.add_argument(
        "--manifest-template",
        default="stage3_target_kws_gsc_marvin_ft2_stage1_{split}.jsonl",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints/speaker_verifier_wavlm_cabin_ft1/best.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "logs/speaker_verifier_wavlm_cabin_ft1_2026-10-11.json",
    )
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    cache_dir = args.cache_dir if args.cache_dir.is_absolute() else REPO_ROOT / args.cache_dir
    manifest_root = args.manifest_root if args.manifest_root.is_absolute() else REPO_ROOT / args.manifest_root
    train_enroll, train_extract, train_labels = load_split(cache_dir, manifest_root, args.manifest_template, "train")
    val_enroll, val_extract, val_labels = load_split(cache_dir, manifest_root, args.manifest_template, "val")
    test_enroll, test_extract, test_labels = load_split(cache_dir, manifest_root, args.manifest_template, "test")

    train_ds = TensorDataset(
        torch.from_numpy(train_enroll.copy()),
        torch.from_numpy(train_extract.copy()),
        torch.from_numpy(train_labels.copy()),
    )
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True)
    model = SpeakerVerifierHead(train_enroll.shape[-1], hidden=int(args.hidden)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_fn = nn.BCEWithLogitsLoss()
    best_auc = float("-inf")
    best_epoch = 0
    bad_epochs = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: list[float] = []
        for enrollment, extracted, labels in train_loader:
            logits = model(enrollment.to(device), extracted.to(device))
            loss = loss_fn(logits, labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_scores = predict(model, torch.from_numpy(val_enroll.copy()), torch.from_numpy(val_extract.copy()), device)
        val_auc = float(roc_summary(val_scores, val_labels)["auc"])
        history.append({"epoch": float(epoch), "train_loss": float(np.mean(losses)), "val_auc": val_auc})
        print(f"epoch={epoch} train_loss={np.mean(losses):.4f} val_auc={val_auc:.4f}", flush=True)
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            bad_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= int(args.patience):
                print(f"early_stop epoch={epoch}", flush=True)
                break
    if best_state is None:
        raise RuntimeError("No verifier head checkpoint was selected")
    model.load_state_dict(best_state)
    val_scores = predict(model, torch.from_numpy(val_enroll.copy()), torch.from_numpy(val_extract.copy()), device)
    test_scores = predict(model, torch.from_numpy(test_enroll.copy()), torch.from_numpy(test_extract.copy()), device)
    selected = choose_threshold(val_scores, val_labels, float(args.target_fpr))
    test_at_selected = threshold_metrics(test_scores, test_labels, selected["threshold"])
    metrics = {
        "benchmark": "wavlm_relation_speaker_verifier_head",
        "device": str(device),
        "cache_dir": str(cache_dir),
        "manifest_template": str(args.manifest_template),
        "embedding_dim": int(train_enroll.shape[-1]),
        "hidden": int(args.hidden),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "training": {
            "best_epoch": int(best_epoch),
            "best_validation_auc": float(best_auc),
            "epochs_requested": int(args.epochs),
            "patience": int(args.patience),
            "seed": int(args.seed),
        },
        "validation_operating_point": {"target_fpr": float(args.target_fpr), **selected},
        "test_at_validation_operating_point": test_at_selected,
        "splits": {
            "train": split_metrics(predict(model, torch.from_numpy(train_enroll.copy()), torch.from_numpy(train_extract.copy()), device), train_labels),
            "val": split_metrics(val_scores, val_labels),
            "test": split_metrics(test_scores, test_labels),
        },
        "history": history,
        "data_contract": {
            "real_driver_evidence": False,
            "real_cabin_evidence": False,
            "wavlm_backbone_frozen": True,
            "production_pipeline_changed": False,
        },
        "decision": "diagnostic_candidate_only; requires real-driver tuning, held-out test, and edge resource audit",
    }
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else REPO_ROOT / args.checkpoint
    save_checkpoint(checkpoint, model, best_epoch, metrics)
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "output": str(output),
                "best_val_auc": best_auc,
                "test_auc": metrics["splits"]["test"]["roc"]["auc"],
                "test_tpr": test_at_selected["tpr"],
                "test_fpr": test_at_selected["fpr"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
