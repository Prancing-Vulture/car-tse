#!/usr/bin/env python
"""Train a compact speaker verifier from frozen WavLM teacher scores."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from car_tse.models.speaker_verifier import CompactSpeakerVerifier  # noqa: E402
from car_tse.train.utils import load_checkpoint, save_checkpoint  # noqa: E402
from scripts.train.train_wavlm_speaker_verifier_head import SpeakerVerifierHead  # noqa: E402
from scripts.eval.calibrate_spex_speaker_verifier import (  # noqa: E402
    choose_threshold,
    roc_summary,
    threshold_metrics,
)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


class WavePairDataset(Dataset):
    def __init__(
        self,
        enrollment: np.ndarray,
        extracted: np.ndarray,
        labels: np.ndarray,
        teacher_probabilities: np.ndarray,
    ) -> None:
        self.enrollment = enrollment
        self.extracted = extracted
        self.labels = labels
        self.teacher_probabilities = teacher_probabilities

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(np.array(self.enrollment[index], dtype=np.float32, copy=True)),
            torch.from_numpy(np.array(self.extracted[index], dtype=np.float32, copy=True)),
            torch.tensor(float(self.labels[index]), dtype=torch.float32),
            torch.tensor(float(self.teacher_probabilities[index]), dtype=torch.float32),
        )


def read_labels(manifest_root: Path, template: str, split: str, expected: int) -> np.ndarray:
    path = manifest_root / split / template.format(split=split)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels = np.asarray([int(row.get("label", 0)) for row in rows], dtype=np.float32)
    if len(labels) != expected:
        raise ValueError(f"{split}: expected {expected} labels, found {len(labels)}")
    return labels


@torch.no_grad()
def teacher_probabilities(
    teacher: SpeakerVerifierHead,
    enrollment: np.ndarray,
    extracted: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    teacher.eval()
    values: list[np.ndarray] = []
    for start in range(0, len(enrollment), 512):
        logits = teacher(
            torch.from_numpy(
                np.asarray(enrollment[start : start + 512], dtype=np.float32).copy()
            ).to(device),
            torch.from_numpy(
                np.asarray(extracted[start : start + 512], dtype=np.float32).copy()
            ).to(device),
        )
        values.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(values) if values else np.zeros(0, dtype=np.float32)


@torch.no_grad()
def evaluate(model: CompactSpeakerVerifier, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    for enrollment, extracted, _labels, _teacher in loader:
        logits, _enrollment, _extracted = model(enrollment.to(device), extracted.to(device))
        values.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(values) if values else np.zeros(0, dtype=np.float32)


def split_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float | None = None) -> dict[str, Any]:
    output: dict[str, Any] = {
        "records": int(len(labels)),
        "positive_count": int((labels == 1).sum()),
        "negative_count": int((labels == 0).sum()),
        "roc": roc_summary(scores, labels),
    }
    if threshold is not None:
        output["at_threshold"] = threshold_metrics(scores, labels, threshold)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wave-cache-dir",
        type=Path,
        default=REPO_ROOT / "data/cache/compact_speaker_verifier_gsc_marvin_2026-10-11",
    )
    parser.add_argument(
        "--extracted-cache-dir",
        type=Path,
        default=REPO_ROOT / "data/cache/stage3_target_kws_gsc_marvin_ft2_stage1_16k",
    )
    parser.add_argument(
        "--teacher-cache-dir",
        type=Path,
        default=REPO_ROOT / "data/cache/wavlm_speaker_verifier_gsc_marvin_2026-10-11",
    )
    parser.add_argument("--manifest-root", type=Path, default=REPO_ROOT / "manifests")
    parser.add_argument(
        "--manifest-template",
        default="stage3_target_kws_gsc_marvin_ft2_stage1_{split}.jsonl",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints/speaker_verifier_wavlm_cabin_ft1/best.pt",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints/compact_speaker_verifier_gsc_ft1/best.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "logs/compact_speaker_verifier_gsc_ft1_2026-10-11.json",
    )
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--distill-weight", type=float, default=0.25)
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
    wave_cache_dir = resolve(args.wave_cache_dir)
    extracted_cache_dir = resolve(args.extracted_cache_dir)
    teacher_cache_dir = resolve(args.teacher_cache_dir)
    manifest_root = resolve(args.manifest_root)

    teacher = SpeakerVerifierHead(embedding_dim=512, hidden=128).to(device)
    load_checkpoint(resolve(args.teacher_checkpoint), teacher)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    datasets: dict[str, WavePairDataset] = {}
    raw: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for split in ("train", "val", "test"):
        enrollment = np.load(wave_cache_dir / f"{split}_enrollment_wav.npy", mmap_mode="r", allow_pickle=False)
        extracted = np.load(extracted_cache_dir / f"{split}_wav_16000hz.npy", mmap_mode="r", allow_pickle=False)
        teacher_enrollment = np.load(teacher_cache_dir / f"{split}_enrollment_embedding.npy", mmap_mode="r", allow_pickle=False)
        teacher_extracted = np.load(teacher_cache_dir / f"{split}_extracted_embedding.npy", mmap_mode="r", allow_pickle=False)
        labels = read_labels(manifest_root, str(args.manifest_template), split, len(enrollment))
        probabilities = teacher_probabilities(teacher, teacher_enrollment, teacher_extracted, device)
        raw[split] = (enrollment, extracted, labels)
        datasets[split] = WavePairDataset(enrollment, extracted, labels, probabilities)

    train_loader = DataLoader(datasets["train"], batch_size=int(args.batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(datasets["val"], batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    test_loader = DataLoader(datasets["test"], batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    model = CompactSpeakerVerifier(
        embedding_dim=int(args.embedding_dim),
        hidden=int(args.hidden),
        sample_rate=16000,
        n_mels=40,
        n_fft=400,
        hop=160,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1.0e-4)
    best_auc = float("-inf")
    best_epoch = 0
    bad_epochs = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: list[float] = []
        for enrollment, extracted, labels, teacher_prob in train_loader:
            logits, student_enrollment, student_extracted = model(
                enrollment.to(device), extracted.to(device)
            )
            classification = F.binary_cross_entropy_with_logits(logits, labels.to(device))
            student_cosine = F.cosine_similarity(student_enrollment, student_extracted, dim=-1)
            teacher_logit = torch.logit(teacher_prob.to(device).clamp(1.0e-4, 1.0 - 1.0e-4))
            distillation = F.mse_loss(logits, teacher_logit) + 0.25 * F.mse_loss(
                student_cosine, (teacher_prob.to(device) * 2.0 - 1.0)
            )
            loss = classification + float(args.distill_weight) * distillation
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_scores = evaluate(model, val_loader, device)
        val_labels = raw["val"][2]
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
        raise RuntimeError("No compact verifier checkpoint was selected")
    model.load_state_dict(best_state)
    val_scores = evaluate(model, val_loader, device)
    test_scores = evaluate(model, test_loader, device)
    val_labels = raw["val"][2]
    test_labels = raw["test"][2]
    selected = choose_threshold(val_scores, val_labels, float(args.target_fpr))
    test_at_selected = threshold_metrics(test_scores, test_labels, selected["threshold"])
    metrics: dict[str, Any] = {
        "benchmark": "compact_speaker_verifier_distilled_from_wavlm",
        "device": str(device),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "embedding_dim": int(args.embedding_dim),
        "hidden": int(args.hidden),
        "training": {
            "best_epoch": int(best_epoch),
            "best_validation_auc": float(best_auc),
            "epochs_requested": int(args.epochs),
            "patience": int(args.patience),
            "distill_weight": float(args.distill_weight),
            "seed": int(args.seed),
        },
        "validation_operating_point": {"target_fpr": float(args.target_fpr), **selected},
        "test_at_validation_operating_point": test_at_selected,
        "splits": {
            "val": split_metrics(val_scores, val_labels),
            "test": split_metrics(test_scores, test_labels),
        },
        "data_contract": {
            "teacher": "frozen WavLM relation head",
            "source": "synthetic CAVEMOVE/GSC-Marvin cabin mixtures",
            "real_driver_evidence": False,
            "real_cabin_evidence": False,
            "production_pipeline_changed": False,
        },
        "decision": "diagnostic_candidate_only; requires real-driver data and edge benchmark",
        "history": history,
    }
    checkpoint = resolve(args.checkpoint)
    save_checkpoint(checkpoint, model, best_epoch, metrics)
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "output": str(output),
                "parameters": metrics["parameter_count"],
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
