#!/usr/bin/env python
"""Train a small Stage 2/3 score-fusion head from labeled runtime windows.

CSV columns: presence_probability,wake_probability,speaker_similarity,label,split
`split` must be `train` or `val`; create splits by speaker/recording before use.
Blank speaker_similarity is allowed and is handled with an explicit mask.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.models.stage23_fusion import Stage23FusionHead
from car_tse.train.utils import get_device, save_checkpoint, write_metrics


def read_rows(path: Path) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    by_split: dict[str, list[tuple[list[float], float]]] = {"train": [], "val": []}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"presence_probability", "wake_probability", "speaker_similarity", "label", "split"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")
        for line, row in enumerate(reader, start=2):
            split = row["split"].strip().lower()
            if split not in by_split:
                raise ValueError(f"row {line}: split must be train or val")
            try:
                presence = float(row["presence_probability"])
                wake = float(row["wake_probability"])
                label = int(row["label"])
                similarity_text = row["speaker_similarity"].strip()
                similarity = float(similarity_text) if similarity_text else 0.0
            except (TypeError, ValueError) as exc:
                raise ValueError(f"row {line}: invalid score or label") from exc
            if label not in (0, 1) or not 0.0 <= presence <= 1.0 or not 0.0 <= wake <= 1.0:
                raise ValueError(f"row {line}: expected label 0/1 and probabilities in [0,1]")
            if similarity_text and not -1.0 <= similarity <= 1.0:
                raise ValueError(f"row {line}: speaker similarity must be in [-1,1]")
            by_split[split].append(([presence, wake, similarity, float(bool(similarity_text))], float(label)))
    result = {}
    for split, rows in by_split.items():
        if not rows:
            raise ValueError(f"CSV has no {split} rows")
        x = torch.tensor([item[0] for item in rows], dtype=torch.float32)
        y = torch.tensor([item[1] for item in rows], dtype=torch.float32)
        if y.unique().numel() != 2:
            raise ValueError(f"{split} split must contain positive and negative wake windows")
        result[split] = (x, y)
    return result


@torch.no_grad()
def evaluate(model: Stage23FusionHead, x: torch.Tensor, y: torch.Tensor, threshold: float) -> dict[str, float]:
    probability = torch.sigmoid(model.net(x).squeeze(-1))
    prediction = probability >= threshold
    truth = y.bool()
    tp = int((prediction & truth).sum())
    fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "f1": 2 * precision * recall / max(precision + recall, 1.0e-8),
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fp / max(int((~truth).sum()), 1),
        "examples": float(y.numel()),
        "true_positive": float(tp),
        "false_positive": float(fp),
        "false_negative": float(fn),
    }


@torch.no_grad()
def best_threshold(model: Stage23FusionHead, x: torch.Tensor, y: torch.Tensor) -> tuple[float, dict[str, float]]:
    candidates = [index / 100.0 for index in range(5, 100, 5)]
    measurements = [(threshold, evaluate(model, x, y, threshold)) for threshold in candidates]
    return max(measurements, key=lambda item: (item[1]["f1"], item[1]["recall"], item[0]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--wandb-project", default="car-tse")
    parser.add_argument("--run-name", default="stage23-score-fusion")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    data = read_rows(args.scores_csv)
    device = get_device()
    model = Stage23FusionHead().to(device)
    train_x, train_y = data["train"]
    val_x, val_y = (tensor.to(device) for tensor in data["val"])
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    pos = max(float(train_y.sum()), 1.0)
    neg = max(float(train_y.numel() - train_y.sum()), 1.0)
    positive_weight = torch.tensor([neg / pos], device=device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run = None
    if not args.no_wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            job_type="stage23-fusion-training",
            config={"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                    "threshold": args.threshold, "train_rows": len(train_y), "val_rows": len(val_y),
                    "score_columns": ["presence_probability", "wake_probability", "speaker_similarity", "speaker_valid"]},
            mode=args.wandb_mode,
        )

    best_f1 = -1.0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = seen = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            p, w, sim, valid = batch_x.unbind(dim=-1)
            logits = model(p, w, sim, valid)
            loss = F.binary_cross_entropy_with_logits(logits, batch_y, pos_weight=positive_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * batch_y.numel()
            seen += batch_y.numel()
        val = evaluate(model, val_x, val_y, args.threshold)
        tuned_threshold, tuned_val = best_threshold(model, val_x, val_y)
        metrics = {
            "epoch": epoch,
            "train_loss": loss_sum / max(seen, 1),
            "val": val,
            "val_best_threshold": tuned_threshold,
            "val_at_best_threshold": tuned_val,
        }
        history.append(metrics)
        write_metrics(args.output_dir / "metrics_last.json", metrics)
        if run is not None:
            run.log({
                "epoch": epoch,
                "train/loss": metrics["train_loss"],
                **{f"val_at_configured_threshold/{k}": v for k, v in val.items()},
                "val/tuned_threshold": tuned_threshold,
                **{f"val_at_tuned_threshold/{k}": v for k, v in tuned_val.items()},
            })
        print(
            f"epoch={epoch} train_loss={metrics['train_loss']:.4f} "
            f"val_f1@{args.threshold:.2f}={val['f1']:.4f} "
            f"val_best_f1={tuned_val['f1']:.4f}@{tuned_threshold:.2f}",
            flush=True,
        )
        if tuned_val["f1"] > best_f1:
            best_f1 = tuned_val["f1"]
            stale = 0
            save_checkpoint(args.output_dir / "best.pt", model, epoch, metrics)
        else:
            stale += 1
            if stale >= args.patience:
                break
    write_metrics(args.output_dir / "metrics_history.json", {"epochs": history, "best_val_f1": best_f1})
    best_metrics = history[max(range(len(history)), key=lambda i: history[i]["val_at_best_threshold"]["f1"])]
    if run is not None:
        artifact = wandb.Artifact(args.run_name, type="model")
        artifact.add_file(str(args.output_dir / "best.pt"), name="best.pt")
        run.log_artifact(artifact)
        run.summary["best_val_f1"] = best_f1
        run.summary["recommended_threshold"] = best_metrics["val_best_threshold"]
        run.finish()
    print(
        f"checkpoint={args.output_dir / 'best.pt'} best_val_f1={best_f1:.4f} "
        f"threshold={best_metrics['val_best_threshold']:.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
