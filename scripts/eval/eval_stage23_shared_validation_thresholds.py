#!/usr/bin/env python
"""Compare Stage 2/3 wake decision approaches on one shared labeled split.

This is a window-level validation analysis. It does not simulate temporal
confirmation or report a final held-out result. The score CSV is also used to
select fusion thresholds, so results must be described as validation metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from car_tse.models.stage23_fusion import Stage23FusionHead
from car_tse.train.utils import load_checkpoint


def metrics(y: torch.Tensor, pred: torch.Tensor) -> dict[str, float | int]:
    y = y.bool()
    pred = pred.bool()
    tp = int((y & pred).sum())
    fp = int((~y & pred).sum())
    fn = int((y & ~pred).sum())
    tn = int((~y & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def markdown_table(rows: list[dict], metric: str, thresholds: list[float]) -> str:
    approaches = ["Stage 2 score", "Stage 3 score", "Stage 2/3 fusion"]
    gate_presence_thresholds = (0.20, 0.35, 0.50, 0.70, 0.90)
    columns = approaches + [f"Hard gate p={value:.2f}" for value in gate_presence_thresholds]
    lines = [f"### {metric.replace('_', ' ').title()}", "", "| Wake-score threshold | " + " | ".join(columns) + " |", "|---:|" + "---:|" * len(columns)]
    for threshold in thresholds:
        cells = []
        for approach in approaches:
            found = next((r for r in rows if r["approach"] == approach and r["threshold"] == f"{threshold:.2f}"), None)
            cells.append("—" if found is None else f"{100 * float(found[metric]):.1f}%")
        for presence_threshold in gate_presence_thresholds:
            found = next((r for r in rows if r["approach"] == "Hard gate sweep"
                          and r["presence_threshold"] == presence_threshold
                          and r["wake_threshold"] == threshold), None)
            cells.append("—" if found is None else f"{100 * float(found[metric]):.1f}%")
        lines.append(f"| {threshold:.2f} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-csv", type=Path, required=True)
    parser.add_argument("--fusion-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "reports" / "evaluation")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()

    with args.scores_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    train_speakers = {row["speaker_id"] for row in rows if row["split"].lower() == "train"}
    val_rows = [row for row in rows if row["split"].lower() == "val"]
    val_speakers = {row["speaker_id"] for row in val_rows}
    if not val_rows or train_speakers & val_speakers:
        raise ValueError("Need validation rows and disjoint train/validation speakers")

    p = torch.tensor([float(row["presence_probability"]) for row in val_rows])
    w = torch.tensor([float(row["wake_probability"]) for row in val_rows])
    sim_text = [row["speaker_similarity"].strip() for row in val_rows]
    sim = torch.tensor([float(value) if value else 0.0 for value in sim_text])
    valid = torch.tensor([1.0 if value else 0.0 for value in sim_text])
    y = torch.tensor([int(row["label"]) for row in val_rows], dtype=torch.bool)
    if y.unique().numel() != 2:
        raise ValueError("Validation split must contain positive and negative windows")

    fusion = Stage23FusionHead(hidden=8)
    load_checkpoint(args.fusion_checkpoint, fusion)
    fusion.eval()
    with torch.no_grad():
        fused = torch.sigmoid(fusion(p, w, sim, valid))

    thresholds = sorted(set([index / 100 for index in range(5, 100, 5)] + [0.29]))
    output_rows: list[dict] = []

    def append(approach: str, threshold: str, result: dict, p_threshold=None, w_threshold=None):
        output_rows.append({
            "approach": approach,
            "threshold": threshold,
            "presence_threshold": p_threshold,
            "wake_threshold": w_threshold,
            **result,
            "examples": len(val_rows),
            "positive_examples": int(y.sum()),
            "negative_examples": int((~y).sum()),
            "split": "speaker-held-out validation",
        })

    # The currently configured hard rule: Stage 2 must pass and then Stage 3
    # must pass. Its two thresholds are held fixed in this row.
    append("Configured hard gate", "fixed", metrics(y, (p >= 0.90) & (w >= 0.29)), 0.90, 0.29)

    for threshold in thresholds:
        append("Stage 2 score", f"{threshold:.2f}", metrics(y, p >= threshold), threshold, None)
        append("Stage 3 score", f"{threshold:.2f}", metrics(y, w >= threshold), None, threshold)
        append("Stage 2/3 fusion", f"{threshold:.2f}", metrics(y, fused >= threshold), None, None)

    # Also sweep the two independent cutoffs of the simple gate. This makes
    # explicit that a gate is a two-threshold rule, not one scalar threshold.
    for p_threshold in (0.20, 0.35, 0.50, 0.70, 0.90):
        for w_threshold in thresholds:
            append(
                "Hard gate sweep",
                f"p={p_threshold:.2f};w={w_threshold:.2f}",
                metrics(y, (p >= p_threshold) & (w >= w_threshold)),
                p_threshold,
                w_threshold,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "stage23_apples_to_apples_validation_2026-10-12.csv"
    json_path = args.output_dir / "stage23_apples_to_apples_validation_2026-10-12.json"
    md_path = args.output_dir / "STAGE23_APPLES_TO_APPLES_VALIDATION_2026-10-12.md"
    fieldnames = list(output_rows[0])
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    best_by_approach = {}
    for approach in ("Stage 2 score", "Stage 3 score", "Stage 2/3 fusion", "Hard gate sweep"):
        candidates = [row for row in output_rows if row["approach"] == approach]
        best_by_approach[approach] = max(candidates, key=lambda row: (row["f1"], row["recall"]))
    summary = {
        "split": "speaker-held-out validation",
        "examples": len(val_rows),
        "positive_windows": int(y.sum()),
        "negative_windows": int((~y).sum()),
        "train_speakers": len(train_speakers),
        "validation_speakers": len(val_speakers),
        "speaker_overlap": sorted(train_speakers & val_speakers),
        "configured_hard_gate": next(row for row in output_rows if row["approach"] == "Configured hard gate"),
        "best_validation_f1_by_approach": best_by_approach,
        "warning": "Validation was used for fusion training checkpoint selection and threshold selection; these are not untouched test metrics. Window-level scores do not include temporal confirmation or event/hour behavior.",
        "csv": csv_path.name,
    }
    json_path.write_text(json.dumps({"summary": summary, "rows": output_rows}, indent=2), encoding="utf-8")

    metric_names = ["f1", "precision", "recall", "specificity", "accuracy", "false_positive_rate", "false_negative_rate"]
    thresholds_to_show = thresholds
    parts = [
        "# Stage 2/3 apples-to-apples validation comparison — 12 October 2026",
        "",
        f"**Shared set:** {len(val_rows)} speaker-held-out windows ({int(y.sum())} wake-positive, {int((~y).sum())} negative), scored once for each approach. Train/validation speaker overlap: {len(train_speakers & val_speakers)}.",
        "",
        "**Scope:** This is a window-level validation comparison, not a final test or event-level stream score. Validation was used to select the fusion checkpoint/threshold. The configured hard gate is Stage 2 probability ≥ 0.90 **and** Stage 3 wake probability ≥ 0.29. Stage 2-only, Stage 3-only, and Stage 2/3 fusion columns use the scalar threshold in the left column; each hard-gate column applies that row's wake threshold plus the named fixed Stage 2 threshold.",
        "",
        "The hard gate is a two-threshold rule, so its fixed operating point is reported separately. Its independent two-threshold sweep is present in the CSV/JSON for full inspection.",
        "",
        "| Fixed approach | Thresholds | TP | FP | FN | TN | F1 | Precision | Recall | Specificity | Accuracy | FPR | FNR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    gate = summary["configured_hard_gate"]
    parts.append("| Configured hard gate | presence 0.90; wake 0.29 | " + " | ".join(str(gate[k]) if k in ("tp", "fp", "fn", "tn") else f"{100*gate[k]:.1f}%" for k in ("tp", "fp", "fn", "tn", "f1", "precision", "recall", "specificity", "accuracy", "false_positive_rate", "false_negative_rate")) + " |")
    for metric in metric_names:
        parts += ["", markdown_table(output_rows, metric, thresholds_to_show)]
    parts += [
        "",
        "## Best validation operating points (descriptive only)",
        "",
        "| Approach | Selected validation threshold(s) | TP | FP | FN | TN | F1 | Precision | Recall | Specificity |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for approach, row in best_by_approach.items():
        parts.append(f"| {approach} | {row['threshold']} | {row['tp']} | {row['fp']} | {row['fn']} | {row['tn']} | {100*row['f1']:.1f}% | {100*row['precision']:.1f}% | {100*row['recall']:.1f}% | {100*row['specificity']:.1f}% |")
    parts += [
        "",
        "## Reproduction",
        "",
        "```powershell",
        "conda run -n audioml_base python scripts/eval/eval_stage23_shared_validation_thresholds.py `",
        "  --scores-csv E:/car-tse-data/logs/stage23_fusion/score_rows.csv `",
        "  --fusion-checkpoint checkpoints/stage23_fusion_candidate_lr02_20261012/best.pt `",
        "  --wandb-mode online",
        "```",
        "",
        "Source score rows were collected from continuous proxy windows and labeled positive when at least half of the Stage 3 crop overlapped the labeled wake. This is a repeatable shared-window comparison, but it does not measure temporal confirmation or full-pipeline intent/action accuracy.",
        "",
    ]
    md_path.write_text("\n".join(parts), encoding="utf-8")

    if args.wandb_mode != "disabled":
        import wandb
        run = wandb.init(
            project="car-tse",
            name="stage23-apples-to-apples-validation-20261012",
            job_type="evaluation",
            mode=args.wandb_mode,
            config={
                "split": "speaker-held-out validation",
                "examples": len(val_rows),
                "positives": int(y.sum()),
                "negatives": int((~y).sum()),
                "hard_gate_presence_threshold": 0.90,
                "hard_gate_wake_threshold": 0.29,
                "fusion_checkpoint": str(args.fusion_checkpoint),
                "thresholds": thresholds,
                "speaker_overlap": len(train_speakers & val_speakers),
            },
        )
        run.log({
            "configured_gate/f1": gate["f1"],
            "configured_gate/precision": gate["precision"],
            "configured_gate/recall": gate["recall"],
            "configured_gate/specificity": gate["specificity"],
            "configured_gate/accuracy": gate["accuracy"],
            "configured_gate/tp": gate["tp"],
            "configured_gate/fp": gate["fp"],
            "configured_gate/fn": gate["fn"],
            "configured_gate/tn": gate["tn"],
            **{f"best_val/{name}/f1": value["f1"] for name, value in best_by_approach.items()},
            "threshold_sweep": wandb.Table(data=[[
                r["approach"], r["threshold"], r["f1"], r["precision"], r["recall"],
                r["specificity"], r["accuracy"], r["false_positive_rate"], r["false_negative_rate"],
                r["tp"], r["fp"], r["fn"], r["tn"],
            ] for r in output_rows], columns=[
                "approach", "threshold", "f1", "precision", "recall", "specificity", "accuracy",
                "fpr", "fnr", "tp", "fp", "fn", "tn",
            ]),
        })
        artifact = wandb.Artifact("stage23-apples-to-apples-validation-20261012", type="evaluation")
        for path in (csv_path, json_path, md_path):
            artifact.add_file(str(path))
        run.log_artifact(artifact)
        run.finish()

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
