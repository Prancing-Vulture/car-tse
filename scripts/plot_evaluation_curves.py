#!/usr/bin/env python
"""Export publication-resolution ROC/PR or intent confusion plots from eval JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def add_binary_curves(axes: Any, name: str, roc: dict, pr: dict, color: str) -> None:
    axes[0].plot(
        roc["false_positive_rate"], roc["true_positive_rate"],
        label=name, linewidth=1.8, color=color,
    )
    axes[1].plot(pr["recall"], pr["precision"], label=name, linewidth=1.8, color=color)


def make_binary_figure(metrics: dict[str, Any], run_name: str, output_dir: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "savefig.dpi": 300})
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    if "roc_curve" in metrics and "pr_curve" in metrics:
        add_binary_curves(axes, "Wake detector", metrics["roc_curve"], metrics["pr_curve"], "#2563eb")
        if "roc_auc" in metrics:
            axes[0].text(0.98, 0.04, f"AUROC = {metrics['roc_auc']:.3f}", transform=axes[0].transAxes, ha="right")
        if "average_precision" in metrics:
            axes[1].text(0.98, 0.04, f"AP = {metrics['average_precision']:.3f}", transform=axes[1].transAxes, ha="right")
    elif "frame_threshold_sweep" in metrics and "chunk_threshold_sweep" in metrics:
        colors = {"frame": "#2563eb", "chunk": "#ea580c"}
        for scope in ("frame", "chunk"):
            points = metrics[f"{scope}_threshold_sweep"]
            roc = {
                "false_positive_rate": [point["false_accept_rate"] for point in points],
                "true_positive_rate": [point["recall"] for point in points],
            }
            pr = {
                "recall": [point["recall"] for point in points],
                "precision": [point["precision"] for point in points],
            }
            add_binary_curves(axes, scope.title(), roc, pr, colors[scope])
    else:
        raise ValueError("Input does not contain Stage 2 sweeps or Stage 3 ROC/PR arrays")

    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1, label="Chance")
    axes[0].set(xlabel="False-accept rate", ylabel="Recall / true-positive rate", title="ROC curve", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall curve", xlim=(0, 1), ylim=(0, 1))
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle(run_name)
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"{run_name}_evaluation_curves.{suffix}", bbox_inches="tight")
    plt.close(figure)


def make_intent_figure(metrics: dict[str, Any], run_name: str, output_dir: Path) -> None:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Intent confusion matrix must be square")
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    class_names = metrics.get("class_names") or [str(index) for index in range(matrix.shape[0])]
    figure, axes = plt.subplots(1, 2, figsize=(max(11, 0.43 * matrix.shape[0]), 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
    image = axes[0].imshow(normalized, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axes[0].set_title("Row-normalized confusion matrix")
    axes[0].set_xlabel("Predicted intent")
    axes[0].set_ylabel("True intent")
    axes[0].set_xticks(range(matrix.shape[0]))
    axes[0].set_yticks(range(matrix.shape[0]))
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = normalized[row, column]
            axes[0].text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=6,
                         color="white" if value > 0.55 else "black")
    figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    class_f1 = metrics.get("per_class_f1")
    if class_f1 is not None:
        axes[1].bar(np.arange(len(class_f1)), class_f1, color="#2563eb")
        axes[1].set_title("Per-intent F1")
        axes[1].set_xlabel("Intent class")
        axes[1].set_ylabel("F1")
        axes[1].set_ylim(0, 1)
        axes[1].set_xticks(np.arange(len(class_f1)))
        axes[1].set_xticklabels(class_names, rotation=90, fontsize=6)
        axes[1].grid(axis="y", alpha=0.25)
    else:
        axes[1].remove()
    figure.suptitle(run_name)
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"{run_name}_intent_confusion.{suffix}", bbox_inches="tight")
    plt.close(figure)


def make_stage1_distribution(metrics: dict[str, Any], run_name: str, output_dir: Path) -> None:
    values = np.asarray(metrics["per_example_si_sdri_db"], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Stage 1 input contains no finite per-example SI-SDRi values")
    summary = metrics.get("metrics", {})
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    axes[0].hist(values, bins="fd", color="#2563eb", alpha=0.85, edgecolor="white")
    axes[0].axvline(float(np.mean(values)), color="#dc2626", linestyle="--", label=f"Mean {np.mean(values):.2f} dB")
    axes[0].axvline(float(np.median(values)), color="#ea580c", linestyle=":", label=f"Median {np.median(values):.2f} dB")
    axes[0].set(xlabel="SI-SDR improvement (dB)", ylabel="Examples", title="Per-example SI-SDRi distribution")
    axes[0].legend(frameon=False)
    ordered = np.sort(values)
    cumulative = np.arange(1, ordered.size + 1) / ordered.size
    axes[1].plot(ordered, cumulative, color="#2563eb", linewidth=1.8)
    for quantile, label in ((0.10, "p10"), (0.50, "p50"), (0.90, "p90")):
        value = float(np.quantile(values, quantile))
        axes[1].axvline(value, color="#555555", linestyle="--", linewidth=0.8)
        axes[1].text(value, 0.04, f"{label} {value:.2f}", rotation=90, va="bottom", ha="right", fontsize=8)
    axes[1].set(xlabel="SI-SDR improvement (dB)", ylabel="Cumulative fraction", title="Empirical CDF", ylim=(0, 1))
    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.suptitle(run_name + (f"  (n={summary['examples']})" if "examples" in summary else ""))
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"{run_name}_sisdr_improvement_distribution.{suffix}", bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    run_name = args.run_name or args.metrics.stem
    if "per_example_si_sdri_db" in metrics:
        make_stage1_distribution(metrics, run_name, args.output_dir)
        result_type = "stage1_sisdr_distribution"
    elif "confusion_matrix" in metrics and "per_class_f1" in metrics:
        make_intent_figure(metrics, run_name, args.output_dir)
        result_type = "intent_confusion"
    else:
        make_binary_figure(metrics, run_name, args.output_dir)
        result_type = "binary_curves"
    print(json.dumps({"input": str(args.metrics.resolve()), "run_name": run_name, "figure_type": result_type,
                      "output_dir": str(args.output_dir.resolve())}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
