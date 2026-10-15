#!/usr/bin/env python
"""Export publication-resolution training curves from a trainer JSONL history."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def flatten_numbers(value: Any, prefix: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            output.update(flatten_numbers(child, name))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric):
            output[prefix] = numeric
    return output


def load_history(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"Training history is empty: {path}")
    if any("epoch" not in row for row in rows):
        raise ValueError(f"Every history row must contain an epoch: {path}")
    # Keep only the last record for an epoch in case a run was resumed and
    # appended a replacement row after an interrupted attempt.
    by_epoch = {int(row["epoch"]): row for row in rows}
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def series(rows: list[dict[str, Any]], key: str) -> list[float | None]:
    values = []
    for row in rows:
        flat = flatten_numbers(row)
        value = flat.get(key)
        values.append(value)
    return values


def first_key(rows: list[dict[str, Any]], *keys: str) -> str | None:
    available = set().union(*(flatten_numbers(row).keys() for row in rows))
    return next((key for key in keys if key in available), None)


def selected_metrics(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    available = set().union(*(flatten_numbers(row).keys() for row in rows))
    specs = [
        ("Training loss", ("train_loss", "train.loss")),
        ("Training accuracy", ("train_accuracy", "train.accuracy")),
        ("Training SI-SDR (dB)", ("train.si_sdr_s1", "train_si_sdr_s1")),
        ("Validation SI-SDR improvement (dB)", ("val.si_sdri", "validation.si_sdri")),
        ("Validation output SI-SDR (dB)", ("val.si_sdr_s1", "validation.si_sdr_s1")),
        ("Validation frame F1", ("validation.best_frame_f1.f1",)),
        ("Validation chunk F1", ("validation.best_chunk_f1.f1",)),
        ("Validation accuracy", ("val.accuracy", "validation.accuracy")),
        ("Validation precision", ("val.precision", "validation.macro_precision")),
        ("Validation recall", ("val.recall", "validation.macro_recall")),
        ("Validation F1", ("val.f1", "validation.macro_f1")),
        ("Validation top-5 accuracy", ("val.top5_accuracy", "validation.top5_accuracy")),
    ]
    selected: list[tuple[str, str]] = []
    for label, candidates in specs:
        key = first_key(rows, *candidates)
        if key is not None:
            selected.append((label, key))
    if {
        "validation.best_chunk_balanced.recall",
        "validation.best_chunk_balanced.specificity",
    }.issubset(available):
        selected.append(("Validation balanced chunk accuracy", "__balanced_chunk_accuracy__"))
    return selected


def metric_values(rows: list[dict[str, Any]], key: str) -> list[float | None]:
    if key == "__balanced_chunk_accuracy__":
        values = []
        for row in rows:
            balanced = row.get("validation", {}).get("best_chunk_balanced", {})
            recall = balanced.get("recall")
            specificity = balanced.get("specificity")
            values.append((float(recall) + float(specificity)) / 2 if recall is not None and specificity is not None else None)
        return values
    return series(rows, key)


def export_presence_threshold_plot(rows: list[dict[str, Any]], run_name: str, output_dir: Path) -> str | None:
    candidates = [
        row for row in rows
        if row.get("validation", {}).get("chunk_threshold_sweep")
        and row.get("validation", {}).get("frame_threshold_sweep")
    ]
    if not candidates:
        return None
    best_row = max(
        candidates,
        key=lambda row: float(row.get("validation", {}).get("best_chunk_f1", {}).get("f1", float("-inf"))),
    )
    validation = best_row["validation"]
    selected_threshold = float(validation["best_chunk_f1"]["threshold"])
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))
    for scope, color in (("frame", "#2563eb"), ("chunk", "#ea580c")):
        points = validation[f"{scope}_threshold_sweep"]
        axes[0].plot([p["false_accept_rate"] for p in points], [p["recall"] for p in points],
                     marker="o", markersize=2.5, label=scope.title(), color=color)
        axes[1].plot([p["recall"] for p in points], [p["precision"] for p in points],
                     marker="o", markersize=2.5, label=scope.title(), color=color)
        axes[2].plot([p["threshold"] for p in points], [p["f1"] for p in points],
                     marker="o", markersize=2.5, label=scope.title(), color=color)
    axes[0].plot([0, 1], [0, 1], "--", color="#777777", linewidth=1)
    axes[0].set(xlabel="False-accept rate", ylabel="Recall", title="ROC by threshold", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall", xlim=(0, 1), ylim=(0, 1))
    axes[2].axvline(selected_threshold, color="#111827", linestyle="--", linewidth=1,
                    label=f"Selected chunk threshold {selected_threshold:.2f}")
    axes[2].set(xlabel="Decision threshold", ylabel="F1", title=f"Validation sweep, epoch {best_row['epoch']}", ylim=(0, 1))
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle(run_name)
    fig.tight_layout()
    stem = output_dir / f"{run_name}_validation_threshold_curves"
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return str(stem.with_suffix(".pdf").resolve())


def export_sisdr_distribution_plot(rows: list[dict[str, Any]], epochs: list[int], run_name: str, output_dir: Path) -> str | None:
    flattened = [flatten_numbers(row) for row in rows]
    keys = ("val.si_sdri_p10", "val.si_sdri_median", "val.si_sdri_p90", "val.si_sdri")
    if not any(key in item for item in flattened for key in keys):
        return None
    p10 = [item.get(keys[0]) for item in flattened]
    median = [item.get(keys[1]) for item in flattened]
    p90 = [item.get(keys[2]) for item in flattened]
    mean = [item.get(keys[3]) for item in flattened]
    fig, axis = plt.subplots(figsize=(5.8, 3.5))
    if all(value is not None for value in p10 + p90):
        axis.fill_between(epochs, p10, p90, alpha=0.2, color="#2563eb", label="10th–90th percentile")
    if any(value is not None for value in median):
        axis.plot(epochs, median, marker="o", color="#ea580c", label="Median SI-SDRi")
    if any(value is not None for value in mean):
        axis.plot(epochs, mean, marker="s", linestyle="--", color="#2563eb", label="Mean SI-SDRi")
    axis.set(title="Validation SI-SDR improvement distribution", xlabel="Epoch", ylabel="SI-SDRi (dB)")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    fig.suptitle(run_name)
    fig.tight_layout()
    stem = output_dir / f"{run_name}_validation_si_sdri_distribution"
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return str(stem.with_suffix(".pdf").resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True, help="metrics_history.jsonl from a training run")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    rows = load_history(args.history)
    metrics = selected_metrics(rows)
    if not metrics:
        raise ValueError(f"No recognized numeric training metrics in {args.history}")
    epochs = [int(row["epoch"]) for row in rows]
    run_name = args.run_name or args.history.parent.name
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "lines.linewidth": 1.8,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = 2
    rows_count = (len(metrics) + columns - 1) // columns
    figure, axes = plt.subplots(rows_count, columns, figsize=(8.0, max(2.8, 2.55 * rows_count)), squeeze=False)
    palette = plt.get_cmap("tab10")
    for index, (label, key) in enumerate(metrics):
        axis = axes.flat[index]
        values = metric_values(rows, key)
        valid = [(epoch, value) for epoch, value in zip(epochs, values) if value is not None]
        if valid:
            xs, ys = zip(*valid)
            axis.plot(xs, ys, marker="o", markersize=3.2, color=palette(index % 10), label=label)
            axis.set_title(label)
            axis.set_xlabel("Epoch")
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.25)
            axis.margins(x=0.04)
    for axis in axes.flat[len(metrics):]:
        axis.remove()
    figure.suptitle(run_name, fontsize=11, y=1.0)
    figure.tight_layout()
    stem = args.output_dir / f"{run_name}_training_curves"
    figure.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    threshold_plot = export_presence_threshold_plot(rows, run_name, args.output_dir)
    sisdr_distribution_plot = export_sisdr_distribution_plot(rows, epochs, run_name, args.output_dir)

    summary = {
        "run_name": run_name,
        "history": str(args.history.resolve()),
        "epochs": epochs,
        "metrics_plotted": {label: key for label, key in metrics},
        "png": str(stem.with_suffix(".png").resolve()),
        "pdf": str(stem.with_suffix(".pdf").resolve()),
        "validation_threshold_plot_pdf": threshold_plot,
        "validation_si_sdri_distribution_pdf": sisdr_distribution_plot,
    }
    (args.output_dir / f"{run_name}_training_curves.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
