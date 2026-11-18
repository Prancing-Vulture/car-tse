#!/usr/bin/env python
"""Render the current Stage-2 dual-domain validation summary.

The current promoted Stage-2 run records a final validation summary rather
than per-epoch metrics.  This renderer intentionally plots that summary
instead of mixing older training histories from different datasets with the
current checkpoint result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("validation"), dict):
        raise ValueError(f"expected a Stage-2 metrics_last.json payload: {path}")
    return payload


def annotate_bars(ax, bars, values: list[float], fmt: str = "{:.3f}") -> None:
    for bar, value in zip(bars, values):
        ax.annotate(
            fmt.format(value),
            xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def grouped_bars(
    ax,
    labels,
    series,
    title,
    ylabel,
    *,
    ylim=(0.0, 1.05),
    legend_loc="lower right",
) -> None:
    x = np.arange(len(labels), dtype=float)
    width = 0.34 if len(series) == 2 else 0.24
    offsets = (np.arange(len(series), dtype=float) - (len(series) - 1) / 2.0) * width
    for offset, (name, values, color) in zip(offsets, series):
        bars = ax.bar(x + offset, values, width=width, label=name, color=color)
        annotate_bars(ax, bars, values)
    ax.set_xticks(x, labels)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc=legend_loc, fontsize=8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdf-output", type=Path, required=True)
    args = parser.parse_args()

    payload = load_metrics(args.metrics)
    validation = payload["validation"]
    views = [
        ("Frame\n@ 0.50", validation["frame_at_0.5"]),
        ("Frame\nbest F1 @ 0.65", validation["best_frame_f1"]),
        ("Chunk\n@ 0.50", validation["chunk_at_0.5"]),
        ("Chunk\nbest F1 @ 0.25", validation["best_chunk_f1"]),
    ]
    labels = [label for label, _ in views]

    colors = ["#2f78b7", "#f28e2b"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Stage-2 target-presence validation: current dual-domain FT1 checkpoint",
        fontsize=16,
        y=0.98,
    )

    grouped_bars(
        axes[0, 0],
        labels,
        [("Accuracy", [float(row["accuracy"]) for _, row in views], "#4e79a7")],
        "Held-out validation accuracy",
        "Accuracy",
    )
    grouped_bars(
        axes[0, 1],
        labels,
        [("F1", [float(row["f1"]) for _, row in views], "#59a14f")],
        "Held-out validation F1",
        "F1",
    )
    axes[0, 1].axhline(0.90, color="#b22222", linestyle="--", linewidth=1, alpha=0.75)
    axes[0, 1].text(
        0.02,
        0.91,
        "0.90 reference",
        transform=axes[0, 1].transAxes,
        color="#8b0000",
        fontsize=8,
    )
    grouped_bars(
        axes[1, 0],
        labels,
        [
            ("Precision", [float(row["precision"]) for _, row in views], colors[0]),
            ("Recall", [float(row["recall"]) for _, row in views], colors[1]),
        ],
        "Held-out precision and recall",
        "Score",
    )
    grouped_bars(
        axes[1, 1],
        labels,
        [
            ("False-accept rate", [float(row["false_accept_rate"]) for _, row in views], "#e15759"),
            ("False-reject rate", [float(row["false_reject_rate"]) for _, row in views], "#9467bd"),
        ],
        "Held-out error rates",
        "Rate",
        ylim=(0.0, 0.12),
        legend_loc="upper left",
    )

    epoch = payload.get("epoch", "?")
    train_examples = payload.get("train_examples", "?")
    val_examples = payload.get("val_examples", "?")
    fig.text(
        0.5,
        0.035,
        (
            f"Checkpoint epoch {epoch}; {train_examples:,} training examples; "
            f"{val_examples:,} validation chunks. Source: {args.metrics.name}. "
            "All views use the same checkpoint; historical per-epoch plots are separate diagnostics."
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.pdf_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    fig.savefig(args.pdf_output, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")
    print(f"wrote {args.pdf_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
