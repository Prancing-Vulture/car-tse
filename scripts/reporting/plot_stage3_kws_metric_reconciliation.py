#!/usr/bin/env python
"""Plot the protocol distinction behind the Stage-3 KWS F1 values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-history", type=Path, required=True)
    parser.add_argument("--negative-history", type=Path, required=True)
    parser.add_argument("--score-fusion", type=Path, required=True)
    parser.add_argument("--current-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdf-output", type=Path, required=True)
    args = parser.parse_args()

    previous = max(float(row["val"]["f1"]) for row in read_jsonl(args.previous_history))
    negative = max(float(row["val"]["f1"]) for row in read_jsonl(args.negative_history))
    score_fusion = json.loads(args.score_fusion.read_text(encoding="utf-8"))
    selected = next(
        row for row in score_fusion["rows"] if abs(float(row["alpha_checkpoint_a"]) - 0.80) < 1e-9
    )
    operating_point = float(selected["best_f1"]["f1"])
    operating_threshold = float(selected["best_f1"]["threshold"])
    current_metrics = json.loads(args.current_metrics.read_text(encoding="utf-8"))
    current = float(current_metrics["val"]["f1"])

    labels = [
        "Previous large\nepoch max @ 0.50",
        "Negative-calibrated\nepoch max @ 0.50",
        "Selected TSE-domain\nthreshold sweep @ 0.44",
        "Current stored target KWS\nvalidation @ 0.50",
    ]
    values = [previous, negative, operating_point, current]
    colors = ["#377eb8", "#ff7f00", "#4daf4a", "#984ea3"]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.barh(labels, values, color=colors, edgecolor="#333333", linewidth=0.6)
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("F1")
    ax.set_title("Stage-3 KWS metric reconciliation: epoch history vs operating point")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(
            min(value + 0.015, 1.02),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f} ({value * 100:.2f}%)",
            va="center",
            ha="left",
            fontsize=10,
        )
    ax.text(
        0.01,
        -0.22,
        "The first two bars are maxima from fixed-threshold epoch logs. The 97.40% bar is a separate "
        f"threshold-swept TSE-domain point (selected threshold {operating_threshold:.2f}); it is not an epoch value.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.98))
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
