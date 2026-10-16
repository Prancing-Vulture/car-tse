#!/usr/bin/env python
"""Plot the separate clean/overlap validation curves of the mixed intent run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def plot(ax, rows: list[dict], fields: list[tuple[str, str]], title: str, ylabel: str) -> None:
    for field, label in fields:
        values = [row[field.split(".")[0]][field.split(".")[1]] for row in rows]
        ax.plot(range(1, len(values) + 1), values, marker="o", linewidth=2, label=label)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdf-output", type=Path, default=None)
    args = parser.parse_args()
    rows = load_rows(args.history)
    if not rows:
        raise ValueError(f"empty history: {args.history}")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    epochs = range(1, len(rows) + 1)
    axes[0, 0].plot(epochs, [row["train_loss"] for row in rows], marker="o", linewidth=2)
    axes[0, 0].set_title("Training loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Cross-entropy")
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 1].plot(epochs, [row["train_accuracy"] for row in rows], marker="o", linewidth=2)
    axes[0, 1].set_title("Training accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].grid(True, alpha=0.25)
    plot(
        axes[1, 0],
        rows,
        [("val_clean.accuracy", "Clean validation"), ("val_overlap.accuracy", "Overlap validation")],
        "Validation accuracy by domain",
        "Accuracy",
    )
    plot(
        axes[1, 1],
        rows,
        [("val_clean.macro_f1", "Clean validation"), ("val_overlap.macro_f1", "Overlap validation")],
        "Validation macro-F1 by domain",
        "Macro-F1",
    )
    fig.suptitle("Stage 4 mixed clean+overlap intent training")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    if args.pdf_output is not None:
        args.pdf_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.pdf_output, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")
    if args.pdf_output is not None:
        print(f"wrote {args.pdf_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
