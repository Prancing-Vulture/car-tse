#!/usr/bin/env python
"""Render paper-ready training curves from the JSONL histories written by trainers.

The trainers intentionally keep the raw JSONL as the source of truth.  This
small renderer handles the nested ``val`` dictionaries used by Stage 2/3/4,
and writes both a high-resolution PNG and a vector PDF when requested.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_history(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    rows: list[dict] = []
    if text.lstrip().startswith("{"):
        # A single JSON object may embed a history list, but JSONL also starts
        # with ``{``.  Try the object form first and fall back to line parsing
        # when the file contains multiple records.
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        embedded = payload.get("history") if isinstance(payload, dict) else None
        if isinstance(embedded, list):
            rows = [row for row in embedded if isinstance(row, dict)]
    if not rows:
        for line in text.splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty training history: {path}")
    return rows


def value(row: dict, key: str):
    current = row
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def series(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for row in rows:
        item = value(row, key)
        if item is None:
            continue
        xs.append(int(row.get("epoch", len(xs) + 1)))
        ys.append(float(item))
    return xs, ys


def plot_one(ax, histories: list[tuple[str, list[dict]]], key: str, title: str, ylabel: str) -> None:
    plotted = False
    for label, rows in histories:
        x, y = series(rows, key)
        if x:
            ax.plot(x, y, marker="o", linewidth=2, markersize=4, label=label)
            plotted = True
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend(frameon=True)
    else:
        ax.text(0.5, 0.5, f"No field: {key}", ha="center", va="center", transform=ax.transAxes)


def plot_first_available(
    ax,
    histories: list[tuple[str, list[dict]]],
    keys: list[tuple[str, str, str]],
) -> None:
    """Plot the first metric family present in the supplied histories."""

    for key, title, ylabel in keys:
        if any(series(rows, key)[0] for _, rows in histories):
            plot_one(ax, histories, key, title, ylabel)
            return
    key, title, ylabel = keys[0]
    plot_one(ax, histories, key, title, ylabel)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="JSONL history with the label used in the legend; repeat for comparisons.",
    )
    parser.add_argument("--output", type=Path, required=True, help="PNG output path")
    parser.add_argument("--pdf-output", type=Path, default=None, help="optional vector PDF output")
    parser.add_argument("--title", default="Training curves")
    args = parser.parse_args()

    histories: list[tuple[str, list[dict]]] = []
    for spec in args.history:
        if "=" not in spec:
            raise ValueError(f"history must be LABEL=PATH: {spec}")
        label, raw_path = spec.split("=", 1)
        histories.append((label, read_history(Path(raw_path))))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    plot_one(axes[0, 0], histories, "train_loss", "Training loss", "Cross-entropy")
    plot_one(axes[0, 1], histories, "train_accuracy", "Training accuracy", "Accuracy")
    plot_first_available(
        axes[1, 0],
        histories,
        [
            ("val.accuracy", "Validation accuracy", "Accuracy"),
            ("val_auc", "Validation ROC-AUC", "ROC-AUC"),
        ],
    )
    plot_first_available(
        axes[1, 1],
        histories,
        [
            ("val.macro_f1", "Validation macro-F1", "Macro-F1"),
            ("val.f1", "Validation F1", "F1"),
            ("val_auc", "Validation ROC-AUC", "ROC-AUC"),
        ],
    )
    fig.suptitle(args.title)
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
