#!/usr/bin/env python
"""Render Stage-2 presence training histories as paper-ready PNG/PDF figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_history(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    history = payload.get("history") if isinstance(payload, dict) else None
    if not isinstance(history, list) or not history:
        raise ValueError(f"expected a non-empty history list in {path}")
    return [row for row in history if isinstance(row, dict)]


def metric(row: dict, key: str) -> float | None:
    value = row
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return None if value is None else float(value)


def plot_metric(ax, histories, key: str, title: str, ylabel: str) -> None:
    plotted = False
    for label, rows in histories:
        xs: list[int] = []
        ys: list[float] = []
        for row in rows:
            value = metric(row, key)
            if value is not None:
                xs.append(int(row.get("epoch", len(xs) + 1)))
                ys.append(value)
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2, markersize=4, label=label)
            plotted = True
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend(frameon=True)
    else:
        ax.text(0.5, 0.5, f"No field: {key}", ha="center", va="center", transform=ax.transAxes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Stage-2 metrics JSON containing a history list; repeat for comparisons.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdf-output", type=Path, required=True)
    parser.add_argument("--title", default="Stage-2 target-presence training comparison")
    args = parser.parse_args()

    histories: list[tuple[str, list[dict]]] = []
    for spec in args.history:
        if "=" not in spec:
            raise ValueError(f"history must be LABEL=PATH: {spec}")
        label, raw_path = spec.split("=", 1)
        histories.append((label, read_history(Path(raw_path))))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    plot_metric(axes[0, 0], histories, "train_loss", "Training loss", "Loss")
    plot_metric(axes[0, 1], histories, "val.accuracy", "Validation accuracy", "Accuracy")
    plot_metric(axes[1, 0], histories, "val.f1", "Validation F1 at 0.5", "F1")
    plot_metric(axes[1, 1], histories, "best_f1.far", "Best-threshold false-accept rate", "FAR")
    fig.suptitle(args.title)
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
