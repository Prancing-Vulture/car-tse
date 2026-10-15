#!/usr/bin/env python
"""Render training curves from the JSONL histories emitted by train scripts.

The plotting script is intentionally model-agnostic: it reads the existing
epoch JSONL artifacts and never re-runs training or evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_history(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"history is empty: {path}")
    return rows


def short_label(path: Path) -> str:
    name = path.parent.name
    return name.removeprefix("stage4_intent_wav2vec_")


def render(histories: list[tuple[str, list[dict]]], output: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = [
        (axes[0, 0], "train_loss", "Training loss", "loss"),
        (axes[0, 1], "train_accuracy", "Training accuracy", "accuracy"),
        (axes[1, 0], "val_accuracy", "Validation accuracy", "accuracy"),
        (axes[1, 1], "val_macro_f1", "Validation macro-F1", "F1"),
    ]
    for label, rows in histories:
        epochs = [int(row["epoch"]) for row in rows]
        values = {
            "train_loss": [float(row["train_loss"]) for row in rows],
            "train_accuracy": [float(row["train_accuracy"]) for row in rows],
            "val_accuracy": [float(row["val"]["accuracy"]) for row in rows],
            "val_macro_f1": [float(row["val"]["macro_f1"]) for row in rows],
        }
        for axis, key, panel_title, ylabel in panels:
            axis.plot(epochs, values[key], marker="o", linewidth=2, label=label)
            axis.set_title(panel_title)
            axis.set_xlabel("Epoch")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.25)
            axis.set_xticks(epochs)
    for axis, *_ in panels:
        axis.legend(loc="best")
    fig.suptitle(title, fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", nargs="+", type=Path, required=True)
    parser.add_argument("--label", nargs="*", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Training curves")
    args = parser.parse_args()
    if args.label is not None and len(args.label) not in {0, len(args.history)}:
        parser.error("--label must be omitted or supplied once per --history")
    labels = args.label if args.label else [short_label(path) for path in args.history]
    histories = [(label, read_history(path)) for label, path in zip(labels, args.history)]
    render(histories, args.output, args.title)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
