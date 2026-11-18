#!/usr/bin/env python
"""Render the fixed-threshold Stage-3 target-speaker KWS epoch history.

The Stage-3 trainer records training accuracy and validation metrics at the
argmax operating point (equivalent to a binary probability threshold of 0.50),
but the historical runs used for the paper figure did not record per-epoch
training loss or per-epoch threshold sweeps.  This renderer therefore uses only
fields present in the source JSONL and labels the fixed-threshold metrics
explicitly.  It must not be used to imply that a separately threshold-tuned
operating point is an epoch metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_history(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty training history: {path}")
    return rows


def series(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for row in rows:
        value = row
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                value = None
                break
            value = value[part]
        if value is not None:
            xs.append(int(row.get("epoch", len(xs) + 1)))
            ys.append(float(value))
    return xs, ys


def plot_lines(ax, histories, specs, title: str, ylabel: str) -> None:
    for label, rows in histories:
        for key, suffix in specs:
            xs, ys = series(rows, key)
            if xs:
                pretty_label = label.replace("_", " ")
                ax.plot(xs, ys, marker="o", linewidth=2, markersize=4, label=f"{pretty_label}{suffix}")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdf-output", type=Path, default=None)
    parser.add_argument(
        "--title",
        default="Target-speaker KWS epoch history at fixed validation threshold 0.50",
    )
    parser.add_argument(
        "--validation-threshold",
        type=float,
        default=0.50,
        help="Threshold used by the logged binary argmax metrics (display only).",
    )
    args = parser.parse_args()

    histories: list[tuple[str, list[dict]]] = []
    for spec in args.history:
        if "=" not in spec:
            raise ValueError(f"history must be LABEL=PATH: {spec}")
        label, raw_path = spec.split("=", 1)
        histories.append((label, read_history(Path(raw_path))))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # These runs do not contain train_loss.  Do not fabricate it or leave an
    # empty panel; training accuracy is the available training-side metric.
    threshold_label = f"{args.validation_threshold:.2f}"
    plot_lines(axes[0, 0], histories, [("train_accuracy", "")], "Training accuracy", "Accuracy")
    plot_lines(
        axes[0, 1],
        histories,
        [("val.accuracy", "")],
        f"Validation accuracy @ threshold {threshold_label}",
        "Accuracy",
    )
    plot_lines(
        axes[1, 0],
        histories,
        [("val.f1", "")],
        f"Validation F1 @ threshold {threshold_label}",
        "F1",
    )
    plot_lines(
        axes[1, 1],
        histories,
        [
            ("val.precision", f" precision @ {threshold_label}"),
            ("val.recall", f" recall @ {threshold_label}"),
        ],
        f"Validation precision and recall @ {threshold_label}",
        "Score",
    )

    fig.suptitle(args.title)
    fig.text(
        0.5,
        0.030,
        "Epoch metrics are from the negative-heavy target-KWS validation split (n=1,629) at the fixed binary argmax threshold. "
        "The reported 97.40% F1 is a separate threshold-swept TSE-domain operating point, not an epoch value from these runs.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.text(
        0.5,
        0.012,
        "Per-epoch training loss and per-epoch threshold sweeps were not recorded; no missing curves are inferred.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0.0, 0.085, 1.0, 0.96))
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
