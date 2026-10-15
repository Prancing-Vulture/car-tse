#!/usr/bin/env python
"""Plot corrected multi-speaker safety/accuracy metrics from evaluator JSON logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(row: dict, key: str) -> float:
    if key == "strict_false_wake_events_per_hour":
        duration_hours = float(row.get("duration_hours", 0.0) or 0.0)
        return float(row.get("strict_false_wake_events", 0.0) or 0.0) / max(duration_hours, 1.0e-9)
    value = row.get(key)
    if isinstance(value, dict):
        value = value.get("value", value.get("count", 0))
    return float(value or 0.0)


def label_for(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"run must be LABEL=PATH: {spec}")
    label, path = spec.split("=", 1)
    return label, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdf-output", type=Path, default=None)
    parser.add_argument("--title", default="Corrected multi-speaker comparison")
    args = parser.parse_args()

    names: list[str] = []
    rows: list[dict] = []
    for spec in args.run:
        name, path = label_for(spec)
        names.append(name)
        rows.append(read(path))

    panels = [
        ("target_intent_accuracy_overall", "Target intent accuracy", True),
        ("fully_correct_scenes", "Fully correct scenes", True),
        ("strict_false_wake_events_per_hour", "Strict false wakes / hour", False),
        ("passenger_only_false_action_events", "Passenger-only false actions", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, (key, title, higher_is_better) in zip(axes.flat, panels):
        values = []
        for row in rows:
            if key == "fully_correct_scenes":
                total = max(metric(row, "scenes"), 1.0)
                values.append(100.0 * metric(row, key) / total)
            else:
                values.append(metric(row, key) * (100.0 if key == "target_intent_accuracy_overall" else 1.0))
        best_index = max(range(len(values)), key=lambda i: values[i]) if higher_is_better else min(range(len(values)), key=lambda i: values[i])
        colors = ["#2f78b7" if i == best_index else "#9aa7b2" for i in range(len(values))]
        bars = ax.bar(names, values, color=colors)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=22)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=9)
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
