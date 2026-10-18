#!/usr/bin/env python
"""Render Stage-1 SI-SDR/SI-SDRi training curves for paper artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: Path) -> list[dict]:
    if any(char in str(path) for char in "*?["):
        matches = sorted(path.parent.glob(path.name))
        if len(matches) != 1:
            raise ValueError(f"expected one history match for {path}, found {len(matches)}")
        path = matches[0]
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty history: {path}")
    return rows


def series(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    out_x: list[int] = []
    out_y: list[float] = []
    for row in rows:
        value = row
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                value = None
                break
            value = value[part]
        if value is not None:
            out_x.append(int(row.get("epoch", len(out_x) + 1)))
            out_y.append(float(value))
    return out_x, out_y


def plot(ax, histories, key: str, title: str, ylabel: str) -> None:
    for label, rows in histories:
        x, y = series(rows, key)
        if x:
            ax.plot(x, y, marker="o", linewidth=2, markersize=4, label=label)
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
    parser.add_argument("--title", default="Stage-1 TSE adaptation curves")
    args = parser.parse_args()
    histories = []
    for spec in args.history:
        label, raw = spec.split("=", 1)
        histories.append((label, read_rows(Path(raw))))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    plot(axes[0, 0], histories, "train_si_sdr_s1", "Training SI-SDR", "SI-SDR (dB)")
    plot(axes[0, 1], histories, "val.si_sdr_s1", "Validation SI-SDR", "SI-SDR (dB)")
    plot(axes[1, 0], histories, "val.si_sdri", "Validation SI-SDR improvement", "SI-SDRi (dB)")
    plot(axes[1, 1], histories, "val.si_sdri_median", "Validation median SI-SDRi", "Median SI-SDRi (dB)")
    fig.suptitle(args.title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    if args.pdf_output:
        args.pdf_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.pdf_output, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")
    if args.pdf_output:
        print(f"wrote {args.pdf_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
