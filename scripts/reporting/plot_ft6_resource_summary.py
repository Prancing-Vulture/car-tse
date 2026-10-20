#!/usr/bin/env python
"""Render the current FT6 resource summary used by the paper.

The latency/parameter values are read from the committed FT6 CUDA benchmark
JSON.  The FLOP estimates are the analytical values documented in
reports/model_architecture_metrics_latency_sota_2026-10-17.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


PARAMETER_LABELS = {
    "stage1": "Stage 1 TSE",
    "presence": "Presence",
    "kws": "General KWS",
    "target_kws": "Target KWS",
    "compact_speaker_verifier": "Speaker verifier",
    "intent": "wav2vec2 + intent",
}

FLOPS_GF = {
    "Stage 1 TSE": 103.945,
    "Presence": 0.020,
    "General KWS": 0.449,
    "Target KWS": 0.152,
    "Speaker verifier": 0.152,
    "wav2vec2 + intent": 25.431,
}

LATENCY_LABELS = {
    "stage1_extraction_3s_4ch": "Stage 1 TSE",
    "presence_head_3s": "Presence",
    "target_kws_1s": "Target KWS",
    "general_kws_1s": "General KWS",
    "compact_speaker_verifier_2x1s": "Speaker verifier",
    "intent_wav2vec_3s": "wav2vec2 + intent",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latency-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdf-output", type=Path, default=None)
    args = parser.parse_args()

    payload = json.loads(args.latency_json.read_text(encoding="utf-8"))
    parameters = payload["parameters"]
    components = payload["components"]

    labels = list(PARAMETER_LABELS.values())
    parameter_m = [parameters[key] / 1_000_000.0 for key in PARAMETER_LABELS]
    latency_mean = [components[key]["mean_ms"] for key in LATENCY_LABELS]
    latency_p95 = [components[key]["p95_ms"] for key in LATENCY_LABELS]
    flops = [FLOPS_GF[label] for label in labels]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    bars = axes[0, 0].bar(labels, parameter_m, color="#2f78b7")
    axes[0, 0].set_title("Parameters by component")
    axes[0, 0].set_ylabel("Millions of parameters")
    axes[0, 0].tick_params(axis="x", rotation=28)
    axes[0, 0].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, parameter_m):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    x = list(range(len(labels)))
    axes[0, 1].bar([i - 0.18 for i in x], latency_mean, width=0.36, label="Mean", color="#2f78b7")
    axes[0, 1].bar([i + 0.18 for i in x], latency_p95, width=0.36, label="P95", color="#e58f2a")
    axes[0, 1].set_title("RTX 4080 FP16 component latency")
    axes[0, 1].set_ylabel("Milliseconds per invocation")
    axes[0, 1].set_xticks(x, labels, rotation=28, ha="right")
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", alpha=0.25)

    bars = axes[1, 0].bar(labels, flops, color="#2f78b7")
    axes[1, 0].set_title("Analytical compute estimate")
    axes[1, 0].set_ylabel("GFLOPs per 3-second call")
    axes[1, 0].set_yscale("log")
    axes[1, 0].tick_params(axis="x", rotation=28)
    axes[1, 0].grid(axis="y", alpha=0.25, which="both")
    for bar, value in zip(bars, flops):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    composite = components["composite_command_stage"]
    axes[1, 1].bar(["Composite wake", "Composite command"], [components["composite_wake_stage"]["mean_ms"], composite["mean_ms"]], color=["#9aa7b2", "#2f78b7"])
    axes[1, 1].set_title("Composite model-forward latency")
    axes[1, 1].set_ylabel("Mean milliseconds")
    axes[1, 1].grid(axis="y", alpha=0.25)
    for bar, value in zip(axes[1, 1].patches, [components["composite_wake_stage"]["mean_ms"], composite["mean_ms"]]):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}", ha="center", va="bottom")

    fig.suptitle("Current FT6 system resource summary (RTX 4080 FP16)")
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
