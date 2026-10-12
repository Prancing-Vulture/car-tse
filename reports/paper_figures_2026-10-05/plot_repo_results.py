"""Generate publication-style figures from metrics reported in this repository.

No model outputs are inferred or synthesized here. Values are transcribed from
the cited benchmark/status reports and artifact metadata. The source notes
created by this script identify which evaluations are proxies or diagnostics.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=0.95)
COLORS = sns.color_palette("colorblind")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def write_csv(name: str, rows: list[dict]) -> None:
    path = OUT / name
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# 1. Held-out CAVEMOVE extraction summary. Error bars show reported 10th–90th
# percentiles; markers show the separately reported mean and median.
extraction = pd.DataFrame([
    {"Signal": "Mixture input", "Mean": -2.41, "Median": -2.37, "P10": -8.32, "P90": 3.30},
    {"Signal": "Extracted output", "Mean": 8.57, "Median": 10.49, "P10": 0.22, "P90": 15.07},
])
write_csv("stage1_extraction_summary.csv", extraction.to_dict(orient="records"))
fig, ax = plt.subplots(figsize=(8.4, 2.7))
for y, row in enumerate(extraction.itertuples(index=False)):
    ax.hlines(y, row.P10, row.P90, color=COLORS[0], lw=7, alpha=0.38,
              label="10th–90th percentile" if y == 0 else None)
    ax.scatter(row.Mean, y, marker="o", s=75, color=COLORS[1], zorder=3,
               label="Mean" if y == 0 else None)
    ax.scatter(row.Median, y, marker="|", s=250, linewidths=2.5, color=COLORS[2], zorder=4,
               label="Median" if y == 0 else None)
ax.set_yticks([0, 1], extraction["Signal"])
ax.set_xlabel("SI-SDR (dB)")
ax.set_title("Stage 1 extraction on held-out synthetic CAVEMOVE mixtures (n=400)")
ax.legend(frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0))
fig.subplots_adjust(right=0.74)
ax.text(0.01, -0.30, "Percentile bars summarize reported quantiles; individual utterance values were unavailable.",
        transform=ax.transAxes, fontsize=8)
save(fig, "fig01_stage1_extraction")


# 2. Front-end / initialization comparison, using SI-SDR improvement values
# reported for the same style of CAVEMOVE test protocol.
frontend = pd.DataFrame([
    {"System": "Cabin-noise mean-pool baseline", "SI-SDRi (dB)": 8.01},
    {"System": "Clean transferred SpEx+ baseline", "SI-SDRi (dB)": 10.73},
    {"System": "SpEx+ + learned cabin front end", "SI-SDRi (dB)": 10.98},
])
write_csv("stage1_frontend_comparison.csv", frontend.to_dict(orient="records"))
fig, ax = plt.subplots(figsize=(7.4, 3.8))
sns.barplot(data=frontend, x="SI-SDRi (dB)", y="System", color=COLORS[0], ax=ax)
for patch, value in zip(ax.patches, frontend["SI-SDRi (dB)"]):
    ax.text(value + 0.08, patch.get_y() + patch.get_height()/2, f"{value:.2f}", va="center", fontsize=9)
ax.set_xlim(0, 12.5)
ax.set_title("Reported Stage 1 baseline comparison")
ax.set_ylabel("")
ax.text(0.01, -0.23, "Values are SI-SDR improvement (SI-SDRi), not absolute extracted SI-SDR.",
        transform=ax.transAxes, fontsize=8)
save(fig, "fig02_stage1_frontend_comparison")


# 3. Input-RMS sensitivity sweep on two distinct held-out sets.
scale = pd.DataFrame([
    {"Policy": "Raw input", "AISHELL-5 (n=73)": 1.47, "CAVEMOVE (n=400)": 10.98},
    {"Policy": "RMS target 0.10", "AISHELL-5 (n=73)": 4.94, "CAVEMOVE (n=400)": 10.91},
    {"Policy": "RMS target 0.20", "AISHELL-5 (n=73)": 6.43, "CAVEMOVE (n=400)": 10.62},
    {"Policy": "Enrollment RMS", "AISHELL-5 (n=73)": 1.11, "CAVEMOVE (n=400)": 10.72},
])
write_csv("stage1_rms_sweep.csv", scale.to_dict(orient="records"))
long_scale = scale.melt(id_vars="Policy", var_name="Evaluation set", value_name="SI-SDRi (dB)")
fig, ax = plt.subplots(figsize=(7.5, 4.0))
sns.lineplot(data=long_scale, x="Policy", y="SI-SDRi (dB)", hue="Evaluation set",
             marker="o", linewidth=2.3, ax=ax)
ax.set_title("Stage 1 input-scale sweep")
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=12)
ax.legend(frameon=False, title="")
ax.text(0.01, -0.32, "Each line uses a different evaluation set; compare trends within each set, not absolute values across sets.",
        transform=ax.transAxes, fontsize=8)
save(fig, "fig03_stage1_rms_sweep")


# 4. Scalar presence validation operating points.
presence = pd.DataFrame([
    {"Threshold": "0.50", "False accept rate": 96.25, "False reject rate": 0.00},
    {"Threshold": "0.90", "False accept rate": 17.50, "False reject rate": 8.88},
])
write_csv("stage2_presence_operating_points.csv", presence.to_dict(orient="records"))
long_presence = presence.melt(id_vars="Threshold", var_name="Error type", value_name="Rate (%)")
fig, ax = plt.subplots(figsize=(6.2, 3.8))
sns.barplot(data=long_presence, x="Threshold", y="Rate (%)", hue="Error type", ax=ax,
            palette=[COLORS[3], COLORS[1]])
ax.set_ylim(0, 105)
ax.set_title("Presence-gate threshold trade-off")
ax.set_xlabel("Decision threshold")
ax.text(0.01, -0.26, "Reported validation split: 800 target-present and 80 target-absent examples.",
        transform=ax.transAxes, fontsize=8)
save(fig, "fig04_stage2_presence_tradeoff")


# 5. KWS operating metrics. Evaluation sizes/distributions vary by candidate.
kws = pd.DataFrame([
    {"Candidate": "Original cabin", "N": 1147, "Precision": 52.29, "Recall": 82.05, "F1": 63.87},
    {"Candidate": "Hard-negative cabin", "N": 3147, "Precision": 32.91, "Recall": 79.49, "F1": 46.55},
    {"Candidate": "TSE-domain", "N": 445, "Precision": 95.94, "Recall": 96.92, "F1": 96.43},
    {"Candidate": "Post-TSE hybrid", "N": 3397, "Precision": 84.48, "Recall": 75.38, "F1": 79.67},
    {"Candidate": "AISHELL-5 augmented", "N": 4551, "Precision": 83.15, "Recall": 78.46, "F1": 80.74},
])
write_csv("stage3_kws_reported_metrics.csv", kws.to_dict(orient="records"))
long_kws = kws.melt(id_vars=["Candidate", "N"], var_name="Metric", value_name="Percent")
fig, ax = plt.subplots(figsize=(9.0, 4.5))
sns.barplot(data=long_kws, x="Percent", y="Candidate", hue="Metric", ax=ax)
ax.set_xlim(0, 105)
ax.set_xlabel("Score (%)")
ax.set_ylabel("")
ax.set_title("Wake-word detection: reported test-set metrics")
ax.legend(frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="")
fig.subplots_adjust(right=0.80)
ax.text(0.01, -0.25, "Test-set sizes and negative composition differ by candidate; scores are not a controlled paired comparison.",
        transform=ax.transAxes, fontsize=8)
save(fig, "fig05_stage3_kws_metrics")


# 6. Intent top-1 accuracy across clean and local-domain evaluations.
intent = pd.DataFrame([
    {"Condition": "FSC official validation (n=3,118)", "Top-1 accuracy": 96.60},
    {"Condition": "FSC official test (n=3,793)", "Top-1 accuracy": 99.31},
    {"Condition": "Local raw-cabin validation (n=2,314)", "Top-1 accuracy": 90.71},
    {"Condition": "Local TSE-domain validation (n=2,314)", "Top-1 accuracy": 90.49},
])
write_csv("stage4_intent_top1.csv", intent.to_dict(orient="records"))
fig, ax = plt.subplots(figsize=(7.7, 4.2))
sns.barplot(data=intent, x="Top-1 accuracy", y="Condition", color=COLORS[0], ax=ax)
for p, v in zip(ax.patches, intent["Top-1 accuracy"]):
    ax.text(v + 0.15, p.get_y() + p.get_height()/2, f"{v:.2f}%", va="center", fontsize=8)
ax.set_xlim(0, 105)
ax.set_xlabel("Top-1 accuracy (%)")
ax.set_ylabel("")
ax.set_title("Stage 4 intent classification")
ax.text(0.01, -0.23, "Clean FSC test performance is distinct from local cabin/TSE-domain validation performance.",
        transform=ax.transAxes, fontsize=8)
save(fig, "fig06_stage4_intent")


# 7. Continuous full-pipeline proxy. Four trials; not speaker-consistent.
pipeline = pd.DataFrame([
    {"Presence threshold": "0.90", "Target wakes": 0, "Trials": 4,
     "Correct intents": 0, "False wake events": 1, "False actions/hour": 64.3},
    {"Presence threshold": "0.50", "Target wakes": 1, "Trials": 4,
     "Correct intents": 0, "False wake events": 1, "False actions/hour": 64.3},
])
pipeline["Wake recall (%)"] = 100 * pipeline["Target wakes"] / pipeline["Trials"]
pipeline["Correct intent rate (%)"] = 100 * pipeline["Correct intents"] / pipeline["Trials"]
write_csv("full_pipeline_proxy.csv", pipeline.to_dict(orient="records"))
fig, ax = plt.subplots(figsize=(6.8, 3.7))
x = range(len(pipeline))
bars = ax.bar(x, pipeline["Wake recall (%)"], color=COLORS[0], width=0.56,
              label="Target wake recall")
ax.scatter(list(x), pipeline["Correct intent rate (%)"], color=COLORS[2], marker="D",
           s=55, zorder=3, label="Correct intent rate")
for i, (_, row) in enumerate(pipeline.iterrows()):
    ax.annotate(f"{int(row['Target wakes'])}/{int(row['Trials'])} wakes\n0/{int(row['Trials'])} intents",
                (i, row["Wake recall (%)"]), xytext=(0, 9), textcoords="offset points",
                ha="center", fontsize=8)
ax.set_xticks(list(x), pipeline["Presence threshold"])
ax.set_ylim(0, 110)
ax.set_xlabel("Presence threshold")
ax.set_title("Continuous cross-corpus proxy: 4 target trials")
ax.legend(frameon=False, loc="upper left")
ax.text(0.01, -0.28, "Speaker match is unverified; diagnostic only. Each threshold had 1 false event in 56 s (~64.3/h).",
        transform=ax.transAxes, fontsize=8)
save(fig, "fig07_full_pipeline_proxy")


# 8. Model-forward timings reported for a one-second benchmark input.
latency = pd.DataFrame([
    {"Device": "RTX 4080 GPU", "Component": "Stage 1", "Latency": 51.60},
    {"Device": "RTX 4080 GPU", "Component": "KWS", "Latency": 1.92},
    {"Device": "RTX 4080 GPU", "Component": "Intent", "Latency": 16.43},
    {"Device": "CPU", "Component": "Stage 1", "Latency": 168.06},
    {"Device": "CPU", "Component": "KWS", "Latency": 2.09},
    {"Device": "CPU", "Component": "Intent", "Latency": 105.19},
])
write_csv("reported_latency.csv", latency.to_dict(orient="records"))
fig, ax = plt.subplots(figsize=(7.1, 4.0))
sns.barplot(data=latency, x="Component", y="Latency", hue="Device", ax=ax)
ax.set_ylabel("Forward-pass time (ms)")
ax.set_title("Reported model-forward latency for a one-second input")
ax.text(0.01, -0.27, "Presence, audio I/O, buffering, resampling, and action execution are excluded.",
        transform=ax.transAxes, fontsize=8)
ax.legend(frameon=False, title="")
save(fig, "fig08_latency")


# 9. Parameter counts from the selected checkpoint bundle.
params = pd.DataFrame([
    {"Component": "Stage 1 TSE + front end", "Parameters (M)": 10.73},
    {"Component": "Stage 2 presence", "Parameters (M)": 0.25},
    {"Component": "Stage 3 KWS", "Parameters (M)": 0.09},
    {"Component": "Wav2vec encoder", "Parameters (M)": 94.37},
    {"Component": "Intent head", "Parameters (M)": 0.41},
])
write_csv("reported_parameter_counts.csv", params.to_dict(orient="records"))
fig, ax = plt.subplots(figsize=(7.2, 3.8))
sns.barplot(data=params, x="Parameters (M)", y="Component", color=COLORS[4], ax=ax)
for p, v in zip(ax.patches, params["Parameters (M)"]):
    ax.text(v + 1, p.get_y() + p.get_height()/2, f"{v:.2f}M", va="center", fontsize=9)
ax.set_xlim(0, 108)
ax.set_title("Parameter distribution in the reported pipeline")
ax.set_ylabel("")
ax.text(0.01, -0.23, "Total: approximately 105.85M parameters; wav2vec dominates model size.",
        transform=ax.transAxes, fontsize=8)
save(fig, "fig09_parameter_counts")


# 10. Long AISHELL-5 target-absent stream: smoothed KWS events per hour.
aishell = pd.DataFrame([
    {"System / threshold": "Prior hybrid KWS, 0.81", "Smoothed events/hour": 9},
    {"System / threshold": "AISHELL-augmented KWS, 0.81", "Smoothed events/hour": 30},
    {"System / threshold": "AISHELL-augmented KWS, 0.90", "Smoothed events/hour": 12},
    {"System / threshold": "AISHELL-augmented KWS, 0.95", "Smoothed events/hour": 3},
    {"System / threshold": "Presence-gated full path, 0.90", "Smoothed events/hour": 0},
])
write_csv("aishell5_target_absent_stream.csv", aishell.to_dict(orient="records"))
fig, ax = plt.subplots(figsize=(8.0, 4.0))
sns.barplot(data=aishell, x="Smoothed events/hour", y="System / threshold", color=COLORS[3], ax=ax)
ax.set_xlabel("Smoothed detections per hour")
ax.set_ylabel("")
ax.set_title("AISHELL-5 target-absent speech stream (20 minutes)")
ax.text(0.01, -0.23, "Negative-only evaluation: no target wake positives, so this figure does not measure recall.",
        transform=ax.transAxes, fontsize=8)
save(fig, "fig10_aishell5_false_events")


sources = """# Figure data provenance and use notes

Generated by `plot_repo_results.py` on 2026-10-05. Every plotted value is
transcribed from repository evidence; see the source files below. The script
exports a CSV beside each figure so the plotted data can be inspected.

## Sources

- `reports/base_project_report_2026-10-04.md`: Stage 1 quantiles, input-scale
  sweep, presence operating points, current pipeline summary, parameter counts.
- `reports/project_benchmark_2026-10-03.md`: extraction baseline comparison,
  KWS candidate metrics, FSC intent results, latency, AISHELL-5 negative-stream
  false-event rates.
- `reports/continuous_full_pipeline_proxy_2026-10-04.md`: 4-trial continuous
  proxy result at presence thresholds 0.50 and 0.90.
- `reports/artifacts/full_pipeline_cavemove_sample_000/full_pipeline.json`:
  one illustrative CAVEMOVE scene; it is not an aggregate accuracy result.

## Interpretation limits

- Stage 1 quantile bars are from reported summary statistics. Individual
  utterance SI-SDR values were not present in this workspace, so no box plot,
  histogram, or significance interval is inferred.
- KWS candidate rows use different test-set sizes and negative compositions;
  the grouped plot is descriptive, not a controlled paired comparison.
- The full-pipeline proxy is cross-corpus and not speaker-consistent. It is
  shown as a diagnostic, not as an estimate of real-driver accuracy.
- The AISHELL-5 stream contains target-absent speech only. Its false-event
  rate does not measure target wake recall.
- Referenced showcase WAV files and the original raw audio were absent from
  the workspace at generation time. No waveform or spectrogram plot was
  fabricated. A before/after spectrogram requires rerunning the model on an
  available, labeled audio example.
- Model-forward latency excludes the presence head and full streaming I/O.

Figures are vector SVG plus 300-dpi PNG. CSVs contain plotted values.
"""
(OUT / "README.md").write_text(sources, encoding="utf-8")
print(f"Wrote figures, CSVs, and provenance notes to {OUT}")
