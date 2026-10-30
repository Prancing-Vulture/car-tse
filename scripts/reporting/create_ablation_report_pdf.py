#!/usr/bin/env python
"""Create the in-car assistant ablation and robustness PDF report."""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output" / "pdf" / "in_car_voice_assistant_ablation_report_2026-10-11.pdf"
TMP = ROOT / "tmp" / "pdfs" / "ablation_report_2026-10-11"

NAVY = colors.HexColor("#102A43")
INK = colors.HexColor("#243B53")
MUTED = colors.HexColor("#627D98")
TEAL = colors.HexColor("#138A8A")
ORANGE = colors.HexColor("#D97706")
RED = colors.HexColor("#C2410C")
PALE_BLUE = colors.HexColor("#EAF2F8")
PALE_TEAL = colors.HexColor("#E6F6F4")
PALE_ORANGE = colors.HexColor("#FFF3D6")
PALE_RED = colors.HexColor("#FDECEC")
GRID = colors.HexColor("#D9E2EC")


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def esc(value: object) -> str:
    return html.escape(str(value), quote=False)


def pct(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.1f}%"


def number(value: object, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def rate(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.1f}/h"


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def table(data: list[list[object]], widths: list[float], header: bool = True, font_size: float = 7.6, row_backgrounds: dict[int, colors.Color] | None = None) -> Table:
    converted: list[list[object]] = []
    for row_index, row in enumerate(data):
        converted_row: list[object] = []
        for cell in row:
            if isinstance(cell, Paragraph):
                converted_row.append(cell)
            else:
                style = TABLE_HEADER if header and row_index == 0 else TABLE_CELL
                converted_row.append(Paragraph(esc(cell), style))
        converted.append(converted_row)
    result = Table(converted, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ])
        start = 1
        for row_index in range(1, len(data)):
            if row_index % 2:
                commands.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#F7FAFC")))
    else:
        start = 0
    if row_backgrounds:
        for row_index, background in row_backgrounds.items():
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), background))
    result.setStyle(TableStyle(commands))
    return result


def section(title: str) -> list[object]:
    return [
        Spacer(1, 5),
        Paragraph(esc(title), H1),
        Spacer(1, 3),
        Table([[""]], colWidths=[170 * mm], rowHeights=[2], style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), TEAL)])),
        Spacer(1, 8),
    ]


def bullet(text: str) -> Paragraph:
    return Paragraph(f"<font color='{TEAL.hexval()}'>&bull;</font> {esc(text)}", BODY)


def make_charts(data: dict[str, dict], threshold: dict[str, dict], passenger: dict[str, dict], scalar: dict[str, dict]) -> list[Path]:
    TMP.mkdir(parents=True, exist_ok=True)
    charts: list[Path] = []

    names = ["Full", "No TSE", "No presence", "No speaker\nmatching", "No smoothing"]
    keys = ["full", "no_tse", "no_presence", "no_speaker_matching", "no_wake_smoothing"]
    wake = [float(data[key]["target_wake_recall"] or 0) * 100 for key in keys]
    intent = [float(data[key]["target_intent_accuracy"] or 0) * 100 for key in keys]
    actions = [float(data[key]["false_action_rate_per_hour"] or 0) for key in keys]
    x = np.arange(len(names))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.55), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor("white")
    axes[0].bar(x - width / 2, wake, width, label="Wake recall", color="#138A8A")
    axes[0].bar(x + width / 2, intent, width, label="Intent accuracy", color="#D97706")
    axes[0].set_ylim(0, 110)
    axes[0].set_ylabel("Percent")
    axes[0].set_xticks(x, names, fontsize=8)
    axes[0].set_title("Detection and command success")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    for values, offset in ((wake, -width / 2), (intent, width / 2)):
        for i, value in enumerate(values):
            axes[0].text(i + offset, value + 3, f"{value:.0f}%", ha="center", va="bottom", fontsize=7)
    axes[1].bar(x, actions, color="#C2410C")
    axes[1].set_ylabel("False actions per hour")
    axes[1].set_xticks(x, names, fontsize=8)
    axes[1].set_title("Safety tradeoff")
    axes[1].grid(axis="y", alpha=0.25)
    for i, value in enumerate(actions):
        axes[1].text(i, value + max(actions) * 0.02 + 2, f"{value:.0f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Frame-presence candidate at passenger SNR 0 dB", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = TMP / "ablation_overview.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    charts.append(path)

    presence_values = [0.30, 0.50, 0.70, 0.90]
    wake_values = []
    intent_values = []
    action_values = []
    for value in presence_values:
        key = f"presence={value:.2f}|kws=0.50|variant=full"
        row = threshold[key]
        wake_values.append(float(row["target_wake_recall"] or 0) * 100)
        intent_values.append(float(row["target_intent_accuracy"] or 0) * 100)
        action_values.append(float(row["false_action_rate_per_hour"] or 0))
    fig, ax1 = plt.subplots(figsize=(8.4, 3.4))
    ax2 = ax1.twinx()
    ax1.plot(presence_values, wake_values, marker="o", linewidth=2.2, color="#138A8A", label="Wake recall")
    ax1.plot(presence_values, intent_values, marker="o", linewidth=2.2, color="#D97706", label="Intent accuracy")
    ax2.plot(presence_values, action_values, marker="s", linewidth=2.2, color="#C2410C", label="False actions/hour")
    ax1.set_xlabel("Presence threshold")
    ax1.set_ylabel("Recall / accuracy (%)")
    ax2.set_ylabel("False actions per hour")
    ax1.set_ylim(0, 110)
    ax2.set_ylim(0, max(action_values) * 1.18 + 1)
    ax1.set_xticks(presence_values)
    ax1.grid(alpha=0.25)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, fontsize=8, loc="center right")
    ax1.set_title("Full path operating-point tradeoff at KWS threshold 0.50")
    fig.tight_layout()
    path = TMP / "threshold_tradeoff.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    charts.append(path)

    x = np.array([-10, 0, 10])
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    for key, label, color in (("full", "Full", "#138A8A"), ("no_presence", "No presence", "#C2410C"), ("no_wake_smoothing", "No smoothing", "#D97706")):
        y = [float(passenger[str(snr)][key]["false_action_rate_per_hour"] or 0) for snr in x]
        ax.plot(x, y, marker="o", linewidth=2.2, label=label, color=color)
    ax.set_xlabel("Passenger SNR relative to target (dB)")
    ax.set_ylabel("False actions per hour")
    ax.set_title("Passenger loudness sensitivity")
    ax.set_xticks(x)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = TMP / "passenger_tradeoff.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    charts.append(path)

    frame_full = data["full"]
    scalar_full = scalar["full"]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    labels = ["Wake recall", "Intent accuracy", "False actions/h"]
    frame_values = [float(frame_full["target_wake_recall"] or 0) * 100, float(frame_full["target_intent_accuracy"] or 0) * 100, float(frame_full["false_action_rate_per_hour"] or 0)]
    scalar_values = [float(scalar_full["target_wake_recall"] or 0) * 100, float(scalar_full["target_intent_accuracy"] or 0) * 100, float(scalar_full["false_action_rate_per_hour"] or 0)]
    x = np.arange(3)
    ax.bar(x - 0.18, frame_values, 0.36, label="Framewise", color="#138A8A")
    ax.bar(x + 0.18, scalar_values, 0.36, label="Scalar", color="#627D98")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Percent, except false actions/hour")
    ax.set_title("Presence-head architecture comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    for values, offset in ((frame_values, -0.18), (scalar_values, 0.18)):
        for i, value in enumerate(values):
            ax.text(i + offset, value + max(frame_values + scalar_values) * 0.02 + 1, f"{value:.0f}", ha="center", fontsize=7)
    fig.tight_layout()
    path = TMP / "presence_head_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    charts.append(path)
    return charts


def on_page(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.4)
    canvas.line(doc.leftMargin, height - 23 * mm, width - doc.rightMargin, height - 23 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, height - 18 * mm, "IN-CAR VOICE ASSISTANT | ABLATION REPORT")
    canvas.drawRightString(width - doc.rightMargin, 12 * mm, f"Page {doc.page}")
    canvas.drawString(doc.leftMargin, 12 * mm, "Engineering proxy results - not real-driver reliability evidence")
    canvas.restoreState()


styles = getSampleStyleSheet()
TITLE = ParagraphStyle("TitleCustom", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=24, leading=29, textColor=NAVY, alignment=TA_LEFT, spaceAfter=8)
SUBTITLE = ParagraphStyle("SubtitleCustom", parent=styles["Normal"], fontName="Helvetica", fontSize=11, leading=15, textColor=MUTED, alignment=TA_LEFT, spaceAfter=12)
H1 = ParagraphStyle("H1Custom", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=NAVY, spaceBefore=8, spaceAfter=6)
H2 = ParagraphStyle("H2Custom", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=10.5, leading=14, textColor=INK, spaceBefore=7, spaceAfter=4)
BODY = ParagraphStyle("BodyCustom", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.1, leading=13.2, textColor=INK, spaceAfter=6)
SMALL = ParagraphStyle("SmallCustom", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.8, leading=10.5, textColor=MUTED, spaceAfter=4)
CAPTION = ParagraphStyle("CaptionCustom", parent=styles["BodyText"], fontName="Helvetica-Oblique", fontSize=7.7, leading=10, textColor=MUTED, spaceBefore=2, spaceAfter=8)
TABLE_HEADER = ParagraphStyle("TableHeader", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=7.1, leading=8.8, textColor=colors.white, alignment=TA_LEFT)
TABLE_CELL = ParagraphStyle("TableCell", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.1, leading=8.8, textColor=INK, alignment=TA_LEFT)
CODE = ParagraphStyle("CodeCustom", parent=styles["Code"], fontName="Courier", fontSize=6.2, leading=8.1, textColor=INK, backColor=colors.HexColor("#F4F7F9"), borderColor=GRID, borderWidth=0.5, borderPadding=7, spaceBefore=4, spaceAfter=8)
CARD = ParagraphStyle("Card", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=14, leading=17, textColor=NAVY, alignment=TA_CENTER)
CARD_LABEL = ParagraphStyle("CardLabel", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.1, leading=9, textColor=MUTED, alignment=TA_CENTER)


def build_pdf() -> Path:
    matrix = load_json("logs/continuous_ablation_matrix_2026-10-09.json")
    passenger_m10 = load_json("logs/continuous_ablation_matrix_passenger_m10_2026-10-09.json")
    passenger_p10 = load_json("logs/continuous_ablation_matrix_passenger_p10_2026-10-09.json")
    threshold = load_json("logs/threshold_ablation_sweep_2026-10-09.json")
    scalar = load_json("logs/continuous_ablation_matrix_scalar_presence_2026-10-09.json")
    baseline = load_json("logs/full_pipeline_proxy_study_2026-10-09.json")
    aishell = load_json("logs/aishell5_target_absent_dual_domain_stage2_2026-10-10.json")
    aishell_target_kws_current = load_json("logs/aishell5_target_absent_dual_domain_target_kws_current_2026-10-10.json")
    aishell_target_kws_current_sweep = load_json("logs/aishell5_target_absent_dual_domain_target_kws_current_sweep_2026-10-10.json")
    aishell_target_kws_hardneg = load_json("logs/aishell5_target_absent_dual_domain_target_kws_hardneg_2026-10-10.json")
    aishell_target_kws_hardneg_sweep = load_json("logs/aishell5_target_absent_dual_domain_target_kws_hardneg_sweep_2026-10-10.json")
    aishell_target_kws_hardneg_thr070 = load_json("logs/aishell5_target_absent_dual_domain_target_kws_hardneg_thr070_2026-10-10.json")
    runtime = load_json("logs/runtime_ablation_sweep_2026-10-10.json")
    stage1_regression = load_json("logs/stage1_dual_domain_regression_full_2026-10-10.json")
    stage2_presence = load_json("logs/stage2_frame_presence_dual_domain_nonwake_ft1/metrics_last.json")
    dual_matrix = load_json("logs/continuous_ablation_matrix_dual_domain_stage2_operating_2026-10-10.json")
    handoff = load_json("logs/continuous_ablation_matrix_dual_domain_stage2_intent_handoff_2026-10-10.json")
    handoff_m10 = load_json("logs/continuous_ablation_matrix_dual_domain_stage2_intent_handoff_passenger_m10_2026-10-10.json")
    handoff_p10 = load_json("logs/continuous_ablation_matrix_dual_domain_stage2_intent_handoff_passenger_p10_2026-10-10.json")
    command_enrollment = load_json("logs/continuous_ablation_matrix_dual_domain_stage2_command_enrollment_2026-10-10.json")
    same_speaker_other = load_json("logs/continuous_ablation_matrix_dual_domain_stage2_same_speaker_other_2026-10-10.json")
    hardneg = load_json("logs/continuous_ablation_matrix_dual_domain_stage2_hardneg_2026-10-10.json")
    hardneg_m10 = load_json("logs/continuous_ablation_matrix_dual_domain_stage2_hardneg_passenger_m10_2026-10-10.json")
    safety_m10 = load_json("logs/continuous_ablation_matrix_stage2_intent_confidence_gate_passenger_m10_2026-10-10.json")
    safety_p10 = load_json("logs/continuous_ablation_matrix_stage2_intent_confidence_gate_passenger_p10_2026-10-10.json")
    ft2_gate = load_json("logs/continuous_ablation_matrix_stage2_ft2_head_same_speaker_gate_sweep_2026-10-10.json")
    safety_v2_m10 = load_json("logs/continuous_full_pipeline_proxy_safety_diagnostic_same_speaker_m10_v2_2026-10-11.json")
    safety_v2_p0 = load_json("logs/continuous_full_pipeline_proxy_safety_diagnostic_same_speaker_p0_presence012_2026-10-10.json")
    safety_v2_p10 = load_json("logs/continuous_full_pipeline_proxy_safety_diagnostic_same_speaker_p10_presence012_2026-10-11.json")
    intent_calibration_ft8 = load_json("logs/intent_confidence_calibration_ft8_tse_domain_waveform_val_2026-10-11.json")
    aishell_adapted = load_json("logs/aishell5_continuous_targetkws_hardneg_aishell5_ft1_p015_sg024_kws050_smooth22_1200s_2026-10-11.json")
    aishell_adapted_long = load_json("logs/aishell5_continuous_targetkws_hardneg_aishell5_ft1_p015_sg024_kws050_smooth22_eval2_all18_2026-10-10.json")
    aishell_adapted_all18 = load_json("logs/aishell5_continuous_targetkws_hardneg_aishell5_ft1_p015_sg024_kws050_smooth22_eval2_all18_dedup_2026-10-11.json")
    adapted_proxy_m10 = load_json("logs/continuous_full_pipeline_proxy_hardneg_aishell5_neg_splitgate_p015_wake024_cmd022_same_speaker_m10_2026-10-11.json")
    adapted_proxy_p0 = load_json("logs/continuous_full_pipeline_proxy_hardneg_aishell5_neg_splitgate_p015_wake024_cmd022_same_speaker_p0_2026-10-10.json")
    adapted_proxy_p10 = load_json("logs/continuous_full_pipeline_proxy_hardneg_aishell5_neg_splitgate_p015_wake024_cmd022_same_speaker_p10_2026-10-11.json")
    aishell_adapted_sg025 = load_json("logs/aishell5_continuous_targetkws_hardneg_aishell5_ft1_p015_sg025_kws050_smooth22_1200s_2026-10-11.json")
    aishell_adapted_sg0245 = load_json("logs/aishell5_continuous_targetkws_hardneg_aishell5_ft1_p015_sg0245_kws050_smooth22_1200s_2026-10-11.json")
    proxy_sg025_m10 = load_json("logs/continuous_full_pipeline_proxy_hardneg_aishell5_neg_splitgate_sg025_p015_wake025_cmd022_same_speaker_m10_2026-10-11.json")
    proxy_sg0245_m10 = load_json("logs/continuous_full_pipeline_proxy_hardneg_aishell5_neg_splitgate_sg0245_p015_wake0245_cmd022_same_speaker_m10_2026-10-11.json")
    mined_v2_proxy50 = load_json("logs/continuous_full_pipeline_proxy_hardneg_aishell5_mined_v2_p015_sg024_kws050_same_speaker_m10_2026-10-11.json")
    mined_v2_proxy40 = load_json("logs/continuous_full_pipeline_proxy_hardneg_aishell5_mined_v2_p015_sg024_kws040_same_speaker_m10_2026-10-11.json")
    mined_v3_proxy = load_json("logs/continuous_full_pipeline_proxy_hardneg_aishell5_mined_v3_from_pos_p015_sg024_kws049_same_speaker_m10_2026-10-11.json")
    mined_v4_proxy = load_json("logs/continuous_full_pipeline_proxy_hardneg_aishell5_mined_v4_full_from_pos_p015_sg024_kws050_same_speaker_m10_2026-10-11.json")
    transfer_v2_proxy = load_json("logs/continuous_full_pipeline_proxy_hardneg_aishell5_v2_full_from_target_p015_sg024_kws050_same_speaker_m10_2026-10-11.json")
    classifier_only_proxy = load_json("logs/continuous_full_pipeline_proxy_hardneg_aishell5_v3_classifier_only_p015_sg024_kws052_same_speaker_m10_2026-10-11.json")
    latency_gpu = load_json("logs/full_pipeline_latency_gpu_2026-10-11.json")
    latency_cpu = load_json("logs/full_pipeline_latency_cpu_2026-10-11.json")
    latency_fp16 = load_json("logs/full_pipeline_latency_gpu_fp16_2026-10-11.json")
    latency_int8 = load_json("logs/full_pipeline_latency_cpu_int8_2026-10-11.json")
    latency_crnn = load_json("logs/full_pipeline_latency_crnn_intent_gpu_2026-10-11.json")
    crnn_proxy = load_json("logs/continuous_full_pipeline_proxy_crnn_intent_candidate_same_speaker_m10_2026-10-11.json")
    latency_distill_crnn = load_json("logs/full_pipeline_latency_norm_crnn_distill_t3_a05_gpu_2026-10-11.json")
    distill_val = load_json("logs/stage4_intent_tse_domain_distill_t3_a05_val.json")
    distill_clean = load_json("logs/stage4_intent_tse_domain_distill_t3_a05_fsc_test_clean.json")
    distill_proxy = load_json("logs/continuous_full_pipeline_proxy_norm_crnn_distill_t3_a05_conf035_same_speaker_m10_2026-10-11.json")
    medium_intent_val_clean = load_json("logs/stage4_intent_tse_domain_distill_medium_t3_a05/metrics_val_clean.json")
    medium_intent_val_cabin = load_json("logs/stage4_intent_tse_domain_distill_medium_t3_a05/metrics_val_cabin.json")
    medium_intent_clean = load_json("logs/stage4_intent_tse_domain_distill_medium_t3_a05/metrics_test_clean.json")
    medium_intent_proxy = load_json("logs/continuous_full_pipeline_proxy_medium_student_10scenes_2026-10-12.json")
    medium_intent_latency = load_json("logs/full_pipeline_latency_medium_student_gpu_2026-10-12.json")
    wav2vec_proxy_32 = load_json("logs/continuous_full_pipeline_proxy_wav2vec_reference_same_speaker_m10_32scenes_2026-10-11.json")
    wav2vec_proxy_kws055 = load_json("logs/continuous_full_pipeline_proxy_wav2vec_reference_kws055_same_speaker_m10_32scenes_2026-10-11.json")
    wav2vec_proxy_kws060 = load_json("logs/continuous_full_pipeline_proxy_wav2vec_reference_kws060_same_speaker_m10_32scenes_2026-10-11.json")
    aishell_kws_sweep = load_json("logs/aishell5_continuous_targetkws_hardneg_aishell5_ft1_p015_sg024_kws055_sweep_eval2_all18_2026-10-11.json")
    readiness = load_json("logs/production_readiness_snapshot_2026-10-12.json")
    smart_negative = load_json("logs/smart_speaker_command_negative_250_2026-10-11.json")
    smart_negative_candidate = load_json("logs/smart_speaker_command_negative_250_candidate_2026-10-11.json")
    smart_negative_dedup = load_json("logs/smart_speaker_command_negative_250_dedup_baseline_2026-10-11.json")
    smart_negative_candidate_dedup = load_json("logs/smart_speaker_command_negative_250_dedup_candidate_2026-10-11.json")
    smart_negative_thin = load_json("logs/smart_speaker_command_negative_250_thin_candidate_2026-10-11.json")
    smart_negative_thin_kws070 = load_json("logs/smart_speaker_command_negative_250_thin_candidate_kws070_2026-10-11.json")
    smart_proxy_full = load_json("logs/continuous_full_pipeline_proxy_smart_speaker_hardneg_same_speaker_m10_32scenes_2026-10-11.json")
    smart_proxy_thin = load_json("logs/continuous_full_pipeline_proxy_smart_speaker_thin_same_speaker_m10_8scenes_2026-10-11.json")
    smart_proxy_thin_kws070 = load_json("logs/continuous_full_pipeline_proxy_smart_speaker_thin_kws070_same_speaker_m10_8scenes_2026-10-11.json")
    smart_proxy_distill = load_json("logs/continuous_full_pipeline_proxy_smart_speaker_distill_same_speaker_m10_8scenes_2026-10-11.json")
    personal_proxy = load_json("logs/continuous_full_pipeline_proxy_personal_presence_same_speaker_m10_8scenes_2026-10-11.json")
    personal_proxy_p040 = load_json("logs/continuous_full_pipeline_proxy_personal_presence_p040_same_speaker_m10_8scenes_2026-10-11.json")
    enrollment_bank1_proxy = load_json("logs/continuous_full_pipeline_proxy_enrollment_bank1_m10_8scenes_2026-10-11.json")
    enrollment_bank3_proxy = load_json("logs/continuous_full_pipeline_proxy_enrollment_bank3_m10_8scenes_2026-10-11.json")
    verifier_calibration = load_json("logs/speaker_verifier_calibration_gsc_cache_stage1_ft2_2026-10-11.json")
    verifier_calibration_proxy = load_json("logs/continuous_full_pipeline_proxy_late_fusion_gsc_cache_calibrated_m10_2026-10-11.json")
    wavlm_verifier = load_json("logs/wavlm_speaker_verifier_cabin_diagnostic_2026-10-11.json")
    wavlm_head = load_json("logs/speaker_verifier_wavlm_cabin_ft1_2026-10-11.json")
    wavlm_latency = load_json("logs/wavlm_speaker_verifier_latency_2026-10-11.json")
    compact_verifier = load_json("logs/compact_speaker_verifier_gsc_ft1_2026-10-11.json")
    compact_latency_gpu = load_json("logs/compact_speaker_verifier_latency_gpu_2026-10-11.json")
    compact_proxy = load_json("logs/continuous_full_pipeline_proxy_compact_verifier_operating_diagnostic_2026-10-11.json")
    anir_presence_proxy = load_json("logs/continuous_full_pipeline_proxy_personal_presence_anir_same_speaker_m10_8scenes_2026-10-11.json")
    personal_activity_train = load_json("logs/stage2_personal_activity_gsc_cavemove_ft1/metrics_last.json")
    personal_activity_p025 = load_json("logs/continuous_full_pipeline_proxy_personal_activity_operating_p025_same_speaker_m10_8scenes_2026-10-12.json")
    personal_activity_p030 = load_json("logs/continuous_full_pipeline_proxy_personal_activity_operating_p030_same_speaker_m10_8scenes_2026-10-12.json")
    personal_activity_p040 = load_json("logs/continuous_full_pipeline_proxy_personal_activity_same_speaker_m10_8scenes_2026-10-12.json")
    personal_activity_group_train = load_json("logs/stage2_personal_activity_gsc_cavemove_group_ft1/metrics_last.json")
    personal_activity_group_p025 = load_json("logs/continuous_full_pipeline_proxy_personal_activity_group_p025_same_speaker_m10_8scenes_2026-10-12.json")
    personal_activity_group_p035 = load_json("logs/continuous_full_pipeline_proxy_personal_activity_group_p035_same_speaker_m10_8scenes_2026-10-12.json")
    personal_activity_aishell = load_json("logs/aishell5_target_absent_personal_activity_group_all18_presence_sweep_2026-10-12.json")
    personal_activity_aishell_margin = load_json("logs/aishell5_target_absent_personal_activity_group_all18_margin_m067_2026-10-12.json")
    personal_activity_aishell_margin_targettop = load_json("logs/aishell5_target_absent_personal_activity_group_all18_margin_targettop_m030_2026-10-12.json")
    remasc_reference_stress = load_json("logs/continuous_full_pipeline_proxy_remasc_vehicle_stress_8scenes_2026-10-12.json")
    remasc_activity_p025_stress = load_json("logs/continuous_full_pipeline_proxy_remasc_personal_activity_group_p025_8scenes_2026-10-12.json")
    remasc_activity_p035_stress = load_json("logs/continuous_full_pipeline_proxy_remasc_personal_activity_group_p035_8scenes_2026-10-12.json")
    remasc_activity_asymmetric_stress = load_json("logs/continuous_full_pipeline_proxy_remasc_personal_activity_group_p025_cmdp035_8scenes_2026-10-12.json")
    remasc_activity_margin_m066 = load_json("logs/continuous_full_pipeline_proxy_remasc_personal_activity_group_p025_margin_m0p66_8scenes_2026-10-12.json")
    remasc_activity_margin_m067 = load_json("logs/continuous_full_pipeline_proxy_remasc_personal_activity_group_p025_margin_m0p67_8scenes_2026-10-12.json")
    remasc_activity_margin_m068 = load_json("logs/continuous_full_pipeline_proxy_remasc_personal_activity_group_p025_margin_m0p68_8scenes_2026-10-12.json")
    remasc_activity_margin_m070 = load_json("logs/continuous_full_pipeline_proxy_remasc_personal_activity_group_p025_margin_m0p70_8scenes_2026-10-12.json")
    remasc_activity_targettop_observe = load_json("logs/continuous_full_pipeline_proxy_remasc_personal_activity_targettop_observe_8scenes_2026-10-12.json")
    remasc_activity_targettop_m030 = load_json("logs/continuous_full_pipeline_proxy_remasc_personal_activity_targettop_m030_8scenes_2026-10-12.json")
    remasc_activity_framewisetop_observe = load_json("logs/continuous_full_pipeline_proxy_remasc_personal_activity_framewisetop_observe_8scenes_2026-10-12.json")
    remasc_activity_framewisetop_m030 = load_json("logs/continuous_full_pipeline_proxy_remasc_personal_activity_framewisetop_m030_8scenes_2026-10-12.json")
    remasc_stream_hop050 = load_json("logs/continuous_full_pipeline_proxy_remasc_personal_activity_margin_m067_hop050_8scenes_2026-10-12.json")
    remasc_stream_hop025 = load_json("logs/continuous_full_pipeline_proxy_remasc_personal_activity_margin_m067_hop025_8scenes_2026-10-12.json")
    remasc_stream_hop025_delay2 = load_json("logs/continuous_full_pipeline_proxy_remasc_personal_activity_margin_m067_hop025_delay2_8scenes_2026-10-12.json")
    remasc_stream_hop025_delay4 = load_json("logs/continuous_full_pipeline_proxy_remasc_personal_activity_margin_m067_hop025_delay4_8scenes_2026-10-12.json")
    remasc_stream_fast = load_json("logs/continuous_full_pipeline_proxy_remasc_personal_activity_margin_m067_fast_stream_hop025_8scenes_2026-10-12.json")
    aishell_stream_ref8 = load_json("logs/aishell5_target_absent_personal_activity_hop025_ref8_1200s_2026-10-12.json")
    aishell_stream_ref16 = load_json("logs/aishell5_target_absent_personal_activity_hop025_ref16_1200s_2026-10-12.json")
    stage1_regression_ft2 = load_json("logs/stage1_dual_domain_regression_ft2_2026-10-11.json")
    ft2_proxy_negative = load_json("logs/continuous_full_pipeline_proxy_ft2_stage1_consistent_negative_same_speaker_m10_32scenes_2026-10-11.json")
    ft2_proxy_positive = load_json("logs/continuous_full_pipeline_proxy_ft2_stage1_consistent_pos_same_speaker_m10_32scenes_2026-10-11.json")
    ft2_presence_proxy = load_json("logs/continuous_full_pipeline_proxy_ft2_stage1_consistent_pos_presence_same_speaker_m10_32scenes_2026-10-11.json")
    ft1_positive_fullmatrix = load_json("logs/continuous_full_pipeline_proxy_ft1_positive_fullmatrix_same_speaker_m10_32scenes_2026-10-11.json")
    ft1_positive_fullmatrix_kws055 = load_json("logs/continuous_full_pipeline_proxy_ft1_positive_fullmatrix_kws055_same_speaker_m10_32scenes_2026-10-11.json")
    aishell_positive_all18 = load_json("logs/aishell5_continuous_targetkws_hardneg_aishell5_pos_ft1_p015_sg022_kws050_sweep_eval2_all18_2026-10-11.json")

    kws_sweep_summary: dict[float, dict[str, float]] = {}
    for index, threshold_value in enumerate((0.50, 0.55, 0.60)):
        rows = [session["threshold_sweep"][index] for session in aishell_kws_sweep["session_results"]]
        duration = float(aishell_kws_sweep["duration_seconds"])
        kws_sweep_summary[threshold_value] = {
            "full_path_smoothed_event_rate_per_hour": sum(row["full_path_smoothed_events"] for row in rows) * 3600.0 / max(duration, 1.0e-6),
            "kws_only_smoothed_event_rate_per_hour": sum(row["kws_only_smoothed_events"] for row in rows) * 3600.0 / max(duration, 1.0e-6),
        }

    charts = make_charts(matrix["summary"], threshold["summary"], {"-10": passenger_m10["summary"], "0": matrix["summary"], "10": passenger_p10["summary"]}, scalar["summary"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=17 * mm,
        leftMargin=17 * mm,
        topMargin=29 * mm,
        bottomMargin=18 * mm,
        title="In-Car Voice Assistant Ablation and Robustness Report",
        author="Codex",
        subject="TSE, presence, speaker matching, wake smoothing, and threshold ablations",
    )
    story: list[object] = []

    # Cover and executive summary.
    story.append(Spacer(1, 17 * mm))
    story.append(Paragraph("In-Car Voice Assistant", TITLE))
    story.append(Paragraph("Ablation and robustness report", ParagraphStyle("CoverSub", parent=SUBTITLE, fontSize=16, leading=20, textColor=TEAL)))
    story.append(Paragraph("Target speaker extraction -> presence -> wake word -> intent", SUBTITLE))
    story.append(Spacer(1, 5 * mm))
    cover_box = Table([
        [Paragraph("Study date", CARD_LABEL), Paragraph("Evidence status", CARD_LABEL), Paragraph("Training", CARD_LABEL)],
        [Paragraph("2026-10-12 update", CARD), Paragraph("Proxy diagnostics", CARD), Paragraph("Bounded candidates complete", CARD)],
    ], colWidths=[54 * mm, 54 * mm, 54 * mm], rowHeights=[11 * mm, 17 * mm])
    cover_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
        ("BOX", (0, 0), (-1, -1), 0.8, TEAL),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(cover_box)
    story.append(Spacer(1, 9 * mm))
    story.append(Paragraph("Executive finding", H1))
    story.append(Paragraph(
        "The current reference path combines dual-domain Stage 1 extraction, framewise presence, target-speaker KWS, a split speaker gate, and the existing intent head. Under the exact 2-window/2-hit runtime it produces 4.628 full-path false wakes/hour over 3.889 hours of target-absent AISHELL-5 Eval2 (96.673/hour at KWS-only) and 100% wake/intent with 0 false actions/hour on the same-speaker CAVEMOVE proxy at passenger SNR -10 and +10 dB. The shorter 1,200-second slice is 3.0/hour. These are not real-driver reliability scores because the workspace still has zero continuous same-driver recordings.",
        BODY,
    ))
    cards = Table([
        [Paragraph("0", CARD), Paragraph("8", CARD), Paragraph("3.889 h", CARD), Paragraph("26 passed", CARD)],
        [Paragraph("real-driver sessions", CARD_LABEL), Paragraph("proxy scenes per run", CARD_LABEL), Paragraph("AISHELL target-absent check", CARD_LABEL), Paragraph("tests (1 skipped)", CARD_LABEL)],
    ], colWidths=[41 * mm] * 4, rowHeights=[13 * mm, 10 * mm])
    cards.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), PALE_RED),
        ("BACKGROUND", (1, 0), (1, -1), PALE_TEAL),
        ("BACKGROUND", (2, 0), (2, -1), PALE_ORANGE),
        ("BACKGROUND", (3, 0), (3, -1), PALE_BLUE),
        ("BOX", (0, 0), (-1, -1), 0.5, GRID),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(cards)
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("Important evidence boundary", H2))
    story.append(Paragraph(
        "The positive scenes use a Marvin clip and an FSC command clip from different public corpora. Their speaker identity is not verified. AISHELL-5 supplies a real in-car target-absent diagnostic only. No result in this PDF should be described as a real-driver end-to-end accuracy or production false-accept rate.",
        BODY,
    ))
    story.append(PageBreak())

    # Study design and data status.
    story.extend(section("1. Study design and data status"))
    story.append(Paragraph("Question", H2))
    story.append(Paragraph(
        "Can the assistant keep a target driver through enrollment, detect Marvin, understand the following command, and avoid acting on passenger or background speech? The requested real-driver study requires the same speaker's enrollment, wake word, and command in continuous cabin recordings, plus passenger and no-command intervals.",
        BODY,
    ))
    data_rows = [
        ["Evidence source", "Used for", "Status / limitation"],
        ["Real custom in-car sessions", "Positive driver evaluation", "0 sessions; collection still required"],
        ["Cross-corpus proxy", "Continuous cascade debugging", "8 scenes per run; speaker consistency false"],
        ["AISHELL-5 Eval2", "Real-cabin target-absent rejection", "14,001.9 seconds; no positive wake/command"],
        ["CAVEMOVE", "Cabin noise and multichannel mixtures", "Augmentation material, not driver sessions"],
    ]
    story.append(table(data_rows, [43 * mm, 55 * mm, 72 * mm]))
    story.append(Spacer(1, 5))
    story.append(Paragraph(
        "The official AISHELL-5 release identifies the dataset as CC BY-SA 4.0: https://us.openslr.org/159/. Commercial or sign-in datasets were not treated as evidence without project permission.",
        SMALL,
    ))
    story.append(Paragraph("Proxy scene construction", H2))
    for item in [
        "14 seconds per scene, with target wake at about 1 second and target command at about 2.1 seconds.",
        "Passenger Marvin near 6 seconds and passenger non-wake speech near 9 seconds.",
        "CAVEMOVE multichannel cabin noise at -5, 0, 10, and 20 dB cabin SNR.",
        "Passenger loudness sweep at -10, 0, and +10 dB relative to the target.",
        "Stage 1 context is 3 seconds and stream hop is 1 second; reported wake latency is window-center latency, not measured wall-clock response time.",
    ]:
        story.append(bullet(item))
    story.append(Spacer(1, 5))
    story.append(Paragraph("Current candidate checkpoints", H2))
    checkpoints = matrix.get("checkpoints", {})
    checkpoint_rows = [["Component", "Checkpoint"]] + [[key, value] for key, value in checkpoints.items()]
    story.append(table(checkpoint_rows, [50 * mm, 120 * mm]))

    # Main ablation matrix.
    story.append(PageBreak())
    story.extend(section("2. Component ablation matrix"))
    story.append(Paragraph(
        "This is the primary same-scene ablation at passenger SNR 0 dB using the target-KWS candidate configuration. Each row uses identical scenes and differs only in the named component. False actions are post-wake command events outside the labeled target command region; they are a proxy diagnostic, not a human action-rate estimate.",
        BODY,
    ))
    s = matrix["summary"]
    rows = [["Variant", "Wake recall", "Intent accuracy", "False actions/h", "Latency mean / p95"]]
    for key, label in [("full", "Full"), ("no_tse", "No TSE"), ("no_presence", "No presence"), ("no_speaker_matching", "No speaker matching"), ("no_wake_smoothing", "No wake smoothing"), ("direct_command_oracle", "Direct command oracle")]:
        row = s[key]
        rows.append([label, pct(row["target_wake_recall"]), pct(row["target_intent_accuracy"]), rate(row["false_action_rate_per_hour"]), f"{number(row['target_wake_latency_mean_seconds'], 2)} / {number(row['target_wake_latency_p95_seconds'], 2)} s"])
    story.append(table(rows, [45 * mm, 26 * mm, 29 * mm, 31 * mm, 39 * mm]))
    story.append(Spacer(1, 4))
    story.append(Image(str(charts[0]), width=170 * mm, height=56 * mm))
    story.append(Paragraph("Figure 1. The current frame-presence candidate is conservative. Disabling presence greatly increases recall but produces many proxy false actions.", CAPTION))
    story.append(Paragraph("Interpretation", H2))
    for item in [
        "Presence checking is safety-critical in this proxy. Removing it produced 100% target wake recall but 417.9 false actions/hour.",
        "One-hit wake detection produced 100% wake recall but 0% intent accuracy in this run, indicating that accepting a wake is not enough; the post-wake command window is still unstable.",
        "Speaker matching did not change the aggregate result at this operating point. That result is inconclusive without same-speaker passenger recordings.",
        "The no-TSE row is an intentionally harsh diagnostic. Its raw mixture is outside the training distribution of downstream heads, so it is not a candidate architecture.",
        "The direct command row uses labeled command segmentation and is an oracle baseline, not a deployable streaming path.",
    ]:
        story.append(bullet(item))

    # Passenger and threshold.
    story.extend(section("3. Passenger loudness ablation"))
    story.append(Paragraph("The passenger loudness sweep tests whether the safety behavior changes when passenger speech is louder or quieter than the target. Negative SNR means the passenger is louder.", BODY))
    passenger_rows = [["Passenger SNR", "Variant", "Wake recall", "Intent accuracy", "False actions/h"]]
    for snr, report in [(-10, passenger_m10), (0, matrix), (10, passenger_p10)]:
        for key, label in [("full", "Full"), ("no_tse", "No TSE"), ("no_presence", "No presence"), ("no_wake_smoothing", "No smoothing")]:
            row = report["summary"][key]
            passenger_rows.append([f"{snr:+d} dB", label, pct(row["target_wake_recall"]), pct(row["target_intent_accuracy"]), rate(row["false_action_rate_per_hour"])])
    story.append(table(passenger_rows, [27 * mm, 39 * mm, 29 * mm, 32 * mm, 42 * mm]))
    story.append(Spacer(1, 4))
    story.append(Image(str(charts[2]), width=170 * mm, height=57 * mm))
    story.append(Paragraph("Figure 2. Louder passenger speech increases proxy false actions, especially when the presence gate is removed.", CAPTION))

    story.extend(section("4. Threshold operating-point ablation"))
    story.append(Paragraph("The grid replayed presence thresholds 0.30, 0.50, 0.70, and 0.90 against KWS thresholds 0.30, 0.40, 0.50, and 0.60. The expensive Stage 1 scores were computed once per scene and the state machine was replayed for each cell.", BODY))
    selected = [
        ("0.30", "0.50"),
        ("0.50", "0.50"),
        ("0.70", "0.50"),
        ("0.90", "0.50"),
    ]
    threshold_rows = [["Presence", "KWS", "Wake recall", "Intent accuracy", "False actions/h"]]
    for p, k in selected:
        row = threshold["summary"][f"presence={p}|kws={k}|variant=full"]
        threshold_rows.append([p, k, pct(row["target_wake_recall"]), pct(row["target_intent_accuracy"]), rate(row["false_action_rate_per_hour"])])
    story.append(table(threshold_rows, [30 * mm, 26 * mm, 39 * mm, 39 * mm, 46 * mm]))
    story.append(Spacer(1, 4))
    story.append(Image(str(charts[1]), width=170 * mm, height=56 * mm))
    story.append(Paragraph("Figure 3. Lowering the presence threshold improves proxy recall, but false actions rise sharply. KWS threshold changes were less influential than presence threshold changes in this grid.", CAPTION))
    story.append(Paragraph("Practical reading", H2))
    story.append(Paragraph("The current p=0.70 operating point is a conservative starting point for safety, not a claim of good recognition. The p=0.50 result is a candidate only for tuning experiments because its false-action rate is unacceptable as a final operating point. Proper threshold selection must use a consented tuning split and then be frozen before held-out testing.", BODY))

    # Presence head architecture.
    story.extend(section("5. Scalar versus framewise presence"))
    story.append(Paragraph("The target-conditioned KWS setup was kept fixed while changing the presence head from the candidate framewise checkpoint to the scalar cabin-trained checkpoint.", BODY))
    frame = matrix["summary"]["full"]
    scalar_row = scalar["summary"]["full"]
    presence_rows = [
        ["Presence head", "Wake recall", "Intent accuracy", "False actions/h", "Latency mean / p95"],
        ["Framewise", pct(frame["target_wake_recall"]), pct(frame["target_intent_accuracy"]), rate(frame["false_action_rate_per_hour"]), f"{number(frame['target_wake_latency_mean_seconds'], 2)} / {number(frame['target_wake_latency_p95_seconds'], 2)} s"],
        ["Scalar", pct(scalar_row["target_wake_recall"]), pct(scalar_row["target_intent_accuracy"]), rate(scalar_row["false_action_rate_per_hour"]), f"{number(scalar_row['target_wake_latency_mean_seconds'], 2)} / {number(scalar_row['target_wake_latency_p95_seconds'], 2)} s"],
    ]
    story.append(table(presence_rows, [42 * mm, 28 * mm, 32 * mm, 35 * mm, 43 * mm]))
    story.append(Spacer(1, 4))
    story.append(Image(str(charts[3]), width=145 * mm, height=54 * mm))
    story.append(Paragraph("Figure 4. Scalar presence improves proxy recall but admits more false actions and has more variable timing.", CAPTION))
    story.append(Paragraph("This is a useful candidate for real-driver tuning, but not a replacement decision. The proxy does not contain the required continuous same-driver enrollment, passenger, and command events.", BODY))

    # New runtime-only sweep.
    story.append(PageBreak())
    story.extend(section("6. Runtime-only ablation sweep"))
    story.append(Paragraph(
        "The 2026-10-10 sweep kept all trained checkpoints fixed and replayed runtime choices on the same eight proxy scenes. It covered 0.5 and 1.0 second hops, four temporal rules, 0.5/1.0/1.5 second enrollment durations, simulated channel subsets, speaker-gate thresholds, and command-capture timeouts. No training was started.",
        BODY,
    ))
    temporal_rows = [["Hop", "Temporal rule", "Wake recall", "Intent accuracy", "False actions/h", "Latency"]]
    temporal_labels = [
        ("one_hit", "one hit"),
        ("two_of_two", "two of two"),
        ("two_of_three_refractory", "two of three + refractory"),
        ("three_of_five_refractory", "three of five + refractory"),
    ]
    for hop_key in ("0.5s", "1s"):
        for rule_key, label in temporal_labels:
            row = runtime["temporal"][hop_key][rule_key]["summary"]["full"]
            temporal_rows.append([
                hop_key,
                label,
                pct(row["target_wake_recall"]),
                pct(row["target_intent_accuracy"]),
                rate(row["false_action_rate_per_hour"]),
                f"{number(row['target_wake_latency_mean_seconds'], 2)} s",
            ])
    story.append(table(temporal_rows, [18 * mm, 48 * mm, 27 * mm, 29 * mm, 30 * mm, 18 * mm]))
    story.append(Spacer(1, 5))
    story.append(Paragraph(
        "Stricter temporal confirmation suppresses proxy false actions but also suppresses the target wake. The 1.0 second, two-of-two setting retained 0 false actions/hour but reached only 25% wake recall and 12.5% intent accuracy.",
        BODY,
    ))
    story.append(Paragraph("Enrollment and front-end results", H2))
    enrollment_rows = [["Enrollment", "Wake recall", "Intent accuracy", "False actions/h"]]
    for key in ("0.5s", "1s", "1.5s"):
        row = runtime["enrollment"][key]["summary"]
        enrollment_rows.append([key, pct(row["target_wake_recall"]), pct(row["target_intent_accuracy"]), rate(row["false_action_rate_per_hour"])])
    story.append(table(enrollment_rows, [34 * mm, 40 * mm, 43 * mm, 53 * mm]))
    story.append(Spacer(1, 5))
    frontend_rows = [["Front-end", "Wake recall", "Intent accuracy", "False actions/h", "Target presence"]]
    for key, label in (("all4", "All 4 channels"), ("mono", "Mono repeated"), ("front_pair", "Front pair"), ("rear_pair", "Rear pair")):
        row = runtime["frontend"][key]["summary"]
        frontend_rows.append([label, pct(row["target_wake_recall"]), pct(row["target_intent_accuracy"]), rate(row["false_action_rate_per_hour"]), pct(row["target_presence_pass_rate_mean"])])
    story.append(table(frontend_rows, [42 * mm, 30 * mm, 34 * mm, 32 * mm, 32 * mm]))
    story.append(Spacer(1, 5))
    story.append(Paragraph(
        "Enrollment duration and simulated channel selection did not materially change the aggregate outcome. These channel modes are simulations, not recordings from alternate vehicle microphone placements.",
        BODY,
    ))
    story.append(PageBreak())
    story.extend(section("7. Speaker gate and timeout diagnostics"))
    story.append(Paragraph(
        "The speaker gate used Stage-1 cosine similarity. Its mean similarity was only 0.031 across proxy windows. A threshold of 0.0 changed no outcome; thresholds of 0.2 and above rejected every target window. Command timeouts of 1, 2, 4, and 6 calls also produced the same aggregate result. These controls require same-driver tuning data before they can be selected.",
        BODY,
    ))
    gate_rows = [["Gate threshold", "Wake recall", "Intent accuracy", "False actions/h", "Gate pass"]]
    for key, label in (("disabled", "Disabled"), ("-0.2", "-0.2"), ("0", "0.0"), ("0.2", "0.2"), ("0.4", "0.4")):
        row = runtime["speaker_gate"][key]["summary"]
        gate_rows.append([label, pct(row["target_wake_recall"]), pct(row["target_intent_accuracy"]), rate(row["false_action_rate_per_hour"]), pct(row["speaker_gate_pass_rate_mean"])])
    story.append(table(gate_rows, [32 * mm, 30 * mm, 34 * mm, 36 * mm, 38 * mm]))
    story.append(Spacer(1, 7))
    story.append(Paragraph("Runtime sweep conclusion", H2))
    for item in [
        "Temporal smoothing is the main runtime precision/recall control, but no tested setting is strong enough to claim reliable end-to-end operation.",
        "The current speaker gate is uncalibrated in this domain; higher thresholds eliminate target detection rather than improving selectivity.",
        "The next decisive experiment remains the real-driver pilot with speaker-disjoint tuning and held-out test sessions.",
    ]:
        story.append(bullet(item))
    story.append(Paragraph("Primary source files", H2))
    sources = [
        "reports/pilot_evaluation_status_2026-10-09.md",
        "reports/additional_ablation_results_2026-10-09.md",
        "logs/continuous_ablation_matrix_2026-10-09.json",
        "logs/continuous_ablation_matrix_passenger_m10_2026-10-09.json",
        "logs/continuous_ablation_matrix_passenger_p10_2026-10-09.json",
        "logs/threshold_ablation_sweep_2026-10-09.json",
        "logs/continuous_ablation_matrix_scalar_presence_2026-10-09.json",
        "logs/aishell5_target_absent_study_2026-10-09.json",
        "reports/runtime_ablation_sweep_2026-10-10.md",
        "logs/runtime_ablation_sweep_2026-10-10.json",
        "reports/stage1_dual_domain_study_2026-10-10.md",
        "reports/dual_domain_stage2_handoff_study_2026-10-10.md",
        "logs/stage1_dual_domain_regression_full_2026-10-10.json",
        "logs/stage2_frame_presence_dual_domain_nonwake_ft1/metrics_last.json",
        "logs/continuous_ablation_matrix_dual_domain_stage2_intent_handoff_2026-10-10.json",
        "logs/continuous_ablation_matrix_dual_domain_stage2_command_enrollment_2026-10-10.json",
        "logs/continuous_ablation_matrix_dual_domain_stage2_same_speaker_other_2026-10-10.json",
        "logs/continuous_ablation_matrix_dual_domain_stage2_hardneg_2026-10-10.json",
        "logs/continuous_ablation_matrix_dual_domain_stage2_hardneg_passenger_m10_2026-10-10.json",
        "reports/joint_safety_threshold_sweep_2026-10-10.md",
        "reports/wake_model_and_stage1_followup_2026-10-10.md",
        "logs/continuous_ablation_matrix_stage2_intent_confidence_gate_passenger_m10_2026-10-10.json",
        "logs/continuous_ablation_matrix_stage2_intent_confidence_gate_passenger_p10_2026-10-10.json",
        "logs/continuous_ablation_matrix_stage2_ft2_head_same_speaker_gate_sweep_2026-10-10.json",
        "logs/aishell5_target_absent_dual_domain_stage2_2026-10-10.json",
        "logs/aishell5_target_absent_dual_domain_target_kws_current_2026-10-10.json",
        "logs/aishell5_target_absent_dual_domain_target_kws_current_sweep_2026-10-10.json",
        "logs/aishell5_target_absent_dual_domain_target_kws_hardneg_2026-10-10.json",
        "logs/aishell5_target_absent_dual_domain_target_kws_hardneg_sweep_2026-10-10.json",
        "logs/aishell5_target_absent_dual_domain_target_kws_hardneg_thr070_2026-10-10.json",
        "reports/production_candidate_status_2026-10-10.md",
        "reports/aishell5_target_kws_adaptation_2026-10-11.md",
        "reports/intent_confidence_calibration_runtime_ft8_2026-10-11.md",
        "logs/continuous_full_pipeline_proxy_safety_diagnostic_same_speaker_m10_v2_2026-10-11.json",
        "logs/continuous_full_pipeline_proxy_safety_diagnostic_same_speaker_p0_presence012_2026-10-10.json",
        "logs/continuous_full_pipeline_proxy_safety_diagnostic_same_speaker_p10_presence012_2026-10-11.json",
        "logs/intent_confidence_calibration_ft8_tse_domain_waveform_val_2026-10-11.json",
        "logs/aishell5_continuous_targetkws_hardneg_aishell5_ft1_p015_sg024_kws050_smooth22_1200s_2026-10-11.json",
        "logs/aishell5_continuous_targetkws_hardneg_aishell5_ft1_p015_sg024_kws050_smooth22_eval2_all18_2026-10-10.json",
        "logs/aishell5_continuous_targetkws_hardneg_aishell5_ft1_p015_sg024_kws050_smooth22_eval2_all18_dedup_2026-10-11.json",
        "logs/full_pipeline_latency_cpu_int8_2026-10-11.json",
        "logs/full_pipeline_latency_norm_crnn_distill_t3_a05_gpu_2026-10-11.json",
        "logs/stage4_intent_tse_domain_distill_t3_a05_val.json",
        "logs/stage4_intent_tse_domain_distill_t3_a05_fsc_test_clean.json",
        "logs/continuous_full_pipeline_proxy_norm_crnn_distill_t3_a05_conf035_same_speaker_m10_2026-10-11.json",
        "reports/research_informed_intent_student_candidate_2026-10-12.md",
        "configs/stage4_intent_tse_domain_norm_student_medium.yaml",
        "configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_norm_crnn_medium_candidate.yaml",
        "checkpoints/stage4_intent_tse_domain_distill_medium_t3_a05/best.pt",
        "logs/stage4_intent_tse_domain_distill_medium_t3_a05/metrics_val_clean.json",
        "logs/stage4_intent_tse_domain_distill_medium_t3_a05/metrics_val_cabin.json",
        "logs/stage4_intent_tse_domain_distill_medium_t3_a05/metrics_test_clean.json",
        "logs/continuous_full_pipeline_proxy_medium_student_10scenes_2026-10-12.json",
        "logs/full_pipeline_latency_medium_student_gpu_2026-10-12.json",
        "src/car_tse/pipeline/in_car_pipeline.py",
        "scripts/eval/eval_continuous_full_pipeline_proxy.py",
        "logs/continuous_full_pipeline_proxy_wav2vec_reference_same_speaker_m10_32scenes_2026-10-11.json",
        "logs/continuous_full_pipeline_proxy_wav2vec_reference_kws055_same_speaker_m10_32scenes_2026-10-11.json",
        "logs/continuous_full_pipeline_proxy_wav2vec_reference_kws060_same_speaker_m10_32scenes_2026-10-11.json",
        "logs/aishell5_continuous_targetkws_hardneg_aishell5_ft1_p015_sg024_kws055_sweep_eval2_all18_2026-10-11.json",
        "logs/continuous_full_pipeline_proxy_hardneg_aishell5_neg_splitgate_p015_wake024_cmd022_same_speaker_m10_2026-10-11.json",
        "logs/continuous_full_pipeline_proxy_hardneg_aishell5_neg_splitgate_p015_wake024_cmd022_same_speaker_p0_2026-10-10.json",
        "logs/continuous_full_pipeline_proxy_hardneg_aishell5_neg_splitgate_p015_wake024_cmd022_same_speaker_p10_2026-10-11.json",
        "logs/continuous_full_pipeline_proxy_enrollment_bank1_m10_8scenes_2026-10-11.json",
        "logs/continuous_full_pipeline_proxy_enrollment_bank3_m10_8scenes_2026-10-11.json",
        "reports/enrollment_bank_candidate_2026-10-11.md",
        "reports/preexisting_systems_review_2026-10-11.md",
        "scripts/data/build_remasc_cabin_stress_bed.py",
        "configs/eval_end_to_end_cavemove_personal_activity_group_p025_cmdp035.yaml",
        "logs/continuous_full_pipeline_proxy_remasc_vehicle_stress_8scenes_2026-10-12.json",
        "logs/continuous_full_pipeline_proxy_remasc_personal_activity_group_p025_8scenes_2026-10-12.json",
        "logs/continuous_full_pipeline_proxy_remasc_personal_activity_group_p035_8scenes_2026-10-12.json",
        "logs/continuous_full_pipeline_proxy_remasc_personal_activity_group_p025_cmdp035_8scenes_2026-10-12.json",
        "configs/eval_end_to_end_cavemove_personal_activity_group_p025_margin_m067.yaml",
        "logs/continuous_full_pipeline_proxy_remasc_personal_activity_group_p025_margin_m0p66_8scenes_2026-10-12.json",
        "logs/continuous_full_pipeline_proxy_remasc_personal_activity_group_p025_margin_m0p67_8scenes_2026-10-12.json",
        "logs/continuous_full_pipeline_proxy_remasc_personal_activity_group_p025_margin_m0p68_8scenes_2026-10-12.json",
        "logs/continuous_full_pipeline_proxy_remasc_personal_activity_group_p025_margin_m0p70_8scenes_2026-10-12.json",
        "logs/aishell5_target_absent_personal_activity_group_all18_margin_m067_2026-10-12.json",
        "reports/personal_activity_margin_aggregation_2026-10-12.md",
        "configs/eval_end_to_end_cavemove_personal_activity_group_p025_margin_targettop_observe.yaml",
        "configs/eval_end_to_end_cavemove_personal_activity_group_p025_margin_framewise_observe.yaml",
        "logs/continuous_full_pipeline_proxy_remasc_personal_activity_targettop_observe_8scenes_2026-10-12.json",
        "logs/continuous_full_pipeline_proxy_remasc_personal_activity_targettop_m030_8scenes_2026-10-12.json",
        "logs/continuous_full_pipeline_proxy_remasc_personal_activity_framewisetop_observe_8scenes_2026-10-12.json",
        "logs/continuous_full_pipeline_proxy_remasc_personal_activity_framewisetop_m030_8scenes_2026-10-12.json",
        "logs/aishell5_target_absent_personal_activity_group_all18_margin_targettop_m030_2026-10-12.json",
        "reports/streaming_latency_candidate_2026-10-12.md",
        "configs/eval_end_to_end_cavemove_personal_activity_group_p025_margin_m067_fast_stream.yaml",
        "logs/continuous_full_pipeline_proxy_remasc_personal_activity_margin_m067_hop050_8scenes_2026-10-12.json",
        "logs/continuous_full_pipeline_proxy_remasc_personal_activity_margin_m067_hop025_8scenes_2026-10-12.json",
        "logs/continuous_full_pipeline_proxy_remasc_personal_activity_margin_m067_hop025_delay2_8scenes_2026-10-12.json",
        "logs/continuous_full_pipeline_proxy_remasc_personal_activity_margin_m067_hop025_delay4_8scenes_2026-10-12.json",
        "logs/continuous_full_pipeline_proxy_remasc_personal_activity_margin_m067_fast_stream_hop025_8scenes_2026-10-12.json",
        "logs/aishell5_target_absent_personal_activity_hop025_ref8_1200s_2026-10-12.json",
        "logs/aishell5_target_absent_personal_activity_hop025_ref16_1200s_2026-10-12.json",
        "reports/late_speaker_fusion_candidate_2026-10-11.md",
        "configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_late_fusion_candidate.yaml",
        "logs/continuous_full_pipeline_proxy_late_fusion_min_c024_s004_m10_2026-10-11.json",
        "logs/continuous_full_pipeline_proxy_late_fusion_min_c026_s004_m10_2026-10-11.json",
        "logs/continuous_full_pipeline_proxy_late_fusion_product_c024_s004_m10_2026-10-11.json",
        "logs/smart_speaker_command_negative_250_late_fusion_min_c024_s004_2026-10-11.json",
        "logs/smart_speaker_command_negative_250_late_fusion_min_c026_s004_2026-10-11.json",
        "reports/speaker_verifier_calibration_2026-10-11.md",
        "scripts/eval/calibrate_cached_spex_speaker_verifier.py",
        "logs/speaker_verifier_calibration_gsc_cache_stage1_ft2_2026-10-11.json",
        "logs/continuous_full_pipeline_proxy_late_fusion_gsc_cache_calibrated_m10_2026-10-11.json",
        "reports/wavlm_speaker_verifier_candidate_2026-10-11.md",
        "scripts/eval/eval_pretrained_wavlm_speaker_verifier.py",
        "scripts/train/train_wavlm_speaker_verifier_head.py",
        "scripts/eval/benchmark_pretrained_wavlm_speaker_verifier.py",
        "logs/wavlm_speaker_verifier_cabin_diagnostic_2026-10-11.json",
        "logs/speaker_verifier_wavlm_cabin_ft1_2026-10-11.json",
        "logs/wavlm_speaker_verifier_latency_2026-10-11.json",
        "reports/compact_speaker_verifier_runtime_candidate_2026-10-11.md",
        "configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_compact_verifier_candidate.yaml",
        "configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_compact_verifier_operating_diagnostic.yaml",
        "logs/compact_speaker_verifier_gsc_ft1_2026-10-11.json",
        "logs/compact_speaker_verifier_latency_gpu_2026-10-11.json",
        "logs/compact_speaker_verifier_latency_cpu_2026-10-11.json",
        "logs/continuous_full_pipeline_proxy_compact_verifier_smoke_2026-10-11.json",
        "logs/continuous_full_pipeline_proxy_compact_verifier_operating_diagnostic_2026-10-11.json",
        "reports/anir_presence_augmentation_candidate_2026-10-11.md",
        "scripts/data/build_anir_multichannel_cache.py",
        "scripts/data/build_anir_augmented_presence_manifest.py",
        "data/manifests/external/anir_v_class_noise_16k_mc.json",
        "manifests/train/stage2_presence_gsc_aishell_anir.jsonl",
        "configs/stage2_personal_presence_dual_domain_nonwake_anir_ft1.yaml",
        "configs/eval_end_to_end_cavemove_personal_presence_anir_candidate.yaml",
        "logs/stage2_personal_presence_dual_domain_nonwake_anir_ft1/metrics_last.json",
        "logs/continuous_full_pipeline_proxy_personal_presence_anir_same_speaker_m10_8scenes_2026-10-11.json",
        "reports/personal_activity_3way_candidate_2026-10-12.md",
        "scripts/data/build_personal_activity_manifest.py",
        "manifests/train/stage2_personal_activity_gsc_cavemove.jsonl",
        "configs/stage2_personal_activity_gsc_cavemove_ft1.yaml",
        "configs/eval_end_to_end_cavemove_personal_activity_candidate.yaml",
        "configs/eval_end_to_end_cavemove_personal_activity_operating_diagnostic.yaml",
        "logs/stage2_personal_activity_gsc_cavemove_ft1/metrics_last.json",
        "logs/continuous_full_pipeline_proxy_personal_activity_operating_p025_same_speaker_m10_8scenes_2026-10-12.json",
        "logs/continuous_full_pipeline_proxy_personal_activity_operating_p030_same_speaker_m10_8scenes_2026-10-12.json",
        "logs/continuous_full_pipeline_proxy_personal_activity_same_speaker_m10_8scenes_2026-10-12.json",
        "logs/smart_speaker_command_negative_250_personal_activity_p025_2026-10-12.json",
        "reports/personal_activity_group_split_candidate_2026-10-12.md",
        "configs/stage2_personal_activity_gsc_cavemove_group_ft1.yaml",
        "configs/eval_end_to_end_cavemove_personal_activity_group_p025.yaml",
        "configs/eval_end_to_end_cavemove_personal_activity_group_p035.yaml",
        "logs/stage2_personal_activity_gsc_cavemove_group_ft1/metrics_last.json",
        "logs/continuous_full_pipeline_proxy_personal_activity_group_p025_same_speaker_m10_8scenes_2026-10-12.json",
        "logs/continuous_full_pipeline_proxy_personal_activity_group_p035_same_speaker_m10_8scenes_2026-10-12.json",
        "scripts/eval/eval_aishell5_continuous_kws.py",
        "reports/aishell5_personal_activity_target_absent_2026-10-12.md",
        "logs/aishell5_target_absent_personal_activity_group_all18_presence_sweep_2026-10-12.json",
        "scripts/train/train_compact_speaker_verifier_distill.py",
        "scripts/eval/benchmark_compact_speaker_verifier.py",
        "reports/automotive_dataset_sourcing_review_2026-10-11.md",
        "data/manifests/external/remasc_sample.json",
        "data/sources.json",
        "data/manifests/external/anir_v_class_impulse_responses.json",
        "data/manifests/external/anir_v_class_noise.json",
        "logs/anir_noise_target_absent_all_conditions_2026-10-11.json",
        "reports/anir_noise_target_absent_2026-10-11.md",
        "logs/stage1_dual_domain_regression_ft2_2026-10-11.json",
        "logs/continuous_full_pipeline_proxy_ft2_stage1_consistent_negative_same_speaker_m10_32scenes_2026-10-11.json",
        "logs/continuous_full_pipeline_proxy_ft2_stage1_consistent_pos_same_speaker_m10_32scenes_2026-10-11.json",
        "logs/continuous_full_pipeline_proxy_ft2_stage1_consistent_pos_presence_same_speaker_m10_32scenes_2026-10-11.json",
        "logs/continuous_full_pipeline_proxy_ft1_positive_fullmatrix_same_speaker_m10_32scenes_2026-10-11.json",
        "logs/continuous_full_pipeline_proxy_ft1_positive_fullmatrix_kws055_same_speaker_m10_32scenes_2026-10-11.json",
        "logs/aishell5_continuous_targetkws_hardneg_aishell5_pos_ft1_p015_sg022_kws050_sweep_eval2_all18_2026-10-11.json",
        "scripts/data/validate_real_driver_sessions.py",
        "tests/test_real_driver_session_validation.py",
        "logs/real_driver_validation_latest_2026-10-11.json",
        "reports/capture_hardware_preflight_2026-10-11.md",
        "tests/test_record_real_driver_capture_preflight.py",
        "scripts/data/record_real_driver_session.py",
        "scripts/eval/eval_real_driver_sessions.py",
        "tests/test_real_driver_threshold_selection.py",
        "tests/test_real_driver_condition_metrics.py",
        "logs/real_driver_pipeline_evaluation_2026-10-12.json",
        "scripts/reporting/audit_production_readiness.py",
        "logs/production_readiness_snapshot_2026-10-12.json",
        "reports/production_readiness_snapshot_2026-10-12.md",
    ]
    story.append(Paragraph("Sources: " + "; ".join(esc(source) for source in sources), SMALL))

    # AISHELL and conclusions.
    story.append(PageBreak())
    story.extend(section("8. Real-cabin target-absent diagnostic"))
    story.append(Paragraph("AISHELL-5 Eval2 session 001 was evaluated for 120 seconds with a fixed LibriSpeech enrollment. The enrolled target is absent, so this tests rejection only.", BODY))
    aishell_rows = [
        ["Metric", "Result"],
        ["Duration", f"{number(aishell['duration_seconds'], 0)} s ({number(aishell['duration_hours'], 4)} h)"],
        ["Speech occupancy", pct(aishell["speech_occupancy"])],
        ["Full-path smoothed false events", rate(aishell["full_path_smoothed_event_rate_per_hour"])],
        ["KWS-only smoothed false events", rate(aishell["kws_only_smoothed_event_rate_per_hour"])],
    ]
    story.append(table(aishell_rows, [85 * mm, 85 * mm]))
    story.append(Spacer(1, 6))
    story.append(Paragraph("This result is encouraging for one target-absent real-cabin diagnostic, but it cannot establish driver wake recall, command accuracy, or speaker rejection of a real passenger saying Marvin.", BODY))
    story.append(Paragraph("Optional target-speaker KWS diagnostic", H2))
    target_kws_rows = [["KWS mode", "Threshold", "Full-path events/h", "KWS-only events/h", "Max score"]]
    for label, report, threshold_value in [
        ("Generic baseline", aishell, 0.50),
        ("Current target-speaker", aishell_target_kws_current, 0.50),
        ("Hard-negative target-speaker", aishell_target_kws_hardneg, 0.50),
        ("Hard-negative target-speaker", aishell_target_kws_hardneg_thr070, 0.70),
    ]:
        row = report["session_results"][0]
        target_kws_rows.append([
            label,
            number(threshold_value, 2),
            rate(row["full_path_smoothed_event_rate_per_hour"]),
            rate(row["kws_only_smoothed_event_rate_per_hour"]),
            number(row["max_kws_score"], 3),
        ])
    story.append(KeepTogether([table(target_kws_rows, [43 * mm, 22 * mm, 36 * mm, 36 * mm, 28 * mm])]))
    story.append(Paragraph("At the inherited 0.50 threshold, the optional target-speaker heads are worse than the generic baseline on this cross-corpus target-absent slice. A 0.70 hard-negative threshold removes events in this slice, but no positive same-driver wake data exists to measure the recall cost; this is a calibration warning, not a promoted operating point.", SMALL))
    current_sweep = {float(row["kws_threshold"]): row for row in aishell_target_kws_current_sweep["session_results"][0]["threshold_sweep"]}
    hardneg_sweep = {float(row["kws_threshold"]): row for row in aishell_target_kws_hardneg_sweep["session_results"][0]["threshold_sweep"]}
    sweep_rows = [["Target-speaker head", "0.50", "0.55", "0.60", "0.65", "0.70"]]
    for label, sweep in [("Current", current_sweep), ("Hard-negative", hardneg_sweep)]:
        sweep_rows.append([label] + [rate(sweep[threshold]["full_path_smoothed_event_rate_per_hour"]) for threshold in (0.50, 0.55, 0.60, 0.65, 0.70)])
    story.append(KeepTogether([table(sweep_rows, [47 * mm, 25 * mm, 25 * mm, 25 * mm, 25 * mm, 25 * mm])]))
    story.append(Paragraph("One-pass replay of the same extracted windows; values are full-path smoothed false events/hour. This does not estimate positive wake recall and is not threshold tuning for deployment.", SMALL))

    story.extend(section("9. Overall assessment and next step"))
    for item in [
        "The TSE/presence cascade is not yet reliable as a single end-to-end system on the current proxy. Its conservative settings suppress many events, including target wakes.",
        "The immediate engineering problem is calibration and post-wake command capture: lower thresholds recover wakes but create false actions; more smoothing increases latency and can lose the command relationship.",
        "The scalar presence head is a promising recall-oriented candidate, but it must be tuned on consented driver/passenger sessions before adoption.",
        "The next highest-value experiment is a small real-driver pilot with speaker-disjoint tuning and test sessions. Without that dataset, further proxy optimization risks overfitting to cross-corpus artifacts.",
    ]:
        story.append(bullet(item))
    story.append(Paragraph("Bounded Stage 1 and Stage 2 training runs were completed before this update. The report documents their fixed-checkpoint regressions, inference-only handoffs, ablations, and threshold replays; it does not claim production readiness.", BODY))

    # Latest bounded dual-domain adaptation and handoff study.
    story.extend(section("10. Dual-domain adaptation and post-wake handoff"))
    story.append(Paragraph(
        "The full 144-row-per-domain Stage 1 regression passes the declared no-regression gate. The new framewise Stage 2 head has strong held-out validation metrics, but its probability scale differs from the prior head; the inherited 0.70 threshold therefore rejected nearly everything. A joint proxy sweep selected presence 0.07, command-presence 0.10, KWS 0.50, and two-hit smoothing. These thresholds are provisional until tuned on permitted same-driver sessions.",
        BODY,
    ))
    baseline_domains = stage1_regression["baseline"]["metrics"]["by_domain"]
    candidate_domains = stage1_regression["candidate"]["metrics"]["by_domain"]
    stage1_rows = [
        ["Domain", "Baseline SI-SDRi", "Candidate SI-SDRi", "Delta"],
        ["AISHELL-5", f"{number(baseline_domains['aishell5']['mean_si_sdri_db'], 3)} dB", f"{number(candidate_domains['aishell5']['mean_si_sdri_db'], 3)} dB", f"{number(stage1_regression['delta_candidate_minus_baseline']['aishell5'], 3)} dB"],
        ["CAVEMOVE", f"{number(baseline_domains['cavemove']['mean_si_sdri_db'], 3)} dB", f"{number(candidate_domains['cavemove']['mean_si_sdri_db'], 3)} dB", f"{number(stage1_regression['delta_candidate_minus_baseline']['cavemove'], 3)} dB"],
        ["Overall", f"{number(stage1_regression['baseline']['metrics']['overall']['mean_si_sdri_db'], 3)} dB", f"{number(stage1_regression['candidate']['metrics']['overall']['mean_si_sdri_db'], 3)} dB", f"{number(stage1_regression['candidate']['metrics']['overall']['mean_si_sdri_db'] - stage1_regression['baseline']['metrics']['overall']['mean_si_sdri_db'], 3)} dB"],
    ]
    story.append(table(stage1_rows, [38 * mm, 43 * mm, 43 * mm, 46 * mm]))
    story.append(Spacer(1, 5))
    validation = stage2_presence["validation"]
    presence_rows = [
        ["Stage 2 validation", "F1", "FAR", "FRR", "Threshold"],
        ["Framewise", pct(validation["frame_at_0.5"]["f1"]), pct(validation["frame_at_0.5"]["false_accept_rate"]), pct(validation["frame_at_0.5"]["false_reject_rate"]), number(validation["frame_at_0.5"]["threshold"], 2)],
        ["Best frame F1", pct(validation["best_frame_f1"]["f1"]), pct(validation["best_frame_f1"]["false_accept_rate"]), pct(validation["best_frame_f1"]["false_reject_rate"]), number(validation["best_frame_f1"]["threshold"], 2)],
        ["Chunk", pct(validation["chunk_at_0.5"]["f1"]), pct(validation["chunk_at_0.5"]["false_accept_rate"]), pct(validation["chunk_at_0.5"]["false_reject_rate"]), number(validation["chunk_at_0.5"]["threshold"], 2)],
    ]
    story.append(KeepTogether([table(presence_rows, [37 * mm, 35 * mm, 35 * mm, 35 * mm, 28 * mm])]))
    story.append(Spacer(1, 5))
    story.append(Paragraph("Post-wake handoff ablation", H2))
    handoff_rows = [["Variant", "Wake recall", "Intent accuracy", "False actions/h"]]
    for key, label in [("full", "Extracted signal"), ("raw_intent_after_wake", "Raw mixture after wake"), ("fused_intent_after_wake", "Mean extracted + raw logits")]:
        row = handoff["summary"][key]
        handoff_rows.append([label, pct(row["target_wake_recall"]), pct(row["target_intent_accuracy"]), rate(row["false_action_rate_per_hour"])])
    row = command_enrollment["summary"]["raw_intent_after_wake"]
    handoff_rows.append(["Exact command reused as enrollment", pct(row["target_wake_recall"]), pct(row["target_intent_accuracy"]), rate(row["false_action_rate_per_hour"])])
    row = same_speaker_other["summary"]["raw_intent_after_wake"]
    handoff_rows.append(["Different utterance, same command speaker", pct(row["target_wake_recall"]), pct(row["target_intent_accuracy"]), rate(row["false_action_rate_per_hour"])])
    story.append(table(handoff_rows, [66 * mm, 35 * mm, 35 * mm, 34 * mm]))
    story.append(Spacer(1, 5))
    harsh = handoff_m10["summary"]["raw_intent_after_wake"]
    standard = handoff_p10["summary"]["raw_intent_after_wake"]
    story.append(Paragraph(
        f"The raw handoff is a research candidate, not a promotion: at passenger SNR -10 dB it reached {pct(harsh['target_intent_accuracy'])} intent accuracy but {rate(harsh['false_action_rate_per_hour'])} false actions, while at +10 dB it retained {rate(standard['false_action_rate_per_hour'])} false actions. Matching the command speaker raised intent to 100.0%, but the different-utterance diagnostic still produced {rate(same_speaker_other['summary']['raw_intent_after_wake']['false_action_rate_per_hour'])}; exact-command enrollment is leakage-prone. The positive clips remain cross-corpus and speaker-inconsistent.",
        BODY,
    ))
    story.append(Paragraph("Hard-negative target-KWS replay", H2))
    hardneg_rows = [["Target KWS", "Passenger SNR", "Wake recall", "Intent accuracy", "False actions/h", "False wakes"]]
    for label, report, snr in [("Current", handoff, "+10 dB"), ("Hard-negative", hardneg, "+10 dB"), ("Current", handoff_m10, "-10 dB"), ("Hard-negative", hardneg_m10, "-10 dB")]:
        row = report["summary"]["full"]
        hardneg_rows.append([label, snr, pct(row["target_wake_recall"]), pct(row["target_intent_accuracy"]), rate(row["false_action_rate_per_hour"]), str(row["false_wake_events"])])
    story.append(KeepTogether([table(hardneg_rows, [36 * mm, 28 * mm, 28 * mm, 31 * mm, 30 * mm, 17 * mm])]))
    story.append(Paragraph("The hard-negative checkpoint reduces false wake events from 5 to 3 at +10 dB and from 6 to 5 at -10 dB, but does not remove the harsh-passenger false-action failure. It is retained as a candidate for real-driver tuning.", SMALL))
    story.append(Paragraph("Joint safety-threshold follow-up", H2))
    safety_rows = [["Operating point", "Wake recall", "Intent accuracy", "False actions/h"]]
    for label, report, key in [
        ("No speaker/confidence gate (-10 dB)", safety_m10, "full"),
        ("Speaker gate 0.22 (-10 dB)", safety_m10, "full_speaker_gate_0p220"),
        ("Speaker 0.22 + confidence 0.53 (-10 dB)", safety_m10, "full_gate_0p220_intent_confidence_0p530"),
        ("Speaker 0.22 + confidence 0.53 (+10 dB)", safety_p10, "full_gate_0p220_intent_confidence_0p530"),
    ]:
        row = report["summary"][key]
        safety_rows.append([label, pct(row["target_wake_recall"]), pct(row["target_intent_accuracy"]), rate(row["false_action_rate_per_hour"])])
    story.append(KeepTogether([table(safety_rows, [78 * mm, 30 * mm, 32 * mm, 30 * mm])]))
    story.append(Paragraph("The 0.22/0.53 pair removed the residual false action in this short partial-speaker proxy without reducing target intent accuracy. It is a diagnostic candidate only; thresholds must be tuned on real-driver data.", SMALL))
    story.append(Paragraph("Additional ft2 Stage-1/head check", H2))
    ft2_rows = [["ft2 operating point", "Wake recall", "Intent accuracy", "False actions/h"]]
    for label, key in [("No speaker gate", "full"), ("Speaker gate 0.18", "full_speaker_gate_0p180"), ("Speaker gate 0.20", "full_speaker_gate_0p200"), ("Speaker gate 0.22", "full_speaker_gate_0p220")]:
        row = ft2_gate["summary"][key]
        ft2_rows.append([label, pct(row["target_wake_recall"]), pct(row["target_intent_accuracy"]), rate(row["false_action_rate_per_hour"])])
    story.append(KeepTogether([table(ft2_rows, [78 * mm, 30 * mm, 32 * mm, 30 * mm])]))
    story.append(Paragraph("The ft2 frontend/head is useful for extraction SI-SDRi, but it is not a proxy end-to-end improvement over the retained ft1-aligned head.", SMALL))
    story.append(Paragraph("Latest runtime safety replay", H2))
    safety_v2_rows = [["Passenger SNR", "Wake recall", "Intent accuracy", "False wakes/h", "False actions/h", "Mean latency"]]
    for label, report in [("-10 dB", safety_v2_m10), ("0 dB", safety_v2_p0), ("+10 dB", safety_v2_p10)]:
        safety_v2_rows.append([
            label,
            pct(report["target_wake_recall"]),
            pct(report["target_intent_accuracy_overall"]),
            rate(report["false_wake_rate_per_hour"]),
            rate(report["false_action_rate_per_hour"]),
            f"{number(report['target_wake_latency_mean_seconds'], 2)} s",
        ])
    story.append(KeepTogether([table(safety_v2_rows, [25 * mm, 28 * mm, 30 * mm, 30 * mm, 34 * mm, 28 * mm])]))
    calibration_053 = next(row for row in intent_calibration_ft8["thresholds"] if abs(float(row["threshold"]) - 0.53) < 1e-6)
    story.append(Paragraph(
        f"The v2 diagnostic raises only pre-wake presence to 0.12 and retains speaker gate 0.22 plus intent confidence 0.53. It reaches 100%/100% with 0 false wakes and actions per hour in all three short proxy conditions, at 2.25 s mean wake latency. The exact runtime ft8 intent model has {pct(intent_calibration_ft8['raw_accuracy'])} raw validation accuracy and {pct(calibration_053['coverage'])} coverage at {number(calibration_053['threshold'], 2)} confidence with {pct(calibration_053['selective_accuracy'])} selective accuracy. These are calibration and proxy diagnostics, not real-driver release metrics.",
        SMALL,
    ))
    story.append(Paragraph("AISHELL hard-negative adaptation and split-gate check", H2))
    adapted_rows = [["Exposure", "Wake / intent", "Full-path false wakes/h", "KWS-only false events/h", "Mean latency"]]
    adapted_rows.append([
        "AISHELL-5 Eval2 target absent (all 18; 3.889 h)",
        "n/a",
        rate(aishell_adapted_all18["full_path_smoothed_event_rate_per_hour"]),
        rate(aishell_adapted_all18["kws_only_smoothed_event_rate_per_hour"]),
        "n/a",
    ])
    adapted_rows.append([
        "AISHELL-5 Eval2 target absent (1,200 s)",
        "n/a",
        rate(aishell_adapted["full_path_smoothed_event_rate_per_hour"]),
        rate(aishell_adapted["kws_only_smoothed_event_rate_per_hour"]),
        "n/a",
    ])
    for label, report in [
        ("Proxy passenger -10 dB (112 s)", adapted_proxy_m10),
        ("Proxy passenger 0 dB (112 s)", adapted_proxy_p0),
        ("Proxy passenger +10 dB (112 s)", adapted_proxy_p10),
    ]:
        adapted_rows.append([
            label,
            f"{pct(report['target_wake_recall'])} / {pct(report['target_intent_accuracy_overall'])}",
            rate(report["false_wake_rate_per_hour"]),
            "n/a",
            f"{number(report['target_wake_latency_mean_seconds'], 2)} s",
        ])
    story.append(KeepTogether([table(adapted_rows, [52 * mm, 30 * mm, 37 * mm, 37 * mm, 25 * mm])]))
    story.append(Paragraph(f"The adapted negative-only target-KWS head plus presence 0.15 and a split speaker gate (0.24 before wake, 0.22 during command capture) produced {rate(aishell_adapted_all18['full_path_smoothed_event_rate_per_hour'])} full-path smoothed false wakes/hour and {rate(aishell_adapted_all18['kws_only_smoothed_event_rate_per_hour'])} KWS-only events/hour over all 18 target-absent Eval2 sessions (3.889 hours). The shorter 1,200-second runtime-matched slice is shown separately. This uses the exact 2-window/2-hit smoothing; an earlier 3-window replay was more optimistic and is not used here. The positive rows are cross-corpus proxy diagnostics; this candidate is frozen for research comparison, not released.", SMALL))
    story.append(Paragraph("The full integration analysis is in reports/dual_domain_stage2_handoff_study_2026-10-10.md.", SMALL))

    story.append(PageBreak())
    story.extend(section("11. Latest candidate update - 2026-10-11"))
    story.append(Paragraph(
        "The previous report version contained a 0/hour AISHELL value produced with the evaluator's 3-window/2-hit default. The runtime candidate is 2-window/2-hit, so the exact replay was rerun before selecting any update. The corrected 1,200-second slice is 3.0 full-path false wakes/hour with 60.0/hour at the KWS-only stage; the subsequent all-18-session Eval2 exposure is the stronger aggregate at 4.628 full-path/hour and 96.673/hour KWS-only.",
        BODY,
    ))
    gate_rows = [["Pre-wake speaker gate", "AISHELL full-path false wakes/h", "Proxy wake recall", "Proxy intent accuracy", "Decision"]]
    for label, aishell_report, proxy_report, decision in [
        ("0.240 (all Eval2)", aishell_adapted_all18, adapted_proxy_m10, "Reference"),
        ("0.245", aishell_adapted_sg0245, proxy_sg0245_m10, "Rejected: recall loss"),
        ("0.250", aishell_adapted_sg025, proxy_sg025_m10, "Rejected: recall loss"),
    ]:
        gate_rows.append([
            label,
            rate(aishell_report["full_path_smoothed_event_rate_per_hour"]),
            pct(proxy_report["target_wake_recall"]),
            pct(proxy_report["target_intent_accuracy_overall"]),
            decision,
        ])
    story.append(KeepTogether([table(gate_rows, [32 * mm, 43 * mm, 32 * mm, 36 * mm, 27 * mm])]))
    story.append(Paragraph(
        "The 0.245 and 0.250 gates eliminate the observed AISHELL event, but they also reject positive target windows in the harsh passenger proxy. This is why the safer-looking gate is not promoted without real-driver tuning data.",
        SMALL,
    ))
    story.append(Paragraph("Hard-negative mining and transfer checks", H2))
    mining_rows = [["Variant", "Validation", "Proxy wake / intent", "Decision"]]
    mining_rows.extend([
        ["Mined v2, threshold 0.50", "F1 46.6% at 0.62", f"{pct(mined_v2_proxy50['target_wake_recall'])} / {pct(mined_v2_proxy50['target_intent_accuracy_overall'])}", "Rejected"],
        ["Mined v2, threshold 0.40", "F1 46.6% at 0.62", f"{pct(mined_v2_proxy40['target_wake_recall'])} / {pct(mined_v2_proxy40['target_intent_accuracy_overall'])}", "Rejected"],
        ["Mined v3, positive-init attempt", "F1 44.0% at 0.49", f"{pct(mined_v3_proxy['target_wake_recall'])} / {pct(mined_v3_proxy['target_intent_accuracy_overall'])}", "Rejected"],
        ["Mined v4, corrected full warm start", "F1 42.2% at 0.50", f"{pct(mined_v4_proxy['target_wake_recall'])} / {pct(mined_v4_proxy['target_intent_accuracy_overall'])}", "Rejected"],
        ["Full transfer, pre-AISHELL head", "F1 46.5% at 0.51", f"{pct(transfer_v2_proxy['target_wake_recall'])} / {pct(transfer_v2_proxy['target_intent_accuracy_overall'])}", "Rejected"],
        ["Classifier-only, audio frozen", "F1 46.9% at 0.52", f"{pct(classifier_only_proxy['target_wake_recall'])} / {pct(classifier_only_proxy['target_intent_accuracy_overall'])}", "Rejected"],
    ])
    story.append(KeepTogether([table(mining_rows, [47 * mm, 36 * mm, 46 * mm, 41 * mm])]))
    story.append(Paragraph(
        "A trainer bug was found during v4: target-speaker checkpoints store audio parameters under audio.* and the relation classifier at the top level, but the old initializer loaded them into model.audio, matching nothing. The trainer now has an explicit full-model warm-start mode. The corrected v4 experiment still lost positive proxy recall, so no mined checkpoint replaces the reference.",
        SMALL,
    ))
    story.append(Paragraph(
        "Current decision: keep the original negative-only AISHELL-adapted target-KWS checkpoint with the 0.24/0.22 split speaker gate as a diagnostic reference, and keep the production gate blocked until permitted same-driver tuning and held-out recordings exist.",
        BODY,
    ))
    story.append(PageBreak())
    story.extend(section("11A. Candidate compression and safety replay"))
    story.append(Paragraph("Compact intent distillation candidate", H2))
    distill_rows = [
        ["Candidate", "Clean FSC top-1", "TSE-domain top-1", "Diagnostic proxy", "Decision"],
        [
            "Normalized + Wav2Vec-logit distilled CRNN",
            pct(distill_clean["accuracy"]),
            pct(distill_val["accuracy"]),
            f"{pct(distill_proxy['target_intent_accuracy_overall'])} overall / {pct(distill_proxy['target_intent_accuracy_conditioned_on_wake'])} conditioned",
            "Edge fallback; not default",
        ],
        [
            "Medium hidden-384 distilled CRNN",
            pct(medium_intent_clean["accuracy"]),
            pct(medium_intent_val_cabin["accuracy"]),
            f"{pct(medium_intent_proxy['target_intent_accuracy_overall'])} overall / {pct(medium_intent_proxy['target_intent_accuracy_conditioned_on_wake'])} conditioned",
            "Rejected: lower quality",
        ],
        [
            "Wav2Vec fp32 reference",
            "n/a",
            "~90.2%",
            f"{pct(wav2vec_proxy_32['target_intent_accuracy_overall'])} overall / {pct(wav2vec_proxy_32['target_intent_accuracy_conditioned_on_wake'])} conditioned",
            "Quality reference",
        ],
    ]
    story.append(KeepTogether([table(distill_rows, [45 * mm, 28 * mm, 31 * mm, 39 * mm, 27 * mm], font_size=6.8)]))
    story.append(Paragraph(
        f"The distilled CRNN uses {number(latency_distill_crnn['total_parameters'] / 1.0e6, 2)}M parameters and {number(latency_distill_crnn['checkpoint_megabytes_decimal'], 1)} MB of checkpoints, with {number(latency_distill_crnn['wake_stage']['p50_ms'], 1)} ms wake and {number(latency_distill_crnn['post_wake_command_stage']['p50_ms'], 1)} ms command p50. Its inherited 0.53 confidence threshold under-reported usable commands; exploratory 0.25/0.35 replay reached {pct(distill_proxy['target_intent_accuracy_overall'])} overall intent with zero proxy false actions. This threshold is not frozen and the candidate is not a Wav2Vec quality replacement.", SMALL,
    ))
    story.append(Paragraph(
        f"A research-informed hidden-384 variant was also trained. It fell to {pct(medium_intent_val_cabin['accuracy'])} held-out cabin/TSE accuracy and {pct(medium_intent_clean['accuracy'])} clean FSC accuracy, with {pct(medium_intent_proxy['target_intent_accuracy_overall'])} overall proxy intent and {pct(medium_intent_proxy['target_intent_accuracy_conditioned_on_wake'])} conditioned on wake. Its 23.94M parameters and {number(medium_intent_latency['checkpoint_megabytes_decimal'], 1)} MB checkpoint did not justify the quality loss, so it is rejected and the normalized hidden-256 student remains the compact fallback.", SMALL,
    ))
    story.append(Paragraph("All-session KWS threshold replay", H2))
    kws_rows = [["KWS threshold", "Full-path target-absent events/h", "KWS-only events/h", "32-scene positive wake recall", "Decision"]]
    for threshold_value, positive_report, decision in [
        (0.50, wav2vec_proxy_32, "Quality reference"),
        (0.55, wav2vec_proxy_kws055, "Rejected: recall loss"),
        (0.60, wav2vec_proxy_kws060, "Rejected: recall loss"),
    ]:
        sweep = kws_sweep_summary[threshold_value]
        kws_rows.append([
            f"{threshold_value:.2f}",
            rate(sweep["full_path_smoothed_event_rate_per_hour"]),
            rate(sweep["kws_only_smoothed_event_rate_per_hour"]),
            pct(positive_report["target_wake_recall"]),
            decision,
        ])
    story.append(KeepTogether([table(kws_rows, [25 * mm, 42 * mm, 38 * mm, 35 * mm, 30 * mm], font_size=6.8)]))
    story.append(Paragraph(
        "The sweep spans all 18 AISHELL-5 Eval2 sessions (3.889 hours) but is target-absent relative to the fixed enrollment. It is useful for safety trend analysis, not a real-driver FAR. The current 0.50 setting remains the reference until a permitted positive tuning split can choose a threshold.", SMALL,
    ))

    story.append(PageBreak())
    story.append(Paragraph("Candidate promotion audit", H2))
    story.append(Paragraph(
        "This audit separates an isolated extraction improvement from a complete-pipeline improvement. The Stage 1 ft2 branch improves the dual-domain SI-SDR regression, but its downstream score distributions do not match the retained heads. The positive-enriched wake candidate improves the short proxy, then fails the longer target-absent exposure. Neither is promoted.", BODY,
    ))
    stage1_rows = [
        ["Stage 1 checkpoint", "AISHELL-5 SI-SDRi", "CAVEMOVE SI-SDRi", "Overall SI-SDRi", "Decision"],
        [
            "ft1 baseline",
            number(stage1_regression_ft2["baseline"]["metrics"]["by_domain"]["aishell5"]["mean_si_sdri_db"], 2) + " dB",
            number(stage1_regression_ft2["baseline"]["metrics"]["by_domain"]["cavemove"]["mean_si_sdri_db"], 2) + " dB",
            number(stage1_regression_ft2["baseline"]["metrics"]["overall"]["mean_si_sdri_db"], 2) + " dB",
            "Reference",
        ],
        [
            "ft2 candidate",
            number(stage1_regression_ft2["candidate"]["metrics"]["by_domain"]["aishell5"]["mean_si_sdri_db"], 2) + " dB",
            number(stage1_regression_ft2["candidate"]["metrics"]["by_domain"]["cavemove"]["mean_si_sdri_db"], 2) + " dB",
            number(stage1_regression_ft2["candidate"]["metrics"]["overall"]["mean_si_sdri_db"], 2) + " dB",
            "Stage 1 pass; downstream reject",
        ],
    ]
    story.append(KeepTogether([table(stage1_rows, [32 * mm, 32 * mm, 32 * mm, 31 * mm, 49 * mm], font_size=6.7)]))
    downstream_rows = [["Pipeline candidate", "Wake recall", "Overall intent", "Conditioned intent", "False actions/h", "Decision"]]
    for label, report, decision in [
        ("Retained ft1 negative-only", wav2vec_proxy_32, "Reference"),
        ("ft1 positive-enriched KWS", ft1_positive_fullmatrix, "Reject: long negative exposure"),
        ("ft2 Stage 1 + negative KWS", ft2_proxy_negative, "Reject: downstream drop"),
        ("ft2 Stage 1 + positive KWS", ft2_proxy_positive, "Reject: downstream drop"),
        ("ft2 Stage 1 + new presence", ft2_presence_proxy, "Reject: score shift"),
    ]:
        downstream_rows.append([
            label,
            pct(report["target_wake_recall"]),
            pct(report["target_intent_accuracy_overall"]),
            pct(report["target_intent_accuracy_conditioned_on_wake"]),
            rate(report["false_action_rate_per_hour"]),
            decision,
        ])
    story.append(KeepTogether([table(downstream_rows, [39 * mm, 23 * mm, 25 * mm, 29 * mm, 27 * mm, 33 * mm], font_size=6.3)]))
    positive_threshold_rows = [["Positive-enriched KWS", "Wake recall", "Overall intent", "Conditioned intent", "False actions/h", "Decision"]]
    for threshold_label, report, decision in [
        ("0.50", ft1_positive_fullmatrix, "Quality point; safety reject"),
        ("0.55", ft1_positive_fullmatrix_kws055, "Safety point; no useful recall"),
    ]:
        positive_threshold_rows.append([
            threshold_label,
            pct(report["target_wake_recall"]),
            pct(report["target_intent_accuracy_overall"]),
            pct(report["target_intent_accuracy_conditioned_on_wake"]),
            rate(report["false_action_rate_per_hour"]),
            decision,
        ])
    story.append(KeepTogether([table(positive_threshold_rows, [39 * mm, 23 * mm, 25 * mm, 29 * mm, 27 * mm, 33 * mm], font_size=6.3)]))
    negative_exposure_rows = [
        ["Candidate", "Exposure", "Full-path events/h", "KWS-only events/h", "Decision"],
        [
            "ft1 negative-only",
            "AISHELL Eval2, 3.889 h",
            rate(kws_sweep_summary[0.50]["full_path_smoothed_event_rate_per_hour"]),
            rate(kws_sweep_summary[0.50]["kws_only_smoothed_event_rate_per_hour"]),
            "Retained reference",
        ],
        [
            "ft1 positive-enriched",
            "AISHELL Eval2, 3.889 h",
            rate(aishell_positive_all18["full_path_smoothed_event_rate_per_hour"]),
            rate(aishell_positive_all18["kws_only_smoothed_event_rate_per_hour"]),
            "Rejected",
        ],
    ]
    story.append(KeepTogether([table(negative_exposure_rows, [35 * mm, 39 * mm, 34 * mm, 34 * mm, 34 * mm], font_size=6.5)]))
    story.append(Paragraph(
        "The positive-enriched candidate reaches 78.1% wake recall and 68.8% overall intent on the 32-scene proxy, versus 68.8% and 59.4% for the retained reference. Its full-path target-absent rate nevertheless rises from 4.63/h to 9.26/h over 3.889 hours. Thresholds 0.55 and 0.60 remove events in this negative exposure but do not supply a useful positive operating point in the available proxy. The result is a rejection based on safety evidence, not a claim that the model cannot be improved with real-driver data.", SMALL,
    ))

    story.append(PageBreak())
    story.extend(section("12. Runtime optimization and data readiness"))
    story.append(Paragraph(
        "The selected Wav2Vec intent candidate was measured through the loaded full pipeline, not only through isolated components. The benchmark uses a 3-second, 4-channel context and reports compute/resource behavior; it is not a speech-quality test.",
        BODY,
    ))
    latency_rows = [["Runtime path", "p50 wake", "p95 wake", "p50 command", "p95 command", "Stored checkpoints", "Decision"]]
    latency_rows.extend([
        [
            "Wav2Vec fp32 / CUDA",
            f"{number(latency_gpu['wake_stage']['p50_ms'], 1)} ms",
            f"{number(latency_gpu['wake_stage']['p95_ms'], 1)} ms",
            f"{number(latency_gpu['post_wake_command_stage']['p50_ms'], 1)} ms",
            f"{number(latency_gpu['post_wake_command_stage']['p95_ms'], 1)} ms",
            f"{number(latency_gpu['checkpoint_megabytes_decimal'], 1)} MB",
            "Quality reference",
        ],
        [
            "Wav2Vec fp32 / CPU",
            f"{number(latency_cpu['wake_stage']['p50_ms'], 1)} ms",
            f"{number(latency_cpu['wake_stage']['p95_ms'], 1)} ms",
            f"{number(latency_cpu['post_wake_command_stage']['p50_ms'], 1)} ms",
            f"{number(latency_cpu['post_wake_command_stage']['p95_ms'], 1)} ms",
            f"{number(latency_cpu['checkpoint_megabytes_decimal'], 1)} MB",
            "Too slow for target",
        ],
        [
            "Wav2Vec dynamic INT8 / CPU",
            f"{number(latency_int8['wake_stage']['p50_ms'], 1)} ms",
            f"{number(latency_int8['wake_stage']['p95_ms'], 1)} ms",
            f"{number(latency_int8['post_wake_command_stage']['p50_ms'], 1)} ms",
            f"{number(latency_int8['post_wake_command_stage']['p95_ms'], 1)} ms",
            f"{number(latency_int8['checkpoint_megabytes_decimal'], 1)} MB",
            "Rejected: marginal; no quality test",
        ],
        [
            "Wav2Vec fp16 / CUDA",
            f"{number(latency_fp16['wake_stage']['p50_ms'], 1)} ms",
            f"{number(latency_fp16['wake_stage']['p95_ms'], 1)} ms",
            f"{number(latency_fp16['post_wake_command_stage']['p50_ms'], 1)} ms",
            f"{number(latency_fp16['post_wake_command_stage']['p95_ms'], 1)} ms",
            f"{number(latency_fp16['checkpoint_megabytes_decimal'], 1)} MB",
            "Rejected: slower",
        ],
        [
            "Compact CRNN / CUDA",
            f"{number(latency_crnn['wake_stage']['p50_ms'], 1)} ms",
            f"{number(latency_crnn['wake_stage']['p95_ms'], 1)} ms",
            f"{number(latency_crnn['post_wake_command_stage']['p50_ms'], 1)} ms",
            f"{number(latency_crnn['post_wake_command_stage']['p95_ms'], 1)} ms",
            f"{number(latency_crnn['checkpoint_megabytes_decimal'], 1)} MB",
            "Rejected: {0}/{1} proxy".format(pct(crnn_proxy["target_wake_recall"]), pct(crnn_proxy["target_intent_accuracy_overall"])),
        ],
        [
            "Distilled normalized CRNN / CUDA",
            f"{number(latency_distill_crnn['wake_stage']['p50_ms'], 1)} ms",
            f"{number(latency_distill_crnn['wake_stage']['p95_ms'], 1)} ms",
            f"{number(latency_distill_crnn['post_wake_command_stage']['p50_ms'], 1)} ms",
            f"{number(latency_distill_crnn['post_wake_command_stage']['p95_ms'], 1)} ms",
            f"{number(latency_distill_crnn['checkpoint_megabytes_decimal'], 1)} MB",
            "Edge fallback; quality lower",
        ],
    ])
    story.append(KeepTogether([table(latency_rows, [34 * mm, 22 * mm, 22 * mm, 25 * mm, 25 * mm, 27 * mm, 35 * mm], font_size=6.4)]))
    story.append(Paragraph(
        f"The selected fp32 CUDA run measured {number(latency_gpu['wake_stage']['p50_ms'], 1)} ms wake p50 / {number(latency_gpu['wake_stage']['p95_ms'], 1)} ms p95 and {number(latency_gpu['post_wake_command_stage']['p50_ms'], 1)} ms command p50 / {number(latency_gpu['post_wake_command_stage']['p95_ms'], 1)} ms p95. The original compact CRNN was rejected on quality; the new distilled normalized CRNN reduces the stored artifacts to {number(latency_distill_crnn['checkpoint_megabytes_decimal'], 1)} MB and reaches {pct(distill_proxy['target_intent_accuracy_overall'])} overall intent on the 32-scene proxy, but remains an edge fallback rather than the quality reference.",
        SMALL,
    ))
    story.append(Paragraph("Production-readiness audit", H2))
    story.append(Paragraph(
        f"The machine-generated audit status is <b>{esc(readiness['status'])}</b>. The selected checkpoints are present, but the session validator found {readiness['real_driver_validation']['session_count']} real-driver sessions. Same-driver continuity, passenger rejection, frozen tuning/test separation, and threshold freeze therefore remain blocked. The 14,001.9-second all-session AISHELL Eval2 exposure and CAVEMOVE same-speaker scenes remain diagnostic proxies only.",
        BODY,
    ))
    story.append(Paragraph(
        "The evidence contract was hardened before pilot collection: consent records must identify the owner, permitted purpose, and agreement; session IDs must match their directories; audio must match the declared 16 kHz/four-channel contract; and the annotations must contain an ordered enrollment -> driver wake -> driver command sequence plus passenger/no-command/noise events. The readiness audit also requires at least three drivers and three complete sessions with non-empty tuning and held-out groups, plus matching capture hardware. The preflight found no four-channel input, and the empty pilot tree still fails validation.",
        BODY,
    ))
    story.append(Paragraph(
        "The real-driver evaluator now records explicit held-out gates: 90% minimum driver-wake recall, 90% minimum command accuracy, and maximum 1.0 conservative unmatched-wake false accept/hour and false action/hour by default. These values are configurable before the held-out test, never after it, and the evaluator reports both conservative and interval-specific false-accept rates.",
        BODY,
    ))
    release = readiness.get("release_manifest", {})
    release_git = release.get("git", {})
    story.append(Paragraph(
        f"The readiness snapshot now carries a release identity: Git revision {esc(release_git.get('revision') or 'unavailable')}, worktree clean={esc(release_git.get('worktree_clean'))}, and SHA-256 hashes for the runtime config and selected checkpoints. The current worktree is still dirty, so reproducible-release sign-off remains blocked even though the hashes are recorded.",
        BODY,
    ))
    story.append(Paragraph(
        "Dataset decision: the official ICSRC 2022 page describes 20 hours of real in-car Mandarin recordings, but its agreement limits use to non-commercial research/education/technology development, prohibits redistribution, and prohibits Test-set training. It lacks English Marvin and same-driver wake/command continuity, so it was not added to the production training path. CI-AVSR is a Cantonese in-car command corpus with clean/noisy/OOD splits; it also lacks the required English wake/enrollment sequence and was not treated as real-driver evidence.",
        BODY,
    ))
    story.append(Paragraph(
        "Capture-contract recheck: the host still exposes only one- or two-channel Fifine input, not the required four-channel cabin array. The recorder now validates the selected device and 16 kHz/channel contract before creating a session directory; an unsupported four-channel request fails cleanly and leaves no misleading pilot artifact. Successful captures record the resolved device capability in session metadata.",
        BODY,
    ))
    story.append(Paragraph(
        "Threshold-selection safety: the real-driver evaluator now requires at least one tuning combination to satisfy both the false-accept and false-action limits. If no combination is safe, it reports a least-unsafe diagnostic fallback, records that no safe tuning point exists, and forces production_valid=false rather than silently freezing an unsafe threshold.",
        BODY,
    ))
    story.append(Paragraph(
        "Condition reporting: each permitted session and the aggregate now include exposure-aware metrics by annotation condition. Driver wake recall and command accuracy are separated from passenger/no-command/noise-only false accepts and false actions, with rates normalized by the union of labeled negative intervals. This supports the planned condition-wise pilot table without changing inference or tuning.",
        BODY,
    ))
    story.append(Paragraph(
        "The next decisive improvement is a permitted pilot dataset: at least three consenting drivers, speaker-disjoint tuning and held-out test sessions, continuous enrollment -> Marvin -> command sequences, passenger Marvin/speech, no-command/noise-only intervals, and condition metadata. Without that evidence, additional cross-corpus training is more likely to optimize proxies than production reliability.",
        BODY,
    ))
    story.append(Paragraph("Pre-existing systems review and Smart Speaker hard-negative extension", H2))
    story.append(Paragraph(
        "Primary references point to a shared enrolled-speaker condition across streaming extraction, personal activity detection, speaker verification, and keyphrase detection. The practical follow-up for this repository is a small causal target/non-target/non-speech activity head, then a two-branch wake decision where speaker verification can veto keyword evidence; spatial or beamforming upgrades should wait for matching four-channel recordings. Relevant references are SpEx+ and SpEx++ for short-reference extraction, Personal VAD 2.0 and TS-VAD for target activity, personalized keyphrase detection and the newer user-defined-keyword/speaker-verification work for speaker-aware wake decisions, BC-ResNet/MatchboxNet for compact KWS backbones, and SpeakerBeam-SS for a future causal low-cost TSE study. The full review and source links are in reports/preexisting_systems_review_2026-10-11.md.",
        BODY,
    ))
    story.append(Paragraph(
        "Dataset sourcing update: the free ANIR V-class impulse-response archive was downloaded and hash-verified at D:/datasets/anir (682 48-kHz .fir files), and its 1,302-file V-class noise archive was also verified across 42 conditions for future spatial/cabin augmentation and hard-negative testing. Neither archive has wake/intent labels, so neither is real-driver evaluation evidence. The 121.7-GB iCarB-Voice corpus is a promising speaker-conditioned in-car resource, but its Zenodo files are restricted to approved non-commercial EULA users, so no access was assumed. The sourcing rationale is in reports/automotive_dataset_sourcing_review_2026-10-11.md.",
        BODY,
    ))
    story.append(Paragraph(
        "A ReMASC sample was also downloaded and SHA-256 verified at D:/datasets/remasc (10.9 MB, 12 WAV files). Its English vehicle commands and genuine/source/replayed multichannel labels are useful for future replay/array stress experiments, but it lacks the continuous same-driver enrollment -> Marvin -> command -> passenger/no-command protocol and was not added to training or real-driver evidence.",
        BODY,
    ))
    story.append(Paragraph(
        "The full ANIR V-class noise archive was also tested as a noise-only target-absent diagnostic across 42 conditions (0.35 h, microphones 0-3). The retained gated path produced 0.0 observed events/hour; KWS-only produced 8.57/hour. This supports hard-negative testing and future augmentation, but contains no speech, passenger, wake, or command evidence and was not added to training.",
        BODY,
    ))
    smart_rows = [["Run", "Duration", "Full-path wakes/h", "KWS-only/h", "Positive proxy", "Decision"]]
    for label, negative, positive, decision in [
        ("Retained reference (merged)", smart_negative_dedup, wav2vec_proxy_32, "Reference; negative stress only"),
        ("Full Smart hard-negative (merged)", smart_negative_candidate_dedup, smart_proxy_full, "Reject: positive loss"),
        ("Thin Smart candidate (legacy)", smart_negative_thin, smart_proxy_thin, "Reject: no negative gain"),
        ("Thin, KWS 0.70 (legacy)", smart_negative_thin_kws070, smart_proxy_thin_kws070, "Reject: 0% wake"),
        ("Distilled Smart candidate", None, smart_proxy_distill, "Reject: low wake; negative run not performed"),
    ]:
        positive_text = f"{pct(positive['target_wake_recall'])} wake / {pct(positive['target_intent_accuracy_overall'])} intent"
        if negative is None:
            duration_text, full_rate, kws_rate = "-", "-", "-"
        else:
            duration_text = f"{number(negative['duration_hours'], 3)} h"
            full_rate = rate(negative["false_wakes_per_hour"])
            kws_rate = rate(negative["kws_only_false_wakes_per_hour"])
            action_rate = negative.get("action_sample", {}).get("false_actions_per_hour")
            if action_rate is not None:
                decision = f"{decision}; {rate(action_rate)} actions"
        smart_rows.append([
            label,
            duration_text,
            full_rate,
            kws_rate,
            positive_text,
            decision,
        ])
    story.append(KeepTogether([table(smart_rows, [32 * mm, 21 * mm, 29 * mm, 27 * mm, 39 * mm, 42 * mm], font_size=6.1)]))
    story.append(Paragraph(
        "The Smart Speaker dataset is clean 48 kHz mono studio audio from seven speakers saying “Hey Alexa” plus a command; it is licensed CC BY 4.0 and is not cabin or multichannel evidence. The clips were duplicated across four channels only for interface compatibility. The full candidate reduced the negative diagnostic but fell to 14.8% wake recall; the thin candidate reached 81.3% wake recall on only 16 synthetic scenes without improving the negative diagnostic. The 0.70 threshold produced no observed negative events and no positive wakes. No candidate replaces the retained reference.",
        SMALL,
    ))
    story.append(Paragraph("Personal activity-head candidate", H2))
    personal_rows = [["Presence operating point", "Wake recall", "Overall intent", "False wakes/h", "False actions/h", "Decision"]]
    for label, report, decision in [
        ("0.15 / command 0.10", personal_proxy, "Reject: unsafe"),
        ("0.40 / command 0.10", personal_proxy_p040, "Reject: recall loss"),
    ]:
        personal_rows.append([
            label,
            pct(report["target_wake_recall"]),
            pct(report["target_intent_accuracy_overall"]),
            rate(report["false_wake_rate_per_hour"]),
            rate(report["false_action_rate_per_hour"]),
            decision,
        ])
    story.append(KeepTogether([table(personal_rows, [39 * mm, 25 * mm, 28 * mm, 31 * mm, 34 * mm, 33 * mm], font_size=6.6)]))
    story.append(Paragraph(
        "The new personal presence head conditions the frame gate on the SpEx+ enrollment embedding. It reaches 81.3% wake recall at the inherited threshold but produces 96.4 false actions/hour; the validation-derived 0.40 threshold removes observed proxy actions but drops wake recall to 50.0%. It remains a rejected research candidate until a real-driver tuning split exists.",
        SMALL,
    ))
    story.append(Paragraph("Enrollment-bank candidate", H2))
    enrollment_rows = [["Reference representation", "Bank size", "Wake recall", "Overall intent", "False actions/h", "Decision"]]
    for label, report, decision in [
        ("Single short utterance", enrollment_bank1_proxy, "Retain as default"),
        ("Uncalibrated mean", enrollment_bank3_proxy, "Reject: recall collapse"),
    ]:
        enrollment_rows.append([
            label,
            str(report.get("enrollment_bank_size", "?")),
            pct(report["target_wake_recall"]),
            pct(report["target_intent_accuracy_overall"]),
            rate(report["false_action_rate_per_hour"]),
            decision,
        ])
    story.append(KeepTogether([table(enrollment_rows, [42 * mm, 20 * mm, 27 * mm, 30 * mm, 32 * mm, 39 * mm], font_size=6.6)]))
    story.append(Paragraph(
        "A three-utterance SpEx embedding bank was tested as a literature-inspired runtime option. It produced zero observed proxy actions only by missing nearly every target wake (6.25% recall), so it is not promoted. The mechanism remains available for same-driver tuning with quality-weighted or learned aggregation.",
        SMALL,
    ))
    story.append(PageBreak())
    story.append(Paragraph("Late speaker-verification fusion candidate", H2))
    late_rows = [
        ["Candidate", "Proxy wake", "Proxy intent", "Smart full-path/h", "Decision"],
        ["Min, center .24", "100.0%", "100.0%", "728.1", "No improvement"],
        ["Min, center .26", "87.5%", "87.5%", "695.3", "Reject: recall loss"],
        ["Product, center .24", "25.0%", "25.0%", "not run", "Reject: severe recall loss"],
    ]
    story.append(KeepTogether([table(late_rows, [38 * mm, 25 * mm, 27 * mm, 38 * mm, 45 * mm], font_size=6.8)]))
    story.append(Paragraph(
        "The opt-in late-fusion path combines target-KWS evidence with a sigmoid-mapped SpEx cosine score before smoothing. Min fusion at cosine center 0.24/scale 0.04 retained 100.0% proxy wake/intent and 0 false actions but matched the retained Smart-negative rate; center 0.26 reduced that rate only to 695.3/hour while dropping proxy recall. Product fusion dropped proxy recall to 25.0%. The raw SpEx similarity is therefore not calibrated enough for promotion; the reference path remains unchanged. Detailed evidence is in reports/late_speaker_fusion_candidate_2026-10-11.md.",
        BODY,
    ))

    story.append(PageBreak())
    story.extend(section("12A. Runtime-domain speaker-verifier calibration audit"))
    story.append(Paragraph(
        "The literature review suggested a separate text-independent speaker branch. Before collecting the real-driver pilot, the frozen SpEx enrollment and extracted-mixture embeddings were calibrated on the existing Stage-3 synthetic CAVEMOVE/GSC-Marvin target-present/target-absent cache. This is closer to the runtime path than clean speaker pairs, but it is still not real-driver or real-passenger evidence.",
        BODY,
    ))
    calibration_test = verifier_calibration["splits"]["test"]
    calibration_val = verifier_calibration["validation_operating_point"]
    calibration_proxy = verifier_calibration_proxy
    verifier_rows = [
        ["Measurement", "Result"],
        ["Calibration rows", "1,208 train / 260 validation / 256 test"],
        ["Held-out test AUC", number(calibration_test["roc"]["auc"], 3)],
        ["Validation operating point", f"{pct(calibration_val['tpr'])} TPR / {pct(calibration_val['fpr'])} FPR"],
        ["Held-out test at selected point", f"{pct(verifier_calibration['test_at_validation_operating_point']['tpr'])} TPR / {pct(verifier_calibration['test_at_validation_operating_point']['fpr'])} FPR"],
        ["End-to-end proxy wake / intent", f"{pct(calibration_proxy['target_wake_recall'])} / {pct(calibration_proxy['target_intent_accuracy_overall'])}"],
        ["End-to-end proxy false actions", rate(calibration_proxy["false_action_rate_per_hour"])],
    ]
    story.append(KeepTogether([table(verifier_rows, [70 * mm, 100 * mm])]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "The fitted mapping was sigmoid((cosine - 0.2661) / 2.7197). The held-out AUC of 0.515 is chance-level; at the validation point chosen for about 1% FPR, test TPR was only 1.56%. In the 16-condition proxy, the candidate retained 56.25% wake recall and 43.75% overall intent with zero observed false events. The zero-negative result is caused by suppressing target events and is not a usable safety result.",
        BODY,
    ))
    story.append(Paragraph(
        "Decision: reject the calibration-only late-fusion candidate. The evidence points to an upstream representation problem, not a missing sigmoid threshold. A production-oriented upgrade needs a separately trained speaker-conditioned activity/verifier head using permitted same-driver enrollment, driver wake/command, passenger speech, passenger Marvin, and no-command intervals, with thresholds frozen on tuning sessions before held-out evaluation.",
        BODY,
    ))
    story.append(Paragraph("Detailed evidence: reports/speaker_verifier_calibration_2026-10-11.md", SMALL))
    story.append(Spacer(1, 3))
    story.append(Paragraph("Pretrained speaker-encoder follow-up", H2))
    wavlm_test = wavlm_verifier["splits"]["test"]
    wavlm_head_test = wavlm_head["splits"]["test"]
    wavlm_head_point = wavlm_head["test_at_validation_operating_point"]
    wavlm_rows = [
        ["Verifier", "Test AUC", "Test TPR", "Test FPR"],
        ["SpEx extracted embedding", number(calibration_test["roc"]["auc"], 3), pct(verifier_calibration["test_at_validation_operating_point"]["tpr"]), pct(verifier_calibration["test_at_validation_operating_point"]["fpr"])],
        ["WavLM cosine", number(wavlm_test["roc"]["auc"], 3), pct(wavlm_verifier["test_at_validation_operating_point"]["tpr"]), pct(wavlm_verifier["test_at_validation_operating_point"]["fpr"])],
        ["Frozen WavLM + relation head", number(wavlm_head_test["roc"]["auc"], 3), pct(wavlm_head_point["tpr"]), pct(wavlm_head_point["fpr"])],
    ]
    story.append(KeepTogether([table(wavlm_rows, [72 * mm, 30 * mm, 32 * mm, 32 * mm], font_size=6.8)]))
    story.append(Paragraph(
        f"WavLM improves representation quality but is not an edge-ready branch: one 1-second forward measured {number(wavlm_latency['p50_ms'], 1)} ms p50 / {number(wavlm_latency['p95_ms'], 1)} ms p95, with {number(wavlm_latency['parameters_m'], 1)}M parameters and {number(wavlm_latency['peak_cuda_memory_mb'], 1)} MB peak CUDA memory. It remains a teacher/diagnostic candidate pending same-driver data, distillation, quantization, and target-device measurements.",
        SMALL,
    ))
    story.append(PageBreak())
    story.append(Paragraph("Compact verifier runtime candidate", H2))
    compact_test = compact_verifier["splits"]["test"]
    compact_point = compact_verifier["test_at_validation_operating_point"]
    compact_rows = [
        ["Measurement", "Result"],
        ["Held-out test AUC", number(compact_test["roc"]["auc"], 3)],
        ["Test EER", pct(compact_test["roc"]["eer"])],
        ["TPR / FPR at validation point", f"{pct(compact_point['tpr'])} / {pct(compact_point['fpr'])}"],
        ["Parameters", f"{compact_verifier['parameter_count']:,}"],
        ["RTX 4080 p50 / p95", f"{number(compact_latency_gpu['p50_ms'], 2)} / {number(compact_latency_gpu['p95_ms'], 2)} ms"],
        ["Proxy wake / intent at relaxed 0.80", f"{pct(compact_proxy['target_wake_recall'])} / {pct(compact_proxy['target_intent_accuracy_overall'])}"],
    ]
    story.append(KeepTogether([table(compact_rows, [80 * mm, 90 * mm], font_size=6.8)]))
    story.append(Paragraph(
        "The compact verifier is a fast, memory-light 176,579-parameter late speaker veto, but its low-FPR operating point transferred poorly. The relaxed full-cascade diagnostic recovered 1/16 wakes and 0/16 correct commands, so it is rejected for promotion. Same-driver data and causal speaker-conditioned training are required before revisiting it.",
        SMALL,
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph("ANIR noise-augmentation follow-up", H2))
    anir_rows = [
        ["Candidate", "Wake", "Intent", "False actions/h"],
        [
            "Existing personal head",
            pct(personal_proxy["target_wake_recall"]),
            pct(personal_proxy["target_intent_accuracy_overall"]),
            rate(personal_proxy["false_action_rate_per_hour"]),
        ],
        [
            "ANIR-augmented personal head",
            pct(anir_presence_proxy["target_wake_recall"]),
            pct(anir_presence_proxy["target_intent_accuracy_overall"]),
            rate(anir_presence_proxy["false_action_rate_per_hour"]),
        ],
    ]
    story.append(KeepTogether([table(anir_rows, [78 * mm, 28 * mm, 28 * mm, 36 * mm], font_size=6.8)]))
    story.append(Paragraph(
        "ANIR contributes 42 matched four-channel, 16-kHz automotive-noise beds, but this augmentation did not change the 16-scene proxy result. The checkpoint is rejected as a production upgrade; the cache is retained for future same-driver training.",
        SMALL,
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Three-way personal-activity follow-up", H2))
    activity_validation = personal_activity_train["validation"]["best_chunk_f1"]
    activity_rows = [
        ["Operating point", "Wake", "Intent", "False wakes/h", "False actions/h", "Decision"],
        [
            "Binary personal head, 0.15",
            pct(personal_proxy["target_wake_recall"]),
            pct(personal_proxy["target_intent_accuracy_overall"]),
            rate(personal_proxy["false_wake_rate_per_hour"]),
            rate(personal_proxy["false_action_rate_per_hour"]),
            "Reference",
        ],
        [
            "3-way activity, 0.25",
            pct(personal_activity_p025["target_wake_recall"]),
            pct(personal_activity_p025["target_intent_accuracy_overall"]),
            rate(personal_activity_p025["false_wake_rate_per_hour"]),
            rate(personal_activity_p025["false_action_rate_per_hour"]),
            "Diagnostic; reject",
        ],
        [
            "3-way activity, 0.30",
            pct(personal_activity_p030["target_wake_recall"]),
            pct(personal_activity_p030["target_intent_accuracy_overall"]),
            rate(personal_activity_p030["false_wake_rate_per_hour"]),
            rate(personal_activity_p030["false_action_rate_per_hour"]),
            "Recall loss",
        ],
        [
            "3-way activity, 0.40",
            pct(personal_activity_p040["target_wake_recall"]),
            pct(personal_activity_p040["target_intent_accuracy_overall"]),
            rate(personal_activity_p040["false_wake_rate_per_hour"]),
            rate(personal_activity_p040["false_action_rate_per_hour"]),
            "Reject; all suppressed",
        ],
    ]
    story.append(KeepTogether([table(activity_rows, [37 * mm, 19 * mm, 21 * mm, 25 * mm, 27 * mm, 41 * mm], font_size=5.9)]))
    story.append(Paragraph(
        f"The opt-in head predicts non-speech, target speech, and non-target speech and reached internal validation chunk F1 {number(activity_validation['f1'], 3)} at 0.40. On the cross-corpus proxy, 0.25 halved observed false actions from {rate(personal_proxy['false_action_rate_per_hour'])} to {rate(personal_activity_p025['false_action_rate_per_hour'])}, but false wakes remained {rate(personal_activity_p025['false_wake_rate_per_hour'])} and target recall fell. It is retained for training on real passenger labels, not promoted.",
        SMALL,
    ))
    story.append(Paragraph(
        "On the 250-file Smart Speaker non-Marvin diagnostic, the same point changed merged full-path events from 728.1/hour to 674.4/hour but left the 14-file action sample at 105.9 false actions/hour. This is only a modest clean-studio change and does not meet the safety bar.",
        SMALL,
    ))
    group_validation = personal_activity_group_train["validation"]["best_chunk_f1"]
    story.append(Paragraph(
        f"A speaker-disjoint retrain used 1,925 training rows across 912 enrollment groups and 483 validation rows across 225 groups, with zero overlap. Its internal chunk F1 was {number(group_validation['f1'], 3)} at 0.45, but the proxy returned {pct(personal_activity_group_p025['target_wake_recall'])} wake / {pct(personal_activity_group_p025['target_intent_accuracy_overall'])} intent / {rate(personal_activity_group_p025['false_action_rate_per_hour'])} false actions/hour at 0.25 and rejected all scenes at 0.35. It is retained as the more honest training baseline, not promoted.",
        SMALL,
    ))
    story.append(Paragraph("AISHELL-5 target-absent follow-up", H2))
    aishell_activity_rows = [
        ["Exposure", "Full-path events/h", "KWS-only events/h"],
        [
            f"{number(personal_activity_aishell['duration_hours'], 3)} h / {personal_activity_aishell['sessions']} sessions",
            rate(personal_activity_aishell['full_path_smoothed_event_rate_per_hour']),
            rate(personal_activity_aishell['kws_only_smoothed_event_rate_per_hour']),
        ],
    ]
    story.append(KeepTogether([table(aishell_activity_rows, [72 * mm, 48 * mm, 50 * mm], font_size=6.8)]))
    aishell_sweep_text = ", ".join(
        f"{number(row['kws_threshold'], 2)}->{rate(row['full_path_smoothed_event_rate_per_hour'])}"
        for row in personal_activity_aishell.get("threshold_sweep", [])
    )
    aishell_presence_sweep_text = ", ".join(
        f"{number(row['presence_threshold'], 2)}->{rate(row['full_path_smoothed_event_rate_per_hour'])}"
        for row in personal_activity_aishell.get("presence_threshold_sweep", [])
    )
    story.append(Paragraph(
        f"Across {number(personal_activity_aishell['duration_hours'], 3)} hours of real four-channel AISHELL-5 target-absent speech, the group-split three-way candidate produced {rate(personal_activity_aishell['full_path_smoothed_event_rate_per_hour'])} full-path events/hour versus {rate(personal_activity_aishell['kws_only_smoothed_event_rate_per_hour'])} KWS-only events/hour at KWS 0.50. The aggregate full-path sweep (KWS threshold->events/hour) was {esc(aishell_sweep_text)}; the presence sweep (presence threshold->events/hour) was {esc(aishell_presence_sweep_text)}. This supports useful rejection on real cabin audio, but contains no same-driver positives and is not a production false-accept or recall claim.",
        SMALL,
    ))
    story.append(Paragraph(
        f"The opt-in target-minus-passenger margin at -0.67 reduced the same target-absent full-path exposure to {rate(personal_activity_aishell_margin['full_path_smoothed_event_rate_per_hour'])} versus {rate(personal_activity_aishell['full_path_smoothed_event_rate_per_hour'])} without changing the KWS-only rate ({rate(personal_activity_aishell_margin['kws_only_smoothed_event_rate_per_hour'])}). Because the margin was selected on the synthetic ReMASC stress proxy and there are no positive driver sessions, this is a negative-rejection diagnostic only.",
        SMALL,
    ))

    story.extend(section("12B. Pre-existing systems reference and ReMASC vehicle stress"))
    story.append(Paragraph(
        "The reference review converges on a small streaming keyword branch plus an independently calibrated speaker-conditioned activity or verification veto. Personal VAD 2.0 and target-speaker VAD motivate the three-way activity head; personalized-keyphrase systems motivate separating keyword evidence from speaker identity; multichannel hotword work motivates ANC or spatial features once matching four-channel recordings exist; and production KWS toolkits treat temporal state, long negative exposure, and the threshold/refractory policy as part of the system. These references support a data-and-calibration upgrade, not an immediate replacement of the intent model.",
        BODY,
    ))
    remasc_rows = [["Configuration", "Wake", "Intent", "False wakes/h", "False actions/h", "Decision"]]
    for label, report, decision in [
        ("Retained frame gate", remasc_reference_stress, "Reference diagnostic"),
        ("3-way activity, p=0.25", remasc_activity_p025_stress, "Reject; passenger actions"),
        ("3-way + margin -0.67", remasc_activity_margin_m067, "Diagnostic; no promotion"),
        ("target-top margin -0.30", remasc_activity_targettop_m030, "Reject; recall loss"),
        ("framewise-top margin -0.30", remasc_activity_framewisetop_m030, "Reject; recall loss"),
        ("fast stream, 0.25 s / delay 4", remasc_stream_fast, "Diagnostic; no promotion"),
        ("3-way activity, p=0.35", remasc_activity_p035_stress, "Reject; all suppressed"),
        ("p=0.25 / command=0.35", remasc_activity_asymmetric_stress, "Reject; no commands"),
    ]:
        remasc_rows.append([
            label,
            pct(report["target_wake_recall"]),
            pct(report["target_intent_accuracy_overall"]),
            rate(report["false_wake_rate_per_hour"]),
            rate(report["false_action_rate_per_hour"]),
            decision,
        ])
    story.append(KeepTogether([table(remasc_rows, [42 * mm, 20 * mm, 24 * mm, 29 * mm, 32 * mm, 43 * mm], font_size=5.8)]))
    story.append(Paragraph(
        f"A 60-second, six-channel ReMASC replay was resampled and tiled as a real-vehicle stress bed, then mixed into 32 synthetic target scenes (448 seconds) at -5/0/5/10 dB cabin SNR. The retained path reached {pct(remasc_reference_stress['target_wake_recall'])} wake and {pct(remasc_reference_stress['target_intent_accuracy_overall'])} intent with {rate(remasc_reference_stress['false_action_rate_per_hour'])} diagnostic false actions/hour. The group-split 3-way head improved intent to {pct(remasc_activity_p025_stress['target_intent_accuracy_overall'])} at p=0.25 but raised false actions to {rate(remasc_activity_p025_stress['false_action_rate_per_hour'])}; p=0.35 rejected every target scene. The asymmetric gate also failed. This is synthetic target-plus-real-cabin stress, not real-driver evidence or a production FAR.",
        SMALL,
    ))
    story.append(Paragraph(
        f"The margin sweep was -0.70/-0.68/-0.67/-0.66. At -0.67 it reached {pct(remasc_activity_margin_m067['target_wake_recall'])} wake, {pct(remasc_activity_margin_m067['target_intent_accuracy_overall'])} intent, {rate(remasc_activity_margin_m067['false_wake_rate_per_hour'])} diagnostic false wakes/hour, and {rate(remasc_activity_margin_m067['false_action_rate_per_hour'])} diagnostic false actions/hour. The -0.68 point restored false actions, while -0.66 reduced wake recall to {pct(remasc_activity_margin_m066['target_wake_recall'])}. The cutoff is therefore retained as an opt-in research diagnostic, not a frozen operating point.",
        SMALL,
    ))
    story.append(Paragraph(
        f"A frame-alignment follow-up compared target and passenger probabilities on the same frames. At a proxy-selected -0.30 cutoff, target-top reached {pct(remasc_activity_targettop_m030['target_wake_recall'])} wake / {pct(remasc_activity_targettop_m030['target_intent_accuracy_overall'])} intent and framewise-top reached {pct(remasc_activity_framewisetop_m030['target_wake_recall'])} / {pct(remasc_activity_framewisetop_m030['target_intent_accuracy_overall'])}; both had {rate(remasc_activity_targettop_m030['false_action_rate_per_hour'])} observed false actions/hour. Target-top reduced all-session AISHELL target-absent events to {rate(personal_activity_aishell_margin_targettop['full_path_smoothed_event_rate_per_hour'])}/hour, only slightly below the independent-top margin result. Both modes are rejected for promotion because of positive proxy recall loss.",
        SMALL,
    ))
    story.append(Paragraph(
        f"A streaming-state follow-up reduced the hop from 1.0 s to 0.25 s and delayed intent capture by four calls so the command window no longer overlaps the wake word. The fast candidate reached {pct(remasc_stream_fast['target_wake_recall'])} wake, {pct(remasc_stream_fast['target_intent_accuracy_overall'])} intent, {rate(remasc_stream_fast['false_wake_rate_per_hour'])} diagnostic false wakes/hour, {rate(remasc_stream_fast['false_action_rate_per_hour'])} false actions/hour, and {number(remasc_stream_fast['target_wake_latency_mean_seconds'], 3)} s mean wake latency ({number(remasc_stream_fast['target_wake_latency_p95_seconds'], 3)} s p95). This improves the one-second reference latency, but remains a proxy result that must be calibrated on real drivers and measured on the target CPU/DSP.",
        SMALL,
    ))
    story.append(Paragraph(
        f"The fast setting was then checked on 1,200 seconds of real four-channel AISHELL-5 target-absent audio. Both eight-call and sixteen-call refractory settings produced {rate(aishell_stream_ref8['full_path_smoothed_event_rate_per_hour'])} full-path events/hour, versus 3.0/hour for the comparable one-second reference; the longer refractory changed KWS-only events but not the full path. The fast setting is therefore rejected as a current safety improvement and remains an opt-in latency study only.",
        SMALL,
    ))
    story.append(Paragraph(
        "Decision: keep the 3-way head as a research implementation, but do not promote it. The next experiment is permitted same-driver continuous data with target/passenger/non-speech labels, a compact speaker-verification branch, and a two-factor action veto calibrated on tuning sessions and frozen before held-out testing. The full reference list and exact commands are in reports/preexisting_systems_review_2026-10-11.md and reports/research_journey.md.",
        BODY,
    ))

    story.append(PageBreak())
    story.extend(section("13 - Reproducibility"))
    commands = r"""# From the repository root
python scripts/eval/eval_continuous_ablation_matrix.py `
  --config configs/eval_end_to_end_cavemove_target_kws_temporal2_candidate.yaml `
  --paths configs/paths.yaml --scenes 2 --cabin-snr-db -5 0 10 20 `
  --passenger-snr-db 0 --output logs/continuous_ablation_matrix_2026-10-09.json

python scripts/eval/eval_threshold_ablation_sweep.py `
  --config configs/eval_end_to_end_cavemove_target_kws_temporal2_candidate.yaml `
  --paths configs/paths.yaml --scenes 2 --cabin-snr-db -5 0 10 20 `
  --passenger-snr-db 0 --presence-thresholds 0.30 0.50 0.70 0.90 `
  --kws-thresholds 0.30 0.40 0.50 0.60 `
  --variants full no_speaker_matching no_wake_smoothing `
  --output logs/threshold_ablation_sweep_2026-10-09.json

python scripts/eval/eval_continuous_ablation_matrix.py `
  --config configs/eval_end_to_end_cavemove_target_kws_scalar_candidate.yaml `
  --paths configs/paths.yaml --scenes 2 --cabin-snr-db -5 0 10 20 `
  --passenger-snr-db 0 --output logs/continuous_ablation_matrix_scalar_presence_2026-10-09.json

python scripts/eval/eval_runtime_ablation_sweep.py `
  --config configs/eval_end_to_end_cavemove_target_kws_temporal2_candidate.yaml `
  --paths configs/paths.yaml --scenes 2 --cabin-snr-db -5 0 10 20 `
  --passenger-snr-db 0 --hops 0.5 1.0 --enrollment-seconds 0.5 1.0 1.5 `
  --output logs/runtime_ablation_sweep_2026-10-10.json

python scripts/eval/eval_stage1_dual_domain_regression.py `
  --max-per-domain 144 `
  --output logs/stage1_dual_domain_regression_full_2026-10-10.json

python scripts/eval/eval_continuous_ablation_matrix.py `
  --config configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_operating_candidate.yaml `
  --paths configs/paths.yaml --scenes 2 --cabin-snr-db -5 0 10 20 `
  --passenger-snr-db 10 `
  --output logs/continuous_ablation_matrix_dual_domain_stage2_intent_handoff_2026-10-10.json

python scripts/eval/eval_continuous_ablation_matrix.py `
  --config configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_operating_candidate.yaml `
  --paths configs/paths.yaml --scenes 2 --cabin-snr-db -5 0 10 20 `
  --passenger-snr-db 10 --enrollment-source same_speaker_other `
  --output logs/continuous_ablation_matrix_dual_domain_stage2_same_speaker_other_2026-10-10.json

python scripts/eval/eval_continuous_ablation_matrix.py `
  --config configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_hardneg_candidate.yaml `
  --paths configs/paths.yaml --scenes 2 --cabin-snr-db -5 0 10 20 `
  --passenger-snr-db 10 `
  --output logs/continuous_ablation_matrix_dual_domain_stage2_hardneg_2026-10-10.json

python scripts/eval/eval_continuous_ablation_matrix.py `
  --config configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_operating_candidate.yaml `
  --output logs/continuous_ablation_matrix_stage2_intent_confidence_gate_passenger_m10_2026-10-10.json `
  --scenes 2 --cabin-snr-db -5 0 10 20 --passenger-snr-db -10 `
  --enrollment-source same_speaker_other --speaker-gate-thresholds 0.22 `
  --intent-confidence-thresholds 0.50 0.53 0.55 0.60"""
    story.append(Preformatted(commands, CODE))
    # Validation status is already stated on the cover and in the assessment;
    # keep the final page from being reduced to a single trailing line.

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    # Keep the final PDF and source report; intermediate charts are disposable.
    shutil.rmtree(TMP, ignore_errors=True)
    return OUTPUT


if __name__ == "__main__":
    result = build_pdf()
    print(result)
