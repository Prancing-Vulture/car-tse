#!/usr/bin/env python
"""Create paper-style TSE and stagewise figures from a recorded CAR-TSE stream."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RATE = 16000
WINDOW_SECONDS = 3.0
HOP_SECONDS = 0.25
PAPER = {
    "ink": "#17243A",
    "blue": "#2474A6",
    "green": "#218A63",
    "orange": "#D78300",
    "red": "#C94C4C",
    "purple": "#8064A2",
    "gray": "#7D8794",
}


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    if audio.size == 0:
        raise ValueError("Audio file is empty: {}".format(path))
    if sample_rate != SAMPLE_RATE:
        from math import gcd

        divisor = gcd(int(sample_rate), SAMPLE_RATE)
        audio = resample_poly(
            audio, SAMPLE_RATE // divisor, int(sample_rate) // divisor, axis=0
        ).astype(np.float32)
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains non-finite samples: {}".format(path))
    return np.asarray(audio, dtype=np.float32), SAMPLE_RATE


def fit_audio(audio: np.ndarray, samples: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) >= samples:
        return np.ascontiguousarray(audio[:samples][None, :])
    return np.pad(audio[None, :], ((0, 0), (0, samples - len(audio))))


def run_tse(
    audio: np.ndarray, enrollment: np.ndarray, model_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Run the retained Stage 1 ONNX graph on the stream's rolling windows."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX Runtime is required when running this plotting helper directly."
        ) from exc
    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    inputs = {item.name: item for item in session.get_inputs()}
    required = {"multichannel_audio", "enrollment_audio", "playback_reference"}
    if not required.issubset(inputs):
        raise ValueError("Stage 1 model inputs do not match the retained ONNX contract")

    channel_count = audio.shape[1]
    mono = audio.mean(axis=1).astype(np.float32)
    if channel_count == 4:
        multichannel = np.asarray(audio, dtype=np.float32)
    else:
        # Match the input preparation used by the existing stream replay script.
        multichannel = np.repeat(mono[:, None], 4, axis=1)

    mixture_samples = int(inputs["multichannel_audio"].shape[-1])
    enrollment_samples = int(inputs["enrollment_audio"].shape[-1])
    if not isinstance(mixture_samples, int) or not isinstance(enrollment_samples, int):
        raise ValueError("Stage 1 ONNX model must declare fixed audio windows")
    if mixture_samples != int(round(WINDOW_SECONDS * SAMPLE_RATE)):
        raise ValueError("Stage 1 ONNX input is not the expected 3-second window")

    enrollment_batch = fit_audio(enrollment, enrollment_samples)
    window_weight = 0.05 + 0.95 * np.hanning(mixture_samples + 2)[1:-1].astype(np.float32)
    accumulated = np.zeros(len(mono), dtype=np.float64)
    weight_sum = np.zeros(len(mono), dtype=np.float64)
    window = int(round(WINDOW_SECONDS * SAMPLE_RATE))
    hop = int(round(HOP_SECONDS * SAMPLE_RATE))
    ends = list(range(window, len(mono) + 1, hop))
    if not ends or ends[-1] != len(mono):
        ends.append(len(mono))

    for end in ends:
        start = max(0, end - window)
        frame = multichannel[start:end]
        if len(frame) < window:
            frame = np.pad(frame, ((window - len(frame), 0), (0, 0)))
        frame_batch = np.ascontiguousarray(frame.T[None, :, :], dtype=np.float32)
        playback = np.zeros((1, window), dtype=np.float32)
        feed = {
            "multichannel_audio": frame_batch,
            "enrollment_audio": enrollment_batch,
            "playback_reference": playback,
        }
        estimate = np.asarray(session.run(None, feed)[0], dtype=np.float32).reshape(-1)
        if len(estimate) < window:
            estimate = np.pad(estimate, (0, window - len(estimate)))
        estimate = estimate[:window]
        count = min(window, len(mono) - start)
        accumulated[start : start + count] += (
            estimate[:count].astype(np.float64) * window_weight[:count]
        )
        weight_sum[start : start + count] += window_weight[:count]

    extracted = (accumulated / np.maximum(weight_sum, 1.0e-12)).astype(np.float32)
    residual = mono - extracted
    return extracted, residual


def paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.edgecolor": "#404040",
            "axes.linewidth": 0.65,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def segment_bounds(
    mixture: np.ndarray, duration: float
) -> list[tuple[float, float, str, str, str]]:
    """Find the two spoken groups and keep model decisions separate from them."""
    mono = np.asarray(mixture, dtype=np.float32).reshape(-1)
    frame_samples = SAMPLE_RATE // 100  # 10 ms RMS frames
    frame_count = int(np.ceil(len(mono) / frame_samples))
    padded = np.pad(mono, (0, frame_count * frame_samples - len(mono)))
    frames = padded.reshape(frame_count, frame_samples)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))

    active = np.flatnonzero(rms >= 0.02)
    if not len(active):
        raise ValueError("Could not detect speech in the recording")

    # Merge short within-utterance gaps, then ignore brief noise bursts.
    runs: list[tuple[int, int]] = []
    start = previous = int(active[0])
    for frame in active[1:]:
        frame = int(frame)
        if frame - previous - 1 > 12:
            runs.append((start, previous))
            start = frame
        previous = frame
    runs.append((start, previous))
    runs = [(left, right) for left, right in runs if right - left + 1 >= 15]
    if len(runs) != 2:
        raise ValueError(
            "Expected two clear spoken groups (wake phrase and command); "
            "found {}. Check the recording or adjust the speech RMS threshold.".format(len(runs))
        )

    pad_seconds = 0.05
    wake_start = max(0.0, runs[0][0] * 0.01 - pad_seconds)
    wake_end = min(duration, (runs[0][1] + 1) * 0.01 + pad_seconds)
    command_start = max(0.0, runs[1][0] * 0.01 - pad_seconds)
    command_end = min(duration, (runs[1][1] + 1) * 0.01 + pad_seconds)
    return [
        (0.0, wake_start, "A", "Context", "#E7E8EA"),
        (wake_start, wake_end, "B", "MARVIN", "#F8E3B8"),
        (wake_end, command_start, "", "Pause", "#F2F2F2"),
        (command_start, command_end, "C", "COMMAND", "#DCEBE3"),
        (command_end, duration, "D", "Monitor", "#E7E8EA"),
    ]


def draw_segment_brackets(ax, segments, y: float, scale: float) -> None:
    cap = 0.025 * scale
    for left, right, letter, label, _color in segments:
        ax.plot([left, right], [y, y], color="#222222", lw=0.65, clip_on=False)
        ax.plot([left, left], [y - cap, y], color="#222222", lw=0.65, clip_on=False)
        ax.plot([right, right], [y - cap, y], color="#222222", lw=0.65, clip_on=False)
        if letter:
            ax.text((left + right) / 2, y + 0.035 * scale, "{}  {}".format(letter, label),
                    ha="center", va="bottom", fontsize=6.7, color="#202020", clip_on=False)


def plot_tse(
    time_s: np.ndarray,
    mixture: np.ndarray,
    extracted: np.ndarray,
    residual: np.ndarray,
    active_mask: np.ndarray,
    timeline: dict,
    duration: float,
    output: Path,
) -> None:
    stride = max(1, SAMPLE_RATE // 4000)
    t = time_s[::stride]
    traces = [mixture[::stride], extracted[::stride], residual[::stride]]
    limit = max(float(np.max(np.abs(trace))) for trace in traces)
    limit = max(limit * 1.08, 1.0e-4)
    active_rms = [
        float(np.sqrt(np.mean(np.square(wave[active_mask], dtype=np.float64))))
        for wave in (mixture, extracted, residual)
    ]
    segments = segment_bounds(mixture, duration)
    fig, ax = plt.subplots(figsize=(7.3, 3.35), constrained_layout=False)
    for left, right, _letter, _label, color in segments:
        ax.axvspan(left, right, color=color, alpha=0.72, lw=0, zorder=0)
    ax.axhline(0, color="#6D737A", lw=0.5, zorder=1)
    ax.plot(t, traces[0], color="#26384D", lw=0.42, alpha=0.48, label="Recorded mixture", rasterized=True, zorder=2)
    ax.plot(t, traces[1], color=PAPER["orange"], lw=0.58, alpha=0.9,
            label="TSE target estimate  (active RMS {:.3f})".format(active_rms[1]), rasterized=True, zorder=3)
    ax.plot(t, traces[2], color=PAPER["blue"], lw=0.62, alpha=0.96,
            label="Residual  (active RMS {:.3f})".format(active_rms[2]), rasterized=True, zorder=4)
    ax.set_xlim(0, duration)
    ax.set_ylim(-limit, limit * 1.36)
    draw_segment_brackets(ax, segments, limit * 1.16, limit)
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Time (s)")
    ax.grid(axis="y", color="#D9D9D9", lw=0.45, alpha=0.75, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.37), ncol=3, frameon=False, handlelength=2.2)
    reduction_db = 20.0 * np.log10(max(active_rms[0], 1.0e-12) / max(active_rms[2], 1.0e-12))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.96, bottom=0.23)
    fig.savefig(output, dpi=450, bbox_inches="tight")
    plt.close(fig)


def plot_stagewise(
    time_s: np.ndarray,
    mixture: np.ndarray,
    timeline: dict,
    output: Path,
) -> None:
    rows = timeline.get("steps", [])
    if not rows:
        raise ValueError("The stream-processing report contains no steps")
    duration = float(timeline.get("duration_seconds", time_s[-1]))
    segments = segment_bounds(mixture, duration)
    stride = max(1, SAMPLE_RATE // 4000)
    t = time_s[::stride]
    waveform = mixture[::stride]
    wave_limit = max(float(np.max(np.abs(waveform))) * 1.08, 1.0e-4)
    fig, (ax, state_ax) = plt.subplots(
        2, 1, figsize=(7.3, 3.55), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 0.25], "hspace": 0.08},
    )
    for left, right, _letter, _label, color in segments:
        ax.axvspan(left, right, color=color, alpha=0.78, lw=0, zorder=0)
    ax.plot(t, waveform, color="#203247", lw=0.42, rasterized=True, zorder=2)
    ax.set_xlim(0, duration)
    ax.set_ylim(-wave_limit, wave_limit * 1.44)
    draw_segment_brackets(ax, segments, wave_limit * 1.20, wave_limit)
    ax.set_ylabel("Amplitude")
    ax.axhline(0, color="#777777", lw=0.45)
    ax.grid(axis="y", color="#D9D9D9", lw=0.45, alpha=0.75, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Utterance blocks remain intact; rolling-window decisions are separate
    # markers because the result can arrive before the spoken phrase ends.
    state_ax.set_ylim(0, 1)
    state_ax.set_yticks([])
    state_ax.set_xlim(0, duration)
    state_ax.set_ylabel("Process", rotation=0, labelpad=25)
    for left, right, letter, label, color in segments:
        state_ax.axvspan(left, right, color=color, alpha=0.94, lw=0)
        state_text = {
            "A": "Context",
            "B": "Wake word",
            "": "Pause",
            "C": "Command",
            "D": "Monitor",
        }[letter]
        state_ax.text((left + right) / 2, 0.5, state_text, ha="center", va="center", fontsize=6.0)

    for event in timeline.get("events", []):
        phase = event.get("phase")
        x = float(event["window_end_seconds"])
        if phase == "wake":
            color = PAPER["orange"]
            score = float(event.get("wake_probability") or 0.0)
            text = "Stage 3 PASSED\nwake score: {:.3f}".format(score)
        elif phase == "intent":
            color = PAPER["green"]
            confidence = float(event.get("intent_confidence") or 0.0)
            text = "Stage 4 RESULT\n{} ({:.3f})".format(event.get("action") or "intent", confidence)
        else:
            continue
        ax.vlines(
            x,
            -wave_limit,
            wave_limit * 0.90,
            color=color,
            lw=1.1,
            linestyles=(0, (2, 2)),
            zorder=4,
        )
        label_y = wave_limit * 1.07
        ax.text(
            x,
            label_y,
            text,
            ha="center",
            va="top",
            fontsize=5.9,
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
            zorder=5,
        )

    for spine in ("top", "right", "left"):
        state_ax.spines[spine].set_visible(False)
    state_ax.spines["bottom"].set_color("#666666")
    state_ax.set_xlabel("Time (s)")
    fig.subplots_adjust(left=0.1, right=0.99, top=0.96, bottom=0.16, hspace=0.1)
    fig.savefig(output, dpi=450, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio",
        type=Path,
        default=REPO_ROOT / "stream_rec" / "stream_trimmed_for_alignment.wav",
    )
    parser.add_argument(
        "--enrollment",
        type=Path,
        default=REPO_ROOT / "stream_rec" / "stream_enrollment.wav",
    )
    parser.add_argument(
        "--timeline",
        type=Path,
        default=REPO_ROOT / "stream_rec" / "stream_processing_aligned.json",
    )
    parser.add_argument(
        "--stage1-model",
        type=Path,
        default=REPO_ROOT
        / "car_tse_jetson_onnx_ft6_2026-10-19_v3"
        / "models"
        / "onnx"
        / "ft6"
        / "stage1.onnx",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "stream_rec")
    args = parser.parse_args()

    audio_path = args.audio.expanduser().resolve()
    enrollment_path = args.enrollment.expanduser().resolve()
    timeline_path = args.timeline.expanduser().resolve()
    model_path = args.stage1_model.expanduser().resolve()
    for path, label in (
        (audio_path, "recording"),
        (enrollment_path, "enrollment audio"),
        (timeline_path, "stream-processing report"),
        (model_path, "Stage 1 ONNX model"),
    ):
        if not path.is_file():
            raise FileNotFoundError("Missing {}: {}".format(label, path))

    audio, _ = read_audio(audio_path)
    enrollment_audio, _ = read_audio(enrollment_path)
    enrollment = enrollment_audio.mean(axis=1)
    with timeline_path.open("r", encoding="utf-8") as handle:
        timeline = json.load(handle)
    if int(timeline.get("sample_rate", SAMPLE_RATE)) != SAMPLE_RATE:
        raise ValueError("The aligned stream report must use 16 kHz audio")
    if abs(float(timeline.get("duration_seconds", 0)) - len(audio) / SAMPLE_RATE) > 1.0 / SAMPLE_RATE:
        raise ValueError("Stream report and selected recording durations do not match")

    print("Running Stage 1 TSE on {}".format(audio_path.name), flush=True)
    mono = audio.mean(axis=1).astype(np.float32)
    raw_extracted, _ = run_tse(audio, enrollment, model_path)
    # SpEx+ can have a large, polarity-ambiguous output gain. Fit one signed
    # scalar on speech-active samples so the displayed residual is meaningful.
    rms_window = max(1, int(round(0.02 * SAMPLE_RATE)))
    power = np.square(mono, dtype=np.float64)
    pad_left = rms_window // 2
    pad_right = rms_window - 1 - pad_left
    padded = np.pad(power, (pad_left, pad_right), mode="edge")
    integral = np.concatenate(([0.0], np.cumsum(padded, dtype=np.float64)))
    smoothed_rms = np.sqrt((integral[rms_window:] - integral[:-rms_window]) / rms_window)
    activity_threshold = max(1.0e-4, 0.10 * float(np.percentile(smoothed_rms, 95)))
    active_mask = smoothed_rms >= activity_threshold
    if int(active_mask.sum()) < SAMPLE_RATE // 4:
        active_mask = smoothed_rms >= max(1.0e-5, 0.03 * float(smoothed_rms.max()))
    scale_denominator = float(np.dot(raw_extracted[active_mask], raw_extracted[active_mask]))
    if scale_denominator <= 1.0e-16:
        raise RuntimeError("Stage 1 returned a silent estimate on the active recording")
    display_gain = float(
        np.dot(mono[active_mask], raw_extracted[active_mask]) / scale_denominator
    )
    extracted = (raw_extracted * display_gain).astype(np.float32)
    residual = (mono - extracted).astype(np.float32)
    time_s = np.arange(len(mono), dtype=np.float64) / SAMPLE_RATE
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    speaker_noise_path = output_dir / "tse_speaker_and_removed_component.png"
    stagewise_path = output_dir / "stagewise_processing_paper_style.png"

    paper_style()
    plot_tse(
        time_s, mono, extracted, residual, active_mask, timeline,
        len(mono) / SAMPLE_RATE, speaker_noise_path
    )
    plot_stagewise(time_s, mono, timeline, stagewise_path)

    def rms(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))

    reduction_db = 20.0 * np.log10(
        max(rms(mono[active_mask]), 1.0e-12) / max(rms(residual[active_mask]), 1.0e-12)
    )
    print("Display gain applied to raw TSE estimate: {:.8g}".format(display_gain))
    print("Active RMS: mixture {:.5f}; target estimate {:.5f}; residual {:.5f} ({:.1f} dB below mixture).".format(
        rms(mono[active_mask]), rms(extracted[active_mask]), rms(residual[active_mask]), reduction_db
    ))
    print("Created: {}".format(speaker_noise_path))
    print("Created: {}".format(stagewise_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
