#!/usr/bin/env python
"""Feed a recorded stream clip into the live-style streaming runner.

The runner receives small capture frames on a wall-clock cadence and keeps a
bounded-lag ring buffer between capture and inference.  This script is a
development harness: it replays a four-channel clip through the same runner a
microphone callback would use, reports the lag/skip accounting, and therefore
shows whether the current model keeps up with real time on this device.

It is not a benchmark and it does not change any model or threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import soundfile as sf
import torch

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.segment import segment_waveform
from car_tse.io.audio import read_mono
from car_tse.pipeline.streaming_runner import StreamingRunner
from car_tse.train.utils import get_device
from eval_continuous_full_pipeline_proxy import load_pipeline, resolve_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "configs/eval_end_to_end_cavemove_target_kws_smart_speaker_compact_cabin_smart_thr030_2026-10-12.yaml",
    )
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.yaml")
    parser.add_argument(
        "--source-wav",
        type=Path,
        default=REPO_ROOT
        / "output/artifacts/continuous_stream_eval_samewake_15min_p025_2026-10-12/stream_4ch.wav",
    )
    parser.add_argument(
        "--enrollment-wav",
        type=Path,
        default=REPO_ROOT
        / "data/raw/librispeech/extracted/LibriSpeech/test-clean/1089/134686/1089-134686-0014.flac",
    )
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--frame-ms", type=float, default=100.0)
    parser.add_argument("--time-scale", type=float, default=1.0, help="1.0 is real time; 0 is as fast as possible")
    parser.add_argument("--hop-seconds", type=float, default=0.25)
    parser.add_argument("--max-lag-seconds", type=float, default=2.0)
    parser.add_argument("--overload-policy", choices=("drop_oldest", "catch_up"), default="drop_oldest")
    parser.add_argument(
        "--capture-thread",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Feed frames from a separate thread/callback while inference runs.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "logs/streaming_runner_simulation_2026-10-12.json",
    )
    args = parser.parse_args()

    cfg = load_stage_config(args.config, args.paths)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    pipeline = load_pipeline(cfg, cfg["paths"], device)
    sample_rate = int(cfg["sample_rate"])

    enrollment_wav, _ = read_mono(resolve_path(args.enrollment_wav), sample_rate)
    enrollment = segment_waveform(
        enrollment_wav, int(round(1.5 * sample_rate)), training=False
    )

    source = resolve_path(args.source_wav)
    audio, source_rate = sf.read(str(source), dtype="float32", always_2d=True)
    if source_rate != sample_rate:
        raise ValueError(f"source must be {sample_rate} Hz, found {source_rate}")
    max_samples = min(len(audio), int(round(float(args.seconds) * sample_rate)))
    audio = np.asarray(audio[:max_samples], dtype=np.float32)
    multichannel = audio
    mixture = audio.mean(axis=1).astype(np.float32)

    runner = StreamingRunner(
        forward=pipeline.forward,
        enrollment=enrollment,
        sample_rate=sample_rate,
        window_seconds=float(cfg.get("segment_seconds", 3.0)),
        hop_seconds=float(args.hop_seconds),
        num_mics=int(cfg.get("num_mics", 4)),
        max_lag_seconds=float(args.max_lag_seconds),
        overload_policy=str(args.overload_policy),
        device=device,
    )

    frame_samples = max(1, int(round(float(args.frame_ms) / 1000.0 * sample_rate)))
    time_scale = float(args.time_scale)
    started = time.perf_counter()
    if bool(args.capture_thread):
        stop = threading.Event()
        producer_error: list[BaseException] = []

        def producer() -> None:
            try:
                launched = time.perf_counter()
                pushed_samples = 0
                for start in range(0, len(mixture), frame_samples):
                    end = min(start + frame_samples, len(mixture))
                    runner.push(mixture[start:end], multichannel[start:end])
                    pushed_samples = end
                    if time_scale > 0.0:
                        deadline = launched + (pushed_samples / sample_rate) / time_scale
                        remaining = deadline - time.perf_counter()
                        if remaining > 0:
                            time.sleep(remaining)
            except BaseException as exc:  # pragma: no cover - surfaced below
                producer_error.append(exc)
            finally:
                stop.set()

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()
        while not stop.is_set():
            runner.process_available()
            time.sleep(0.005)
        thread.join()
        if producer_error:
            raise producer_error[0]
    else:
        pushed_samples = 0
        for start in range(0, len(mixture), frame_samples):
            end = min(start + frame_samples, len(mixture))
            runner.push(mixture[start:end], multichannel[start:end])
            runner.process_available()
            pushed_samples = end
            if time_scale > 0.0:
                deadline = started + (pushed_samples / sample_rate) / time_scale
                remaining = deadline - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
    runner.process_available()
    wall = time.perf_counter() - started
    stats = runner.stats()
    stats.update(
        {
            "benchmark": "streaming_runner_simulation",
            "device": str(device),
            "source_wav": str(source),
            "time_scale": time_scale,
            "capture_thread": bool(args.capture_thread),
            "overload_policy": str(args.overload_policy),
            "wall_seconds": wall,
            "seconds_replayed": len(mixture) / sample_rate,
        }
    )
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
