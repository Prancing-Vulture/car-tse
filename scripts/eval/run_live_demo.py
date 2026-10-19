#!/usr/bin/env python3
"""Run the four-stage CAR-TSE pipeline on live microphone audio.

The selected microphone must provide the configured number of simultaneous
channels at the model sample rate. This script prints recognized intent and
stub action strings; it does not control vehicle hardware.
"""

from __future__ import annotations

import argparse
import inspect
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "eval_end_to_end_cavemove.yaml",
    )
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--enrollment", type=Path, default=REPO_ROOT / "data" / "enrollment.wav")
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "car_tse_spexplus_from_stage1.pt",
        help="Base SpEx+ weights used to construct Stage 1.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Compute device for the models (default: CUDA when available).",
    )
    parser.add_argument(
        "--mic-device",
        default=None,
        help="Input device number or name from --list-devices; default input device if omitted.",
    )
    parser.add_argument(
        "--input-channels",
        type=int,
        default=None,
        help="Number of hardware input channels to open (defaults to the model channel count).",
    )
    parser.add_argument(
        "--channel-indices",
        default=None,
        help="Comma-separated hardware channels passed to the model, e.g. 0,2,4,6 for an 8-channel device.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
        help="Audio window length (defaults to segment_seconds in the config).",
    )
    parser.add_argument("--hop-seconds", type=float, default=0.25)
    parser.add_argument("--max-lag-seconds", type=float, default=2.0)
    parser.add_argument(
        "--overload-policy",
        choices=("drop_oldest", "catch_up"),
        default="drop_oldest",
        help="How to recover if model inference falls behind live capture.",
    )
    parser.add_argument("--blocksize", type=int, default=1024)
    parser.add_argument(
        "--status-interval",
        type=float,
        default=10.0,
        help="Seconds between live throughput summaries; set to 0 to disable.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print model scores for each window.")
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List input devices and exit without loading checkpoints.",
    )
    return parser


def _parse_device(value: Optional[str]) -> Any:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _load_sounddevice() -> Any:
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        raise SystemExit(
            "Cannot load python-sounddevice. Install it in this venv with "
            "'python -m pip install sounddevice'. If the error mentions PortAudio, "
            "install its runtime with 'sudo apt install libportaudio2'."
        ) from exc
    return sd


def _list_input_devices(sd: Any) -> int:
    devices = sd.query_devices()
    default_devices = sd.default.device
    default_input = default_devices[0] if isinstance(default_devices, (tuple, list)) else None
    found = False
    print("Input devices:")
    for index, info in enumerate(devices):
        channels = int(info.get("max_input_channels", 0))
        if channels < 1:
            continue
        found = True
        marker = " (default)" if index == default_input else ""
        print(
            "  {}: {} | max input channels: {} | default rate: {} Hz{}".format(
                index,
                info.get("name", "unknown"),
                channels,
                int(round(float(info.get("default_samplerate", 0)))),
                marker,
            )
        )
    if not found:
        print("  No input-capable devices were found.")
        return 1
    return 0


def _parse_channel_indices(value: Optional[str], num_mics: int, input_channels: int) -> List[int]:
    if value is None:
        indices = list(range(num_mics))
    else:
        try:
            indices = [int(part.strip()) for part in value.split(",") if part.strip()]
        except ValueError as exc:
            raise ValueError("--channel-indices must be comma-separated integers") from exc
    if len(indices) != num_mics:
        raise ValueError(
            "the model expects {} channels, but --channel-indices selects {}".format(
                num_mics, len(indices)
            )
        )
    if len(set(indices)) != len(indices):
        raise ValueError("--channel-indices cannot contain duplicates")
    if any(index < 0 or index >= input_channels for index in indices):
        raise ValueError(
            "each --channel-indices value must be between 0 and {}".format(input_channels - 1)
        )
    return indices


def _print_score(name: str, value: Optional[float]) -> str:
    return "{}={}".format(name, "-" if value is None else "{:.3f}".format(float(value)))


def _print_live_output(output: Any, was_pending: bool, verbose: bool) -> bool:
    pending = bool(getattr(output, "command_pending", False))
    if bool(getattr(output, "wake_detected", False)):
        print("[WAKE] Wake word accepted. Speak your command.", flush=True)
    elif pending and not was_pending:
        print("[LISTENING] Speak your command now.", flush=True)

    if bool(getattr(output, "intent_ready", False)):
        action = getattr(output, "action", None)
        if action is None:
            print("[COMMAND] Intent accepted, but no action label was returned.", flush=True)
        else:
            confidence = getattr(output, "intent_confidence", None)
            confidence_text = "-" if confidence is None else "{:.3f}".format(float(confidence))
            print(
                "[COMMAND] intent_id={} action={} confidence={}".format(
                    action.intent_id, action.action, confidence_text
                ),
                flush=True,
            )
    elif bool(getattr(output, "command_suppressed", False)):
        print("[COMMAND] Candidate was suppressed by the configured pipeline gates.", flush=True)

    if verbose:
        print(
            "[SCORES] accepted={} {} {} {}".format(
                bool(getattr(output, "accepted", False)),
                _print_score("presence", getattr(output, "presence_probability", None)),
                _print_score("wake", getattr(output, "wake_probability", None)),
                _print_score("speaker_similarity", getattr(output, "speaker_similarity", None)),
            ),
            flush=True,
        )
    return pending


def _print_summary(runner: Any, callback_stats: dict[str, Any], callback_lock: threading.Lock) -> None:
    stats = runner.stats()
    print(
        "[SUMMARY] windows={} inference_rtf={:.3f} lag={:.2f}s skipped={:.2f}s "
        "wake_events={} intent_events={}".format(
            int(stats["processed_windows"]),
            stats["real_time_factor"],
            stats["lag_seconds"],
            stats["skipped_seconds"],
            int(stats["wake_events"]),
            int(stats["intent_events"]),
        ),
        flush=True,
    )
    with callback_lock:
        status_count = int(callback_stats["status_count"])
        last_status = callback_stats["last_status"]
    if status_count:
        print(
            "[AUDIO] PortAudio reported {} input status event(s); last: {}".format(
                status_count, last_status
            ),
            flush=True,
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    sd = _load_sounddevice()
    if args.list_devices:
        return _list_input_devices(sd)

    if args.blocksize < 0:
        parser.error("--blocksize must be 0 or greater")
    if args.status_interval < 0:
        parser.error("--status-interval must be 0 or greater")
    if args.max_lag_seconds <= 0:
        parser.error("--max-lag-seconds must be positive")
    if args.window_seconds is not None and args.window_seconds <= 0:
        parser.error("--window-seconds must be positive")
    if args.hop_seconds <= 0:
        parser.error("--hop-seconds must be positive")

    # Make the repo and vendored SpEx+ importable even when PYTHONPATH was not
    # exported in the shell.
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "eval"))
    sys.path.insert(0, str(REPO_ROOT / "third_party" / "speaker_extraction_SpEx"))
    sys.path.insert(0, str(REPO_ROOT / "src"))

    import numpy as np
    import torch

    from benchmark_stagewise_latency import load_pipeline, repo_path
    from car_tse.config import load_stage_config
    from car_tse.io.audio import read_mono
    from car_tse.pipeline.in_car_pipeline import PipelineState
    from car_tse.pipeline.streaming_runner import StreamingRunner

    config_path = repo_path(args.config)
    paths_file = repo_path(args.paths)
    enrollment_path = repo_path(args.enrollment)
    base_checkpoint = repo_path(args.base_checkpoint)
    if not paths_file.is_file():
        example_paths = REPO_ROOT / "configs" / "paths.example.yaml"
        if example_paths.is_file():
            paths_file = example_paths

    for path, label in (
        (config_path, "pipeline config"),
        (enrollment_path, "enrollment WAV"),
        (base_checkpoint, "base SpEx+ checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError("Missing {}: {}".format(label, path))

    cfg = load_stage_config(config_path, paths_file)
    sample_rate = int(cfg.get("sample_rate", 16000))
    num_mics = int(cfg.get("num_mics", 4))
    window_seconds = float(
        cfg.get("segment_seconds", 3.0) if args.window_seconds is None else args.window_seconds
    )
    if window_seconds <= 0:
        parser.error("the configured window length must be positive")
    if args.hop_seconds > window_seconds:
        parser.error("--hop-seconds must not exceed the audio window length")

    input_channels = num_mics if args.input_channels is None else args.input_channels
    if input_channels < 1:
        parser.error("--input-channels must be positive")
    try:
        channel_indices = _parse_channel_indices(args.channel_indices, num_mics, input_channels)
    except ValueError as exc:
        parser.error(str(exc))

    microphone = _parse_device(args.mic_device)
    try:
        device_info = sd.query_devices(microphone, "input")
        max_input_channels = int(device_info["max_input_channels"])
    except Exception as exc:
        raise SystemExit("Could not query the selected microphone: {}".format(exc)) from exc
    if input_channels > max_input_channels:
        raise SystemExit(
            "Selected microphone exposes {} input channel(s), but --input-channels={} was requested. "
            "Run --list-devices and select a multichannel input device.".format(
                max_input_channels, input_channels
            )
        )
    try:
        sd.check_input_settings(
            device=microphone,
            channels=input_channels,
            dtype="float32",
            samplerate=sample_rate,
        )
    except Exception as exc:
        raise SystemExit(
            "The selected microphone does not support {} input channel(s) at {} Hz: {}. "
            "Run --list-devices, then choose a compatible device or audio-system setting.".format(
                input_channels, sample_rate, exc
            )
        ) from exc

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false")
    if args.device == "auto":
        compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        compute_device = torch.device(args.device)

    enrollment_np, _ = read_mono(enrollment_path, sample_rate)
    if enrollment_np.size == 0 or not np.isfinite(enrollment_np).all():
        raise ValueError("Enrollment WAV is empty or contains NaN/infinite samples")

    print("[STARTUP] Loading four-stage models on {}...".format(compute_device), flush=True)
    pipeline = load_pipeline(cfg, paths_file, base_checkpoint, compute_device)
    enrollment = torch.from_numpy(np.ascontiguousarray(enrollment_np)).unsqueeze(0).to(compute_device)

    enrollment_embedding = None
    speaker_embedding = getattr(pipeline.stage1, "speaker_embedding", None)
    if callable(speaker_embedding):
        with torch.inference_mode():
            enrollment_embedding = speaker_embedding(enrollment)
        if enrollment_embedding is not None:
            enrollment_embedding = enrollment_embedding.detach()
    presence_embedding = (
        enrollment_embedding if getattr(pipeline.presence, "requires_enrollment", False) else None
    )
    state = PipelineState(
        reference_speaker_embedding=enrollment_embedding,
        presence_enrollment_embedding=presence_embedding,
        target_kws_enrollment_embedding=enrollment_embedding,
    )

    try:
        forward_parameters = inspect.signature(pipeline.forward).parameters
        forward_accepts_extra_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in forward_parameters.values()
        )
    except (TypeError, ValueError):
        forward_parameters = {}
        forward_accepts_extra_kwargs = True

    captured_outputs: deque[Any] = deque()

    def forward_and_record(mixture: Any, enroll: Any, current_state: Any = None, **kwargs: Any) -> Any:
        if not forward_accepts_extra_kwargs:
            kwargs = {key: value for key, value in kwargs.items() if key in forward_parameters}
        if compute_device.type == "cuda":
            torch.cuda.synchronize(compute_device)
        result = pipeline.forward(mixture, enroll, current_state, **kwargs)
        if compute_device.type == "cuda":
            torch.cuda.synchronize(compute_device)
        captured_outputs.append(result[0])
        return result

    runner = StreamingRunner(
        forward=forward_and_record,
        enrollment=enrollment_np,
        sample_rate=sample_rate,
        window_seconds=window_seconds,
        hop_seconds=args.hop_seconds,
        num_mics=num_mics,
        max_lag_seconds=args.max_lag_seconds,
        overload_policy=args.overload_policy,
        device=compute_device,
        state=state,
    )

    callback_stats = {"status_count": 0, "last_status": None, "error": None}
    callback_lock = threading.Lock()

    def audio_callback(indata: Any, _frames: int, _time_info: Any, status: Any) -> None:
        if status:
            with callback_lock:
                callback_stats["status_count"] += 1
                callback_stats["last_status"] = str(status)
        try:
            selected = np.ascontiguousarray(indata[:, channel_indices], dtype=np.float32)
            mono = np.mean(selected, axis=1, dtype=np.float32)
            runner.push(mono, multichannel=selected)
        except Exception as exc:
            with callback_lock:
                if callback_stats["error"] is None:
                    callback_stats["error"] = exc

    print(
        "[STARTUP] Models ready. Mic='{}', channels={}, model channels={}, sample_rate={} Hz.".format(
            device_info.get("name", "default"), input_channels, channel_indices, sample_rate
        ),
        flush=True,
    )
    print(
        "[STARTUP] Window={:.2f}s, hop={:.2f}s. First inference starts after the window fills. "
        "Press Ctrl+C to stop.".format(window_seconds, args.hop_seconds),
        flush=True,
    )

    pending = False
    next_status = time.monotonic() + args.status_interval if args.status_interval else None
    try:
        with sd.InputStream(
            device=microphone,
            channels=input_channels,
            samplerate=sample_rate,
            blocksize=args.blocksize,
            dtype="float32",
            callback=audio_callback,
        ):
            print("[LISTENING] Live audio capture is active.", flush=True)
            while True:
                runner.process_available()
                while captured_outputs:
                    output = captured_outputs.popleft()
                    pending = _print_live_output(output, pending, args.verbose)

                with callback_lock:
                    callback_error = callback_stats["error"]
                if callback_error is not None:
                    raise RuntimeError("Microphone callback failed: {}".format(callback_error))

                now = time.monotonic()
                if next_status is not None and now >= next_status:
                    stats = runner.stats()
                    print(
                        "[STATUS] windows={} inference_rtf={:.3f} lag={:.2f}s skipped={:.2f}s".format(
                            int(stats["processed_windows"]),
                            stats["real_time_factor"],
                            stats["lag_seconds"],
                            stats["skipped_seconds"],
                        ),
                        flush=True,
                    )
                    next_status = now + args.status_interval
                time.sleep(0.005)
    except KeyboardInterrupt:
        print("\n[STOP] Stopping live capture...", flush=True)
    except sd.PortAudioError as exc:
        raise SystemExit("Could not start microphone capture: {}".format(exc)) from exc
    finally:
        _print_summary(runner, callback_stats, callback_lock)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
