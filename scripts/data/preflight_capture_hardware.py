#!/usr/bin/env python
"""Probe audio inputs against the real-driver capture contract.

This command is read-only: it enumerates input devices and calls PortAudio's
format check without creating a session directory or recording audio.  The
default contract is the current model input (4 channels at 16 kHz).  Use
``--channel-count 6`` when checking the recommended production capture
interface with four cabin microphones plus two playback-reference channels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _device_name(info: dict[str, Any], fallback: Any) -> str:
    return str(info.get("name", fallback))


def probe_device(
    sounddevice: Any,
    device: int | str,
    *,
    sample_rate: int,
    channel_count: int,
) -> dict[str, Any]:
    """Return a JSON-serializable capability result for one input device."""

    info = sounddevice.query_devices(device, "input")
    max_channels = int(info.get("max_input_channels", 0) or 0)
    result: dict[str, Any] = {
        "device": device,
        "name": _device_name(info, device),
        "max_input_channels": max_channels,
        "requested_channels": int(channel_count),
        "requested_sample_rate_hz": int(sample_rate),
        "supported": False,
    }
    if max_channels < int(channel_count):
        result["reason"] = (
            f"device exposes {max_channels} input channel(s), "
            f"but {int(channel_count)} requested"
        )
        return result
    try:
        sounddevice.check_input_settings(
            device=device,
            channels=int(channel_count),
            samplerate=int(sample_rate),
            dtype="float32",
        )
    except Exception as exc:  # PortAudio uses backend-specific exception types.
        result["reason"] = f"format rejected: {exc}"
        return result
    result["supported"] = True
    result["reason"] = "format accepted"
    return result


def preflight(
    sounddevice: Any,
    *,
    sample_rate: int = 16000,
    channel_count: int = 4,
) -> dict[str, Any]:
    """Probe every visible input device without opening a recording stream."""

    if int(sample_rate) <= 0 or int(channel_count) <= 0:
        raise ValueError("sample_rate and channel_count must be positive")

    devices = sounddevice.query_devices()
    results: list[dict[str, Any]] = []
    for index, info in enumerate(devices):
        if int(info.get("max_input_channels", 0) or 0) <= 0:
            continue
        results.append(
            probe_device(
                sounddevice,
                index,
                sample_rate=int(sample_rate),
                channel_count=int(channel_count),
            )
        )
    return {
        "sample_rate_hz": int(sample_rate),
        "channel_count": int(channel_count),
        "device_count": len(results),
        "supported_devices": [row for row in results if row["supported"]],
        "devices": results,
        "recorded": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channel-count", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.sample_rate <= 0 or args.channel_count <= 0:
        parser.error("--sample-rate and --channel-count must be positive")
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice is required; install it with python -m pip install sounddevice", file=sys.stderr)
        return 2
    report = preflight(
        sd,
        sample_rate=int(args.sample_rate),
        channel_count=int(args.channel_count),
    )
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["supported_devices"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
