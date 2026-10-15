#!/usr/bin/env python
"""Benchmark exported ONNX components on the local ONNX Runtime provider."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def benchmark(
    manifest_path: Path,
    warmup: int,
    repeats: int,
    intra_op_threads: int | None = None,
    inter_op_threads: int | None = None,
) -> dict[str, Any]:
    import onnxruntime as ort

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    components = manifest.get("components", [])
    if not components and manifest.get("component"):
        # The exporter writes a single-component manifest when invoked with a
        # specific component; accept that form for candidate benchmarks too.
        components = [manifest]
    if not components:
        raise ValueError("ONNX manifest contains no components")
    rng = np.random.default_rng(20261014)
    audio_seconds = {
        "stage1": 3.0,
        "speaker_embedding": 3.0,
        "kws": 1.0,
        "presence": 3.0,
        "target_kws": 1.0,
        "compact_verifier": 1.0,
        "intent": 2.0,
    }
    rows: list[dict[str, Any]] = []
    session_options = ort.SessionOptions()
    if intra_op_threads is not None:
        session_options.intra_op_num_threads = max(1, int(intra_op_threads))
    if inter_op_threads is not None:
        session_options.inter_op_num_threads = max(1, int(inter_op_threads))
    for component in components:
        model_path = Path(str(component["output"]))
        session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        shapes = [tuple(int(value) for value in shape) for shape in component["input_shapes"]]
        feed = {
            input_meta.name: rng.standard_normal(shape, dtype=np.float32)
            for input_meta, shape in zip(session.get_inputs(), shapes)
        }
        for _ in range(max(0, int(warmup))):
            session.run(None, feed)
        elapsed_ms: list[float] = []
        for _ in range(max(1, int(repeats))):
            start = time.perf_counter()
            session.run(None, feed)
            elapsed_ms.append((time.perf_counter() - start) * 1000.0)
        seconds = float(audio_seconds.get(str(component["component"]), 1.0))
        rows.append(
            {
                "component": component["component"],
                "model": str(model_path),
                "bytes": int(model_path.stat().st_size),
                "provider": "CPUExecutionProvider",
                "warmup": int(warmup),
                "repeats": int(repeats),
                "p50_ms": float(np.percentile(elapsed_ms, 50)),
                "p95_ms": float(np.percentile(elapsed_ms, 95)),
                "p99_ms": float(np.percentile(elapsed_ms, 99)),
                "mean_ms": float(np.mean(elapsed_ms)),
                "input_audio_seconds": seconds,
                "real_time_factor_p95": float(np.percentile(elapsed_ms, 95) / 1000.0 / seconds),
            }
        )
    return {
        "benchmark": "onnx_component_cpu_latency",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "python": platform.python_version(),
        "provider": "CPUExecutionProvider",
        "runtime_options": {
            "intra_op_threads": None if intra_op_threads is None else max(1, int(intra_op_threads)),
            "inter_op_threads": None if inter_op_threads is None else max(1, int(inter_op_threads)),
        },
        "components": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--intra-op-threads", type=int, default=None)
    parser.add_argument("--inter-op-threads", type=int, default=None)
    args = parser.parse_args()
    result = benchmark(
        args.manifest.resolve(),
        args.warmup,
        args.repeats,
        args.intra_op_threads,
        args.inter_op_threads,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
