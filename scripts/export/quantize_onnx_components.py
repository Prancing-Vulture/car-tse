#!/usr/bin/env python
"""Run a dynamic-int8 ONNX feasibility preflight.

Dynamic quantization is intentionally diagnostic here.  It quantizes weights
without production calibration data and does not replace the retained models;
static calibration/QAT on representative cabin audio is still required for a
release.
"""

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


def _parity(reference: np.ndarray, quantized: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float32)
    quantized = np.asarray(quantized, dtype=np.float32)
    difference = np.abs(reference - quantized)
    relative_l2 = float(
        np.linalg.norm(reference - quantized)
        / max(np.linalg.norm(reference), 1.0e-12)
    )
    ref_flat = reference.reshape(-1)
    quant_flat = quantized.reshape(-1)
    cosine = float(
        np.dot(ref_flat, quant_flat)
        / max(np.linalg.norm(ref_flat) * np.linalg.norm(quant_flat), 1.0e-12)
    )
    return {
        "max_abs_error": float(difference.max()) if difference.size else 0.0,
        "relative_l2_error": relative_l2,
        "cosine_similarity": cosine,
    }


def _latency(session: Any, feed: dict[str, np.ndarray], warmup: int, repeats: int) -> dict[str, float]:
    for _ in range(max(0, int(warmup))):
        session.run(None, feed)
    times: list[float] = []
    for _ in range(max(1, int(repeats))):
        start = time.perf_counter()
        session.run(None, feed)
        times.append((time.perf_counter() - start) * 1000.0)
    return {
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "mean_ms": float(np.mean(times)),
    }


def quantize_manifest(
    manifest_path: Path,
    output_dir: Path,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    components = manifest.get("components", [])
    if not components and manifest.get("component"):
        # Single-component exporter manifests are valid diagnostic inputs.
        components = [manifest]
    rng = np.random.default_rng(20261014)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for component in components:
        name = str(component["component"])
        source = Path(str(component["output"]))
        target = output_dir / f"{name}.dynamic_int8.onnx"
        row: dict[str, Any] = {
            "component": name,
            "source": str(source),
            "source_sha256": _sha256(source),
            "output": str(target.resolve()),
        }
        try:
            quantize_dynamic(
                str(source),
                str(target),
                weight_type=QuantType.QInt8,
                per_channel=True,
            )
            float_session = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
            int8_session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
            shapes = [tuple(int(value) for value in shape) for shape in component["input_shapes"]]
            feed = {
                input_meta.name: rng.standard_normal(shape, dtype=np.float32)
                for input_meta, shape in zip(float_session.get_inputs(), shapes)
            }
            float_output = float_session.run(None, feed)[0]
            int8_output = int8_session.run(None, feed)[0]
            row.update(
                {
                    "status": "ok",
                    "output_sha256": _sha256(target),
                    "source_bytes": int(source.stat().st_size),
                    "output_bytes": int(target.stat().st_size),
                    "size_ratio": float(target.stat().st_size / max(source.stat().st_size, 1)),
                    "parity": _parity(float_output, int8_output),
                    "float_cpu": _latency(float_session, feed, warmup, repeats),
                    "int8_cpu": _latency(int8_session, feed, warmup, repeats),
                }
            )
        except Exception as exc:  # keep a component-level audit trail
            row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        rows.append(row)
    return {
        "benchmark": "onnx_dynamic_int8_feasibility",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "provider": "CPUExecutionProvider",
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "calibration": "none; diagnostic weight-only dynamic quantization",
        "components": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    result = quantize_manifest(
        args.manifest.resolve(), args.output_dir.resolve(), args.warmup, args.repeats
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if all(row["status"] == "ok" for row in result["components"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
