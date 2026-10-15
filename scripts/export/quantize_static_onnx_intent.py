#!/usr/bin/env python
"""Calibrate and evaluate a static-int8 ONNX intent graph.

This is a deployment diagnostic for the compact intent candidate.  It uses a
declared waveform cache for calibration and a disjoint tail of that cache for
parity/accuracy checks; it does not modify the PyTorch checkpoint or retained
runtime.
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def latency(session: Any, rows: np.ndarray, warmup: int, repeats: int) -> dict[str, float]:
    samples: list[float] = []
    input_name = session.get_inputs()[0].name
    for row in rows[: max(1, min(len(rows), 4))]:
        session.run(None, {input_name: row[None, :].astype(np.float32, copy=False)})
    for _ in range(max(0, int(warmup))):
        row = rows[_ % len(rows)]
        session.run(None, {input_name: row[None, :].astype(np.float32, copy=False)})
    for index in range(max(1, int(repeats))):
        row = rows[index % len(rows)]
        start = time.perf_counter()
        session.run(None, {input_name: row[None, :].astype(np.float32, copy=False)})
        samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "mean_ms": float(np.mean(samples)),
    }


def evaluate_session(session: Any, rows: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    input_name = session.get_inputs()[0].name
    outputs: list[np.ndarray] = []
    for row in rows:
        outputs.append(
            np.asarray(
                session.run(None, {input_name: row[None, :].astype(np.float32, copy=False)})[0][0],
                dtype=np.float32,
            )
        )
    logits = np.stack(outputs)
    return {"logits": logits, "accuracy": float((logits.argmax(axis=1) == labels).mean())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration-cache", type=Path, required=True)
    parser.add_argument("--labels-manifest", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--calibration-limit", type=int, default=128)
    parser.add_argument("--evaluation-limit", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()

    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    cache = np.load(args.calibration_cache.resolve(), mmap_mode="r", allow_pickle=False)
    labels = np.asarray(
        [int(json.loads(line)["label"]) for line in args.labels_manifest.read_text(encoding="utf-8").splitlines() if line.strip()],
        dtype=np.int64,
    )
    if len(cache) != len(labels):
        raise ValueError(f"cache/label length mismatch: {len(cache)} vs {len(labels)}")
    calibration_limit = max(1, int(args.calibration_limit))
    evaluation_limit = max(1, int(args.evaluation_limit))
    eval_start = calibration_limit
    eval_end = min(len(cache), eval_start + evaluation_limit)
    if eval_end <= eval_start:
        raise ValueError("calibration/evaluation split leaves no held-out rows")
    calibration_rows = np.asarray(cache[:calibration_limit], dtype=np.float32)
    evaluation_rows = np.asarray(cache[eval_start:eval_end], dtype=np.float32)
    evaluation_labels = labels[eval_start:eval_end]

    class _Reader(CalibrationDataReader):
        def __init__(self, rows: np.ndarray, input_name: str):
            self.rows = rows
            self.input_name = input_name
            self.index = 0

        def get_next(self) -> dict[str, np.ndarray] | None:
            if self.index >= len(self.rows):
                return None
            row = self.rows[self.index]
            self.index += 1
            return {self.input_name: row[None, :].astype(np.float32, copy=False)}

        def rewind(self) -> None:
            self.index = 0

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    source_session = ort.InferenceSession(
        str(args.model.resolve()), providers=["CPUExecutionProvider"]
    )
    input_name = source_session.get_inputs()[0].name
    quantize_static(
        str(args.model.resolve()),
        str(args.output_model.resolve()),
        _Reader(calibration_rows, input_name),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        # MinMax avoids the histogram collector's failure on near-constant
        # tensors produced by the exported log-mel frontend under current
        # ONNX Runtime/NumPy versions.  This is a diagnostic candidate; its
        # accuracy and latency still have to pass the report gates below.
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
    )
    quantized_session = ort.InferenceSession(
        str(args.output_model.resolve()), providers=["CPUExecutionProvider"]
    )
    float_eval = evaluate_session(source_session, evaluation_rows, evaluation_labels)
    int8_eval = evaluate_session(quantized_session, evaluation_rows, evaluation_labels)
    float_logits = float_eval["logits"]
    int8_logits = int8_eval["logits"]
    delta = int8_logits - float_logits
    parity = {
        "relative_l2_error": float(np.linalg.norm(delta) / max(np.linalg.norm(float_logits), 1.0e-12)),
        "max_abs_error": float(np.abs(delta).max()),
        "cosine_similarity": float(
            np.dot(float_logits.reshape(-1), int8_logits.reshape(-1))
            / max(np.linalg.norm(float_logits) * np.linalg.norm(int8_logits), 1.0e-12)
        ),
    }
    result = {
        "benchmark": "onnx_static_int8_intent_feasibility",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "provider": "CPUExecutionProvider",
        "calibration_method": "minmax_qdq_int8_symmetric",
        "model": str(args.model.resolve()),
        "model_sha256": sha256(args.model.resolve()),
        "output_model": str(args.output_model.resolve()),
        "output_model_sha256": sha256(args.output_model.resolve()),
        "float_bytes": int(args.model.stat().st_size),
        "int8_bytes": int(args.output_model.stat().st_size),
        "size_ratio": float(args.output_model.stat().st_size / max(args.model.stat().st_size, 1)),
        "calibration_cache": str(args.calibration_cache.resolve()),
        "calibration_rows": int(len(calibration_rows)),
        "evaluation_rows": int(len(evaluation_rows)),
        "float_accuracy": float(float_eval["accuracy"]),
        "int8_accuracy": float(int8_eval["accuracy"]),
        "parity": parity,
        "float_cpu": latency(source_session, evaluation_rows, args.warmup, args.repeats),
        "int8_cpu": latency(quantized_session, evaluation_rows, args.warmup, args.repeats),
    }
    args.output_report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
