#!/usr/bin/env python
"""Validate variable-duration inference for the exported speaker encoder.

The retained streaming design may receive enrollment and extracted chunks of
different durations.  The ordinary ONNX parity check uses only the export
example lengths, so this diagnostic deliberately exercises shorter and longer
inputs and compares them with the checkpoint-backed PyTorch module.

This is a deployment-contract check, not a speaker-recognition accuracy
benchmark and not real-driver evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "export"))

from car_tse.config import load_stage_config  # noqa: E402
from car_tse.train.utils import write_metrics  # noqa: E402
from eval_onnx_pipeline_golden_parity import compare  # noqa: E402
from export_onnx_components import _load_component  # noqa: E402


DEFAULT_MANIFEST = REPO_ROOT / "output/onnx/retained_ckptfix3_2026-10-14/manifest.json"


def _speaker_model_path(manifest_path: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("components", [])
    if not rows and manifest.get("component"):
        rows = [manifest]
    for row in rows:
        if str(row.get("component")) == "speaker_embedding":
            path = Path(str(row["output"]))
            if not path.exists():
                raise FileNotFoundError(f"speaker embedding ONNX file is missing: {path}")
            return path
    raise ValueError("ONNX manifest does not contain a speaker_embedding component")


def _run_case(
    model: torch.nn.Module,
    session: Any,
    enrollment_samples: int,
    extracted_samples: int,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    enrollment = torch.randn((1, enrollment_samples), generator=generator)
    extracted = torch.randn((1, extracted_samples), generator=generator)
    with torch.inference_mode():
        reference = model(enrollment, extracted).detach().cpu().numpy()
    names = [meta.name for meta in session.get_inputs()]
    actual = np.asarray(
        session.run(
            None,
            {
                names[0]: enrollment.numpy(),
                names[1]: extracted.numpy(),
            },
        )[0],
        dtype=np.float32,
    )
    result = compare(reference, actual)
    result.update(
        {
            "enrollment_samples": int(enrollment_samples),
            "extracted_samples": int(extracted_samples),
            "enrollment_seconds": float(enrollment_samples / 16000.0),
            "extracted_seconds": float(extracted_samples / 16000.0),
        }
    )
    return result


def run(
    *,
    config_path: Path,
    paths_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    import onnxruntime as ort

    cfg = load_stage_config(config_path, paths_path)
    model, _, _, _, checkpoint = _load_component(cfg, "speaker_embedding")
    model.eval().cpu()
    model_path = _speaker_model_path(manifest_path)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    length_pairs = ((16000, 32000), (24000, 48000), (32000, 64000))
    cases: list[dict[str, Any]] = []
    for index, (enrollment_samples, extracted_samples) in enumerate(length_pairs):
        try:
            cases.append(
                _run_case(
                    model,
                    session,
                    enrollment_samples,
                    extracted_samples,
                    seed=20261014 + index,
                )
            )
        except Exception as exc:  # keep the failure reproducible in the artifact
            cases.append(
                {
                    "passed": False,
                    "enrollment_samples": int(enrollment_samples),
                    "extracted_samples": int(extracted_samples),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "benchmark": "onnx_speaker_embedding_variable_length",
        "status": "ok" if all(bool(case.get("passed")) for case in cases) else "variable_length_failed",
        "config": str(config_path.resolve()),
        "paths": str(paths_path.resolve()),
        "checkpoint": str(checkpoint),
        "onnx_manifest": str(manifest_path.resolve()),
        "onnx_model": str(model_path.resolve()),
        "provider": "CPUExecutionProvider",
        "cases": cases,
        "limitations": (
            "Deterministic random waveform parity only; this validates ONNX input-shape "
            "behavior and numerical agreement, not speaker-recognition accuracy, "
            "quantization, target-SoC performance, or real-driver behavior."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "configs/eval_end_to_end_cavemove_target_kws_smart_speaker_compact_cabin_smart_thr030_2026-10-12.yaml",
    )
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.yaml")
    parser.add_argument("--onnx-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "logs/onnx_speaker_embedding_variable_length_2026-10-14.json",
    )
    args = parser.parse_args()
    result = run(
        config_path=args.config.resolve(),
        paths_path=args.paths.resolve(),
        manifest_path=args.onnx_manifest.resolve(),
    )
    write_metrics(args.output.resolve(), result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
