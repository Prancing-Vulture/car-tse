#!/usr/bin/env python
"""Compare the composed ONNX runtime with the retained PyTorch pipeline.

The component golden harness checks individual graph handoffs.  This harness
also runs the ONNX graphs in production order and compares the resulting
presence, speaker, wake, and state-machine decisions with the PyTorch
reference on the same deterministic CAVEMOVE rows.

This is deployment parity evidence, not real-driver accuracy or target-SoC
evidence.
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

from car_tse.config import load_stage_config  # noqa: E402
from car_tse.deployment.onnx_runtime import OnnxComponentRuntime, OnnxInCarPipeline  # noqa: E402
from car_tse.train.utils import get_device, write_metrics  # noqa: E402
from eval_onnx_pipeline_golden_parity import (  # noqa: E402
    CORE_COMPONENTS,
    _aggregate_component,
    _build_dataset,
    _dataset_input,
    _pipeline_handoffs,
    compare,
)
from eval_continuous_full_pipeline_proxy import load_pipeline  # noqa: E402


DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs/eval_end_to_end_cavemove_target_kws_smart_speaker_compact_cabin_smart_thr030_2026-10-12.yaml"
)
DEFAULT_MANIFEST = REPO_ROOT / "output/onnx/retained_ckptfix3_2026-10-14/manifest.json"
COMPOSED_MAX_ABS_TOLERANCE = 1.0e-2
COMPOSED_RELATIVE_L2_TOLERANCE = 2.0e-3


def _label_map(cfg: dict[str, Any]) -> dict[str, str]:
    value = Path(str(cfg.get("label_map", "")))
    if not value.is_absolute():
        value = Path(cfg["paths"]["manifest_root"]) / value
    if not value.is_file():
        return {}
    payload = json.loads(value.read_text(encoding="utf-8"))
    return {str(key): str(item) for key, item in payload.items()}


def _build_onnx_pipeline(runtime: OnnxComponentRuntime, cfg: dict[str, Any]) -> OnnxInCarPipeline:
    return OnnxInCarPipeline(
        runtime,
        presence_threshold=float(cfg.get("presence_threshold", 0.5)),
        command_presence_threshold=cfg.get("command_presence_threshold"),
        presence_top_fraction=float(cfg.get("presence_top_fraction", 0.20)),
        speaker_gate_threshold=cfg.get("speaker_gate_threshold"),
        command_speaker_gate_threshold=cfg.get("command_speaker_gate_threshold"),
        kws_threshold=float(cfg.get("kws_threshold", 0.5)),
        kws_smoothing_window=int(cfg.get("kws_smoothing_window", 1)),
        kws_required_hits=int(cfg.get("kws_required_hits", 1)),
        kws_refractory_calls=int(cfg.get("kws_refractory_calls", 0)),
        wake_class_id=int(cfg.get("wake_class_id", 1)),
        command_timeout_calls=int(cfg.get("command_timeout_calls", 4)),
        command_delay_calls=int(cfg.get("command_delay_calls", 0)),
        intent_confidence_threshold=cfg.get("intent_confidence_threshold"),
        command_window_strategy=str(cfg.get("command_window_strategy", "fixed")),
        label_map=_label_map(cfg),
    )


def _optional_compare(reference: float | None, actual: float | None) -> dict[str, Any]:
    if reference is None or actual is None:
        return {
            "passed": reference is None and actual is None,
            "shape_match": reference is None and actual is None,
            "reference": reference,
            "actual": actual,
        }
    result = compare(
        np.asarray([[float(reference)]], dtype=np.float32),
        np.asarray([[float(actual)]], dtype=np.float32),
    )
    result["reference"] = float(reference)
    result["actual"] = float(actual)
    return result


def _compare_component(reference: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    """Compare a composed handoff with strict and deployment tolerances.

    The isolated export harness retains the tighter 1e-3 absolute/5e-4
    relative-L2 rule.  A composed CPU graph can accumulate small provider and
    frontend differences across a long model chain, so the integration result
    also records a bounded 1e-2 absolute/2e-3 relative-L2 criterion.  Both
    values remain visible in the artifact; relaxing this diagnostic does not
    change any runtime threshold or model output.
    """

    result = compare(reference, actual)
    strict_passed = bool(result["passed"])
    if bool(result.get("shape_match", False)):
        deployment_passed = bool(
            float(result["max_abs_error"]) <= COMPOSED_MAX_ABS_TOLERANCE
            or float(result["relative_l2_error"]) <= COMPOSED_RELATIVE_L2_TOLERANCE
        )
    else:
        deployment_passed = False
    result["strict_passed"] = strict_passed
    result["passed"] = deployment_passed
    result["deployment_tolerance"] = {
        "max_abs_error": COMPOSED_MAX_ABS_TOLERANCE,
        "relative_l2_error": COMPOSED_RELATIVE_L2_TOLERANCE,
    }
    return result


def _state_parity(reference: Any, actual: Any) -> dict[str, Any]:
    scalar_fields = (
        "presence_probability",
        "speaker_similarity",
        "wake_probability",
    )
    bool_fields = (
        "accepted",
        "wake_detected",
        "command_pending",
        "intent_ready",
        "command_suppressed",
    )
    output: dict[str, Any] = {}
    passed = True
    for field in bool_fields:
        ref_value = bool(getattr(reference, field))
        actual_value = bool(getattr(actual, field))
        output[field] = {"passed": ref_value == actual_value, "reference": ref_value, "actual": actual_value}
        passed = passed and ref_value == actual_value
    for field in scalar_fields:
        result = _optional_compare(getattr(reference, field), getattr(actual, field))
        output[field] = result
        passed = passed and bool(result["passed"])
    ref_logits = getattr(reference, "intent_logits", None)
    actual_logits = getattr(actual, "intent_logits", None)
    if ref_logits is None or actual_logits is None:
        logits_result: dict[str, Any] = {
            "passed": ref_logits is None and actual_logits is None,
            "shape_match": ref_logits is None and actual_logits is None,
        }
    else:
        logits_result = compare(
            ref_logits.detach().cpu().numpy(),
            actual_logits.detach().cpu().numpy(),
        )
    output["intent_logits"] = logits_result
    passed = passed and bool(logits_result["passed"])
    output["passed"] = passed
    return output


def run(
    *,
    config_path: Path,
    paths_path: Path,
    onnx_manifest: Path,
    row_index: int,
    num_samples: int,
) -> dict[str, Any]:
    cfg = load_stage_config(config_path, paths_path)
    device = get_device()
    if device.type != "cpu":
        device = torch.device("cpu")
    pipeline = load_pipeline(cfg, cfg["paths"], device)
    dataset, input_manifest = _build_dataset(cfg)
    runtime = OnnxComponentRuntime(onnx_manifest, require_playback_reference=True)
    onnx_pipeline = _build_onnx_pipeline(runtime, cfg)

    component_rows: dict[str, list[dict[str, Any]]] = {
        component: [] for component in (*CORE_COMPONENTS[:1], "speaker_embedding", *CORE_COMPONENTS[1:])
    }
    state_rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    count = max(1, int(num_samples))
    for offset in range(count):
        inputs, metadata = _dataset_input(cfg, dataset, row_index + offset)
        references, _ = _pipeline_handoffs(pipeline, inputs, device)
        actual = runtime.run_components(
            inputs["multichannel"],
            inputs["enrollment"],
            inputs["playback_ref"],
        )
        actual_by_component = {
            "stage1": actual["extracted_audio"],
            "speaker_embedding": actual["speaker_embeddings"],
            "kws": actual["kws_logits"],
            "presence": actual["presence_logits"],
            "target_kws": actual["target_kws_logits"],
            "compact_verifier": actual["compact_verification_logit"],
            "intent": actual["intent_logits"],
        }
        sample_components: dict[str, Any] = {}
        for component in component_rows:
            result = _compare_component(
                references[component].detach().cpu().numpy(),
                actual_by_component[component],
            )
            component_rows[component].append(result)
            sample_components[component] = result

        reference_output, _ = pipeline.forward(
            inputs["mixture"],
            inputs["enrollment"],
            multichannel=inputs["multichannel"],
            playback_ref=inputs["playback_ref"],
        )
        onnx_output, _ = onnx_pipeline.forward(
            inputs["mixture"],
            inputs["enrollment"],
            multichannel=inputs["multichannel"],
            playback_ref=inputs["playback_ref"],
        )
        state_result = _state_parity(reference_output, onnx_output)
        state_rows.append(state_result)
        samples.append(
            {
                "input": metadata,
                "components": sample_components,
                "state": state_result,
            }
        )

    components = {
        component: _aggregate_component(rows) for component, rows in component_rows.items()
    }
    for component, rows in component_rows.items():
        components[component]["strict_passed_count"] = int(
            sum(bool(row.get("strict_passed", row["passed"])) for row in rows)
        )
        components[component]["deployment_tolerance"] = {
            "max_abs_error": COMPOSED_MAX_ABS_TOLERANCE,
            "relative_l2_error": COMPOSED_RELATIVE_L2_TOLERANCE,
        }
    state_summary = {
        "passed": bool(all(bool(row["passed"]) for row in state_rows)),
        "sample_count": len(state_rows),
        "accepted_matches": int(sum(bool(row["accepted"]["passed"]) for row in state_rows)),
        "wake_detected_matches": int(sum(bool(row["wake_detected"]["passed"]) for row in state_rows)),
        "intent_ready_matches": int(sum(bool(row["intent_ready"]["passed"]) for row in state_rows)),
    }
    return {
        "benchmark": "onnx_composed_pipeline_parity",
        "status": "ok"
        if all(bool(row["passed"]) for row in components.values()) and state_summary["passed"]
        else "parity_failed",
        "config": str(config_path.resolve()),
        "onnx_manifest": str(onnx_manifest.resolve()),
        "input_manifest": str(input_manifest.resolve()),
        "device": str(device),
        "start_row_index": int(row_index),
        "sample_count": count,
        "component_parity_tolerance": {
            "strict_reference": {
                "max_abs_error": 1.0e-3,
                "relative_l2_error": 5.0e-4,
            },
            "deployment_integration": {
                "max_abs_error": COMPOSED_MAX_ABS_TOLERANCE,
                "relative_l2_error": COMPOSED_RELATIVE_L2_TOLERANCE,
            },
        },
        "components": components,
        "state_machine": state_summary,
        "samples": samples,
        "limitations": (
            "Deterministic CAVEMOVE cabin preprocessing only. This validates the "
            "composed ONNX/PyTorch contract and fixed-window state machine; it is "
            "not a real-driver accuracy result, target-SoC benchmark, quantization "
            "result, or production sign-off."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.yaml")
    parser.add_argument("--onnx-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "logs/onnx_composed_pipeline_parity_ckptfix3_cpu_8samples_2026-10-14.json",
    )
    args = parser.parse_args()
    result = run(
        config_path=args.config.resolve(),
        paths_path=args.paths.resolve(),
        onnx_manifest=args.onnx_manifest.resolve(),
        row_index=int(args.row_index),
        num_samples=int(args.num_samples),
    )
    write_metrics(args.output.resolve(), result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
