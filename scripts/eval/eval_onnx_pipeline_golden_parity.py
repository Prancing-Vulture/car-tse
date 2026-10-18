#!/usr/bin/env python
"""Check ONNX component handoffs against the retained PyTorch pipeline.

The existing exporter verifies each graph with independent random inputs. This
check goes one step further: it loads deterministic CAVEMOVE cabin samples
through the same dataset preprocessing used by the retained runtime, then
compares the PyTorch and ONNX outputs at each handoff. It is a deployment
parity diagnostic, not a new accuracy or real-driver result.
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
from car_tse.data.datasets import TripletExtractionDataset  # noqa: E402
from car_tse.pipeline.in_car_pipeline import normalize_extracted_waveform  # noqa: E402
from car_tse.train.utils import get_device, write_metrics  # noqa: E402
from eval_continuous_full_pipeline_proxy import load_pipeline, resolve_path  # noqa: E402


CORE_COMPONENTS = ("stage1", "kws", "presence", "target_kws", "compact_verifier", "intent")
OPTIONAL_COMPONENTS = ("speaker_embedding",)
COMPONENTS = CORE_COMPONENTS + OPTIONAL_COMPONENTS


def compare(reference: np.ndarray, actual: np.ndarray) -> dict[str, float | bool | list[int]]:
    """Return the same numerical parity diagnostics used by the exporter."""

    reference = np.asarray(reference, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    if reference.shape != actual.shape:
        return {
            "passed": False,
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "actual_shape": list(actual.shape),
            "max_abs_error": float("inf"),
            "relative_l2_error": float("inf"),
            "cosine_similarity": 0.0,
        }
    difference = np.abs(reference - actual)
    denominator = np.maximum(np.abs(reference), 1.0e-6)
    reference_flat = reference.reshape(-1)
    actual_flat = actual.reshape(-1)
    cosine = float(
        np.dot(reference_flat, actual_flat)
        / max(np.linalg.norm(reference_flat) * np.linalg.norm(actual_flat), 1.0e-12)
    )
    relative_l2 = float(
        np.linalg.norm(reference - actual)
        / max(np.linalg.norm(reference), 1.0e-12)
    )
    max_abs = float(difference.max()) if difference.size else 0.0
    return {
        "passed": bool(max_abs <= 1.0e-3 or relative_l2 <= 5.0e-4),
        "shape_match": True,
        "reference_shape": list(reference.shape),
        "actual_shape": list(actual.shape),
        "max_abs_error": max_abs,
        "max_relative_error": float((difference / denominator).max()) if difference.size else 0.0,
        "relative_l2_error": relative_l2,
        "cosine_similarity": cosine,
    }


def _session_map(manifest_path: Path) -> dict[str, Any]:
    import onnxruntime as ort

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("components", [])
    if not rows and manifest.get("component"):
        rows = [manifest]
    sessions: dict[str, Any] = {}
    for row in rows:
        component = str(row["component"])
        if component not in COMPONENTS:
            continue
        model_path = Path(str(row["output"]))
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model for {component} is missing: {model_path}")
        sessions[component] = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
    missing = [component for component in CORE_COMPONENTS if component not in sessions]
    if missing:
        raise ValueError(f"ONNX manifest is missing components: {missing}")
    return sessions


def _run(session: Any, values: tuple[np.ndarray, ...]) -> np.ndarray:
    names = [meta.name for meta in session.get_inputs()]
    if len(names) != len(values):
        raise ValueError(f"ONNX input count mismatch: expected {names}, got {len(values)} values")
    return np.asarray(session.run(None, dict(zip(names, values)))[0], dtype=np.float32)


def _build_dataset(cfg: dict[str, Any]) -> tuple[TripletExtractionDataset, Path]:
    manifest_root = Path(cfg["paths"]["manifest_root"])
    manifest_path = manifest_root / str(cfg.get("split", "test")) / str(cfg["manifest"])
    if not manifest_path.exists():
        raise FileNotFoundError(f"golden manifest not found: {manifest_path}")
    cache_dir = cfg.get("cabin_noise_cache_dir", "data/cache/cavemove_16k_mc")
    cache_dir = resolve_path(cache_dir)
    dataset = TripletExtractionDataset(
        manifest_path,
        sample_rate=int(cfg.get("sample_rate", 16000)),
        segment_seconds=float(cfg.get("segment_seconds", 3.0)),
        training=False,
        cabin_noise=bool(cfg.get("cabin_noise", False)),
        num_mics=int(cfg.get("num_mics", 4)),
        cabin_noise_channels=cfg.get("cabin_noise_channels"),
        cabin_noise_snr_db=cfg.get("cabin_noise_snr_db"),
        cabin_noise_cache_size=int(cfg.get("cabin_noise_cache_size", 32)),
        cabin_noise_cache_dir=cache_dir,
    )
    if not dataset:
        raise ValueError(f"golden manifest contains no rows: {manifest_path}")
    return dataset, manifest_path


def _dataset_input(
    cfg: dict[str, Any],
    dataset: TripletExtractionDataset,
    row_index: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    index = int(row_index) % len(dataset)
    item = dataset[index]
    mixture = item["mixture"].float().unsqueeze(0)
    enrollment = item["enrollment"].float().unsqueeze(0)
    multichannel = item.get("multichannel")
    if multichannel is None:
        multichannel = mixture.repeat(int(cfg.get("num_mics", 4)), 1)
    multichannel = multichannel.float().unsqueeze(0)
    playback = item.get("playback_ref")
    if playback is None:
        playback = torch.zeros(mixture.shape[-1], dtype=torch.float32)
    playback = playback.float().unsqueeze(0)
    row = dataset.rows[index]
    return {
        "mixture": mixture,
        "enrollment": enrollment,
        "multichannel": multichannel,
        "playback_ref": playback,
    }, {
        "index": index,
        "id": row.get("id", index),
        "condition": row.get("condition"),
        "cabin_noise_snr_db": row.get("cabin_noise_snr_db"),
        "speaker_id": row.get("speaker_id"),
    }


def _pipeline_handoffs(
    pipeline: Any,
    inputs: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, tuple[torch.Tensor, ...]]]:
    """Return PyTorch references and the exact tensors fed to each ONNX graph."""

    mixture = inputs["mixture"].to(device)
    enrollment = inputs["enrollment"].to(device)
    multichannel = inputs["multichannel"].to(device)
    playback = inputs["playback_ref"].to(device)
    with torch.inference_mode():
        extracted, _, _, _ = pipeline.stage1(
            mixture,
            enrollment,
            multichannel=multichannel,
            playback_ref=playback,
        )
        downstream = normalize_extracted_waveform(extracted)
        presence_logits = pipeline.presence(extracted)
        kws_input = pipeline._prepare_kws_input(downstream)
        kws_logits = pipeline.kws(kws_input)
        enrollment_embedding = pipeline.stage1.speaker_embedding(enrollment)
        extracted_embedding = pipeline.stage1.speaker_embedding(extracted)
        if pipeline.target_kws is None or enrollment_embedding is None or extracted_embedding is None:
            raise RuntimeError("retained runtime must provide target-KWS and speaker embeddings")
        target_kws_logits = pipeline.target_kws(
            kws_input,
            enrollment_embedding,
            extracted_embedding,
        )
        if pipeline.compact_speaker_verifier is None:
            raise RuntimeError("retained runtime must provide the compact speaker verifier")
        compact_enrollment = pipeline._prepare_compact_speaker_input(enrollment)
        compact_extracted = pipeline._prepare_compact_speaker_input(downstream)
        compact_logit = pipeline.compact_speaker_verifier(
            compact_enrollment,
            compact_extracted,
        )[0]
        intent_input = pipeline._prepare_intent_input(downstream)
        intent_logits = pipeline.intent(intent_input)

    references = {
        "stage1": extracted,
        "kws": kws_logits,
        "presence": presence_logits,
        "target_kws": target_kws_logits,
        "compact_verifier": compact_logit,
        "intent": intent_logits,
        "speaker_embedding": torch.cat((enrollment_embedding, extracted_embedding), dim=0),
    }
    feeds = {
        "stage1": tuple(inputs[name] for name in ("multichannel", "enrollment", "playback_ref")),
        "kws": (kws_input,),
        "presence": (extracted,),
        "target_kws": (kws_input, enrollment_embedding, extracted_embedding),
        "compact_verifier": (compact_enrollment, compact_extracted),
        "intent": (intent_input,),
        "speaker_embedding": (enrollment, extracted),
    }
    return references, feeds


def _compare_sample(
    pipeline: Any,
    sessions: dict[str, Any],
    inputs: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, dict[str, float | bool | list[int]]]:
    references, feeds = _pipeline_handoffs(pipeline, inputs, device)
    outputs: dict[str, dict[str, float | bool | list[int]]] = {}
    components = list(CORE_COMPONENTS)
    if "speaker_embedding" in sessions:
        speaker_actual = _run(
            sessions["speaker_embedding"],
            tuple(value.detach().cpu().numpy() for value in feeds["speaker_embedding"]),
        )
        outputs["speaker_embedding"] = compare(
            references["speaker_embedding"].detach().cpu().numpy(),
            speaker_actual,
        )
        # Exercise the exported embedding handoff for the target-KWS graph;
        # the comparison remains against the PyTorch reference logits.
        feeds["target_kws"] = (
            feeds["target_kws"][0],
            torch.from_numpy(speaker_actual[:1]).to(device),
            torch.from_numpy(speaker_actual[1:]).to(device),
        )
        components.insert(1, "speaker_embedding")
    for component in components:
        reference = references[component].detach().cpu().numpy()
        actual = _run(
            sessions[component],
            tuple(value.detach().cpu().numpy() for value in feeds[component]),
        )
        outputs[component] = compare(reference, actual)
    return outputs


def _aggregate_component(
    rows: list[dict[str, float | bool | list[int]]],
) -> dict[str, float | bool | list[int]]:
    first = rows[0]
    return {
        "passed": bool(all(bool(row["passed"]) for row in rows)),
        "sample_count": len(rows),
        "shape_match": bool(all(bool(row["shape_match"]) for row in rows)),
        "reference_shape": first["reference_shape"],
        "actual_shape": first["actual_shape"],
        "max_abs_error": float(max(float(row["max_abs_error"]) for row in rows)),
        "max_relative_error": float(max(float(row["max_relative_error"]) for row in rows)),
        "relative_l2_error": float(max(float(row["relative_l2_error"]) for row in rows)),
        "relative_l2_error_mean": float(
            np.mean([float(row["relative_l2_error"]) for row in rows])
        ),
        "cosine_similarity": float(min(float(row["cosine_similarity"]) for row in rows)),
    }


def run(
    *,
    config_path: Path,
    paths_path: Path,
    onnx_manifest: Path,
    row_index: int,
    num_samples: int = 1,
    device_name: str | None = None,
) -> dict[str, Any]:
    cfg = load_stage_config(config_path, paths_path)
    device = torch.device(device_name) if device_name else get_device()
    pipeline = load_pipeline(cfg, cfg["paths"], device)
    dataset, input_manifest = _build_dataset(cfg)
    sessions = _session_map(onnx_manifest)
    count = max(1, int(num_samples))
    samples: list[dict[str, Any]] = []
    active_components = list(CORE_COMPONENTS)
    if "speaker_embedding" in sessions:
        active_components.insert(1, "speaker_embedding")
    component_rows: dict[str, list[dict[str, float | bool | list[int]]]] = {
        component: [] for component in active_components
    }
    for offset in range(count):
        inputs, row_metadata = _dataset_input(cfg, dataset, row_index + offset)
        sample_components = _compare_sample(pipeline, sessions, inputs, device)
        for component in active_components:
            component_rows[component].append(sample_components[component])
        samples.append({"input": row_metadata, "components": sample_components})
    outputs = {
        component: _aggregate_component(rows) for component, rows in component_rows.items()
    }
    speaker_note = (
        "The exported speaker-embedding graph is included and its output is fed into "
        "the target-KWS parity handoff."
        if "speaker_embedding" in sessions
        else "Speaker embeddings remain the retained PyTorch reference because the "
        "ONNX manifest does not export that encoder."
    )
    return {
        "benchmark": "onnx_pipeline_golden_parity",
        "status": "ok" if all(bool(row["passed"]) for row in outputs.values()) else "parity_failed",
        "config": str(config_path.resolve()),
        "onnx_manifest": str(onnx_manifest.resolve()),
        "device": str(device),
        "input_manifest": str(input_manifest.resolve()),
        "start_row_index": int(row_index),
        "sample_count": count,
        "inputs": samples,
        "components": outputs,
        "limitations": (
            "These are deterministic CAVEMOVE preprocessing/parity samples. "
            "It is not an accuracy benchmark, target-SoC benchmark, quantization result, "
            f"or real-driver evidence. {speaker_note}"
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
    parser.add_argument(
        "--onnx-manifest",
        type=Path,
        default=REPO_ROOT / "output/onnx/retained_ckptfix3_2026-10-14/manifest.json",
    )
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="number of consecutive deterministic test rows to compare",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="reference PyTorch device; defaults to the configured available device",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "logs/onnx_pipeline_golden_parity_2026-10-14.json",
    )
    args = parser.parse_args()
    result = run(
        config_path=args.config.resolve(),
        paths_path=args.paths.resolve(),
        onnx_manifest=args.onnx_manifest.resolve(),
        row_index=int(args.row_index),
        num_samples=int(args.num_samples),
        device_name=args.device,
    )
    write_metrics(args.output.resolve(), result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())

