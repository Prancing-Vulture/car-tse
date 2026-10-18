#!/usr/bin/env python
"""Export small pipeline components to ONNX and verify numerical parity.

This is an embedded-deployment preflight, not a claim that the Python
pipeline is already production-portable.  The current TSE and wav2vec stages
may contain operators unsupported by a target vendor runtime; each component
is therefore exported and reported independently.  The script never replaces
the retained PyTorch checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from car_tse.config import load_stage_config  # noqa: E402
from car_tse.models.kws_model import KWSModel, TargetSpeakerKWSModel  # noqa: E402
from car_tse.models.intent_classifier import IntentClassifier  # noqa: E402
from car_tse.models.presence_head import (  # noqa: E402
    FramePresenceHead,
    PersonalActivityHead,
    PersonalFramePresenceHead,
    PresenceHead,
)
from car_tse.models.speaker_verifier import CompactSpeakerVerifier  # noqa: E402
from car_tse.models.stage1_system import Stage1System  # noqa: E402
from car_tse.models.wav2vec_intent import Wav2VecIntentClassifier  # noqa: E402
from car_tse.train.utils import load_checkpoint  # noqa: E402


COMPONENTS = (
    "stage1",
    "speaker_embedding",
    "kws",
    "presence",
    "target_kws",
    "compact_verifier",
    "intent",
)


class _CompactVerifierLogit(nn.Module):
    """Expose only the verifier logit for a stable ONNX output contract."""

    def __init__(self, model: CompactSpeakerVerifier) -> None:
        super().__init__()
        self.model = model

    def forward(self, enrollment: torch.Tensor, extracted: torch.Tensor) -> torch.Tensor:
        return self.model(enrollment, extracted)[0]


class _Stage1ExtractedSignal(nn.Module):
    """Expose only the extracted waveform from the four-output Stage 1 model."""

    def __init__(self, model: Stage1System) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        multichannel: torch.Tensor,
        enrollment: torch.Tensor,
        playback_ref: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            multichannel.mean(dim=1),
            enrollment,
            multichannel=multichannel,
            playback_ref=playback_ref,
        )[0]


class _SpeakerEmbeddingPair(nn.Module):
    """Expose enrollment and extracted-signal embeddings as one ONNX output."""

    def __init__(self, model: Stage1System) -> None:
        super().__init__()
        self.model = model

    def forward(self, enrollment: torch.Tensor, extracted: torch.Tensor) -> torch.Tensor:
        enrollment_embedding = self.model.speaker_embedding(enrollment)
        extracted_embedding = self.model.speaker_embedding(extracted)
        if enrollment_embedding is None or extracted_embedding is None:
            raise RuntimeError("the retained Stage 1 backend has no speaker embedding")
        return torch.cat((enrollment_embedding, extracted_embedding), dim=0)


def _checkpoint_target(model: nn.Module, component: str) -> nn.Module:
    """Return the underlying checkpoint-bearing module for wrapped exports.

    Stage 1 and the compact verifier expose a reduced ONNX output through a
    wrapper. Their training checkpoints contain the bare model keys, not the
    wrapper's ``model.*`` prefix. Loading into the wrapper with ``strict=False``
    would otherwise silently load zero trained parameters and make the export
    disagree with the Python runtime on real inputs.
    """

    if component in {"stage1", "speaker_embedding", "compact_verifier"} and hasattr(model, "model"):
        return getattr(model, "model")
    return model


def _resolve_checkpoint(cfg: dict[str, Any], key: str) -> Path:
    value = Path(str(cfg["checkpoints"][key]))
    if not value.is_absolute():
        value = Path(cfg["paths"]["checkpoint_root"]) / value
    return value.resolve()


def _load_component(
    cfg: dict[str, Any], component: str
) -> tuple[nn.Module, tuple[torch.Tensor, ...], list[str], list[str], Path]:
    torch.manual_seed(20261014)
    sample_rate = int(cfg.get("sample_rate", 16_000))
    checkpoint_key = component
    if component == "stage1":
        checkpoint_key = "spex"
    if component == "compact_verifier":
        checkpoint_key = "compact_speaker_verifier"
    if component == "speaker_embedding":
        checkpoint_key = "spex"

    if component in {"stage1", "speaker_embedding"}:
        stage1_config_path = Path(str(cfg["stage1_config"]))
        if not stage1_config_path.is_absolute():
            stage1_config_path = REPO_ROOT / stage1_config_path
        stage1_cfg = load_stage_config(stage1_config_path.resolve(), REPO_ROOT / "configs" / "paths.yaml")
        pretrained = Path(str(stage1_cfg["pretrained_checkpoint"]))
        if not pretrained.is_absolute():
            pretrained = REPO_ROOT / pretrained
        stage1_model = Stage1System(
            spex_cfg=stage1_cfg["spex"],
            sample_rate=int(stage1_cfg.get("sample_rate", sample_rate)),
            use_cabin_frontend=bool(stage1_cfg.get("use_cabin_frontend", False)),
            num_mics=int(stage1_cfg.get("num_mics", 4)),
            frontend_cfg=stage1_cfg.get("frontend", {}),
            pretrained_dir=str(pretrained.resolve()),
            backend=str(stage1_cfg.get("spex_backend", "vendor_8k")),
        )
        if component == "stage1":
            model = _Stage1ExtractedSignal(stage1_model)
            samples = int(round(sample_rate * float(cfg.get("segment_seconds", 3.0))))
            inputs = (
                torch.randn(1, int(stage1_cfg.get("num_mics", 4)), samples, dtype=torch.float32),
                torch.randn(1, int(round(sample_rate * 1.5)), dtype=torch.float32),
                torch.randn(1, samples, dtype=torch.float32),
            )
            names = ["multichannel_audio", "enrollment_audio", "playback_reference"]
            outputs = ["extracted_audio"]
        else:
            model = _SpeakerEmbeddingPair(stage1_model)
            enrollment_samples = int(round(sample_rate * 1.5))
            extracted_samples = int(round(sample_rate * float(cfg.get("segment_seconds", 3.0))))
            inputs = (
                torch.randn(1, enrollment_samples, dtype=torch.float32),
                torch.randn(1, extracted_samples, dtype=torch.float32),
            )
            names = ["enrollment_audio", "extracted_audio"]
            outputs = ["speaker_embeddings"]
    elif component == "kws":
        model = KWSModel(
            sample_rate=sample_rate,
            n_mels=int(cfg.get("kws_n_mels", 40)),
            channels=cfg.get("kws_channels", [32, 64, 128]),
            num_classes=int(cfg.get("kws_num_classes", 2)),
            exportable_frontend=True,
        )
        inputs = (torch.randn(1, int(round(sample_rate * float(cfg.get("kws_input_seconds", 1.0)))), dtype=torch.float32),)
        names = ["audio"]
        outputs = ["logits"]
    elif component == "presence":
        presence_type = str(cfg.get("presence_type", "scalar")).lower()
        kwargs = {
            "sample_rate": int(cfg.get("presence_sample_rate", sample_rate)),
            "n_mels": int(cfg.get("presence_n_mels", 40)),
            "hidden": int(cfg.get("presence_hidden", 64)),
            "n_fft": int(cfg.get("presence_n_fft", 512)),
            "hop": int(cfg.get("presence_hop", 160)),
        }
        if presence_type == "frame":
            model = FramePresenceHead(**kwargs, exportable_frontend=True)
        elif presence_type == "personal_frame":
            model = PersonalFramePresenceHead(
                **kwargs,
                speaker_embedding_dim=int(cfg.get("presence_speaker_embedding_dim", 256)),
                exportable_frontend=True,
            )
        elif presence_type in {"personal_activity", "personal_activity_3way"}:
            model = PersonalActivityHead(
                **kwargs,
                speaker_embedding_dim=int(cfg.get("presence_speaker_embedding_dim", 256)),
                exportable_frontend=True,
            )
        elif presence_type == "scalar":
            model = PresenceHead(hidden=int(cfg.get("presence_hidden", 128)))
        else:
            raise ValueError(f"Unsupported presence_type for export: {presence_type}")
        samples = int(round(sample_rate * float(cfg.get("segment_seconds", 3.0))))
        inputs = (torch.randn(1, samples, dtype=torch.float32),)
        names = ["extracted_audio"]
        outputs = ["presence_logits"]
        if getattr(model, "requires_enrollment", False):
            embedding_dim = int(cfg.get("presence_speaker_embedding_dim", 256))
            inputs = (inputs[0], torch.randn(1, embedding_dim, dtype=torch.float32))
            names.append("enrollment_embedding")
    elif component == "target_kws":
        embedding_dim = int(cfg.get("target_kws_speaker_embedding_dim", 256))
        model = TargetSpeakerKWSModel(
            sample_rate=sample_rate,
            n_mels=int(cfg.get("target_kws_n_mels", 40)),
            channels=cfg.get("target_kws_channels", [32, 64, 128]),
            num_classes=int(cfg.get("kws_num_classes", 2)),
            speaker_embedding_dim=embedding_dim,
            hidden=int(cfg.get("target_kws_hidden", 256)),
            exportable_frontend=True,
        )
        samples = int(round(sample_rate * float(cfg.get("kws_input_seconds", 1.0))))
        inputs = (
            torch.randn(1, samples, dtype=torch.float32),
            torch.randn(1, embedding_dim, dtype=torch.float32),
            torch.randn(1, embedding_dim, dtype=torch.float32),
        )
        names = ["audio", "enrollment_embedding", "extracted_embedding"]
        outputs = ["logits"]
    elif component == "compact_verifier":
        embedding_dim = int(cfg.get("compact_speaker_verifier_embedding_dim", 128))
        model = _CompactVerifierLogit(
            CompactSpeakerVerifier(
                embedding_dim=embedding_dim,
                hidden=int(cfg.get("compact_speaker_verifier_hidden", 128)),
                sample_rate=int(cfg.get("compact_speaker_verifier_sample_rate", sample_rate)),
                exportable_frontend=True,
            )
        )
        samples = int(
            round(
                int(cfg.get("compact_speaker_verifier_sample_rate", sample_rate))
                * float(cfg.get("compact_speaker_verifier_input_seconds", 1.0))
            )
        )
        inputs = (
            torch.randn(1, samples, dtype=torch.float32),
            torch.randn(1, samples, dtype=torch.float32),
        )
        names = ["enrollment_audio", "extracted_audio"]
        outputs = ["verification_logit"]
    elif component == "intent":
        intent_type = str(cfg.get("intent_type", "wav2vec")).lower()
        if intent_type == "wav2vec":
            encoder_checkpoint = Path(str(cfg["intent_encoder_checkpoint"]))
            if not encoder_checkpoint.is_absolute():
                encoder_checkpoint = Path(cfg["paths"]["checkpoint_root"]) / encoder_checkpoint
            model = Wav2VecIntentClassifier(
                encoder_name=str(cfg.get("intent_encoder", "facebook/wav2vec2-base-960h")),
                encoder_checkpoint=encoder_checkpoint.resolve(),
                head_checkpoint=_resolve_checkpoint(cfg, "intent"),
                num_classes=int(cfg.get("intent_num_classes", 31)),
                input_sample_rate=sample_rate,
                input_seconds=float(cfg.get("intent_input_seconds", 2.0)),
            )
        elif intent_type == "crnn":
            model = IntentClassifier(
                sample_rate=sample_rate,
                n_mels=int(cfg.get("intent_n_mels", cfg.get("model", {}).get("n_mels", 80))),
                hidden=int(cfg.get("intent_hidden", cfg.get("model", {}).get("hidden", 256))),
                num_layers=int(cfg.get("intent_num_layers", cfg.get("model", {}).get("num_layers", 2))),
                num_classes=int(cfg.get("intent_num_classes", 31)),
                normalize_input=bool(cfg.get("intent_normalize_input", False)),
                exportable_frontend=True,
                bidirectional=bool(cfg.get("intent_bidirectional", True)),
            )
        else:
            raise ValueError(f"Unsupported intent_type for export: {intent_type}")
        samples = int(round(sample_rate * float(cfg.get("intent_input_seconds", 2.0))))
        inputs = (torch.randn(1, samples, dtype=torch.float32),)
        names = ["audio"]
        outputs = ["intent_logits"]
    else:  # pragma: no cover - guarded by argparse and tests
        raise ValueError(f"Unsupported component: {component}")

    checkpoint = _resolve_checkpoint(cfg, checkpoint_key)
    load_checkpoint(checkpoint, _checkpoint_target(model, component))
    model.eval().cpu()
    return model, inputs, names, outputs, checkpoint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _flatten_output(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, (tuple, list)) and value:
        return _flatten_output(value[0])
    raise TypeError(f"Expected tensor-like model output, got {type(value)!r}")


def _parity(
    model: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    session: Any,
    *,
    waveform_output: bool = False,
) -> dict[str, float | bool]:
    with torch.inference_mode():
        reference = _flatten_output(model(*inputs)).numpy()
    actual = session.run(
        None,
        {input_meta.name: value.numpy() for input_meta, value in zip(session.get_inputs(), inputs)},
    )[0]
    reference = np.asarray(reference, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    difference = np.abs(reference - actual)
    denominator = np.maximum(np.abs(reference), 1.0e-6)
    reference_flat = reference.reshape(-1)
    actual_flat = actual.reshape(-1)
    cosine = float(
        np.dot(reference_flat, actual_flat)
        / max(np.linalg.norm(reference_flat) * np.linalg.norm(actual_flat), 1.0e-12)
    )
    max_abs = float(difference.max()) if difference.size else 0.0
    max_relative = float((difference / denominator).max()) if difference.size else 0.0
    relative_l2 = float(
        np.linalg.norm(reference - actual)
        / max(np.linalg.norm(reference), 1.0e-12)
    )
    # Pointwise error is the right criterion for logits. For waveform outputs
    # from a large separator, kernels can differ slightly at individual
    # samples while preserving the signal to well below 0.1% in relative L2.
    # Keep that exception explicit and limited to the waveform-producing
    # separator instead of weakening checks for logits or gates.
    passed = bool(max_abs <= 1.0e-3 or relative_l2 <= 5.0e-4)
    if waveform_output:
        passed = bool(passed or (relative_l2 <= 1.0e-3 and cosine >= 0.99999))
    return {
        "passed": passed,
        "max_abs_error": max_abs,
        "max_relative_error": max_relative,
        "relative_l2_error": relative_l2,
        "cosine_similarity": cosine,
    }


def export_component(
    config: Path,
    paths: Path,
    component: str,
    output_dir: Path,
    opset: int,
    verify: bool,
) -> dict[str, Any]:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - covered by CLI smoke tests outside this env
        raise RuntimeError(
            "ONNX export requires optional packages 'onnx' and 'onnxruntime'. "
            "Install them with: python -m pip install onnx onnxruntime"
        ) from exc

    cfg = load_stage_config(config, paths)
    model, inputs, input_names, output_names, checkpoint = _load_component(cfg, component)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{component}.onnx"
    dynamic_axes = {name: {0: "batch"} for name in input_names + output_names}
    if component == "speaker_embedding":
        # Enrollment and extracted windows are not guaranteed to have the
        # same duration at runtime; the speaker encoder pools over time.
        # These inputs may intentionally have different lengths.  Distinct
        # symbolic names prevent ONNX Runtime from treating them as one
        # shared dimension and warning when the 1.5 s enrollment and 3 s
        # extracted windows are fed in the same call.
        dynamic_axes["enrollment_audio"][1] = "enrollment_time"
        dynamic_axes["extracted_audio"][1] = "extracted_time"
        dynamic_axes["speaker_embeddings"] = {0: "pair", 1: "embedding"}
    torch.onnx.export(
        model,
        inputs,
        str(output_path),
        opset_version=int(opset),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    parity: dict[str, Any] | None = None
    if verify:
        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        parity = _parity(model, inputs, session, waveform_output=component == "stage1")
    result: dict[str, Any] = {
        "component": component,
        "status": "ok" if parity is None or parity["passed"] else "parity_failed",
        "config": str(config.resolve()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "output": str(output_path.resolve()),
        "output_sha256": _sha256(output_path),
        "opset": int(opset),
        "input_names": input_names,
        "input_shapes": [list(value.shape) for value in inputs],
        "output_names": output_names,
        "onnx_ir_version": int(onnx_model.ir_version),
        "onnxruntime_providers": ort.get_available_providers(),
        "dynamic_axes": dynamic_axes,
        "parity": parity,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=(*COMPONENTS, "all"), required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/eval_end_to_end_cavemove_target_kws_smart_speaker_compact_cabin_smart_thr030_2026-10-12.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.yaml")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output" / "onnx")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()
    components = COMPONENTS if args.component == "all" else (args.component,)
    results = [
        export_component(
            args.config.resolve(),
            args.paths.resolve(),
            component,
            args.output_dir.resolve(),
            int(args.opset),
            verify=not args.no_verify,
        )
        for component in components
    ]
    result: dict[str, Any] = results[0] if len(results) == 1 else {
        "status": "ok" if all(row["status"] == "ok" for row in results) else "parity_failed",
        "components": results,
        "config": str(args.config.resolve()),
        "opset": int(args.opset),
    }
    manifest_name = "manifest.json" if args.component == "all" else f"{args.component}.json"
    manifest_path = args.output_dir.resolve() / manifest_name
    manifest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
