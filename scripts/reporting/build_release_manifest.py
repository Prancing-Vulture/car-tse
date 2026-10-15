#!/usr/bin/env python
"""Build a content-addressed manifest for one evaluated runtime configuration.

The manifest is evidence for reproducibility, not a release sign-off.  A dirty
worktree, missing artifact, or missing dataset manifest is recorded explicitly
so a later report cannot silently turn an incomplete snapshot into a release.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from car_tse.config import load_stage_config  # noqa: E402


def resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def file_record(path: Path) -> dict[str, Any]:
    digest = None
    exists = path.is_file()
    if exists:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
    return {
        "path": str(path),
        "exists": exists,
        "bytes": int(path.stat().st_size) if exists else 0,
        "sha256": digest,
    }


def git_state(repo_root: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "worktree_clean": False, "changed_file_count": None}
    changed = [line for line in status.splitlines() if line.strip()]
    return {
        "revision": revision or None,
        "worktree_clean": not changed,
        "changed_file_count": len(changed),
    }


def package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_manifest(
    *,
    repo_root: Path,
    config_path: Path,
    paths_path: Path,
    onnx_manifest_path: Path | None = None,
    onnx_benchmark_path: Path | None = None,
    onnx_parity_path: Path | None = None,
    onnx_composed_parity_path: Path | None = None,
    onnx_streaming_benchmark_path: Path | None = None,
    onnx_cadence_benchmark_path: Path | None = None,
    speaker_shape_check_path: Path | None = None,
) -> dict[str, Any]:
    cfg = load_stage_config(config_path, paths_path)
    paths = cfg["paths"]
    checkpoint_root = Path(paths["checkpoint_root"])
    manifest_root = Path(paths["manifest_root"])

    artifacts: list[dict[str, Any]] = [
        {"role": "runtime_config", **file_record(config_path)},
        {"role": "paths_config", **file_record(paths_path)},
    ]

    stage1_config = cfg.get("stage1_config")
    if stage1_config:
        stage1_path = resolve(repo_root, str(stage1_config))
        artifacts.append({"role": "stage1_config", **file_record(stage1_path)})

    label_map = cfg.get("label_map")
    if label_map:
        label_value = Path(str(label_map))
        label_path = (
            label_value
            if label_value.is_absolute()
            else manifest_root / label_value
        )
        artifacts.append({"role": "intent_label_map", **file_record(label_path)})

    dataset_manifest = cfg.get("manifest")
    if dataset_manifest:
        split = str(cfg.get("split", "")).strip()
        manifest_value = Path(str(dataset_manifest))
        if split and not manifest_value.is_absolute():
            manifest_path = manifest_root / split / manifest_value
        else:
            manifest_path = resolve(manifest_root, str(dataset_manifest))
        artifacts.append({"role": "runtime_dataset_manifest", **file_record(manifest_path)})

    for component, relative in sorted(cfg.get("checkpoints", {}).items()):
        if relative:
            checkpoint_path = resolve(checkpoint_root, str(relative))
            artifacts.append(
                {
                    "role": "checkpoint",
                    "component": component,
                    **file_record(checkpoint_path),
                }
            )

    intent_encoder = cfg.get("intent_encoder_checkpoint")
    if intent_encoder:
        encoder_path = resolve(checkpoint_root, str(intent_encoder))
        artifacts.append({"role": "intent_encoder_checkpoint", **file_record(encoder_path)})

    if onnx_manifest_path is not None:
        onnx_manifest_path = onnx_manifest_path.resolve()
        artifacts.append({"role": "onnx_export_manifest", **file_record(onnx_manifest_path)})
        if onnx_manifest_path.is_file():
            export_payload = json.loads(onnx_manifest_path.read_text(encoding="utf-8"))
            for export in export_payload.get("components", []):
                output = export.get("output")
                if output:
                    artifacts.append(
                        {
                            "role": "onnx_model",
                            "component": export.get("component"),
                            **file_record(Path(str(output))),
                        }
                    )

    if onnx_benchmark_path is not None:
        onnx_benchmark_path = onnx_benchmark_path.resolve()
        artifacts.append(
            {"role": "onnx_latency_benchmark", **file_record(onnx_benchmark_path)}
        )

    if onnx_parity_path is not None:
        onnx_parity_path = onnx_parity_path.resolve()
        artifacts.append(
            {"role": "onnx_golden_parity", **file_record(onnx_parity_path)}
        )

    if onnx_composed_parity_path is not None:
        onnx_composed_parity_path = onnx_composed_parity_path.resolve()
        artifacts.append(
            {
                "role": "onnx_composed_pipeline_parity",
                **file_record(onnx_composed_parity_path),
            }
        )

    if onnx_streaming_benchmark_path is not None:
        onnx_streaming_benchmark_path = onnx_streaming_benchmark_path.resolve()
        artifacts.append(
            {
                "role": "onnx_streaming_runner_benchmark",
                **file_record(onnx_streaming_benchmark_path),
            }
        )

    if onnx_cadence_benchmark_path is not None:
        onnx_cadence_benchmark_path = onnx_cadence_benchmark_path.resolve()
        artifacts.append(
            {
                "role": "onnx_streaming_cadence_benchmark",
                **file_record(onnx_cadence_benchmark_path),
            }
        )

    if speaker_shape_check_path is not None:
        speaker_shape_check_path = speaker_shape_check_path.resolve()
        artifacts.append(
            {
                "role": "onnx_speaker_variable_length_check",
                **file_record(speaker_shape_check_path),
            }
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "runtime_config": str(config_path),
        "git": git_state(repo_root),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(
            ("torch", "torchaudio", "transformers", "soundfile", "onnx", "onnxruntime")
        ),
        "artifacts": artifacts,
        "all_artifacts_present": all(row["exists"] and row["sha256"] for row in artifacts),
        "release_signoff_ready": False,
        "release_signoff_note": (
            "This is an evaluated-state manifest only. Production sign-off still "
            "requires a clean tagged worktree, frozen split, matching capture "
            "hardware, and real-driver held-out evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "configs"
        / "eval_end_to_end_cavemove_target_kws_smart_speaker_compact_cabin_smart_thr030_2026-10-12.yaml",
    )
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument(
        "--onnx-manifest",
        type=Path,
        default=None,
        help="Optional ONNX export manifest whose models should be hashed into this state.",
    )
    parser.add_argument(
        "--onnx-benchmark",
        type=Path,
        default=None,
        help="Optional ONNX latency benchmark to hash into this state.",
    )
    parser.add_argument(
        "--onnx-parity",
        type=Path,
        default=None,
        help="Optional ONNX golden-parity report to hash into this state.",
    )
    parser.add_argument(
        "--onnx-composed-parity",
        type=Path,
        default=None,
        help="Optional composed ONNX/PyTorch pipeline parity report to hash into this state.",
    )
    parser.add_argument(
        "--onnx-streaming-benchmark",
        type=Path,
        default=None,
        help="Optional ONNX StreamingRunner benchmark to hash into this state.",
    )
    parser.add_argument(
        "--onnx-cadence-benchmark",
        type=Path,
        default=None,
        help="Optional alternate-hop ONNX StreamingRunner benchmark to hash into this state.",
    )
    parser.add_argument(
        "--speaker-shape-check",
        type=Path,
        default=None,
        help="Optional variable-length speaker-embedding report to hash into this state.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "logs" / "release_manifest_current.json",
    )
    args = parser.parse_args()
    manifest = build_manifest(
        repo_root=REPO_ROOT,
        config_path=args.config,
        paths_path=args.paths,
        onnx_manifest_path=args.onnx_manifest,
        onnx_benchmark_path=args.onnx_benchmark,
        onnx_parity_path=args.onnx_parity,
        onnx_composed_parity_path=args.onnx_composed_parity,
        onnx_streaming_benchmark_path=args.onnx_streaming_benchmark,
        onnx_cadence_benchmark_path=args.onnx_cadence_benchmark,
        speaker_shape_check_path=args.speaker_shape_check,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
