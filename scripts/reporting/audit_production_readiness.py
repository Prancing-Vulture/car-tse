#!/usr/bin/env python
"""Create a machine-checkable production-readiness snapshot.

The snapshot deliberately separates measured engineering artifacts from the
evidence needed for a production claim.  In particular, synthetic
cross-corpus proxy results never satisfy the same-driver or held-out
real-driver gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from car_tse.config import REPO_ROOT, load_stage_config
from validate_real_driver_sessions import validate


def resolve(repo_root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else repo_root / value


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def file_record(path: Path) -> dict[str, Any]:
    digest = None
    if path.exists() and path.is_file():
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": int(path.stat().st_size) if path.exists() and path.is_file() else 0,
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
    changed_file_count = len([line for line in status.splitlines() if line.strip()])
    return {
        "revision": revision or None,
        "worktree_clean": changed_file_count == 0,
        "changed_file_count": changed_file_count,
    }


def build_snapshot(
    *,
    repo_root: Path,
    config_path: Path,
    paths_path: Path,
    sessions_root: Path,
    tuning_drivers: set[str],
    test_drivers: set[str],
) -> dict[str, Any]:
    cfg = load_stage_config(config_path, paths_path)
    paths = cfg["paths"]
    checkpoint_root = Path(paths["checkpoint_root"])
    manifest_root = Path(paths["manifest_root"])

    checkpoints: list[dict[str, Any]] = []
    for key, relative in cfg.get("checkpoints", {}).items():
        if relative:
            checkpoints.append(file_record(resolve(checkpoint_root, relative)))
            checkpoints[-1]["component"] = key
    encoder_relative = cfg.get("intent_encoder_checkpoint")
    if encoder_relative:
        checkpoints.append(file_record(resolve(checkpoint_root, encoder_relative)))
        checkpoints[-1]["component"] = "intent_encoder"

    validation = validate(sessions_root, tuning_drivers, test_drivers)
    source_state = git_state(repo_root)
    selected_short_log = repo_root / "logs" / "aishell5_continuous_targetkws_hardneg_aishell5_ft1_p015_sg024_kws050_smooth22_1200s_2026-10-11.json"
    selected_long_log = repo_root / "logs" / "aishell5_continuous_targetkws_hardneg_aishell5_ft1_p015_sg024_kws050_smooth22_eval2_all18_2026-10-10.json"
    selected_long_log_dedup = repo_root / "logs" / "aishell5_continuous_targetkws_hardneg_aishell5_ft1_p015_sg024_kws050_smooth22_eval2_all18_dedup_2026-10-11.json"
    selected_log = (
        selected_long_log_dedup
        if selected_long_log_dedup.exists()
        else selected_long_log
        if selected_long_log.exists()
        else selected_short_log
    )
    selected_proxy_logs = [
        repo_root / "logs" / "continuous_full_pipeline_proxy_hardneg_aishell5_neg_splitgate_p015_wake024_cmd022_same_speaker_m10_2026-10-11.json",
        repo_root / "logs" / "continuous_full_pipeline_proxy_hardneg_aishell5_neg_splitgate_p015_wake024_cmd022_same_speaker_p10_2026-10-11.json",
    ]
    latency_log = repo_root / "logs" / "full_pipeline_latency_2026-10-11.json"
    latency_gpu_log = repo_root / "logs" / "full_pipeline_latency_gpu_2026-10-11.json"
    latency_cpu_log = repo_root / "logs" / "full_pipeline_latency_cpu_2026-10-11.json"
    regression_logs = [
        repo_root / "logs" / "stage1_dual_domain_regression_full_2026-10-10.json",
        repo_root / "logs" / "stage1_dual_domain_regression_ft2_2026-10-11.json",
    ]

    observed_drivers = set(validation["drivers"])
    expected_channel_count = int(cfg.get("num_mics", 0) or 0)
    observed_channel_counts = sorted({int(row.get("channel_count", 0)) for row in validation["sessions"]})
    capture_hardware_match = (
        validation["valid"]
        and expected_channel_count > 0
        and bool(validation["sessions"])
        and all(int(row.get("recording_sample_rate_hz", 0)) == 16000 for row in validation["sessions"])
        and all(int(row.get("channel_count", 0)) == expected_channel_count for row in validation["sessions"])
    )
    release_artifacts_complete = bool(source_state["revision"]) and all(
        row["exists"] and bool(row["sha256"])
        for row in checkpoints
    ) and config_path.exists() and paths_path.exists()
    split_ready = (
        validation["valid"]
        and bool(tuning_drivers)
        and bool(test_drivers)
        and not bool(tuning_drivers & test_drivers)
        and observed_drivers.issubset(tuning_drivers | test_drivers)
    )
    study_size_ready = (
        validation["valid"]
        and validation["session_count"] >= 3
        and len(observed_drivers) >= 3
        and bool(tuning_drivers)
        and bool(test_drivers)
    )
    critical = {
        "same_driver_continuity": study_size_ready,
        "passenger_rejection": study_size_ready,
        "study_size": study_size_ready,
        "capture_hardware_match": capture_hardware_match,
        "reproducible_release_state": release_artifacts_complete and bool(source_state["worktree_clean"]),
        "frozen_split": split_ready,
        "threshold_freeze": split_ready,
    }
    measured = {
        "no_command_safety": selected_log.exists(),
        "latency": latency_log.exists(),
        "resource_budget": latency_log.exists(),
        "regression": any(path.exists() for path in regression_logs),
    }
    production_ready = all(critical.values()) and all(measured.values())
    snapshot = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "production_ready" if production_ready else "research_engineering_preview",
        "production_ready": production_ready,
        "config": str(config_path),
        "checkpoints": checkpoints,
        "real_driver_validation": validation,
        "driver_split": {
            "tuning_drivers": sorted(tuning_drivers),
            "test_drivers": sorted(test_drivers),
            "disjoint": not bool(tuning_drivers & test_drivers),
            "all_observed_drivers_assigned": observed_drivers.issubset(tuning_drivers | test_drivers),
        },
        "study_design": {
            "minimum_sessions": 3,
            "minimum_drivers": 3,
            "required_nonempty_splits": True,
            "observed_sessions": validation["session_count"],
            "observed_drivers": len(observed_drivers),
            "expected_channel_count": expected_channel_count,
            "observed_channel_counts": observed_channel_counts,
        },
        "release_manifest": {
            "git": source_state,
            "config": file_record(config_path),
            "paths_config": file_record(paths_path),
            "checkpoints": checkpoints,
        },
        "critical_gates": {
            name: {"status": "pass" if value else "blocked"}
            for name, value in critical.items()
        },
        "measured_gates": {
            name: {"status": "evidence_present" if value else "missing"}
            for name, value in measured.items()
        },
        "proxy_evidence": {
            "target_absent_log": file_record(selected_log),
            "target_absent_short_log": file_record(selected_short_log),
            "target_absent_long_log": file_record(selected_long_log),
            "target_absent_dedup_log": file_record(selected_long_log_dedup),
            "same_speaker_proxy_logs": [file_record(path) for path in selected_proxy_logs],
            "latency_log": file_record(latency_log),
            "latency_gpu_log": file_record(latency_gpu_log),
            "latency_cpu_log": file_record(latency_cpu_log),
            "regression_logs": [file_record(path) for path in regression_logs],
        },
        "decision": (
            "Do not label production-ready: the minimum speaker-disjoint pilot, matching capture hardware, and/or a frozen held-out test are missing."
            if not production_ready
            else "All configured gates have evidence; review the held-out metrics and sign off before release."
        ),
    }
    return snapshot


def render_markdown(snapshot: dict[str, Any]) -> str:
    lines = [
        "# Production-readiness snapshot",
        "",
        f"Generated: `{snapshot['generated_at_utc']}`",
        "",
        f"Status: **{snapshot['status']}**",
        "",
        "This report is generated from the selected runtime configuration and the real-driver session validator. Proxy data is listed as diagnostic evidence only.",
        "",
        "## Gate status",
        "",
        "| Gate | Status |",
        "|---|---|",
    ]
    for name, row in snapshot["critical_gates"].items():
        lines.append(f"| {name} | **{row['status']}** |")
    for name, row in snapshot["measured_gates"].items():
        lines.append(f"| {name} | {row['status']} |")
    release = snapshot["release_manifest"]
    lines.extend(
        [
            "",
            "## Release identity",
            "",
            f"- Git revision: `{release['git']['revision'] or 'unavailable'}`",
            f"- Worktree clean: **{release['git']['worktree_clean']}**",
            f"- Changed files: **{release['git']['changed_file_count'] if release['git']['changed_file_count'] is not None else 'unavailable'}**",
            f"- Runtime config SHA-256: `{release['config']['sha256'] or 'unavailable'}`",
            "",
            "Checkpoint SHA-256 values are stored in the machine-readable JSON snapshot. A production sign-off must archive this revision, configuration hash, and checkpoint manifest together.",
        ]
    )
    validation = snapshot["real_driver_validation"]
    lines.extend(
        [
            "",
            "## Real-driver audit",
            "",
            f"- Session count: **{validation['session_count']}**",
            f"- Drivers: `{', '.join(validation['drivers']) or 'none'}`",
            f"- Validator valid: **{validation['valid']}**",
            f"- Errors: {len(validation['errors'])}",
            "",
            "The evaluator must not be described as a real-driver benchmark until this validator passes on consented tuning and held-out test sessions.",
            "",
            "## Reproducibility artifacts",
            "",
            f"- Runtime config: `{snapshot['config']}`",
            f"- Target-absent diagnostic: `{snapshot['proxy_evidence']['target_absent_log']['path']}`",
            f"- Full-pipeline latency: `{snapshot['proxy_evidence']['latency_log']['path']}`",
            f"- GPU latency detail: `{snapshot['proxy_evidence']['latency_gpu_log']['path']}`",
            f"- CPU latency detail: `{snapshot['proxy_evidence']['latency_cpu_log']['path']}`",
            "- Regression artifacts: " + "; ".join(f"`{row['path']}`" for row in snapshot["proxy_evidence"]["regression_logs"] if row["exists"]),
            "",
            f"Decision: {snapshot['decision']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "configs"
        / "eval_end_to_end_cavemove_target_kws_dual_domain_stage2_hardneg_aishell5_neg_safety_splitgate_candidate.yaml",
    )
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument(
        "--sessions-root",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "custom_incar" / "sessions",
    )
    parser.add_argument("--tuning-drivers", nargs="*", default=[])
    parser.add_argument("--test-drivers", nargs="*", default=[])
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "logs" / "production_readiness_snapshot_2026-10-11.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=REPO_ROOT / "reports" / "production_readiness_snapshot_2026-10-11.md",
    )
    args = parser.parse_args()
    snapshot = build_snapshot(
        repo_root=REPO_ROOT,
        config_path=args.config,
        paths_path=args.paths,
        sessions_root=args.sessions_root,
        tuning_drivers=set(args.tuning_drivers),
        test_drivers=set(args.test_drivers),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(snapshot), encoding="utf-8")
    print(json.dumps(snapshot, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
