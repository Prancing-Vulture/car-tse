"""Small optional W&B adapter for reproducible evaluation runs."""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any


def log_evaluation(
    *,
    project: str,
    run_name: str,
    job_type: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    output_path: Path,
    mode: str | None = None,
    directory: Path | None = None,
    confusion_matrix: list[list[int]] | None = None,
    class_names: list[str] | None = None,
    scene_rows: list[dict[str, Any]] | None = None,
    window_rows: list[dict[str, Any]] | None = None,
    curves: dict[str, list[dict[str, Any]]] | None = None,
    distributions: dict[str, list[float]] | None = None,
    report_artifact: bool = False,
) -> None:
    """Log aggregate and row-level evaluation results to W&B.

    If no API key is configured, use W&B's local offline mode so the run can
    be synced later with ``wandb sync``.
    """

    import wandb

    selected_mode = mode or ("online" if os.environ.get("WANDB_API_KEY") else "offline")
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=project,
        name=run_name,
        job_type=job_type,
        config=config,
        mode=selected_mode,
        dir=str(directory) if directory is not None else None,
    )
    scalar_metrics = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float, bool)) and value is not None
    }
    run.log(scalar_metrics)
    run.summary["report_path"] = output_path.name

    for distribution_name, values in (distributions or {}).items():
        if values:
            run.log({f"distribution/{distribution_name}": wandb.Histogram(values)})

    for curve_name, points in (curves or {}).items():
        if not points:
            continue
        x_key = str(points[0].get("x_key", "x"))
        y_key = str(points[0].get("y_key", "y"))
        table = wandb.Table(columns=[x_key, y_key])
        for point in points:
            table.add_data(float(point[x_key]), float(point[y_key]))
        run.log({
            f"curve_data/{curve_name}": table,
            f"curve/{curve_name}": wandb.plot.line(table, x_key, y_key, title=curve_name.replace("_", " ").title()),
        })

    if confusion_matrix is not None and class_names is not None:
        table = wandb.Table(columns=["true_class", "predicted_class", "count"])
        for true_index, row in enumerate(confusion_matrix):
            for predicted_index, count in enumerate(row):
                table.add_data(class_names[true_index], class_names[predicted_index], int(count))
        run.log({"confusion_matrix_counts": table})

    if scene_rows:
        table = wandb.Table(columns=["scene_index", "snr_db", "wake_success", "intent_correct", "false_wakes", "false_actions"])
        for row in scene_rows:
            table.add_data(
                int(row.get("scene_index", -1)),
                float(row.get("snr_db", 0.0)),
                bool(row.get("target_wake_detected", False)),
                bool(row.get("target_intent_correct", False)),
                int(row.get("false_wake_events", 0)),
                int(row.get("false_action_events", 0)),
            )
        run.log({"scene_outcomes": table})

    if window_rows:
        columns = sorted({key for row in window_rows for key in row})
        table = wandb.Table(columns=columns)
        for row in window_rows:
            table.add_data(*[
                json.dumps(row.get(key), sort_keys=True)
                if isinstance(row.get(key), (dict, list, tuple))
                else row.get(key)
                for key in columns
            ])
        run.log({"window_traces": table})

    if report_artifact and output_path.is_file():
        artifact = wandb.Artifact(
            name=f"{run_name}-report",
            type="evaluation-report",
            metadata={"job_type": job_type, "run_name": run_name},
        )
        artifact.add_file(str(output_path), name=output_path.name)
        run.log_artifact(artifact)

    run.finish()

