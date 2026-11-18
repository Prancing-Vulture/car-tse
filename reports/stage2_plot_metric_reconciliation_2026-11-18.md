# Stage-2 plot metric reconciliation

Date: 2026-11-18  
Status: figure/reporting correction; no model training or new evaluation run

## Issue

The paper figure `stage2_presence_training_comparison_2026-10-20` showed
validation F1 values as low as approximately 0.5--0.7 for some curves, while
the paper reported Stage-2 frame F1 above 0.90. The two numbers were not from
the same checkpoint or evaluation artifact.

## Cause

The old figure was a historical training-history comparison generated from
three separate legacy runs:

- `stage2_presence_gsc_aishell_cached_ft1`;
- `stage2_presence_gsc_aishell_richer_ft1`; and
- `stage2_presence_aishell_target_ft1`.

It plotted each run's `val.f1` at a fixed threshold of 0.50. The weakest
`AISHELL_TARGET` run was a separate target-domain diagnostic and was not the
promoted dual-domain Stage-2 checkpoint. The current dual-domain run does not
contain a per-epoch history; it stores a final summary in
`logs/stage2_frame_presence_dual_domain_nonwake_ft1/metrics_last.json`.

The earlier figure was therefore not mathematically wrong for those legacy
logs, but it was misleading when presented as the current Stage-2 training
curve.

## Correct current-checkpoint metrics

All values below come from the same current checkpoint:
`checkpoints/stage2_frame_presence_dual_domain_nonwake_ft1/best.pt`.

| View | Threshold | Accuracy | Precision | Recall | F1 | FAR | FRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frame | 0.50 | 98.84% | 86.71% | 94.51% | **90.44%** | 0.90% | 5.49% |
| Frame, best F1 | 0.65 | 98.98% | 90.50% | 92.10% | **91.29%** | 0.60% | 7.90% |
| Chunk | 0.50 | 99.79% | 99.18% | 100.00% | **99.59%** | 0.28% | 0.00% |
| Chunk, best F1 | 0.25 | 99.79% | 99.18% | 100.00% | **99.59%** | 0.28% | 0.00% |

The paper's standalone Stage-2 claim should use the **frame** row, not the
chunk row: frame F1 is 90.44% at the default standalone threshold and 91.29%
at the validation-selected best-F1 threshold. Chunk F1 is a coarser aggregation
and should be reported separately.

## Corrected artifacts

- `reports/figures/stage2_presence_current_dual_domain_validation_2026-11-18.{png,pdf}`
- `reports/paper_figures_latest_2026-10-20/stage2_presence_current_dual_domain_validation_2026-11-18.{png,pdf}`
- `reports/figures/stage2_presence_training_comparison_2026-10-20.{png,pdf}`
  (compatibility alias to the corrected figure)
- `reports/figures/stage2_presence_historical_training_comparison_2026-10-20.{png,pdf}`
  (preserved old diagnostic)

The reproducible renderer is
`scripts/reporting/plot_stage2_current_summary.py`. The correction only
changes figure selection/rendering and documentation; it does not change the
checkpoint or claim a new training result.
