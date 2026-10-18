# Production-readiness snapshot

Generated: `2026-10-10T23:24:05.196382+00:00`

Status: **research_engineering_preview**

This report is generated from the selected runtime configuration and the real-driver session validator. Proxy data is listed as diagnostic evidence only.

## Gate status

| Gate | Status |
|---|---|
| same_driver_continuity | **blocked** |
| passenger_rejection | **blocked** |
| study_size | **blocked** |
| capture_hardware_match | **blocked** |
| reproducible_release_state | **blocked** |
| frozen_split | **blocked** |
| threshold_freeze | **blocked** |
| no_command_safety | evidence_present |
| latency | evidence_present |
| resource_budget | evidence_present |
| regression | evidence_present |

## Release identity

- Git revision: `a2e62deae868b31826dfd111f3bb5ba769e2656c`
- Worktree clean: **False**
- Changed files: **176**
- Runtime config SHA-256: `9e2bcd4afac645a3d961103e86028880378c53f4ecaa6362853419abf5610f67`

Checkpoint SHA-256 values are stored in the machine-readable JSON snapshot. A production sign-off must archive this revision, configuration hash, and checkpoint manifest together.

## Real-driver audit

- Session count: **0**
- Drivers: `none`
- Validator valid: **False**
- Errors: 1

The evaluator must not be described as a real-driver benchmark until this validator passes on consented tuning and held-out test sessions.

## Reproducibility artifacts

- Runtime config: `D:\Prawin_New\tse\car-tse-main\configs\eval_end_to_end_cavemove_target_kws_dual_domain_stage2_hardneg_aishell5_neg_safety_splitgate_candidate.yaml`
- Target-absent diagnostic: `D:\Prawin_New\tse\car-tse-main\logs\aishell5_continuous_targetkws_hardneg_aishell5_ft1_p015_sg024_kws050_smooth22_eval2_all18_dedup_2026-10-11.json`
- Full-pipeline latency: `D:\Prawin_New\tse\car-tse-main\logs\full_pipeline_latency_2026-10-11.json`
- GPU latency detail: `D:\Prawin_New\tse\car-tse-main\logs\full_pipeline_latency_gpu_2026-10-11.json`
- CPU latency detail: `D:\Prawin_New\tse\car-tse-main\logs\full_pipeline_latency_cpu_2026-10-11.json`
- Regression artifacts: `D:\Prawin_New\tse\car-tse-main\logs\stage1_dual_domain_regression_full_2026-10-10.json`; `D:\Prawin_New\tse\car-tse-main\logs\stage1_dual_domain_regression_ft2_2026-10-11.json`

Decision: Do not label production-ready: the minimum speaker-disjoint pilot, matching capture hardware, and/or a frozen held-out test are missing.
