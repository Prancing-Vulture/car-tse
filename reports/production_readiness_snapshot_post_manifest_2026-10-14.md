# Production-readiness snapshot

Generated: `2026-10-13T19:15:16.772245+00:00`

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

- Git revision: `1c165b7ba59ed34de6efa6cd899f0fdd718b896d`
- Worktree clean: **False**
- Changed files: **50**
- Runtime config SHA-256: `0a379583636894e475904164029e0265b377d316bc285febde09e26e5b164d69`
- Evaluated-state manifest: `D:\Prawin_New\tse\car-tse-main\logs\release_manifest_current.json` (present=True)

Checkpoint SHA-256 values are stored in the machine-readable JSON snapshot. A production sign-off must archive this revision, configuration hash, and checkpoint manifest together.

## Real-driver audit

- Session count: **0**
- Drivers: `none`
- Validator valid: **False**
- Errors: 1

The evaluator must not be described as a real-driver benchmark until this validator passes on consented tuning and held-out test sessions.

## Reproducibility artifacts

- Runtime config: `D:\Prawin_New\tse\car-tse-main\configs\eval_end_to_end_cavemove_target_kws_smart_speaker_compact_cabin_smart_thr030_2026-10-12.yaml`
- Target-absent diagnostic: `D:\Prawin_New\tse\car-tse-main\logs\aishell5_target_absent_compact_cabin_smart_thr030_hop025_hits4_all18_2026-10-13.json`
- Full-pipeline latency: `D:\Prawin_New\tse\car-tse-main\logs\full_pipeline_latency_compact_cabin_smart_cpu_2026-10-12.json`
- GPU latency detail: `D:\Prawin_New\tse\car-tse-main\logs\full_pipeline_latency_compact_cabin_smart_gpu_2026-10-12.json`
- CPU latency detail: `D:\Prawin_New\tse\car-tse-main\logs\component_latency_cpu_fp32_2026-10-12.json`
- Regression artifacts: `D:\Prawin_New\tse\car-tse-main\logs\stage1_dual_domain_regression_full_2026-10-10.json`; `D:\Prawin_New\tse\car-tse-main\logs\stage1_dual_domain_regression_ft2_2026-10-11.json`

Decision: Do not label production-ready: the minimum speaker-disjoint pilot, matching capture hardware, and/or a frozen held-out test are missing.
