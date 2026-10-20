# Project plot index

The `multi-speaker-eval` branch and its remote counterpart contain the complete
tracked figure archive under `reports/figures/` plus the compact paper-ready
selection under `reports/paper_figures_latest_2026-10-20/`. The latter is the
recommended starting point for the paper's main figures and appendix. The
archive also retains historical candidates so the experiments remain
auditable; historical figures must not be presented as the current FT6 result.

## Current best-system figures

- `reports/figures/ft6_current_system_e2e_2026-10-20.*` — clean target-only,
  corrected sequential-passenger, and corrected simultaneous-overlap proxies.
- `reports/figures/ft6_corrected_multispeaker_1000_2026-10-20.*` — primary
  current FT6 1,000-scene passenger/overlap comparison.
- `reports/figures/ft6_resource_summary_2026-10-20.*` — current FT6 parameter,
  RTX 4080 FP16 latency, compute, and composite-latency summary.
- `reports/paper_figures_latest_2026-10-20/README.md` — curated pack with the
  exact source logs and evidence caveats.

## Stage 1 — target-speaker extraction

- `reports/figures/stage1_spex_fsc_passenger_adaptation_comparison_2026-10-16.*`
- `reports/figures/stage1_spex_fsc_passenger_cochannel_ft1_training_2026-10-16.*`
- `reports/figures/stage1_spex_fsc_passenger_unfrozen_vs_frozen_2026-10-16.*`
- `reports/figures/stage1_spex_passenger_adaptation_all_candidates_2026-10-16.*`
- `reports/figures/stage1_spex_passenger_large_vs_mixed_training_2026-10-16.*`

## Stage 2 — target-presence gate

- `reports/figures/stage2_presence_training_comparison_2026-10-20.*` (new
  comparison generated from the logged training histories)

The current dual-domain FT1 log stores only its final summary, so the new
training-history figure compares the Stage-2 runs that retain epoch histories
(`gsc_aishell_cached`, `gsc_aishell_richer`, and `aishell_target`). Their
source JSON histories remain the reproducibility source of truth.

## Stage 3 — wake-word / target-conditioned KWS

- `reports/figures/stage3_target_kws_negative_calibrated_comparison_2026-10-16.*`

## Stage 4 — intent classification

- `reports/figures/stage4_intent_frozen_tse_downstream_comparison_2026-10-16.*`
- `reports/figures/stage4_intent_hierarchical_jitter_ft7_2026-10-17.*`
- `reports/figures/stage4_intent_joint_tse_overlap_training_2026-10-16.*`
- `reports/figures/stage4_intent_large_passenger_joint_tse_training_2026-10-16.*`
- `reports/figures/stage4_intent_mixed_overlapencoder_domain_curves_2026-10-16.*`
- `reports/figures/stage4_intent_mixed_overlapencoder_training_2026-10-16.*`
- `reports/figures/stage4_intent_overlap_ft1_ft2_training_curves_2026-10-16.*`
- `reports/figures/stage4_intent_runtime_aligned_*.*`
- `reports/figures/stage4_intent_runtime_jitter_*.*`

## Full pipeline, ablations, latency, and benchmark comparisons

- `reports/figures/multispeaker_*.*`
- `reports/figures/postcommand_*.*`
- `reports/figures/runtime_aligned_*.*`
- `reports/figures/strongest_*.*`
- `reports/figures/speech_commands_*.*`
- `reports/figures/slurp_*.*`
- `reports/figures/ft6_current_system_e2e_2026-10-20.*`
- `reports/figures/ft6_corrected_multispeaker_1000_2026-10-20.*`
- `reports/figures/ft6_resource_summary_2026-10-20.*`
- `fig1_system.png`

The PNG and PDF variants are both tracked where available. The current
end-to-end figures use the corrected FT6 1,000-scene logs, not the older
candidate comparison plots. The plots remain synthetic/clean proxy evidence;
they do not establish same-driver real-cabin reliability or Jetson latency.
