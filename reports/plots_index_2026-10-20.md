# Project plot index

The `multi-speaker-eval` branch and its remote counterpart contain the
paper/research figures below. There are currently 90 tracked plot/PDF/SVG
artifacts. Most are under `reports/figures/`; the older paper-ready set is
under `reports/paper_figures_2026-10-05/`.

## Stage 1 — target-speaker extraction

- `reports/figures/stage1_spex_fsc_passenger_adaptation_comparison_2026-10-16.*`
- `reports/figures/stage1_spex_fsc_passenger_cochannel_ft1_training_2026-10-16.*`
- `reports/figures/stage1_spex_fsc_passenger_unfrozen_vs_frozen_2026-10-16.*`
- `reports/figures/stage1_spex_passenger_adaptation_all_candidates_2026-10-16.*`
- `reports/figures/stage1_spex_passenger_large_vs_mixed_training_2026-10-16.*`
- `reports/paper_figures_2026-10-05/fig01_stage1_extraction.*`
- `reports/paper_figures_2026-10-05/fig02_stage1_frontend_comparison.*`
- `reports/paper_figures_2026-10-05/fig03_stage1_rms_sweep.*`
- `reports/paper_figures_2026-10-05/fig11_stage1_domain_transfer.*`

## Stage 2 — target-presence gate

- `reports/figures/stage2_presence_training_comparison_2026-10-20.*` (new
  comparison generated from the logged training histories)
- `reports/paper_figures_2026-10-05/fig04_stage2_presence_operating_points.*`
- `reports/paper_figures_2026-10-05/fig12_stage2_operating_points.*`

The current dual-domain FT1 log stores only its final summary, so the new
training-history figure compares the Stage-2 runs that retain epoch histories
(`gsc_aishell_cached`, `gsc_aishell_richer`, and `aishell_target`). Their
source JSON histories remain the reproducibility source of truth.

## Stage 3 — wake-word / target-conditioned KWS

- `reports/figures/stage3_target_kws_negative_calibrated_comparison_2026-10-16.*`
- `reports/paper_figures_2026-10-05/fig05_stage3_kws_metrics.*`
- `reports/paper_figures_2026-10-05/fig13_stage3_stream_threshold.*`
- `reports/paper_figures_2026-10-05/fig14a_stage3_proxy_recall.*`
- `reports/paper_figures_2026-10-05/fig14b_stage3_proxy_false_events.*`

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
- `reports/paper_figures_2026-10-05/fig06_stage4_intent.*`
- `reports/paper_figures_2026-10-05/fig15_stage4_domain_accuracy.*`

## Full pipeline, ablations, latency, and benchmark comparisons

- `reports/figures/multispeaker_*.*`
- `reports/figures/postcommand_*.*`
- `reports/figures/runtime_aligned_*.*`
- `reports/figures/strongest_*.*`
- `reports/figures/speech_commands_*.*`
- `reports/figures/slurp_*.*`
- `reports/paper_figures_2026-10-05/fig07_full_pipeline_proxy.*`
- `reports/paper_figures_2026-10-05/fig08_latency.*`
- `reports/paper_figures_2026-10-05/fig09_parameter_counts.*`
- `reports/paper_figures_2026-10-05/fig10_aishell5_false_events.*`
- `reports/paper_figures_2026-10-05/overall_system_architecture.svg`

The PNG and PDF variants are both tracked where available. The remote branch
is synchronized with the local branch at the commit that adds this index.
