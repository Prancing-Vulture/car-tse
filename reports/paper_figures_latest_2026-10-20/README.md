# Paper figure pack — 2026-10-20

This directory is the compact paper-ready selection from the complete tracked
figure archive in `reports/figures/`. It contains both PNG figures for quick
inspection and vector-friendly PDF versions where available.

## Current best system figures

- `ft6_current_system_e2e_2026-10-20.*` — clean target-only, corrected
  sequential passenger, and corrected simultaneous-overlap proxies.
- `ft6_corrected_multispeaker_1000_2026-10-20.*` — the primary current
  1,000-scene FT6 passenger/overlap comparison.
- `ft6_resource_summary_2026-10-20.*` — current FT6 parameter, RTX 4080
  FP16 latency, compute, and composite-latency summary.

The two end-to-end figures are generated directly from:

- `logs/clean_e2e_v2_knnvc_matched_postaction16_128_2026-10-16.json`
- `logs/multispeaker_passenger_ft6_p0_dynamic_full_1000_2026-10-17.json`
- `logs/multispeaker_overlap_ft6_p0_dynamic_full_1000_2026-10-17.json`

The FT6 runtime configuration is documented in
`reports/model_architecture_metrics_latency_sota_2026-10-17.md` and the
recommended soft-gate configuration is
`configs/eval_end_to_end_fscpassenger_identity_runtimealignedintent_ft6_jitter_prewakebuffer_gate015_rescue085_postcommand08_2026-10-17.yaml`.

## Component figures

- `fig1_system.png` — architecture overview.
- `stage1_*` — TSE adaptation and training evidence.
- `stage1_spex_fsc_passenger_cochannel_ft1_training_2026-10-16.*` was
  regenerated on 2026-10-20 with `scripts/reporting/plot_stage1_history.py`.
  The original artifact was empty because a generic classification-curve
  renderer was pointed at Stage-1 SI-SDR logs.
- `stage2_presence_current_dual_domain_validation_2026-11-18.*` — authoritative
  Stage-2 figure for the current dual-domain FT1 checkpoint. It is rendered
  from `logs/stage2_frame_presence_dual_domain_nonwake_ft1/metrics_last.json`
  and reports frame F1 0.904 at threshold 0.50, best frame F1 0.913 at 0.65,
  and chunk F1 0.996.
- `stage2_presence_training_comparison_2026-10-20.*` — compatibility alias to
  the corrected current-validation figure above. The filename is retained so
  existing paper references do not resolve to the misleading historical plot.
- `stage2_presence_historical_training_comparison_2026-10-20.*` — preserved
  three-run training-history diagnostic. It combines older, different
  datasets and checkpoints; its lower fixed-threshold F1 values must not be
  cited as the current dual-domain FT1 result.
- `stage3_target_kws_negative_calibrated_comparison_2026-10-16.*` — target-KWS
  candidate training/validation comparison. It was regenerated on 2026-11-18
  with `scripts/reporting/plot_stage3_target_kws_history.py`: the old blank
  panel expected `train_loss`, which these historical Stage-3 logs never
  recorded. The repaired figure uses training accuracy plus validation
  accuracy, F1, precision, and recall at the fixed binary argmax threshold and
  explicitly states that loss and per-epoch threshold sweeps were not inferred.
  Its roughly 0.70 epoch F1 is not the separate 97.40% threshold-swept
  TSE-domain operating point reported in the paper.
- `stage3_kws_epoch_vs_operating_point_reconciliation_2026-11-18.*` — direct
  visual reconciliation of the fixed-threshold epoch maxima, the selected
  97.40% threshold-swept TSE-domain point, and the current stored target-KWS
  validation point. The protocols are labeled separately and must not be
  treated as one curve.
- `compact_speaker_verifier_*` — speaker-verifier training evidence.
- `stage4_intent_runtime_jitter_clean_overlap_ft6_fromft4_2026-10-17.*` —
  current FT6 intent fine-tuning curves.
- `stage4_intent_frozen_tse_downstream_comparison_*` and the WavLM figure —
  downstream/baseline comparisons.

## Ablations and external reference comparisons

The gate, post-command refractory, Speech Commands, and SLURP figures are
included for the ablation and apples-to-apples comparison sections. The full
historical archive remains in `reports/figures/`; older candidates must not be
presented as the current FT6 result.

## Evidence boundary

These figures use clean or synthetic proxy data. They do not establish
same-driver real-cabin reliability, passenger rejection in a measured vehicle,
or Jetson/automotive-device latency. The latest quantitative caveats are in
`reports/model_architecture_metrics_latency_sota_2026-10-17.md`.
