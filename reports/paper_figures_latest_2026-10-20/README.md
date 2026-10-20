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
- `stage2_presence_training_comparison_2026-10-20.*` — retained Stage-2
  training-history comparison. The current dual-domain FT1 log contains only
  a final summary, so this is not a current FT1 epoch curve.
- `stage3_target_kws_negative_calibrated_comparison_2026-10-16.*` — target-KWS
  candidate training/validation comparison.
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
