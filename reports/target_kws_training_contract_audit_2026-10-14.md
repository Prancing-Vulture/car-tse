# Target-KWS training/runtime contract audit — 2026-10-14

## Scope

This audit checks whether the retained hard-negative target-speaker KWS model
is being evaluated with the same input contract it was trained on, and whether
an existing candidate checkpoint should be promoted before collecting
speaker-consistent cabin data.

## Contract check

The retained configuration and runtime agree on the important dimensions:

- sample rate: 16 kHz;
- training/cache segment: 1.0 s = 16,000 samples;
- runtime KWS input: 1.0 s;
- mel features: 40;
- target-KWS audio channels: `[32, 64, 128]`;
- enrollment and extracted embeddings: 256 dimensions;
- cache arrays: `wav (N, 16000)`, `enrollment (N, 256)`,
  `extracted (N, 256)`.

The retained cache has 4,684 train, 1,629 validation, and 1,625 test rows.
The training split contains 604 positives and 4,080 negatives; validation has
130/1,499 and test has 128/1,497. The trainer uses a balanced sampler, so the
optimization batches are balanced even though the stored corpus is negative
heavy.

This means the earlier one-second crop/scale mismatch has been corrected for
the retained checkpoint. The current weakness is not an unverified tensor
shape or crop mismatch.

## Cached calibration

The retained checkpoint's held-out target-KWS test calibration reports:

- best clip-level F1: 0.523 at score threshold 0.51;
- recall at 0.95 target: 0.961, with clip-level false accepts equivalent to
  817.64/hour under a one-second-window normalization;
- the validation best F1 is 0.474.

These clip metrics predict the continuous open-set problem: the model's score
distribution is not sufficiently separated for a low-FAR wake detector.

## Candidate decision

Existing larger, thin, distillation, and positive-heavy checkpoints were not
promoted. The larger candidate reached cached test F1 0.830 at threshold 0.725
but failed to transfer to the cross-corpus positive proxy and was historically
86.5/hour on a pre-correction Smart Speaker subset. The positive-heavy
warm-start reached lower validation F1 than the retained checkpoint. These are
not grounds for replacing the retained model without same-driver evidence.

## Decision

No checkpoint is changed by this audit. The next training run should wait for a
speaker-consistent English tuning split containing the real wake phrase,
same-driver commands, passenger wake negatives, and cabin/playback conditions.
That split can be used to train or recalibrate the detector while preserving a
separate held-out driver test.

## Sources

- `configs/stage3_target_kws_gsc_marvin_hardneg_aishell5_smart_speaker.yaml`
- `data/cache/stage3_target_kws_gsc_marvin_hardneg_aishell5_smart_speaker_16k/`
- `logs/stage3_target_kws_gsc_marvin_hardneg_aishell5_smart_speaker_ft1/threshold_calibration_test_2026-10-14.json`
- `reports/target_kws_large_candidate_2026-10-14.md`
- `reports/target_kws_checkpoint_ablation_2026-10-14.md`
