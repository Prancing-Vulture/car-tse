# Compact speaker-gate backend correction and recalibration — 2026-10-14

## Why this rerun was required

The earlier Smart Speaker report labelled a batched result as a compact
speaker-gate result. The batched evaluator received the speaker threshold but
did not pass the loaded compact verifier into its scoring function, so the
reported result could silently use the default SpEx similarity path. The old
full-context/tail-window correction remains valid, but the backend identity of
that compact result was not.

The evaluator is now corrected in
`scripts/eval/eval_smart_speaker_command_negative.py`: it passes the compact
verifier only when `speaker_gate_backend: compact` is selected, and it supports
shared-forward-pass speaker-threshold sweeps. The historical logs are retained
and superseded rather than rewritten.

## Corrected negative evaluation

Dataset: University of Rochester Smart Speaker Command Dataset, 2,110 files,
3.1925356 hours. The clean single-channel recordings were duplicated across
four channels only to satisfy the pipeline interface; they are not a real
multi-microphone cabin benchmark.

Reproduction:

```powershell
python scripts/eval/eval_smart_speaker_command_negative.py `
  --config configs/eval_end_to_end_cavemove_target_kws_smart_speaker_compact_cabin_smart_thr030_2026-10-12.yaml `
  --max-files 0 --hop-seconds 0.25 --batch-size 32 `
  --kws-thresholds 0.45 0.46 0.47 0.48 0.49 0.50 `
  --action-sample-files 14 `
  --output logs/smart_speaker_compact_verifier_corrected_threshold_sweep_045_050_fullwindows_2026-10-14.json
```

| KWS threshold | Compact gate | Full-path events | Full-path events/hour | KWS-only events/hour |
|---:|---:|---:|---:|---:|
| 0.45 | 0.30 | 547 | 171.34 | 697.25 |
| 0.46 | 0.30 | 494 | 154.74 | 592.95 |
| 0.47 | 0.30 | 402 | 125.92 | 500.86 |
| 0.48 | 0.30 | 287 | 89.90 | 360.84 |
| 0.49 | 0.30 | 127 | 39.78 | 184.18 |
| 0.50 | 0.30 | 6 | **1.88** | 38.21 |

The sequential 14-file action sample at the retained compact configuration
produced 2 false wakes and 0 false actions. This is a small diagnostic sample,
not an action-safety estimate.

At fixed KWS threshold 0.46, raising the compact verifier threshold reduced
but did not eliminate open-set events:

| Compact gate threshold | Full-path events/hour |
|---:|---:|
| 0.30 | 154.74 |
| 0.40 | 129.36 |
| 0.50 | 102.11 |
| 0.60 | 81.75 |
| 0.70 | 55.44 |
| 0.80 | 24.12 |

The shared-forward-pass gate sweep is in
`logs/smart_speaker_compact_verifier_gate_threshold_sweep_kws046_fullwindows_2026-10-14.json`.

## Positive and noisy diagnostics

The positive proxy uses each scene's wake clip as enrollment, so wake and
enrollment are speaker-consistent for the wake decision. The command remains
cross-corpus and is not used as same-driver intent evidence.

| Configuration | Wake recall | Intent interpretation |
|---|---:|---|
| No compact gate, KWS 0.50 | 15/16 = 93.75% | not interpretable |
| Compact 0.30, KWS 0.50 | 3/16 = 18.75% | not interpretable |
| Compact 0.30, KWS 0.45 | 15/16 = 93.75% | not interpretable |
| Compact 0.30, KWS 0.46 | 15/16 = 93.75% | not interpretable |
| Compact 0.30, KWS 0.47 | 13/16 = 81.25% | not interpretable |
| Compact 0.30, KWS 0.48 | 8/16 = 50.00% | not interpretable |

On the noisy CAVEMOVE proxy, compact 0.30 plus KWS 0.46 produced 20/32 wake
decisions (62.5%), 1/32 intent rows (3.125%), and 0 false actions. The proxy
has `speaker_consistent=false`, so these are transfer diagnostics only.

Raw positive logs:

- `logs/clean_compact_gate_wake_enrollment_kws045_fullwindows_16scenes_2026-10-14.json`
- `logs/clean_compact_gate_wake_enrollment_kws046_fullwindows_16scenes_2026-10-14.json`
- `logs/clean_compact_gate_wake_enrollment_kws047_fullwindows_16scenes_2026-10-14.json`
- `logs/clean_compact_gate_wake_enrollment_kws048_fullwindows_16scenes_2026-10-14.json`
- `logs/noisy_compact_gate_wake_enrollment_kws046_fullwindows_16scenes_2026-10-14.json`

## Decision

The compact verifier is not promoted or frozen. The corrected negative result
shows that the earlier apparent 0/hour safety result was an evaluator/backend
misattribution, and the actual compact path remains far above the provisional
0.1/hour target on this open-set corpus. The current evidence supports three
actions:

1. Keep the retained configuration unchanged until the evaluator correction is
   reflected in the release artifacts.
2. Use same-driver cabin tuning data to calibrate the verifier and KWS jointly;
   do not select a threshold from Smart Speaker alone.
3. Treat hard-negative verifier/KWS retraining as the next model improvement
   only after the permitted driver/cabin split exists, with a frozen held-out
   test and separate replay/echo negatives.

This correction improves the reliability of the study, but it is not itself a
model-quality improvement.
