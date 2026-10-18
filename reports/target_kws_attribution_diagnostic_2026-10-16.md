# Target-KWS attribution diagnostic

Date: 2026-10-16  
Branch: `multi-speaker-eval`  
Status: diagnostic only; no weights were trained or changed

## Purpose

The evaluator now records `target_kws_probability` separately from the final
`wake_probability`. The former is the enrollment-conditioned target-KWS score
before optional speaker-score fusion. This makes it possible to distinguish a
target-KWS attribution problem from a later command-window or intent problem.

The two runs below use the same frozen stack, matched FSC speakers, full-length
commands, 0 dB passenger/cabin settings, 0.5 s hops, and 128 scenes. They are
synthetic proxy scenes, not real-driver evidence.

## End-to-end results

| Condition | Target wake recall | Fully correct | False actions | Wrong executed actions | Strict false wakes/h | Strict false commands/h |
|---|---:|---:|---:|---:|---:|---:|
| Sequential / passenger-only | 100.0% | 77.34% (99/128) | 3 | 6 | 128.57 | 10.04 |
| Simultaneous overlap | 100.0% | 75.78% (97/128) | 0 | 9 | 60.27 | 0.00 |

The sequential run has 3 passenger-only false actions. The overlap run is
safer at the action level, but still loses 31 scenes to a wrong or missing
target command; 9 committed actions have the wrong intent. Wake recall is not
the limiting metric in either run.

Raw logs:

- `logs/multispeaker_passenger_mixedintent_confmargin_instrumented_128_2026-10-16.json`
- `logs/multispeaker_overlap_mixedintent_confmargin_instrumented_128_2026-10-16.json`

## Raw target-KWS score distributions

The following categories are assigned with a fixed priority for each window:
target wake, target command, passenger wake only, passenger non-wake only, and
background. In simultaneous overlap, passenger wake windows overlap the target
and therefore are not listed as a separate passenger-only category.

| Condition / window category | Windows | Mean | P10 | Median | P90 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Sequential target wake | 480 | 0.4468 | 0.3974 | 0.4459 | 0.4980 | 0.5228 |
| Sequential target command | 323 | 0.4525 | 0.3959 | 0.4547 | 0.5086 | 0.5257 |
| Sequential passenger wake only | 531 | 0.4440 | 0.3832 | 0.4534 | 0.5007 | 0.5275 |
| Sequential passenger non-wake only | 778 | 0.4261 | 0.3792 | 0.4070 | 0.4848 | 0.5279 |
| Sequential background | 216 | 0.3848 | 0.3797 | 0.3850 | 0.3916 | 0.3916 |
| Overlap target wake | 484 | 0.4442 | 0.3968 | 0.4447 | 0.4927 | 0.5235 |
| Overlap target command | 320 | 0.4711 | 0.4178 | 0.4828 | 0.5108 | 0.5230 |
| Overlap passenger non-wake only | 78 | 0.3983 | 0.3797 | 0.3885 | 0.4316 | 0.4481 |
| Overlap background | 1,629 | 0.3845 | 0.3797 | 0.3850 | 0.3907 | 0.3916 |

## Interpretation

1. In the sequential condition, the passenger-wake-only median (0.4534) is
   slightly above the target-wake median (0.4459). A single higher wake-score
   threshold cannot preserve target recall while reliably rejecting passenger
   wake words.
2. The overlap condition has no passenger-only actions in this run, but its
   75.78% fully-correct rate is limited by command acceptance/intent errors,
   not by target wake recall. The system often reaches a wake and then fails to
   commit the correct target command.
3. The next targeted improvement should therefore combine an independently
   calibrated target-speaker gate or seat-zone cue with target-conditioned
   command endpointing. Wake-score threshold sweeps alone are not a sufficient
   fix.

## Reproduction

From `D:\Prawin_New\tse\car-tse-main`:

```powershell
python scripts/eval/eval_continuous_full_pipeline_proxy.py `
  --config configs/eval_end_to_end_fscpassenger_identity_mixedintent_confmargin_commandgate_2026-10-16.yaml `
  --wake-manifest manifests/clean_e2e_v2/wake_knnvc_same_speaker_20x10_test_2026-10-14.jsonl `
  --passenger-manifest manifests/test/stage3_kws_cabin_test.jsonl `
  --scene-sampling matched_speaker --scenes 128 --passenger-only `
  --full-length-commands --cabin-snr-db 0 --passenger-snr-db 0 `
  --enrollment-source same_speaker_other --hop-seconds 0.5 `
  --output logs/multispeaker_passenger_mixedintent_confmargin_instrumented_128_2026-10-16.json `
  --artifact-dir reports/artifacts/multispeaker_passenger_mixedintent_confmargin_instrumented_128_2026-10-16

python scripts/eval/eval_continuous_full_pipeline_proxy.py `
  --config configs/eval_end_to_end_fscpassenger_identity_mixedintent_confmargin_commandgate_2026-10-16.yaml `
  --wake-manifest manifests/clean_e2e_v2/wake_knnvc_same_speaker_20x10_test_2026-10-14.jsonl `
  --passenger-manifest manifests/test/stage3_kws_cabin_test.jsonl `
  --scene-sampling matched_speaker --scenes 128 --overlap `
  --full-length-commands --cabin-snr-db 0 --passenger-snr-db 0 `
  --enrollment-source same_speaker_other --hop-seconds 0.5 `
  --output logs/multispeaker_overlap_mixedintent_confmargin_instrumented_128_2026-10-16.json `
  --artifact-dir reports/artifacts/multispeaker_overlap_mixedintent_confmargin_instrumented_128_2026-10-16
```

