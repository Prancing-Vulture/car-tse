# Compact cabin+Smart speaker gate full-context recheck — superseded 2026-10-14

> **Superseded:** the batched evaluator did not pass the loaded compact
> verifier into its speaker-score path, so the compact-backend label in this
> report is not reliable. Use
> `reports/compact_gate_backend_correction_2026-10-14.md` and its corrected raw
> logs instead. The historical result is retained for traceability.

## Question

The compact cabin+Smart verifier had previously reduced Smart Speaker
false-wake events from the retained SpEx-gated path. That result was produced
before the shared full-context evaluator correction. This rerun checks the
candidate on the complete corpus without scoring incomplete zero-padded tails.

## Reproduction

```powershell
python scripts/eval/eval_smart_speaker_command_negative.py `
  --config configs/eval_end_to_end_cavemove_target_kws_smart_speaker_compact_cabin_smart_thr030_2026-10-12.yaml `
  --max-files 0 --hop-seconds 0.25 --batch-size 32 `
  --action-sample-files 14 `
  --output logs/smart_speaker_compact_cabin_smart_thr030_fullwindows_2026-10-14.json
```

## Results

| Configuration | Full corpus | Full-path events/hour | KWS-only events/hour | Action sample |
|---|---:|---:|---:|---:|
| Retained hard-negative KWS, no compact gate | 2,110 files / 3.193 h | 23.81 | 49.80 | 0/14 false actions |
| Compact cabin+Smart gate, threshold 0.30 | 2,110 files / 3.193 h | **0.00** | **38.21** | **0/14 false actions** |

The compact gate recorded zero full-path events over the complete corpus and
zero false actions in the 14-file sequential sample. It also reduced KWS-only
smoothed events from 159 to 122 (23.3%), while the retained target-KWS model
without the gate removed none at that stage.

## Positive-side wake/enrollment check

To test whether the safety gain came from rejecting all target speech, the same
16-scene clean proxy was run with each scene's wake clip as its enrollment
clip. This makes wake and enrollment speaker-consistent for the wake decision;
the following command remains cross-corpus, so intent is not interpreted.

| Configuration | Target wakes | Wake recall | Intent result |
|---|---:|---:|---:|
| Retained hard-negative KWS, no compact gate | 15/16 | 93.75% | not interpretable |
| Compact cabin+Smart gate, threshold 0.30 | 3/16 | 18.75% | not interpretable |

The compact gate therefore over-suppresses the current positive proxy despite
its strong negative-domain rejection. It is not a drop-in production
improvement yet.

This is a meaningful negative-domain improvement, but it is not a production
result. The corpus is clean, single-channel studio audio duplicated to four
channels, and contains no driver positives. Earlier same-speaker synthetic
proxy runs with compact gating recorded only 37.5% target-wake recall, so the
gate is not promoted until it is retested on speaker-consistent cabin positives.

## Decision

Keep the compact cabin+Smart checkpoint as the leading safety candidate for
the next real-driver pilot, but do not replace the current retained config or
freeze threshold 0.30. The next decisive experiment is a same-driver tuning
run comparing: no compact gate, compact gate at 0.20/0.30/0.50, and a
command-only compact gate. The held-out driver split must remain untouched.

Raw result:

- `logs/smart_speaker_compact_cabin_smart_thr030_fullwindows_2026-10-14.json`
- `logs/clean_compact_gate_wake_enrollment_fullwindows_16scenes_2026-10-14.json`
- `logs/clean_nogate_wake_enrollment_fullwindows_16scenes_2026-10-14.json`
