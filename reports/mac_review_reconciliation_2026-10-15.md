# Mac review reconciliation and P0 audit

Date: 2026-10-15  
Source: Mac review of `multi-speaker-eval` at `b9ec808`, plus the corrected
re-score requested in GitHub issue #3.  No model training was performed for
this reconciliation.

## Bottom line

The Mac review is correct about the important limitation: the earlier
multi-speaker reports mixed legacy event-association metrics with a benchmark
that did not preserve enough scene metadata.  The old numbers are useful as
historical diagnostics, but the passenger-safety conclusions must use
action-level and source-attributed metrics first.

The clean, matched-speaker 1,000-scene result remains a valid *synthetic clean
proxy* result.  It is not real-driver evidence.  The passenger and overlap
results below are corrected and should replace the corresponding headline
claims in the earlier issue reports.

## Re-scored sequential passenger result

Source log:
`logs/multispeaker_speech_only_e2e_v2_knnvc_matched_buffer4_1000_2026-10-15.json`

Reproduction:

```powershell
python scripts/eval/rescore_multispeaker_log.py `
  logs/multispeaker_speech_only_e2e_v2_knnvc_matched_buffer4_1000_2026-10-15.json `
  --output reports/rescore_multispeaker_speech_only_2026-10-15.json
```

| Metric | Corrected result |
|---|---:|
| Scenes / synthetic duration | 1,000 / 3.889 h |
| Total emitted action events | 1,587 |
| Scenes with at least one wrong-intent action | **617 / 1,000** |
| Executed non-target actions | **616** |
| Fully correct scenes (`target intent correct` and exactly one command event) | **326 / 1,000** |
| Legacy false wakes | 1,102 / 283.37 h |
| Legacy false actions | 99 / 25.46 h |

The source-attributed wake counts are:

- 500 passenger-wake events outside the old target tolerance;
- 602 passenger-nonwake events outside the old target tolerance; and
- 1,546 events absorbed by the historical broad target-wake tolerance.

The last category is why the old false-wake counter is not a passenger
rejection metric.  It includes windows that overlap the target wake association
interval, even when the passenger may be the actual acoustic source.

For completeness, the rescorer also reports 645 scenes with *any* wrong
executed event when extra correct-intent actions are included.  The 617 figure
is the review's primary wrong-intent-scene metric and should be reported first;
the two definitions must not be conflated.

## Overlap correction

The overlap logs use the same synthetic passenger construction and the same
effective three-call command buffer.  A filename containing `buffer4` does not
prove that four calls were used; every report must expose the effective buffer
configuration.  The current overlap JSON records
`effective_command_buffer_calls: 3`.

The earlier comparison also mixed different scene prefixes.  The bypass value
of 45.31% was a 128-scene prefix, not a matched comparison against the full
1,000-scene result.  The review's matched-prefix comparison is the one to use:
the baseline and Stage-1-bypass values were 52.34% and 45.31%, respectively,
on the same scenes.  That is a **7.03-point Stage-1 effect**, not evidence that
the two paths were identical.  The existing FT1/FT2 overlap pilot values
(57.03% and 57.81%) remain exploratory only because their cache construction
used the evaluation passenger manifest and therefore needs a clean rebuild
before it can support a training claim.

## Corrections applied to the evaluator

`scripts/eval/eval_continuous_full_pipeline_proxy.py` now:

1. samples passenger wake and non-wake speakers per scene, avoiding the target
   speaker where possible;
2. records passenger source IDs, source paths, interval endpoints, and the
   effective configuration hash;
3. supports `--full-length-commands` so command audio is not silently cut to
   two seconds;
4. attributes emitted wake windows to target, passenger wake, passenger
   non-wake, overlap, or legacy target-tolerance support;
5. reports executed actions, non-target actions, wrong-intent actions,
   wrong-action scenes, fully-correct scenes, and source-attributed wake
   counts; and
6. exposes `--kws-refractory-calls` / `--kws-refractory-seconds` for the P2
   one-action-per-wake experiment without changing the canonical configuration.

The standalone `scripts/eval/rescore_multispeaker_log.py` reproduces the key
correction from old logs without loading checkpoints or running inference.

The in-progress P0 reproduction uses the following command; its output is
written under a new filename so the historical log remains immutable:

```powershell
python scripts/eval/eval_continuous_full_pipeline_proxy.py `
  --config configs/eval_end_to_end_clean_english_identity_targetkws_confidence_buffered_hybrid_run3.yaml `
  --wake-manifest manifests/clean_e2e_v2/wake_knnvc_same_speaker_20x10_test_2026-10-14.jsonl `
  --passenger-manifest manifests/test/stage3_kws_cabin_test.jsonl `
  --scene-sampling matched_speaker --scenes 1000 `
  --passenger-only --full-length-commands --cabin-snr-db 0 --passenger-snr-db 0 `
  --enrollment-source same_speaker_other --hop-seconds 0.5 `
  --output logs/multispeaker_speech_only_p0_dynamic_full_1000_2026-10-15.json `
  --artifact-dir reports/artifacts/multispeaker_speech_only_p0_dynamic_full_1000_2026-10-15
```

## Remaining P0–P5 order

The review's order remains the project plan:

| Phase | Status / next action |
|---|---|
| P0 evaluator | Implemented in the working tree; rerun the 1,000-scene sequential and overlap studies with dynamic passengers, full-length commands, and saved effective metadata. |
| P1 speaker gate | Tune/log only on FSC validation speakers; keep test speakers untouched. No calibrated gate is currently active in the canonical run. |
| P2 wake/action policy | Evaluate a post-wake refractory period and one committed action per wake. Keep the variant separate until safety and target recall are compared. |
| P3 endpointing | Replace the fixed post-wake command window with wake-end plus presence/VAD endpointing, capped at six seconds. |
| P4 overlap cache | Rebuild from train-split passenger speakers and mix before Stage 1; do not promote the current FT1/FT2 overlap candidates. |
| P5 weak speakers | Inspect `NgQEvO2x7Vh3xy2xz` and `oOK5kxoW7dskMbaK` after P0–P3. |

No further intent fine-tuning should be interpreted as a fix until P0–P3 are
complete.  The next report should place fully-correct scenes, wrong-intent
scenes, non-target actions/hour, and source-attributed false wakes before the
legacy false-wake/false-action counters.

## What is still not established

- There are no consented real-driver continuous sessions in the repository.
- There is no true six-channel car capture with four cabin microphones and two
  playback-reference channels.
- Passenger speech is currently synthetic and does not provide a spatial seat
  cue; the overlap stress test sums channels and is intentionally harsher than
  a real array in one respect and less representative in another.
- Production false-accept, passenger-rejection, device-latency, privacy, and
  safety claims remain blocked until the frozen pilot protocol and target
  hardware evaluation exist.
