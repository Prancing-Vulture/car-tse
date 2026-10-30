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

## Corrected P0 sequential result (completed 2026-10-15)

The first corrected rerun completed with 1,000 scenes using dynamic passenger
selection, 136 unique passenger speakers for both wake and non-wake pools, the
full command audio, the canonical three-call command buffer, and no active
speaker gate or refractory period.  This is still a synthetic speech-only
proxy, not a real-driver result.

| Metric | Result |
|---|---:|
| Scenes / synthetic duration | 1,000 / 3.889 h |
| Target wake recall | **99.90%** |
| Target intent accuracy | **94.00%** |
| Intent accuracy conditioned on target wake | **94.09%** |
| Fully correct scenes | **498 / 1,000 (49.80%)** |
| Strict fully-correct action scenes | **497 / 1,000 (49.70%)** |
| Executed action events | 1,506 |
| Executed non-target actions | 527 |
| Wrong-intent action events | 525 |
| Scenes with a wrong-intent action | 452 / 1,000 |
| Passenger-only false actions | **91** |
| False-action events | 99 / 3.889 h (25.46/h) |
| Strict false-wake events | 2,445 / 3.889 h (628.71/h) |
| Target-wake latency mean / p95 | 1.668 s / 2.000 s |

The result separates the current bottlenecks: target wake recall and
conditional intent recognition are already high on this proxy, while the
guarded action policy is unsafe because passenger/non-target wake evidence is
still converted into many action attempts.  The large strict false-wake count
also means the current source-association and wake policy need to be fixed
before model fine-tuning can be interpreted as a production improvement.

Reproduction artifact:
`logs/multispeaker_speech_only_p0_dynamic_full_1000_2026-10-15.json`

Effective configuration SHA256:
`3fc7a20b28769698d6a21b337f2a667b88b9bc8570abb869b736f6cd5b9dc8bb`

## Corrected P0 overlap result (completed 2026-10-15)

The matched overlap rerun used the same 1,000 scenes and configuration as the
sequential result, but summed the target and passenger speech into a
co-channel mixture.  It has no spatial seat cue and is therefore a stress
proxy, not a real-array cabin measurement.

| Metric | Result |
|---|---:|
| Scenes / synthetic duration | 1,000 / 3.889 h |
| Target wake recall | **100.00%** |
| Target intent accuracy | **61.50%** |
| Fully correct scenes | **534 / 1,000 (53.40%)** |
| Executed action events | 968 |
| Executed non-target actions | 94 |
| Wrong-intent action events | 324 |
| Scenes with a wrong-intent action | 315 / 1,000 |
| Wrong executed action events | 353 |
| Scenes with a wrong executed action | 340 / 1,000 |
| Strict false-wake events | 829 / 3.889 h (213.17/h) |
| Strict false-command events | 35 / 3.889 h (9.00/h) |
| Passenger-overlap command events | 865 |
| Ambiguous overlap command events | 865 |
| Passenger-only false actions | 0* |

\* The zero is not a safety pass: in co-channel overlap, the passenger and
target intervals are intentionally simultaneous, so most events are ambiguous
under source attribution and fall inside the broad target association.  The
wrong-action and strict false-wake metrics are the useful diagnostics here.

Reproduction artifact:
`logs/multispeaker_overlap_p0_dynamic_full_1000_2026-10-15.json`

The identical effective configuration SHA256 was recorded in both runs:
`3fc7a20b28769698d6a21b337f2a667b88b9bc8570abb869b736f6cd5b9dc8bb`.

## P1 compact speaker-gate comparison: sequential result

The first runtime improvement was evaluated without retraining.  It enables
the existing cabin+Smart compact verifier at threshold 0.30 for both the
pre-wake and command speaker checks, while keeping the P0 scene set, target
checkpoints, command buffer, and full-length command construction unchanged.

| Metric | No gate | Compact gate 0.30 | Change |
|---|---:|---:|---:|
| Target wake recall | 99.90% | 96.40% | -3.50 pp |
| Target intent accuracy overall | 94.00% | 62.20% | -31.80 pp |
| Target intent accuracy conditioned on wake | 94.09% | 64.52% | -29.57 pp |
| Fully correct scenes | 49.80% | **57.60%** | **+7.80 pp** |
| Passenger-only false actions | 91 | **22** | **-75.82%** |
| Wrong executed actions | 566 | **79** | **-86.04%** |
| Scenes with wrong executed action | 486 | **73** | **-84.98%** |
| Strict false wakes | 2,445 | **988** | **-59.59%** |
| False-action events | 99 | 22 | -77.78% |

The gate improves the safety-weighted exact-scene metric and sharply reduces
non-target actions, but the 0.30 threshold rejects too much target speech in
this clean synthetic proxy.  It is retained as a candidate, not promoted as
the default.  The next sweep must select a threshold on tuning data and
report the full driver-safety Pareto curve before any checkpoint or default
configuration changes.

Artifact:
`logs/multispeaker_speech_only_p0_compact030_full_1000_2026-10-15.json`

Candidate configuration:
`configs/eval_end_to_end_clean_english_identity_targetkws_confidence_buffered_hybrid_compact030_2026-10-15.yaml`

## P1 compact speaker-gate comparison: simultaneous-overlap result (completed 2026-10-16)

The same compact-gate candidate was then evaluated on the corrected 1,000-scene
co-channel overlap proxy.  The target and passenger tracks are simultaneous,
all four synthetic channels contain the same sum, and no spatial cue is
available.  This is intentionally a stress test, not a spatial-array result.

| Metric | No gate overlap | Compact gate 0.30 overlap |
|---|---:|---:|
| Target wake recall | 100.00% | 96.60% |
| Target intent accuracy overall | 61.50% | **37.10%** |
| Fully correct scenes | 53.40% | 37.00% |
| Executed non-target actions | 94 | **1** |
| Wrong executed actions | 353 | 153 |
| Strict false wakes | 829 (213.17/h) | **451 (115.97/h)** |
| Strict false commands | 35 (9.00/h) | **2 (0.51/h)** |
| Passenger-only false actions | 0* | 0* |

The gate therefore improves action suppression but fails the usefulness side of
the trade-off under co-channel overlap: it rejects target evidence together
with passenger evidence.  It is rejected as a production operating point and
will not be promoted.  The next candidate must improve the target-conditioned
representation or use an actual spatial/seat cue rather than lowering the
threshold blindly.

Artifact:
`logs/multispeaker_overlap_p0_compact030_full_1000_2026-10-15.json`

Comparison figure (PNG/PDF):
`reports/figures/multispeaker_gate_comparison_2026-10-16.png`
and `reports/figures/multispeaker_gate_comparison_2026-10-16.pdf`.

## Train-split-safe overlap intent adaptation (started 2026-10-16)

The earlier FT1/FT2 overlap candidates are exploratory because their passenger
augmentation pool came from the held-out KWS test manifest.  A new append-only
cache was therefore built from the frozen Stage-1 intent cache using only the
KWS train passenger pool (`stage3_kws_cabin_train.jsonl`).  The train and held-
out passenger audio paths have zero intersection (8,628 train files versus
1,147 held-out files).

| Artifact | Value |
|---|---|
| Train/validation rows | 16,384 / 4,096 (balanced clean + overlap) |
| Passenger SNR | 0 dB |
| Cache | `data/cache/stage4_intent_overlap_trainsplit_0db_2026-10-16_16k/` |
| Train manifest | `manifests/train/stage4_intent_overlap_trainsplit_0db_2026-10-16.jsonl` |
| Validation manifest | `manifests/val/stage4_intent_overlap_trainsplit_0db_2026-10-16.jsonl` |
| Config | `configs/stage4_intent_overlap_trainsplit_0db_2026-10-16.yaml` |

The candidate was initialized from the retained clean cabin-domain intent head
and encoder; the Stage-1 extractor and wake model remained frozen.  It trained
for eight epochs with mixed precision and selected epoch 6 by validation
macro-F1:

| Metric | Best validation value |
|---|---:|
| Accuracy | 75.5% |
| Macro-F1 | 75.2% |
| Top-5 accuracy | 92.7% |
| Train accuracy | 77.0% |

Raw history:
`logs/stage4_intent_wav2vec_overlap_trainsplit_0db_2026-10-16/epoch_metrics_c32e45fc3af240788a7b7bd87fd66b12.jsonl`

Training curves (PNG/PDF):
`reports/figures/stage4_intent_overlap_trainsplit_0db_training_curves_2026-10-16.png`
and `reports/figures/stage4_intent_overlap_trainsplit_0db_training_curves_2026-10-16.pdf`.

The head checkpoint SHA256 is
`D99BEBAB14D014F47CFF07F988912EE84E2E2D941E5F6B697F9B169B82BB2ACC` and the
encoder checkpoint SHA256 is
`E550FACB7ED4174BE96DB0E6D6AAD359F12D374206E5F0DEDE43012DE80F4064`.
These are validation results only; the candidate remains unpromoted until the
clean, sequential, and simultaneous-overlap end-to-end regressions complete.

## Train-split-safe candidate end-to-end regression (completed 2026-10-16)

The new head/encoder was evaluated with the same frozen Stage-1, presence,
target-KWS, command-buffer, and confidence settings as the P0 baseline.  The
only changed learned component is the Stage-4 intent head/upper encoder.  Each
condition contains 1,000 matched-speaker scenes (3.889 synthetic hours).

| Metric | Clean control | Sequential passenger | Simultaneous co-channel overlap |
|---|---:|---:|---:|
| Target wake recall | 99.90% | 99.90% | 100.00% |
| Target intent accuracy | **90.60%** | **94.10%** | **70.40%** |
| Fully correct scenes | **90.60%** | **58.60%** | **62.60%** |
| Passenger-only false actions | 0 | 84 | 0* |
| False-action rate | 0/h | 22.63/h | 0/h |
| Executed non-target actions | 0 | 403 | 89 |
| Wrong executed actions | 39 | 440 | 307 |
| Strict false wakes | 176.91/h | 628.71/h | 213.17/h |
| Strict false commands | 2.31/h | 76.63/h | 7.97/h |
| Target-wake latency mean / p95 | 1.664 s / 2.000 s | 1.668 s / 2.000 s | 1.726 s / 3.000 s |

Relative to the no-gate baseline, the candidate keeps the clean score at
90.60%, raises the sequential exact-scene rate from 49.80% to 58.60% (+8.80
points), and raises the overlap exact-scene rate from 53.40% to 62.60% (+9.20
points).  Overlap target intent accuracy rises from 61.50% to 70.40% (+8.90
points).  Sequential passenger-only false actions fall from 91 to 84 and
executed non-target actions from 527 to 403, but 22.63 false actions/hour is
still far above a safe operating target.  The Stage-4 adaptation is therefore
useful and retained as a research candidate, but it is not production-ready.

The co-channel zero in the passenger-only column is not a safety pass: target
and passenger intervals overlap by construction.  The strict false-command,
wrong-action, and ambiguous-window metrics are the meaningful diagnostics for
that condition.

Raw end-to-end artifacts:

- `logs/clean_trainsplit_intent_1000_2026-10-16.json`
- `logs/multispeaker_speech_only_trainsplit_intent_1000_2026-10-16.json`
- `logs/multispeaker_overlap_trainsplit_intent_1000_2026-10-16.json`

Comparison figure (PNG/PDF):
`reports/figures/multispeaker_candidate_comparison_2026-10-16.png`
and `reports/figures/multispeaker_candidate_comparison_2026-10-16.pdf`.

All three runs used effective configuration SHA256
`d5120d89681e86afebb9731de840db55f5b99e3ccc68cf1c344ec7ea867eff8c` and
`production_valid: false`.  The next improvement is a calibrated target
speaker/seat gate and one-action-per-wake policy, evaluated on tuning data
before another held-out test.
