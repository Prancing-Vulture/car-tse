# E2E v2 clean-data results and implementation-plan audit

Date: 2026-10-15  
Status: clean-data engineering target met on a synthetic, speaker-consistent
proxy; production readiness is not established.

## Executive result

The balanced 1,000-scene English clean benchmark reached **90.1% end-to-end
task success** with the existing frozen checkpoints and the existing
`confidence_buffered_hybrid` command decoder. The 95% Wilson lower confidence
bound is approximately **88.1%**, which meets the implementation plan's
provisional acceptance target of a point estimate at least 90% and a lower
bound at least 88%.

The result is not a real-driver or production result. The wake word was
converted with kNN-VC into the voice of the FSC command speaker, and the
scenes were assembled synthetically. There is no natural continuous recording
containing enrollment, wake word, command, passenger speech, and vehicle
playback in this benchmark.

## Primary benchmark

| Metric | Result |
|---|---:|
| Scenes | 1,000 |
| Synthetic duration | 14,000 s / 3.889 h |
| FSC command speakers | 10, 100 scenes each |
| Unique converted wake rows used | 200 |
| Unique FSC command rows used | 1,000 |
| Wake recall | 99.9% (999/1,000) |
| End-to-end task success | **90.1% (901/1,000)** |
| Intent accuracy given wake | 90.19% (901/999) |
| 95% Wilson lower bound, task success | **88.1%** |
| False wake events | 0 observed |
| False action events | 0 observed |
| Wake false-event rate in this stream | 0.0/h observed |
| Action false-event rate in this stream | 0.0/h observed |
| Mean wake latency | 1.664 s |
| P95 wake latency | 2.000 s |
| Device | CUDA GPU |

The zero false-event counts are descriptive only. This clean positive-scene
stream does not replace a long held-out negative/open-set test; with 0 events
over 3.889 h, a rough one-sided 95% Poisson upper bound is about 0.77 events/h,
far above a production target such as 0.1/h.

## Per-speaker result

The matched sampler interleaves speaker rows so the first 1,000 scenes cover
all ten FSC test speakers rather than being dominated by the first few speakers
in lexicographic order.

| FSC speaker ID | Scenes | Wake recall | Task success |
|---|---:|---:|---:|
| 4BrX8aDqK2cLZRYl | 100 | 100% | 96% |
| 7B4XmNppyrCK977p | 100 | 100% | 99% |
| NgQEvO2x7Vh3xy2xz | 100 | 99% | 52% |
| Q4vMvpXkXBsqryvZ | 100 | 100% | 92% |
| V4ZbwLm9G5irobWn | 100 | 100% | 92% |
| V4ejqNL4xbUKkYrV | 100 | 100% | 98% |
| aokxBz9LxXHzZzay | 100 | 100% | 94% |
| k5bqyxx2lzIbrlg9 | 100 | 100% | 98% |
| oOK5kxoW7dskMbaK | 100 | 100% | 82% |
| ppymZZDb2Bf4NQnE | 100 | 100% | 98% |

The main remaining weakness is speaker-dependent intent robustness. There was
one wake miss; the other 99 task failures were after wake, comprising 62 cases
with no committed command event and 37 wrong intent actions. The weakest
intent groups included `change language|Chinese`, `change language|Korean`,
and bedroom-specific light/heat commands. This points to command capture and
intent-domain robustness, not a current clean wake-word recall problem.

## Comparison of clean decoder candidates

These runs use the same frozen component checkpoints and the same matched,
balanced 100-scene pilot unless noted.

| Candidate | Scenes | Wake recall | Task success | Notes |
|---|---:|---:|---:|---|
| Endpoint trim (`kws_trim`) | 100 | 100% | 90% | Baseline candidate |
| Endpoint trim with 5 s crop | 100 | 100% | 87% | Longer padding hurt; not retained |
| Confidence-buffered hybrid, 3 calls | 100 | 100% | 94% | Selected for confirmation |
| Confidence-buffered hybrid, balanced confirmation | 1,000 | 99.9% | **90.1%** | Current clean candidate |
| Endpoint-trim exploratory run, speaker-clustered | 1,000 | 100% | 84.8% | Not balanced; retained only as a diagnostic |

The previously cited 84.4% result came from a smaller cross-corpus proxy. It
should not be compared as if it were a real-driver test. The new benchmark
fixes the largest validity issue by pairing wake and command rows using the
same explicit FSC `speaker_id`, while still labeling the scenes synthetic.

## What was implemented from the new plan

### Benchmark/data changes

- Added an official kNN-VC utility to convert held-out `marvin` clips into the
  ten FSC test speakers:
  `scripts/data/build_knnvc_same_speaker_wakes.py`.
- Generated 200 converted wake clips (20 per FSC speaker) and a manifest:
  `manifests/clean_e2e_v2/wake_knnvc_same_speaker_20x10_test_2026-10-14.jsonl`.
- Added strict `matched_speaker` sampling to
  `scripts/eval/eval_continuous_full_pipeline_proxy.py`.
- Added a separate passenger/negative manifest so a target-positive manifest
  can be evaluated without inventing negative rows:
  `--passenger-manifest manifests/test/stage3_kws_cabin_test.jsonl`.
- Balanced the matched sampler by interleaving speakers and report unique
  source-row counts so repeated converted wakes are not mistaken for new
  speakers.

### Runtime/evaluation changes

- Used the existing command decoder candidate
  `confidence_buffered_hybrid`, with a three-call confidence buffer and
  commit threshold 1.0.
- Kept the TSE, presence, KWS, target-KWS, and intent checkpoints frozen.
- No checkpoint was overwritten and no model training was started in this
  run.

## Exact primary reproduction command

From `D:\Prawin_New\tse\car-tse-main`:

```powershell
python scripts/eval/eval_continuous_full_pipeline_proxy.py `
  --config configs/eval_end_to_end_clean_english_identity_targetkws_confidence_buffered_hybrid_run3.yaml `
  --wake-manifest manifests/clean_e2e_v2/wake_knnvc_same_speaker_20x10_test_2026-10-14.jsonl `
  --passenger-manifest manifests/test/stage3_kws_cabin_test.jsonl `
  --scene-sampling matched_speaker `
  --scenes 1000 `
  --clean `
  --cabin-snr-db 0 `
  --enrollment-source same_speaker_other `
  --hop-seconds 0.5 `
  --output logs/clean_e2e_v2_knnvc_matched_balanced_buffer3_1000_2026-10-15.json `
  --artifact-dir reports/artifacts/clean_e2e_v2_knnvc_matched_balanced_buffer3_1000_2026-10-15
```

Primary raw result:
`logs/clean_e2e_v2_knnvc_matched_balanced_buffer3_1000_2026-10-15.json`.

The report contains every scene, expected/predicted intent, speaker IDs,
source paths, wake/command event windows, confidence, latency, and all
effective thresholds.

## Audit of the implementation plan in `new info`

The plan is useful and directionally correct. Its four explanations for the
old 84.4% score were borne out to different degrees:

1. **Wrong speaker enrolled:** addressed for the clean proxy by converting the
   wake word to the command speaker and using a different utterance from that
   speaker for enrollment. This is synthetic identity consistency, not proof
   of natural same-driver continuity.
2. **Fixed post-wake command window:** the existing confidence-buffered hybrid
   improves the balanced pilot from 90% to 94% by delaying commitment across
   overlapping post-wake frames.
3. **Pre-truncated commands:** the old two-second assumption is no longer the
   primary clean configuration; the retained candidate uses the existing
   buffered decoder. The residual no-event errors show that true endpointed
   buffering still needs improvement.
4. **TSE artifacts:** still present. The frozen intent checkpoint was trained
   on TSE-domain data, but the 10-speaker spread shows that the current
   clean-domain cascade is not uniformly robust. Intent-domain adaptation is
   the next model-side experiment.

### WP status

| Work package | Status |
|---|---|
| WP0 benchmark | Complete for a 1,000-scene synthetic same-speaker proxy; real-driver data still missing |
| WP1 command capture | Partially complete; buffered decoder candidate works, but no true continuous endpoint/VAD buffer yet |
| WP2 intent adaptation | Not started in this run; frozen intent model already clears 90.1% on the new proxy |
| WP3 wake-word check | Clean matched proxy: 99.9% recall; open-set and real-cabin rejection remain unproven |
| WP4 acceptance | Proxy point and lower-bound targets met; production gates remain blocked |

## Data provenance and license notes

- Target wake source: held-out Google Speech Commands `marvin` rows already
  represented in the repository manifests.
- Command and enrollment source: held-out Fluent Speech Commands English test
  rows, with explicit speaker IDs.
- Voice conversion: official kNN-VC repository installed at
  `D:\datasets\knn-vc`; repository license is MIT. Its pretrained models and
  the underlying speech corpora retain their own terms and must be checked
  before redistribution.
- Generated audio is marked `synthetic: true` and `real_driver: false` in the
  generated manifest metadata.
- No consented driver recording, car microphone capture, passenger recording,
  or playback-reference channel was added.

## What this result does and does not establish

It establishes that the English cascade can exceed 90% on a reproducible clean
same-speaker *synthetic* benchmark without retraining the component models,
and that the buffered command decoder is a better clean candidate than the
simple endpoint crop.

It does not establish production reliability, real-cabin robustness, safe
false-accept performance, or broad speaker generalization. The next decisive
step remains a consented driver-disjoint pilot with the same driver's
enrollment, wake phrase, command, passenger negatives, silence, and mandatory
playback/echo-reference channels. After that, the intent adaptation should be
trained using a driver-disjoint tuning split and frozen before test evaluation.

## Cabin-noise transfer pilot

After the clean confirmation, the same frozen candidate was evaluated without
retraining on 100 matched synthetic scenes at each of -5 dB and 0 dB CAVEMOVE
cabin SNR, with passenger wake and passenger non-wake speech enabled. Across
200 rows:

| Metric | Result |
|---|---:|
| Wake recall | 45.5% |
| End-to-end task success | 31.0% |
| Intent accuracy given wake | 68.1% |
| False wake events | 0 observed |
| False action events | 0 observed |

Raw result:
`logs/noisy_e2e_v2_knnvc_matched_balanced_buffer3_pilot100_2026-10-15.json`.

This clean-to-cabin drop is the next improvement target. It indicates that
the clean decoder result should not be transferred directly to noisy release
claims; the next work should focus on cabin-domain Stage 1/TSE and KWS
robustness, then recalibrate the presence/target-speaker gates on a
driver-disjoint cabin set. The short synthetic positive-scene pilot is not a
false-accept-rate study.

## Dual-domain TSE comparison

As a controlled follow-up, the same 200 noisy rows were run with the retained
dual-domain Stage-1/TSE and cabin-trained KWS configuration
(`eval_end_to_end_cavemove_target_kws_dual_domain_stage2_hardneg_candidate.yaml`),
while retaining the three-call buffered command decoder. This substantially
improved positive detection but exposed a severe open-set safety failure:

| Metric | Clean-path candidate | Dual-domain cabin candidate |
|---|---:|---:|
| Wake recall | 45.5% | **99.5%** |
| End-to-end task success | 31.0% | **81.0%** |
| Intent accuracy given wake | 68.1% | 81.4% |
| False wake events | 0 | **254** |
| False wake rate | 0/h observed | **326.6/h** |
| False action events | 0 | **76** |
| False action rate | 0/h observed | **97.7/h** |

The dual-domain TSE is therefore the correct direction for recovering noisy
recall, but this operating point cannot be promoted. The next experiment is a
matched threshold/gate calibration using long negative cabin streams and
passenger-wake examples, followed by a regression on the positive rows. It is
not safe to optimize noisy task success alone.

## Long-negative threshold and speaker-gate calibration

The next calibration pass used all 2,110 files from the University of
Rochester Smart Speaker Command Dataset (3.193 hours). These are clean,
single-channel, non-Marvin studio recordings duplicated across the four input
channels; they are useful open-set diagnostics but are not real-cabin or
multi-microphone evidence.

The no-speaker-gate dual-domain candidate was unsafe at every useful KWS
threshold in the tested range: false wakes fell from 739.2/h at KWS 0.50 to
154.1/h at 0.75, and reached 0/h only at 0.80, with no positive-wake recall
measurement at that threshold. The split-gate candidate reduced the base
operating point to 94.9/h at KWS 0.50 and speaker threshold 0.24. Its separate
one-dimensional sweeps reached 9.1/h at KWS 0.75 and 8.8/h at speaker threshold
0.30; both are still far above a production target.

However, the speaker gate also over-suppressed the clean matched-speaker
positive proxy. On a 100-scene pilot with the split-gate configuration at KWS
0.50 and speaker threshold 0.24, wake recall was only 52.0% and end-to-end
task success was 41.0%. The gate therefore cannot be promoted merely because
it lowers Smart Speaker false wakes. Thresholds must be calibrated jointly on
driver-consistent positives, passenger wakes, and long negatives.

Raw logs:

- `logs/smart_speaker_command_negative_dualdomain_kws_sweep_2026-10-15.json`
- `logs/smart_speaker_command_negative_dualdomain_splitgate_sweep_2026-10-15.json`
- `logs/clean_e2e_v2_splitgate_base_pilot100_2026-10-15.json`

No checkpoints were changed. The current conclusion is that the safety gate
needs speaker-consistent training/calibration data, not just a higher fixed
threshold.

## Existing target-KWS candidate regression

Two previously trained target-KWS candidates were replayed on the new
100-scene clean matched-speaker pilot to avoid selecting from isolated cache
F1 alone. The larger hard-negative head reached only 10.0% wake recall and
2.0% end-to-end task success. The clean-identity cabin head reached 76.0%
wake recall and 33.0% task success. The retained no-gate clean candidate is
therefore still the strongest clean operating point; neither checkpoint was
promoted.

| Candidate | Wake recall | Task success | Decision |
|---|---:|---:|---|
| Retained clean E2E v2 candidate | 99.9% on 1,000 scenes | 90.1% | Retained for clean baseline |
| Large Smart Speaker hard-negative target KWS | 10.0% on 100-scene pilot | 2.0% | Rejected |
| Clean-identity cabin target KWS | 76.0% on 100-scene pilot | 33.0% | Rejected |

Raw logs:

- `logs/clean_e2e_v2_large_targetkws_pilot100_2026-10-15.json`
- `logs/clean_e2e_v2_cleanidentity_targetkws_pilot100_2026-10-15.json`

## Command-buffer follow-up

Increasing the confidence-buffer horizon from three to four post-wake
evaluations produced a small but reproducible clean improvement. On the full
1,000-scene benchmark, buffer four reached 906/1,000 task successes (90.6%),
with 999/1,000 wake recall and an approximate 88.6% Wilson lower bound. The
per-speaker weakness remained concentrated in `NgQEvO2x7Vh3xy2xz` (49/100),
so buffering does not solve the underlying speaker/domain mismatch.

The same sweep on the 200-row noisy dual-domain pilot exposed the safety
trade-off:

| Buffer calls | Task success | False wakes/hour | False actions/hour |
|---:|---:|---:|---:|
| 3 | 81.0% | 326.6 | 97.7 |
| 4 | 84.0% | 230.1 | 88.7 |
| 5 | 80.5% | 212.1 | 5.1 |

Buffer four is retained as the clean candidate because it improves task
success without changing checkpoints. Buffer five is not promoted: its lower
false-action rate is accompanied by lower task success and still an unsafe
false-wake rate. The noisy numbers remain synthetic and are not a substitute
for long real-cabin negative evaluation.

Raw logs:

- `logs/clean_e2e_v2_knnvc_matched_balanced_buffer4_1000_2026-10-15.json`
- `logs/noisy_e2e_v2_knnvc_matched_balanced_dualdomain_buffer4_pilot100_2026-10-15.json`
- `logs/noisy_e2e_v2_knnvc_matched_balanced_dualdomain_buffer5_pilot100_2026-10-15.json`

## LibriSpeech hard-negative target-KWS candidate

To test whether the open-set false-wake problem was primarily caused by too
little negative speech diversity, a new target-KWS candidate was initialized
from the retained hard-negative checkpoint and trained with an append-only
cache containing 36,000 LibriSpeech train-clean-100 negative windows, 7,200
validation windows, and 36,000 held-out test windows. The LibriSpeech speakers
were kept disjoint from the KWS positives and are clean read speech, not cabin
recordings. Existing checkpoints and caches were preserved.

The candidate trained for 12 epochs. Its best cached validation result was
0.493 F1 with 0.962 recall. On the 2,110-file Smart Speaker negative corpus
(3.193 hours), it produced 1 false wake at KWS threshold 0.50, or 0.31/h, and
0 events at thresholds 0.60–0.80. This is a large negative-side improvement
over the earlier dual-domain candidate, but it is not sufficient by itself:
the clean 1,000-scene matched-speaker E2E regression detected only 68.0% of
target wakes and produced 0/1,000 task successes. The candidate is therefore
rejected for promotion and retained only as an experiment artifact.

| Candidate | Smart Speaker FA rate | Clean wake recall | Clean task success | Decision |
|---|---:|---:|---:|---|
| Retained clean E2E v2 | 3.45/h on prior gate study | 99.9% | 90.6% | Retained baseline |
| LibriSpeech hard-negative KWS | 0.31/h at KWS 0.50 | 68.0% | 0.0% | Rejected |

The result is important: adding large amounts of generic clean speech can
reduce false wakes while destroying the voice/domain match needed for the
driver wake word. The next KWS training set must mix hard negatives with
speaker-consistent, TSE-processed positive wakes and use a joint positive/long
negative selection criterion. LibriSpeech is useful as an auxiliary negative
source, but it cannot be the dominant training signal.

A threshold-only rescue check on a separate 100-scene clean pilot increased
wake recall from 68.0% at KWS 0.45 to 89.0% at KWS 0.25, but still produced
0/100 task successes. Therefore the failure is not just a poorly chosen KWS
threshold; the candidate's target-speaker conditioning/gating behavior is
incompatible with the retained clean cascade.

Raw artifacts:

- `data/cache/stage3_target_kws_librispeech_negative_2026-10-15_10h2h10h_16k/`
- `data/cache/stage3_target_kws_gsc_marvin_librispeech_hardneg_2026-10-15_v2_16k/`
- `checkpoints/stage3_target_kws_gsc_marvin_librispeech_hardneg_2026-10-15_v2/best.pt`
- `logs/smart_speaker_command_negative_librispeech_hardneg_candidate_2026-10-15.json`
- `logs/clean_e2e_v2_knnvc_matched_librispeech_hardneg_buffer4_1000_2026-10-15.json`

## 2026-10-15 multi-speaker clean speech-only follow-up

To test passenger rejection without adding cabin or road noise, the retained
clean configuration was rerun on 1,000 synthetic scenes with matched target
speaker identities and a passenger speech track. The passenger track included
passenger wake-word and non-wake speech, while the cabin-noise component was
forced to zero (`--passenger-only`). This is a sequential multi-speaker proxy,
not a real continuous recording and not an overlapping-speech benchmark.

| Metric | Result |
|---|---:|
| Scenes / synthetic duration | 1,000 / 3.889 h |
| Target wake recall | 99.9% (999/1,000) |
| Overall target command accuracy | 90.2% (902/1,000) |
| Command accuracy conditioned on detected target wake | 90.29% |
| False wake events | 1,102 (283.37/h) |
| False action events | 99 (25.46/h) |
| Target-wake latency | 1.664 s mean, 2.000 s p95 |

The result is a confirmed safety gap: high target recall and command accuracy
coexist with a very high passenger/ non-target false-action rate. Therefore the
90.6% target-only clean result must not be described as multi-speaker or
passenger-safe. The run is useful for debugging the target-selection and event
association path, but it does not isolate TSE overlap suppression because the
passenger utterances are assembled sequentially rather than overlapping the
target command.

No checkpoint was changed. The run used CUDA, the retained `run3` checkpoint
set, and `confidence_buffered_hybrid` with buffer four. The complete raw result
and per-scene artifacts are preserved at:

- `logs/multispeaker_speech_only_e2e_v2_knnvc_matched_buffer4_1000_2026-10-15.json`
- `reports/artifacts/multispeaker_speech_only_e2e_v2_knnvc_matched_buffer4_1000_2026-10-15/`

The next engineering issue is to reduce passenger false actions while keeping
driver recall, then repeat the measurement with true simultaneous target and
passenger speech. This synthetic result remains `production_valid: false`.

## 2026-10-15 simultaneous target/passenger overlap

The next run placed a target wake at 1.0 s, a target command at 2.1 s, a
passenger `Marvin` at 2.4 s, and passenger non-wake speech at 3.2 s. The
passenger tracks therefore overlap the target command; cabin/road noise was
disabled. All four synthetic microphone channels receive the same summed
waveform, so this is a target-conditioned co-channel stress test rather than a
spatial-array separation benchmark.

The retained checkpoint set produced the following result on 1,000 matched
speaker scenes:

| Metric | Retained baseline |
|---|---:|
| Scenes / synthetic duration | 1,000 / 3.889 h |
| Target wake recall | 100.0% |
| Overall target command accuracy | 45.3% |
| Legacy outside-target counters (3 s / 3 s support tolerance) | 0 wake / 0 action |
| Strict interval false-wake support | 802 events (206.23/h) |
| Mean / p95 target-wake latency | 1.702 s / 2.525 s |

The outside-target counters are not a passenger-safety result in this
condition. The target command interval and passenger intervals overlap, so
many emitted command windows are inherently ambiguous. Offline analysis of the
raw log found 802 wake windows outside the exact target-wake interval and 766
command windows that simultaneously supported the target
command and passenger activity (196.97 ambiguous command-window supports per
hour). The evaluator now records these contaminated/ambiguous windows
explicitly rather than presenting the legacy outside-target counter as a
false-action rate.

The 128-scene Stage-1 bypass control also scored 45.31% with 100% wake recall.
This indicates that the first failure is not explained solely by the learned
extractor removing the command; the downstream intent/state path has not been
trained for co-channel overlap, and the current three-second support windows
are contaminated by the passenger tracks.

Raw overlap artifacts:

- `logs/multispeaker_overlap_speech_only_e2e_v2_knnvc_matched_buffer4_1000_2026-10-15.json`
- `logs/multispeaker_overlap_bypass_stage1_128_2026-10-15.json`
- `reports/artifacts/multispeaker_overlap_speech_only_e2e_v2_knnvc_matched_buffer4_1000_2026-10-15/`

## 2026-10-15 overlap-aware intent candidate

An append-only Stage-4 candidate was initialized from the retained Wav2Vec
intent head/encoder. Its cache contains 2,048 clean and 2,048 overlap train
examples, plus 512 clean and 512 overlap validation examples. Overlap samples
mix two held-out KWS speech pools into the frozen Stage-1 intent cache at 0 dB
passenger SNR. The candidate trained for four epochs; its best mixed
validation accuracy was 73.8% with macro-F1 0.741.

On a 128-scene pilot using the same held-out matched-speaker scenes, the
candidate reached 57.03% overlap command accuracy (up from 45.31%, +11.72
points) with 100% target wake recall. Its clean control reached 94.53% intent
accuracy with 100% wake recall. These are encouraging adaptation results but
not a promotion: the overlap score remains far below the 90% research target,
the scenes are synthetic, and no real-driver or spatial-array evidence exists.

Candidate artifacts:

- `scripts/data/build_overlap_intent_cache.py`
- `configs/stage4_intent_overlap_wav2vec_ft1.yaml`
- `configs/eval_end_to_end_clean_english_overlap_intent_ft1.yaml`
- `data/cache/stage4_intent_overlap_ft1_16k/`
- `checkpoints/stage4_intent_wav2vec_overlap_ft1_2026-10-15/`
- `logs/clean_overlap_intent_ft1_128_2026-10-15.json`
- `logs/multispeaker_overlap_intent_ft1_128_2026-10-15.json`

The next decision is held out until the 1,000-scene candidate regression and
the clean regression complete. The retained baseline remains the reference
configuration.

## Candidate regression status

The planned 1,000-scene overlap candidate regression was interrupted before a
JSON result was written. A smaller follow-up was subsequently canceled before
completion, also without a result file. Neither run contributes a score or
promotion decision. Training and evaluation are currently paused; the
retained baseline remains the reference configuration until an explicitly
authorized held-out regression is completed.
