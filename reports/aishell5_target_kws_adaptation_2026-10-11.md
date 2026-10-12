# AISHELL-5 target-speaker KWS adaptation and safety freeze

Status: bounded negative-only adaptation and proxy safety study; not a
real-driver evaluation and not a production release.

## Objective

The target-speaker wake head had a cross-corpus calibration problem: the
generic KWS baseline produced 0 smoothed events/hour on a 120-second
target-absent AISHELL-5 slice, while the original target-speaker head produced
630/hour at threshold 0.50. This study adds permitted AISHELL-5 cabin speech
as a hard-negative domain and checks whether the full gated path rejects a
longer real-cabin target-absent exposure without sacrificing the existing
positive proxy.

## Data separation

- AISHELL-5 Train, Dev, and Eval1 were used to construct the adaptation caches;
  each split contains 768 real four-channel cabin windows.
- AISHELL-5 Eval2 was kept out of cache construction and used only for the
  final continuous negative exposure: 1,200 seconds total across sessions 001
  and 002, with 85.5% speech occupancy.
- The fixed LibriSpeech enrollment speaker is absent from AISHELL-5 Eval2.
  Therefore Eval2 measures rejection only; it contains no positive driver wake
  or command examples.
- The synthetic target-present variant places speaker-consistent Marvin
  wake/enrollment clips into AISHELL cabin backgrounds at target SNRs from
  -5 to +10 dB. It is useful for controlled proxy recall, not real-driver
  evidence.

## Checkpoints and validation

| Variant | Initialization/data | Validation result | Decision |
|---|---|---:|---|
| Negative-only AISHELL adaptation | Prior hard-negative target-KWS checkpoint + 768 AISHELL negatives/split | F1 45.6% at threshold 0.54; F1 44.9% at 0.50 | Selected for negative rejection with runtime gates |
| AISHELL target-present adaptation | Negative-only checkpoint + 768 synthetic target-present AISHELL rows/split | F1 78.6% at threshold 0.50 | Retained experiment; threshold 0.55 rejected the positive proxy |

The standalone validation F1 is not the selection criterion. The selected
negative-only head is chosen because it improves the held-out real-cabin
negative exposure when combined with presence and speaker gates. The
target-present adaptation demonstrates that a higher offline F1 does not
guarantee a better cross-corpus operating point.

## Eval2 target-absent exposure

| Path | Presence | Speaker gate | KWS-only smoothed events/hour | Full-path smoothed events/hour |
|---|---:|---:|---:|---:|
| Adapted head, command gate 0.22 | 0.15 | 0.22 | 60.0 | 3.0 |
| Adapted head, split gate | 0.15 | 0.24 wake / 0.22 command | 60.0 | **3.0** |
| Adapted + prior ensemble | 0.15 | 0.22 | 60.0 | 3.0 |

The split-gate candidate is the selected diagnostic path. It rejects all but
one smoothed false wake in 1,200 seconds of speech-rich real-cabin
target-absent audio, while the KWS-only rate remains 60/hour. An earlier
0/hour report used 3-window/2-hit smoothing rather than the runtime's frozen
2-window/2-hit setting and is superseded. The corrected result supports the
combined safety cascade, not a claim that the wake head alone is solved.

## Positive proxy check

The selected split-gate candidate was replayed on eight cross-corpus scenes
(112 seconds per passenger condition) using a different utterance from the
command speaker as enrollment. It reached:

| Passenger SNR | Wake recall | Intent accuracy | False wakes/hour | False actions/hour | Mean latency |
|---:|---:|---:|---:|---:|---:|
| -10 dB | 100% | 100% | 0.0 | 0.0 | 2.5 s |
| 0 dB | 100% | 100% | 0.0 | 0.0 | 2.5 s |
| +10 dB | 100% | 100% | 0.0 | 0.0 | 2.5 s |

These are cross-corpus proxy values. The Marvin wake recording and FSC command
recording are not verified to be the same speaker, and no real driver supplied
the enrollment/wake/command sequence.

The target-present AISHELL adaptation retained the proxy at KWS threshold
0.50 but collapsed to 0% wake recall at 0.55. Three-hit temporal confirmation
also reduced the proxy to 37.5% wake recall and 12.5% intent accuracy. Neither
variant is promoted.

## Hard-negative mining retries

To test whether the remaining AISHELL false accepts could be removed without
losing positive behavior, the top-scoring AISHELL negative windows were mined
from the existing negative cache and added to the training split.

| Variant | Initialization | Validation | Same-speaker proxy | Decision |
|---|---|---:|---:|---|
| Mined v2 | negative-only adapted head | F1 46.6% at 0.62 | 62.5% wake / 53.1% intent at 0.50; 75.0% / 65.6% at 0.40 | Rejected |
| Mined v3 | positive AISHELL-adapted head | F1 44.0% at 0.49 | 68.8% wake / 59.4% intent at 0.49 | Rejected |

Mined v2 reduced the KWS-only AISHELL rate to 48/hour at threshold 0.50,
but both retries lost too much positive-proxy recall relative to the reference
model. They remain reproducibility artifacts and are not in the selected
pipeline.

## Narrow speaker-gate replay

The exact runtime reference uses a 0.24 pre-wake speaker gate and 0.22 command
gate. Raising only the pre-wake gate removed the one observed full-path
AISHELL event, but the positive tradeoff was unfavorable:

| Pre-wake gate | AISHELL full-path false wakes/hour | Harsh proxy wake recall | Harsh proxy intent accuracy | Decision |
|---:|---:|---:|---:|---|
| 0.240 | 3.0 | 100.0% | 100.0% | Reference |
| 0.245 | 0.0 | 65.6% | 50.0% | Rejected |
| 0.250 | 0.0 | 62.5% | 46.9% | Rejected |

This demonstrates why the 0/hour negative-only result cannot be promoted by
itself: a tiny threshold increase can remove the observed negative event while
also rejecting genuine target windows in the cross-corpus proxy.

## Corrected full-model transfer check

The trainer initially loaded every initializer into the acoustic submodule.
That is correct for an ordinary KWS checkpoint, but target-speaker checkpoints
store the acoustic parameters under `audio.*` and the enrollment relation head
at the model root. The old path therefore silently loaded no target-head
weights when asked to initialize from a target-speaker checkpoint. An explicit
full-model warm-start option was added and verified by the v4 run. The
corrected v4 checkpoint reached validation F1 42.2% at threshold 0.50 and only
62.5% wake / 53.1% intent on the harsh positive proxy, so it is rejected. The
code fix remains because future transfer experiments would otherwise be
misleading.

Two control runs separated the result from the mined-cache composition:

- A corrected full transfer from the pre-AISHELL target head reached validation
  F1 46.5% at threshold 0.51 and 62.5% wake / 53.1% intent on the harsh proxy.
- A classifier-only adaptation with the acoustic branch frozen reached
  validation F1 46.9% at threshold 0.52 and the same 62.5% / 53.1% proxy
  result.

Both controls are rejected. The unchanged negative-only AISHELL-adapted head
remains the reference because it is the only tested variant that retains the
100% positive-proxy result at the selected threshold.

## Decision

Keep `stage3_target_kws_gsc_marvin_hardneg_aishell5_ft1` and the split-gate
configuration as research candidates for permitted real-driver tuning. Do not
freeze their thresholds or call the 3/hour Eval2 result a production FAR.
The next decisive evidence remains a consented same-driver continuous pilot
with speaker-disjoint tuning/test sessions, passenger wake negatives,
passenger speech, no-command intervals, and labeled response times.

## Reproducibility artifacts

- `scripts/data/build_target_kws_aishell5_negative_cache.py`
- `scripts/data/build_target_kws_aishell5_positive_cache.py`
- `scripts/eval/calibrate_target_kws_threshold.py`
- `configs/stage3_target_kws_gsc_marvin_hardneg_aishell5.yaml`
- `configs/stage3_target_kws_gsc_marvin_hardneg_aishell5_pos.yaml`
- `configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_hardneg_aishell5_neg_safety_splitgate_candidate.yaml`
- `checkpoints/stage3_target_kws_gsc_marvin_hardneg_aishell5_ft1/best.pt`
- `checkpoints/stage3_target_kws_gsc_marvin_hardneg_aishell5_pos_ft1/best.pt`
- `configs/stage3_target_kws_gsc_marvin_hardneg_aishell5_mined_v2.yaml`
- `configs/stage3_target_kws_gsc_marvin_hardneg_aishell5_mined_v3_from_pos.yaml`
- `checkpoints/stage3_target_kws_gsc_marvin_hardneg_aishell5_mined_v2_ft1/best.pt`
- `checkpoints/stage3_target_kws_gsc_marvin_hardneg_aishell5_mined_v3_from_pos_ft1/best.pt`
- `configs/stage3_target_kws_gsc_marvin_hardneg_aishell5_v2_full_from_target.yaml`
- `configs/stage3_target_kws_gsc_marvin_hardneg_aishell5_v3_classifier_only.yaml`
- `checkpoints/stage3_target_kws_gsc_marvin_hardneg_aishell5_v2_full_from_target_ft1/best.pt`
- `checkpoints/stage3_target_kws_gsc_marvin_hardneg_aishell5_v3_classifier_only_ft1/best.pt`
- `logs/aishell5_continuous_targetkws_hardneg_aishell5_ft1_p015_sg024_kws050_smooth22_1200s_2026-10-11.json`
- `logs/aishell5_continuous_targetkws_hardneg_aishell5_mined_v2_p015_sg024_smooth22_kws_sweep_1200s_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_hardneg_aishell5_v2_full_from_target_p015_sg024_kws050_same_speaker_m10_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_hardneg_aishell5_v3_classifier_only_p015_sg024_kws052_same_speaker_m10_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_hardneg_aishell5_neg_splitgate_p015_wake024_cmd022_same_speaker_m10_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_hardneg_aishell5_neg_splitgate_p015_wake024_cmd022_same_speaker_p0_2026-10-10.json`
- `logs/continuous_full_pipeline_proxy_hardneg_aishell5_neg_splitgate_p015_wake024_cmd022_same_speaker_p10_2026-10-11.json`

## All-session Eval2 negative exposure

After the shorter 1,200-second check, the same frozen candidate was replayed
over all 18 AISHELL-5 Eval2 sessions. The exposure totaled 14,001.9 seconds
(3.889 hours) with 80.999% speech occupancy. Under the exact runtime
2-window/2-hit smoothing, it produced 18 full-path smoothed events:
4.628 full-path false wakes/hour. KWS-only smoothed events totaled 376, or
96.673/hour. This longer aggregate supersedes the shorter slice for future
negative-domain comparisons, while preserving the 3.0/hour value as a
separately documented 1,200-second result. It remains target-absent,
cross-corpus evidence and is not a real-driver FAR or production threshold
freeze.

Artifact: `logs/aishell5_continuous_targetkws_hardneg_aishell5_ft1_p015_sg024_kws050_smooth22_eval2_all18_2026-10-10.json`.
