# Production-readiness gates

These gates are the pre-declared acceptance criteria for the in-car voice
assistant. They are intentionally stricter than component benchmark scores.
Exact numeric targets should be finalized on the tuning split before opening
the held-out test; the rows below define the required evidence and minimum
reporting structure.

| Gate | Required evidence | Current status |
|---|---|---|
| Same-driver continuity | Enrollment, driver Marvin, and driver command in one continuous recording | Blocked: 0 sessions |
| Passenger rejection | Passenger Marvin, passenger speech, and command-like negatives | Blocked: 0 sessions |
| Study size | At least 3 drivers and 3 complete sessions, with non-empty tuning and held-out test groups | Blocked: 0 drivers / 0 sessions |
| Capture hardware match | Pilot audio uses the configured four-microphone cabin layout | Blocked: no four-channel input detected |
| Reproducible release state | Clean source revision plus hashed config/checkpoint manifest archived together | Blocked: working tree contains changes |
| No-command safety | Noise-only, ordinary speech, playback/echo, and silence exposure | Partial: AISHELL target-absent only |
| Frozen split | Speaker/session-disjoint tuning and held-out test | Blocked |
| Threshold freeze | Presence, command-presence, KWS, speaker-gate, and intent-confidence settings selected only on tuning sessions, subject to both false-accept and false-action limits | Prepared in evaluator; blocked by 0 sessions |
| Wake recall | Held-out driver wake recall >= 90% by cabin/noise condition | Proxy only |
| Command accuracy | Held-out correct intent/slots >= 90% after a correctly detected driver wake | Proxy only |
| False accepts | Conservative unmatched-wake rate <= 1.0/hour; tune and freeze before test | Proxy/target-absent only |
| False actions | Actions during ineligible intervals <= 1.0/hour | Proxy only |
| Latency | p50 and p95 wake-to-action plus end-to-end wall-clock timing | Partial; proxy window latency only |
| Resource budget | CPU/GPU latency, memory, model size, and streaming stability | Partial |
| Regression | Candidate must not regress accepted-domain TSE, intent, or rejection metrics | Required for every candidate |

## Required test protocol

Thresholds, smoothing, speaker gate, intent-confidence gate, and
command-capture settings are selected only with tuning sessions. The held-out
test is then run without changing settings. Results must include denominators, exposure hours, confidence
intervals where sample size permits, condition breakdowns, and raw event logs.

## Release decision

If any safety gate fails, the system remains research/engineering-preview
only, regardless of clean intent or isolated wake-word scores.

The evaluator threshold search is safety-constrained on both conservative
unmatched-wake false accepts/hour and false actions/hour. This prevents a
high-recall point that satisfies only the action constraint from being
selected.

## Rate-denominator audit (2026-10-11)

The real-driver evaluator was corrected so the safety decision uses
`false_accepts_per_negative_hour` and `false_actions_per_negative_hour`, where
the denominator is the union of labeled passenger/no-command/noise/playback
intervals. The historical `false_accepts_per_hour` and
`false_actions_per_hour` fields remain in the JSON as total-recording rates for
backward comparison. This distinction matters when a session contains both
positive command material and negative exposure: dividing by full-session
duration can understate the rate over the intervals where an unwanted action
could occur.

The evaluator records both denominators, uses the negative-exposure rates for
tuning selection and held-out production gates, and falls back to the legacy
fields only for older synthetic test summaries that do not contain exposure
metadata. It now also emits 95% Wilson intervals for wake/command proportions
and 95% Poisson-rate intervals for false accepts/actions, including the
condition breakdown. The empty pilot recheck still fails before scoring because
there are zero permitted sessions.

Artifacts:

- `scripts/eval/eval_real_driver_sessions.py`
- `tests/test_real_driver_session_validation.py`
- `logs/real_driver_pipeline_evaluation_ratefix_empty.json`
- `logs/production_readiness_snapshot_ratefix.json`

## Candidate promotion audit (2026-10-11)

The Stage 1 ft2 checkpoint improved the isolated dual-domain regression from
6.432 dB to 8.496 dB mean SI-SDR improvement overall (+4.014 dB on AISHELL-5,
+0.115 dB on CAVEMOVE), but its downstream handoff was worse: the existing
heads produced 62.5% wake recall and 43.8% overall intent on the 32-scene
proxy. Retraining target-KWS and presence against ft2 outputs did not recover
the drop. The ft2 branch is not promoted.

The existing ft1 positive-enriched target-KWS candidate reached 78.1% wake
recall, 68.8% overall intent, and 88.0% conditioned intent on the 32-scene
proxy with zero proxy false actions. However, the 3.889-hour target-absent
AISHELL-5 Eval2 replay produced 9.256 full-path events/hour at KWS 0.50 (and
147.323 KWS-only events/hour). At 0.55/0.60 it produced no events in that
negative exposure, but the full 32-scene positive replay at 0.55 produced
0.0% wake recall and 0.0% overall intent. It is rejected as a production
candidate.

The retained diagnostic reference is the negative-only ft1-aligned target-KWS
configuration: presence 0.15, KWS 0.50 with 2-of-2 smoothing, wake speaker
similarity 0.24, post-wake command similarity 0.22, and intent confidence
0.53. It produced 68.75% wake recall, 59.375% overall intent, 86.364%
conditioned intent, and 0 proxy false actions/hour on the 32-scene matrix,
plus 4.628 full-path events/hour over the 3.889-hour target-absent AISHELL
diagnostic. This is the safer reference, not a release claim.

The all-session AISHELL replay is still not a real-driver FAR: the enrollment
speaker is absent, there are no passenger-wake labels or command/action
labels, and the recordings are not continuous sessions captured for this
assistant. Thresholds remain unfrozen until a permitted real-driver tuning
split exists.

## Runtime/resource follow-up (2026-10-10)

The full loaded pipeline was benchmarked with the same configuration rather
than only timing isolated components. In a clean RTX 4080 rerun, the current
fp32 Wav2Vec candidate measured approximately 48.2 ms wake-stage p50 / 51.6
ms p95 and 62.6 ms post-wake command p50 / 64.4 ms p95 for a 3-second,
4-channel context. The earlier GPU run was made under AISHELL evaluator
contention and is retained in the research log as historical context. A CPU
check measured approximately 522.2 ms wake-stage p50 and 641.7 ms
command-stage p50. The model bundle is approximately 424.5 MB, with 106.0M
parameters. These are compute measurements on synthetic random tensors, not
quality scores; the clean GPU timing is within the project’s sub-100-ms
engineering target, but the resource footprint and evidence gates remain
unresolved.

FP16 autocast was tested and was slower on this stack (approximately 153.0 ms
wake p50 and 190.3 ms command p50), so it was rejected. The first compact CRNN
intent alternative reduced stored checkpoints to approximately 75.1 MB, but
its same-speaker proxy fell to 68.75% target wake recall and 56.25% overall
intent accuracy. A separate normalized/logit-distilled CRNN reaches 80.5%
clean FSC top-1 and 56.3% TSE-domain top-1; at an exploratory 0.35
intent-confidence threshold it matches the Wav2Vec candidate on the 32-scene
proxy (59.4% overall intent, 86.4% conditioned on wake) with 75.1 MB of
stored checkpoints and 18.68M parameters. It is retained as an edge candidate,
not promoted, because direct TSE-domain quality remains lower and no
real-driver tuning/test set exists.

The all-session KWS threshold replay showed that raising KWS from 0.50 to
0.55/0.60 reduces KWS-only target-absent events from 96.673/hour to
60.420/39.080/hour, but positive proxy wake recall drops from 68.75% to
62.50%. The default 0.50 remains the quality reference until a permitted
driver tuning split can select the safety operating point.

Machine-readable artifacts:

- `logs/full_pipeline_latency_gpu_2026-10-11.json`
- `logs/full_pipeline_latency_cpu_2026-10-11.json`
- `logs/full_pipeline_latency_gpu_fp16_2026-10-11.json`
- `logs/full_pipeline_latency_cpu_int8_2026-10-11.json`
- `logs/full_pipeline_latency_crnn_intent_gpu_2026-10-11.json`
- `logs/full_pipeline_latency_norm_crnn_distill_t3_a05_gpu_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_crnn_intent_candidate_same_speaker_m10_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_norm_crnn_distill_t3_a05_conf035_same_speaker_m10_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_wav2vec_reference_same_speaker_m10_32scenes_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_wav2vec_reference_kws055_same_speaker_m10_32scenes_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_wav2vec_reference_kws060_same_speaker_m10_32scenes_2026-10-11.json`
- `logs/aishell5_continuous_targetkws_hardneg_aishell5_ft1_p015_sg024_kws055_sweep_eval2_all18_2026-10-11.json`
- `logs/production_readiness_snapshot_2026-10-11.json`
- `reports/production_readiness_snapshot_2026-10-11.md`
- `reports/dataset_audit_2026-10-10.md`
- `logs/aishell5_continuous_targetkws_hardneg_aishell5_ft1_p015_sg024_kws050_smooth22_eval2_all18_2026-10-10.json`
- `logs/stage1_dual_domain_regression_ft2_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_ft2_stage1_consistent_negative_same_speaker_m10_32scenes_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_ft2_stage1_consistent_pos_same_speaker_m10_32scenes_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_ft2_stage1_consistent_pos_presence_same_speaker_m10_32scenes_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_ft1_positive_fullmatrix_same_speaker_m10_32scenes_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_ft1_positive_fullmatrix_kws055_same_speaker_m10_32scenes_2026-10-11.json`
- `logs/aishell5_continuous_targetkws_hardneg_aishell5_pos_ft1_p015_sg022_kws050_sweep_eval2_all18_2026-10-11.json`

## Evidence-contract hardening (2026-10-11)

The real-driver validator was tightened before pilot collection. A session
must now contain non-empty consent-record metadata, a directory-matching
session ID, a non-empty driver ID, valid 16 kHz/channel-matched audio, all
required event classes, and an ordered enrollment -> driver wake -> driver
command sequence. This is a data-integrity gate; it does not make synthetic
proxy results count as real-driver evidence. The current session root is empty,
so validation remains correctly invalid.

Unit coverage: 18 tests passed and 1 skipped after adding acceptance and
rejection cases for this contract.

Artifacts:

- `scripts/data/validate_real_driver_sessions.py`
- `tests/test_real_driver_session_validation.py`
- `logs/real_driver_validation_latest_2026-10-11.json`

The evaluator now records explicit held-out gates: minimum 90% driver-wake
recall, minimum 90% command accuracy, maximum 1.0 conservative unmatched-wake
false accept/hour, and maximum 1.0 false action/hour by default. The values
are configurable only before the held-out test is opened and are written into
the evaluation report. It also refuses to mark a run production-valid without
the minimum three-driver/three-session study structure.

Artifacts:

- `scripts/eval/eval_real_driver_sessions.py`
- `reports/real_driver_pilot_protocol_2026-10-09.md`
- `reports/capture_hardware_preflight_2026-10-11.md`
