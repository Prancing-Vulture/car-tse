# Audit of the earlier pipeline-problem note

Date: 2026-10-14

This audit checks the problem list in the earlier handoff note against the
current English-only workspace. It distinguishes code/reporting corrections
from limitations that still require data or hardware. Existing historical
artifacts were not rewritten; where an old result used a pre-fix evaluator it
should be treated as stale.

## Confirmed test, data, and reporting problems

| # | Status now | Current conclusion |
|---:|---|---|
| 1 | Fixed for new runs | `eval_stage2_presence.py` now passes the configured Stage-1 backend, fails on a missing requested manifest, and only permits the legacy generated validation split through explicit `--allow-generated-val`. Pre-fix Stage-2 logs remain stale. |
| 2 | Fixed for new runs | Official Fluent Speech Commands validation/test manifests exist, and the wav2vec evaluator records `split=test` plus the actual manifest. The corrected official test artifact is `logs/stage4_intent_wav2vec_ft8_resume_tse_domain/metrics_official_test_v2.json`. |
| 3 | Fixed for the retained KWS path | Runtime-matched one-second crops and peak conditioning are used by the corrected target-KWS cache/evaluations. Older KWS results made with the scale/crop mismatch are not directly comparable. |
| 4 | Corrected in the AISHELL audit path; broader regression remains | Fixed-crop offsets, labeled-region seeking, and microphone-count handling were corrected for the AISHELL audit. This does not prove every historical Stage-1 manifest or every external loader path is clean. |
| 5 | Fixed in later reporting | The later target-KWS reports identify the actual checkpoint paths. Any old comparison with mislabeled checkpoint names must be ignored. |
| 6 | Fixed | Continuous proxy event association uses interval overlap (`interval_overlap_v2`) and records window start/end. `tests/test_proxy_event_association.py` covers a boundary-crossing event. |
| 7 | Superseded, not retroactively repaired | The v9 command-capture timing figures remain invalid. The v10 trace and later interval-overlap artifacts are the authoritative diagnostics; v9 numbers must not be cited. |
| 8 | Still open; metadata guard strengthened | Public Marvin wake and FSC command clips are not verified same-speaker recordings. New proxy output now derives `speaker_consistent` from wake/command identities and reports `false` for the current cross-corpus setup, even when enrollment uses another utterance from the command speaker. |
| 9 | Still open | The proxy remains a small engineering probe, not a production FAR study. The latest clean extension is 64 scenes / 896 seconds; the noisy extension is 32 scene/SNR rows / 896 seconds. Hourly rates from these short probes are diagnostic only. |
| 10 | Fixed for new training runs | Stage 2/3 training now writes run metadata and per-epoch JSONL histories. Missing historical W&B histories cannot be reconstructed. |
| 11 | Partially resolved | A compatible hard-negative target-KWS cache and candidate now exist, but the originally expected manifest path is still not a universal input. Any new experiment must name its exact cache and manifest in the run metadata. |

## Performance weaknesses: current status

- Presence remains a weak calibrated component. The current English clean-identity
  Stage-2 log reports frame F1 0.522 and frame FAR 0.221 at threshold 0.50;
  chunk metrics are unstable because the validation set has only eight negative
  chunks (`chunk_at_0.5` F1 0.582, FAR 0.970). These are stage metrics, not
  end-to-end production rates.
- Wake detection still has an open-set false-accept problem. After excluding
  invalid zero-padded tails, the retained hard-negative target-KWS path
  produced 76 false wakes over 3.193 hours of Smart Speaker negatives
  (23.81/hour). The larger cached-F1 candidate was rejected after cross-corpus
  and negative-subset transfer checks. The earlier 48.55/hour value included
  the old tail-window evaluator and is superseded.
- No current wake threshold is release-ready. The 0.45 setting is retained as
  a diagnostic operating point. Earlier 0.50/0.55 threshold sweeps were run
  with the partial-tail scorer and must be replayed under the full-context
  policy before they can support a threshold decision.
- Clean full-command scoring improved after the duration correction: 64 clean
  proxy scenes reached 100% wake recall and 100% intent accuracy, with one
  event-level false wake and zero false actions. This is not same-driver
  evidence because the wake and command speakers remain cross-corpus.
- Noisy CAVEMOVE transfer remains the main measured model weakness: the current
  full-command, full-context proxy reached 75.0% wake recall, 70.3% overall
  intent, 93.8% intent given wake, and zero observed false wakes in 896 seconds.
  The earlier 84.4/hour value was entirely caused by scoring a zero-padded tail
  and is superseded. The zero is still only a short proxy observation, not a
  production FAR bound.
- Stage-1 domain transfer remains a research gap. The retained English clean
  candidate improved AISHELL metrics only modestly and must not be presented as
  a real-cabin or target-chip result.

## What is still required before a production claim

The decisive missing evidence is a consented, speaker-consistent recording in a
vehicle or equivalent cabin: the same driver must provide enrollment, Marvin,
and the command, with passenger Marvin, passenger speech, no-command periods,
vehicle noise, and playback/echo where applicable. Tuning and final-test
drivers/sessions must be disjoint. The workstation preflight found no device
supporting the required four- or six-channel capture contract, so this cannot
be closed by another synthetic proxy run.

## Verification performed in this audit

- Corrected component latency benchmark now honors the configured 3.25-second
  intent window (`logs/component_latency_clean_longcommand_3p25_2026-10-14.json`).
- A post-fix clean proxy run records `speaker_consistent=false` and
  `passenger_wake_ground_truth_present=false` for the clean control:
  `logs/clean_proxy_speaker_provenance_fix_final_2026-10-14.json`.
- Full post-fix clean and noisy proxy reruns preserve the prior scores while
  correcting the provenance metadata:
  `logs/clean_interval_overlap_endpoint_trim_fullcommand_64scenes_v2_provenance_2026-10-14.json`
  and
  `logs/noisy_interval_overlap_endpoint_trim_fullcommand_32scenes_v2_provenance_2026-10-14.json`.
- A second rerun corrected the shared continuous window policy so incomplete
  zero-padded tails are not scored. The v3 artifacts report the same clean
  score and the same noisy wake/intent score, but noisy false wakes change from
  21 to 0. The Smart Speaker hard-negative corpus was also rerun: 76 false
  wakes over 3.193 hours, or 23.81/hour:
  `logs/clean_interval_overlap_endpoint_trim_fullcommand_64scenes_v3_fullwindows_2026-10-14.json`,
  `logs/noisy_interval_overlap_endpoint_trim_fullcommand_32scenes_v3_fullwindows_2026-10-14.json`,
  and
  `logs/smart_speaker_command_negative_endpoint08_hardneg_nogate_fullwindows_v2_2026-10-14.json`.
- The real-driver evaluator now also associates labeled events by model-window
  overlap, records `window_end_seconds`, and tests the boundary case. This
  removes the same centre-time ambiguity from future permitted-session runs;
  it does not create any real-driver data.
- The refreshed readiness snapshot remains `research_engineering_preview` with
  `production_ready=false`; it records zero permitted sessions and blocked
  same-driver, hardware, split, and threshold gates:
  `reports/production_readiness_snapshot_current_2026-10-14_v4.md`.
- Regression suite: **95 passed, 1 skipped**.

## New open-set threshold sweep

The Smart Speaker negative corpus was replayed at six KWS thresholds using one
shared full-context forward pass. At threshold 0.50 the corrected full-path
rate is 23.81 events/hour over 3.193 hours; at 0.55 and 0.60 it is zero on this
corpus. This does not make either threshold production-safe: the noisy proxy
at 0.55 loses all target wakes, and the Smart Speaker clips are clean,
single-channel studio negatives. The complete result and reproduction command
are in `reports/smart_speaker_threshold_sweep_2026-10-14.md`.

The compact cabin+Smart speaker-verifier candidate was then rechecked after an
additional evaluator audit. The earlier apparent 0.00/hour result was a
backend-attribution error: the batched scorer had not received the loaded
compact verifier and could silently use the SpEx scorer. With the actual
compact backend, the complete 3.193-hour Smart Speaker corpus produced 6
full-path events at KWS 0.50 (1.88/hour) and 494 events at KWS 0.46
(154.74/hour) with gate 0.30. A gate-threshold sweep at KWS 0.46 remained
24.12/hour even at gate 0.80. The compact candidate is therefore not
promoted. The correction and raw logs are documented in
`reports/compact_gate_backend_correction_2026-10-14.md`.

