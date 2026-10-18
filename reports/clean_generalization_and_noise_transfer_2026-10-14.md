# Clean generalization and noise-transfer follow-up — 2026-10-14

Status: diagnostic evidence only; no checkpoint or retained release config was
overwritten.

## Question

The earlier 32-scene clean proxy made a confidence-buffered command decoder
look nearly perfect. This follow-up checks the same policy on a larger set and
then transfers it to matched CAVEMOVE cabin noise.

The evaluator's `--scenes 64` creates 64 target wake groups and runs each at
the two default SNR labels (`-5` and `0` dB), so each report contains 128
scene rows. With `--clean`, both rows are actually noise-free; the SNR labels
are only duplicated condition labels. The wake clip and command clip still
come from different public corpora, so these are not same-driver recordings.

## Operating point

- Stage 1: `stage1_spex_english_clean_identity_ft1`
- KWS: clean-control KWS plus retained target-KWS
- Intent: retained Wav2Vec FSC model
- Command strategy: `confidence_buffered_hybrid`
- Hop: 1.0 s
- Candidate buffer: 2 calls
- Commit confidence: 0.90
- Enrollment: wake clip unless noted

The two-call/0.90 policy was selected only as a diagnostic because it was the
least aggressive buffered setting that improved the earlier clean proxy. It is
not a frozen threshold.

## Results

| Evaluation | Rows | Wake recall | Intent overall | Intent given wake | False wakes/h | False actions/h |
|---|---:|---:|---:|---:|---:|---:|
| Earlier clean proxy, 32 rows | 32 | 100.0% | 96.9% | 96.9% | 0.0 | 0.0 |
| Larger clean proxy, wake enrollment | 128 | 87.5% | 76.6% | 87.5% | 0.0 | 0.0 |
| Larger clean proxy, same-command-speaker enrollment diagnostic | 128 | 84.4% | 73.4% | 87.0% | 0.0 | 0.0 |
| Matched CAVEMOVE cabin noise, wake enrollment | 128 | 45.3% | 25.8% | 56.9% | 0.0 | 0.0 |

An additional three-candidate/0.90 run on the larger clean split reached
87.5% wake, 60.9% overall intent, and 69.6% conditioned intent, so the
two-candidate setting is retained as the better diagnostic operating point.

The larger clean result disproves the earlier impression that the endpoint
policy alone had solved clean operation. The same-command-speaker enrollment
diagnostic did not improve it, so cross-corpus enrollment mismatch is not the
only limiting factor. The noisy transfer remains substantially worse, showing
that a clean Stage-1 path cannot replace the retained cabin-trained path.

The zero false-event rates are not release evidence: the proxy is short,
synthetic, and contains no consented continuous drivers. The noisy rows do
contain passenger/noise segments, but the wake and command speakers are not
verified to be the same person.

## Decision

The two-call buffered decoder remains a clean-path experimental candidate, not
a promoted runtime policy. The current evidence supports:

1. retaining the dual-domain Stage 1 for cabin-noisy work;
2. keeping endpoint/buffered decoding behind an experiment flag until it is
   tuned on same-driver recordings; and
3. prioritizing a speaker-consistent English cabin set and target-hardware
   capture before further checkpoint promotion.

No training run from this study was promoted. More clean-only tuning would
mostly risk fitting the proxy; the next training data must connect the same
driver's enrollment, wake, command, passenger speech, and cabin noise.

## Endpoint-trim cross-check

Because the buffered policy is sensitive to the 1 s evaluation cadence, the
same 128 noise-free rows were replayed at a 500 ms hop with the existing
`kws_trim` endpoint policy (`endpoint_margin_seconds=0.80`). This is a decoder
comparison only; the model checkpoints were unchanged.

| Policy | Rows | Wake recall | Intent overall | Intent given wake | False wakes/h | False actions/h |
|---|---:|---:|---:|---:|---:|---:|
| Confidence-buffered hybrid, 1 s hop | 128 | 87.5% | 76.6% | 87.5% | 0.0 | 0.0 |
| Endpoint trim, 500 ms hop | 128 | **100.0%** | **84.4%** | **84.4%** | 0.0 | 0.0 |

The endpoint-trim result clears the project's provisional 80% clean proxy
target, but it remains cross-corpus evidence and not a production claim. The
128 scenes contain a synthetic sequence and no consented same-driver
recording, and the zero false-event count is not a sufficiently long negative
test. The endpoint policy is therefore the strongest clean diagnostic
candidate, not a frozen release setting.

## Open-set negative and speaker-gate cross-check

The positive-only endpoint result cannot be used as a release operating point,
so the same configuration was tested on the University of Rochester Smart
Speaker Command Dataset. These are clean single-channel non-Marvin clips
(duplicated to four channels for interface compatibility), not car recordings.
At the endpoint candidate's `kws_threshold=0.45` with no speaker gate, 2,110
files provided 3.193 hours of negative exposure and produced:

- 3,861 full-path false wakes: **1,209.38 false wakes/hour**;
- 4,116 KWS-only smoothed wake events: **1,289.26/hour**;
- on the 14-file sequential action sample, 8 false actions over 68.65 s:
  **419.51 false actions/hour**.

This is a safety failure even though the positive proxy exceeded 80%.

As a diagnostic trade-off, the clean positive scenes were rerun with the
existing TSE-domain KWS checkpoint and SpEx speaker gates (`0.24` wake,
`0.22` command). Wake recall fell to **4.69%** and intent to **0%**, with no
false events. That conservative result is dominated by the known invalid
cross-corpus enrollment mismatch: the wake enrollment and FSC command source
are not the same speaker. It should not be interpreted as the performance of a
proper same-driver gate, but it proves that the current clean proxy cannot
select a safe production threshold.

The required next experiment is a speaker-consistent English recording set
with the same enrollment, wake phrase, command, passenger negatives, and
vehicle playback/noise. Until then, do not promote the clean-only KWS model or
the hard-gated diagnostic.

## Reproduction artifacts

- `logs/clean_confidence_buffered_hybrid_candidate_64scenes_clean_2026-10-14.json`
- `logs/noisy_confidence_buffered_hybrid_candidate_64scenes_cleanpolicy_2026-10-14.json`
- `logs/clean_confidence_buffered_hybrid_same_speaker_enroll_64scenes_2026-10-14.json`
- `logs/clean_confidence_buffered_hybrid_candidate_buffer3_64scenes_clean_2026-10-14.json`
- `configs/eval_end_to_end_clean_english_identity_targetkws_confidence_buffered_hybrid_candidate.yaml`
- `configs/eval_end_to_end_clean_english_identity_targetkws_endpoint08.yaml`
- `logs/clean_endpoint08_64scenes_clean_hop05_2026-10-14.json`
- `logs/smart_speaker_command_negative_endpoint08_2026-10-14.json`
- `configs/eval_end_to_end_clean_english_identity_targetkws_endpoint08_hardneg_gate_candidate.yaml`
- `logs/clean_endpoint08_hardneg_gate_64scenes_clean_hop05_2026-10-14.json`
- `configs/eval_end_to_end_clean_english_identity_targetkws_endpoint08_hardneg_nogate_candidate.yaml`
- `logs/smart_speaker_command_negative_endpoint08_hardneg_nogate_250_2026-10-14.json`
- `logs/smart_speaker_command_negative_endpoint08_hardneg_nogate_all_2026-10-14.json`
- `scripts/eval/eval_continuous_full_pipeline_proxy.py`

## Full open-set hard-negative sweep

The hard-negative/TSE-domain KWS candidate was evaluated on the complete
University of Rochester non-Marvin corpus using the same 500 ms endpoint
evaluation, with no explicit speaker gate. This isolates the contribution of
the presence stage while avoiding the invalid cross-corpus enrollment-gate
comparison above.

| Path | Files | Hours | Events | Events/hour |
|---|---:|---:|---:|---:|
| KWS-only, smoothed | 2,110 | 3.193 | 234 | 73.30 |
| Full path with presence gate | 2,110 | 3.193 | 155 | 48.55 |

The presence gate removed 79 of 234 KWS-only events (33.8%). Rates varied by
speaker: the full-path rate ranged from 10.25/hour to 205.27/hour, with the
largest concentration in Speaker6. The 14-file sequential action diagnostic
observed zero false actions, but it covered only 68.65 seconds and therefore
cannot support a safety-rate claim.

This is a clear improvement over the clean-only KWS candidate (1,209.38
full-path false wakes/hour on the same corpus), but 48.55/hour is still far
from the provisional production target of at most 0.1/hour. The result also
uses clean studio audio duplicated to four channels, so it is open-set speech
evidence rather than real-cabin evidence. The next meaningful improvement is
speaker-consistent cabin data and calibration of the seat/presence/target
speaker checks; another clean positive-only threshold sweep is not justified.

## Correction: full-context window scoring

The continuous evaluator used for the rates above scored an incomplete
zero-padded tail window. All reported tail-sensitive rates in this section are
historical pre-fix diagnostics. With the shared full-context window policy,
the retained hard-negative path on all 2,110 files reports 76 false wakes over
3.193 hours (**23.81/hour**) and 0 false actions in the 14-file sequential
sample; see
`logs/smart_speaker_command_negative_endpoint08_hardneg_nogate_fullwindows_v2_2026-10-14.json`.
The open-set conclusion is unchanged: 23.81/hour is still unsafe and does not
constitute real-cabin evidence.

