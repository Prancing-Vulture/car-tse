# In-car voice assistant research journey

This log records the engineering path toward a production-readiness decision.
It is deliberately evidence-first: proxy, target-absent, and real-driver
results are kept separate, and a result is not promoted merely because it is
better on one dataset.

## 2026-10-10 - Production-readiness work opened

User objective: improve the end-to-end in-car assistant continuously, preserve
all experiments, and document the process for a future research paper.

Current evidence boundary:

- There are no permitted real-driver continuous sessions in the workspace.
- Existing positive end-to-end scenes are cross-corpus proxies: the wake-word
  and command clips are not verified to be from the same speaker.
- AISHELL-5 provides a real-cabin target-absent diagnostic, not a positive
  driver wake/command benchmark.

Current baseline to preserve:

- Stage 1 TSE: approximately +10.98 dB mean SI-SDR improvement on 400 held-out
  CAVEMOVE samples.
- Clean FSC intent: approximately 99.24% top-1; cabin/TSE-domain intent is
  approximately 90% top-1.
- Wake-word component: approximately 97.4% F1 at one operating point, with
  approximately 28.8 false accepts/hour.
- Full continuous cross-corpus proxy at the conservative candidate operating
  point: approximately 25% target-wake recall, 12.5% intent accuracy, and
  0 false actions/hour.
- Lowering thresholds can reach approximately 37.5% proxy intent accuracy,
  but false actions rise to approximately 128.6/hour.
- Runtime-only ablations did not reveal a safe improvement from smoothing,
  enrollment length, simulated channel selection, speaker-gate threshold, or
  command timeout.

## Decision rules

Every candidate must record:

1. exact config and checkpoint paths;
2. code revision or dirty-worktree status;
3. dataset provenance and permission status;
4. train/validation/test speaker split;
5. target recall and command accuracy;
6. false accepts and false actions with exposure hours;
7. p50/p95 latency and resource footprint;
8. regressions on the prior accepted candidate.

No candidate becomes the default merely because it improves a proxy metric.
The current work prioritizes domain alignment and false-action safety before
raw wake recall.

## Planned sequence

1. Bounded dual-domain Stage 1 frontend adaptation using existing AISHELL-5
   target-present rows and CAVEMOVE rows; keep the SpEx encoder frozen and
   compare against both domains before promotion. **Completed.**
2. Re-evaluate presence/wake/command handoff with cached features and fixed
   checkpoints. **Completed as a proxy study; intent handoff remains open.**
3. Collect permitted same-driver continuous cabin sessions. Keep tuning
   drivers/sessions separate from held-out test drivers/sessions.
4. Tune thresholds and command-capture policy only on tuning sessions, freeze
   them, then run the held-out test once.
5. Run component ablations on the same held-out examples.
6. Make a production-readiness decision only if the explicit gates are met.

## Experiment entries

### E-2026-10-10-dual-domain-stage1

A bounded two-epoch frontend-only adaptation was run using balanced AISHELL-5
and CAVEMOVE rows. The pilot 64-per-domain check improved AISHELL-5 SI-SDRi
from 2.46 dB to 6.05 dB (+3.59 dB), while CAVEMOVE changed from 11.14 dB to
11.12 dB (-0.02 dB). The subsequent full 144-per-domain check improved
AISHELL-5 from 2.385 dB to 6.243 dB (+3.858 dB) and CAVEMOVE from 10.478 dB
to 10.592 dB (+0.114 dB). It therefore passes the bounded Stage 1
no-regression gate and is retained for integration, but is not yet the default
production checkpoint.

- `reports/stage1_dual_domain_study_2026-10-10.md`
- `logs/stage1_dual_domain_regression_2026-10-10.json`
- `logs/stage1_dual_domain_regression_full_2026-10-10.json`

### E-2026-10-10-dual-domain-stage2-handoff

The new framewise presence head achieved 0.904 frame F1 at threshold 0.50
and 0.996 chunk F1 at threshold 0.50 on its held-out validation set. Its
output scale differed from the prior head, so the inherited 0.70 threshold
was invalid. A bounded joint sweep selected a provisional presence threshold
of 0.07, command-presence threshold 0.10, KWS threshold 0.50, and two-hit
smoothing.

On eight cross-corpus continuous proxy scenes, the full path reached 100%
wake recall, 12.5% intent accuracy, and 0 false actions/hour. Removing TSE
raised proxy intent accuracy to 75% but caused 96.4 false actions/hour;
removing presence caused 450 false actions/hour. A post-wake raw-mixture
handoff preserved 100% wake recall and 0 false actions/hour while improving
proxy intent accuracy to 37.5%. This is a candidate handoff only: the clips
are not speaker-consistent, and no real-driver session has been evaluated.

Repeating the handoff at passenger SNR -10 dB kept raw-handoff intent at 37.5%
but also produced 32.1 false actions/hour, so the apparent intent gain is not
yet a safe improvement under the harsher passenger condition. At passenger SNR
+10 dB, the raw handoff had 0 false actions/hour in the same small proxy.

A partial speaker-consistency diagnostic then reused a different FSC utterance
from the command speaker for enrollment. It reached 100% proxy intent accuracy
but 64.3 false actions/hour. Reusing the exact command clip as enrollment also
reached 100% intent with 32.1 false actions/hour, but is explicitly leakage-prone.
These results show that command-speaker consistency is a major source of the
12.5% intent bottleneck, while the false-action gate remains unresolved. Neither
diagnostic is a real-driver evaluation because the wake-word speaker is not
verified to match and the scene is synthetically assembled.

An explicit speaker-similarity gate was then swept on the same-speaker-other
diagnostic. Thresholds 0.15–0.22 preserved 100% wake recall and intent
accuracy while reducing false actions from 64.3/hour to 0/hour. On the
cross-corpus wake/command proxy, however, even 0.10 reduced wake recall to
12.5% and thresholds at or above 0.12 reduced it to 0%, confirming that the
gate cannot be selected until same-driver tuning data exists.

At passenger SNR -10 dB, the same-speaker diagnostic required a trade-off:
0.22 retained 100% wake/intent accuracy but left 32.1 false actions/hour,
while 0.27 reached 0 false actions/hour at 75% wake recall and 37.5% intent
accuracy. This is useful for designing a future safety operating curve, not a
production threshold.

The pre-existing hard-negative target-KWS checkpoint was then replayed with
the dual-domain Stage 1/Stage 2 candidate. At passenger SNR +10 dB it reduced
false wake events from 5 to 3 while preserving 100% wake recall, 12.5% intent
accuracy, and 0 false actions/hour. At -10 dB it reduced false wake events from
6 to 5, but false actions remained 32.1/hour. It is retained as a wake-stage
candidate for the real-driver tuning split, not promoted from this small proxy.

One 120-second AISHELL-5 target-absent session produced 0 full-path smoothed
triggers/hour at the provisional operating point. This is a short diagnostic,
not a production false-accept estimate.

The same AISHELL-5 slice was also used to smoke-test the optional
target-speaker KWS evaluator. At the inherited 0.50 threshold, the current
target-speaker head produced 630 full-path smoothed events/hour and the
hard-negative head produced 150/hour, versus 0/hour for the generic KWS
baseline. Raising the hard-negative head to a diagnostic 0.70 threshold
reduced this target-absent slice to 0/hour, but there are no positive
same-driver wake examples with which to measure the resulting recall. This is
therefore a calibration warning, not evidence that target-speaker KWS is ready
to replace the generic head.

An optional one-pass threshold replay was added to the AISHELL evaluator so
multiple KWS thresholds reuse the same Stage 1/KWS scores. On the same
target-absent slice, full-path smoothed events/hour for the current target-KWS
head were 630, 30, 0, 0, 0 at thresholds 0.50/0.55/0.60/0.65/0.70; the
hard-negative head produced 150, 60, 30, 0, 0. This is a reproducibility and
calibration diagnostic only: it contains no positive wake examples.

- `reports/dual_domain_stage2_handoff_study_2026-10-10.md`
- `logs/stage2_frame_presence_dual_domain_nonwake_ft1/metrics_last.json`
- `logs/continuous_ablation_matrix_dual_domain_stage2_intent_handoff_2026-10-10.json`
- `logs/continuous_ablation_matrix_dual_domain_stage2_intent_handoff_passenger_m10_2026-10-10.json`
- `logs/continuous_ablation_matrix_dual_domain_stage2_intent_handoff_passenger_p10_2026-10-10.json`
- `logs/continuous_ablation_matrix_dual_domain_stage2_same_speaker_other_2026-10-10.json`
- `logs/continuous_ablation_matrix_stage2_same_speaker_gate_sweep_2026-10-10.json`
- `logs/continuous_ablation_matrix_stage2_cross_corpus_gate_sweep_2026-10-10.json`
- `logs/continuous_ablation_matrix_stage2_same_speaker_gate_passenger_m10_2026-10-10.json`
- `logs/continuous_ablation_matrix_stage2_same_speaker_gate_passenger_m10_high_2026-10-10.json`
- `configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_hardneg_candidate.yaml`
- `logs/continuous_ablation_matrix_dual_domain_stage2_hardneg_2026-10-10.json`
- `logs/continuous_ablation_matrix_dual_domain_stage2_hardneg_passenger_m10_2026-10-10.json`
- `logs/aishell5_target_absent_dual_domain_stage2_2026-10-10.json`
- `scripts/eval/eval_aishell5_continuous_kws.py`
- `logs/aishell5_target_absent_dual_domain_target_kws_current_2026-10-10.json`
- `logs/aishell5_target_absent_dual_domain_target_kws_current_sweep_2026-10-10.json`
- `logs/aishell5_target_absent_dual_domain_target_kws_hardneg_2026-10-10.json`
- `logs/aishell5_target_absent_dual_domain_target_kws_hardneg_sweep_2026-10-10.json`
- `logs/aishell5_target_absent_dual_domain_target_kws_hardneg_thr070_2026-10-10.json`

### E-2026-10-10-runtime-ablation

The fixed-checkpoint runtime sweep tested temporal hop/smoothing, enrollment
duration, simulated channel subsets, speaker-gate thresholds, and command
timeouts across eight continuous proxy scenes. It produced no safe candidate
upgrade. Full details are in:

- `logs/runtime_ablation_sweep_2026-10-10.json`
- `reports/runtime_ablation_sweep_2026-10-10.md`

### E-2026-10-10-real-driver-audit

The workspace was rechecked for usable recordings. Result: zero session
directories. A guarded recorder and stricter validator were added, but no
audio was fabricated and no real-driver result was claimed.

- `scripts/data/record_real_driver_session.py`
- `scripts/data/validate_real_driver_sessions.py`
- `logs/real_driver_validation_2026-10-10.json`

### E-2026-10-10-real-driver-evaluator

The real-driver evaluator is now prepared and guarded by the session
validator. It refuses to run scoring on an invalid or unconsented session
tree, tunes presence/KWS thresholds only on the declared tuning drivers, and
scores held-out test drivers with frozen settings. With the current empty
session tree it records a blocked-input result rather than fabricating metrics.

- `scripts/eval/eval_real_driver_sessions.py`
- `logs/real_driver_pipeline_evaluation_2026-10-10.json`

### E-2026-10-10-dual-domain-stage1-intent-alignment

The accepted dual-domain Stage 1 candidate was used to regenerate the full
CAVEMOVE/FSC intent cache (20,818 train and 2,314 validation examples). A
frozen wav2vec intent head was then adapted to that exact output distribution.
The aligned head reached 90.3% validation accuracy and preserved 99.3% top-1
accuracy on the official clean FSC test. On the same eight-scene
cross-corpus proxy, however, the full cascade remained at 87.5% target-wake
recall, 12.5% intent accuracy, and 0 false actions/hour. The direct labeled
command-window oracle also remained at 12.5%, so this experiment did not
solve the proxy's main limitation.

- `configs/stage4_intent_tse_dual_domain.yaml`
- `data/cache/stage4_intent_tse_dual_domain_16k_v1/`
- `checkpoints/stage4_intent_wav2vec_dual_domain_head_v1/`
- `logs/continuous_ablation_matrix_dual_domain_intent_v1_2026-10-10.json`

### E-2026-10-10-speaker-consistency-diagnostic

The same scenes were rerun with an explicitly diagnostic enrollment made from
the command speaker. This is not a deployable evaluation because it reuses
command-speaker audio as enrollment, but it isolates the cross-corpus
confound. Full-cascade intent became 100% across all eight scenes with 0
false actions/hour; the realistic wake-speaker enrollment remained 12.5%.
This large gap demonstrates that the current proxy is primarily measuring
speaker mismatch between enrollment/wake and command, not the final driver's
ability to issue a command. It also confirms that the intent head can classify
the extracted command when target-speaker continuity is present.

- `logs/continuous_ablation_matrix_dual_domain_intent_command_enroll_2026-10-10.json`
- `scripts/eval/eval_continuous_ablation_matrix.py` (`--enrollment-source`)

### E-2026-10-10-real-cabin-negative-regression-and-runtime

The dual-domain Stage 1 candidate was run for 120 seconds on a speech-rich
AISHELL-5 Eval2 target-absent stream. The full presence-gated path produced
0 smoothed events/hour; KWS-only smoothed events were also 0/hour for this
short check. This supports no immediate negative-stream regression but is not
a positive wake benchmark. A CUDA latency smoke test measured approximately
41.4 ms for Stage 1, 1.6 ms for KWS, and 12.7 ms for wav2vec intent on one
second of audio (about 55.7 ms for those components together, excluding I/O,
buffering, and full application overhead).

- `logs/aishell5_target_absent_dual_domain_stage1_2026-10-10.json`
- `logs/latency_dual_domain_intent_v1_2026-10-10.json`

The guarded real-driver recorder is now runnable after installing the optional
`sounddevice` dependency. The machine exposes a one-channel Fifine input and
a two-channel WASAPI input, but no recording was started because no operator
consent/session content was supplied. The workspace still contains zero
real-driver sessions.

The real-driver evaluator was extended so its tuning split can jointly select
presence, command-presence, KWS, and speaker-gate thresholds, then freeze all
four before held-out scoring. With the empty session tree it correctly refused
to run and emitted a blocked-input report; no real-driver metric was inferred.

- `logs/real_driver_pipeline_evaluation_gate_tuning_2026-10-10.json`
- `scripts/eval/eval_real_driver_sessions.py`

### E-2026-10-10-dual-domain-stage1-ft2

Two additional low-rate frontend-only epochs initialized from the retained
dual-domain Stage 1 checkpoint improved the AISHELL-5 target-present sweep
from approximately 4.85 to 5.38 dB SI-SDRi while keeping CAVEMOVE at
approximately 10.88 dB. The end-to-end cross-corpus proxy did not change
(87.5% wake recall, 12.5% intent accuracy, 0 false actions/hour), confirming
that the proxy's command failure is not fixed by a small frontend refinement.
The ft2 checkpoint is retained as the stronger Stage 1 candidate for future
same-driver evaluation; it is not declared production-ready.

- `configs/stage1_spex_dual_domain_rms010_ft2.yaml`
- `checkpoints/stage1_spex_dual_domain_rms010_ft2/best.pt`
- `logs/stage1_dual_domain_rms010_ft2_aishell_test.json`
- `logs/stage1_dual_domain_rms010_ft2_cavemove_test.json`
- `logs/continuous_ablation_matrix_dual_domain_ft2_intent_v1_2026-10-10.json`

The ft2-aligned downstream cache/head was also trained from the ft2 output
distribution. Its held-out TSE-domain accuracy was 90.0% and its continuous
cross-corpus result remained 12.5% intent accuracy with 0 false actions/hour.
It is therefore not a measurable end-to-end upgrade over the ft1-aligned
head, although both are retained for the future same-driver study.

- `configs/stage4_intent_tse_dual_domain_ft2.yaml`
- `checkpoints/stage4_intent_wav2vec_dual_domain_ft2_head_v1/`
- `logs/continuous_ablation_matrix_dual_domain_ft2_head_v1_2026-10-10.json`

### E-2026-10-10-followup-hardneg-and-ft2-operating-point

Two controlled follow-up comparisons were run on the eight-scene
same-command-speaker enrollment proxy. First, the hard-negative-trained
target-speaker KWS checkpoint was substituted into the current dual-domain
Stage-2 operating configuration. It produced the same baseline result as the
original target-KWS checkpoint: 100% wake recall, 100% intent accuracy, and
64.3 false actions/hour over 112 seconds. With the speaker-similarity gate at
0.18, 0.20, or 0.22, both checkpoints reached 100% wake/intent and 0 false
actions/hour; at 0.22 each had one false wake event. The hard-negative model
is therefore retained for reproducibility but not promoted as an end-to-end
improvement.

Second, the ft2 Stage-1 frontend and its separately aligned intent head were
evaluated with the same operating point. The ungated result was again 100%
wake/intent and 64.3 false actions/hour. At speaker-gate 0.18 it remained
100%/100% with 0 false actions/hour, but at 0.20 and 0.22 it dropped to
87.5% intent and then 87.5% wake/intent. The ft2 extraction improvement is
still useful for AISHELL-5 SI-SDRi, but its downstream head is not a better
full-pipeline candidate on this proxy.

- `reports/wake_model_and_stage1_followup_2026-10-10.md`
- `configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_hardneg_candidate.yaml`
- `configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_ft2_head_operating_candidate.yaml`
- `logs/continuous_ablation_matrix_stage2_hardneg_same_speaker_gate_sweep_2026-10-10.json`
- `logs/continuous_ablation_matrix_stage2_ft2_head_same_speaker_gate_sweep_2026-10-10.json`

### E-2026-10-10-joint-safety-thresholds

The continuous evaluator was extended with an optional post-wake intent
confidence gate and a joint speaker-gate/command-presence sweep. On the
harsher same-speaker diagnostic (passenger SNR -10 dB, eight scenes, 112
seconds), the existing speaker gate at 0.22 reduced false actions from 192.9
to 32.1 per hour while retaining 100% wake and intent accuracy. Adding an
intent confidence threshold of 0.53 removed the remaining false action in
this proxy while retaining 100% wake and intent accuracy. Threshold 0.55 or
0.60 was safe but suppressed one target command, giving 87.5% intent
accuracy. A post-wake presence threshold of 0.15 also removed the false
action but likewise reduced intent accuracy to 87.5%.

The runtime now supports `intent_confidence_threshold` and reports
`command_suppressed`; the default is unset so prior behavior is unchanged.
The 0.22/0.53 pair is retained only as a diagnostic candidate until tuning
and held-out real-driver sessions exist. A second replay at passenger SNR
+10 dB retained 100% wake/intent and 0 false actions/hour; the ungated 0.22
speaker-gate baseline was already safe in that milder condition.

- `reports/joint_safety_threshold_sweep_2026-10-10.md`
- `configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_safety_diagnostic_candidate.yaml`
- `logs/continuous_ablation_matrix_stage2_joint_gate_command_presence_passenger_m10_2026-10-10.json`
- `logs/continuous_ablation_matrix_stage2_intent_confidence_gate_passenger_m10_2026-10-10.json`
- `logs/continuous_ablation_matrix_stage2_intent_confidence_gate_passenger_p10_2026-10-10.json`
- `src/car_tse/pipeline/in_car_pipeline.py`

The real-driver evaluator was then extended to include the intent-confidence
gate in the same tuning-only threshold grid. It now freezes presence,
command-presence, KWS, speaker-gate, and intent-confidence settings from the
tuning split and reuses all five unchanged on held-out test sessions. The
empty permitted-session tree was rechecked with the new evaluator; it still
returns a blocked-input report and runs no fabricated scoring.

- `scripts/eval/eval_real_driver_sessions.py`
- `logs/real_driver_pipeline_evaluation_gate_tuning_confidence_2026-10-10.json`

The new safety diagnostic was additionally run through the actual
`InCarPipeline.forward` path. With same-speaker-other enrollment, passenger
SNR -10 dB, and CAVEMOVE cabin SNR -5/0/+10/+20 dB, it reached 100% wake
recall, 100% intent accuracy, and 0 false actions/hour over 112 seconds. It
generated 3 false wake events (96.4/hour equivalent) and mean wake latency
2.125 seconds. This validates the runtime implementation of the safety gate,
but remains a short cross-corpus proxy and does not close the real-driver
acceptance gates.

- `logs/continuous_full_pipeline_proxy_safety_diagnostic_same_speaker_m10_2026-10-10.json`
- `scripts/eval/eval_continuous_full_pipeline_proxy.py` (`--enrollment-source`)

### E-2026-10-11-intent-confidence-calibration

The frozen ft1-aligned intent head was evaluated on all 2,314 examples in the
dual-domain TSE validation cache. Raw top-1 accuracy was 90.32%, 15-bin ECE
was 0.0240, and the Brier score was 0.1393. At confidence threshold 0.53,
92.13% of examples remained actionable and their selective accuracy was
94.75%; threshold 0.60 gave 90.10% coverage and 95.78% selective accuracy.
This supports using confidence as a safety signal, but the threshold remains
diagnostic until real-driver tuning/test data are available.

- `reports/intent_confidence_calibration_2026-10-11.md`
- `scripts/eval/eval_intent_confidence_calibration.py`
- `logs/intent_confidence_calibration_dual_domain_ft1_val_2026-10-11.json`

The calibration was then repeated for the exact `ft8` runtime intent
encoder/head and its waveform cache. Raw accuracy was 90.49%, 15-bin ECE
0.0230, and at confidence 0.53 coverage was 94.12% with 93.66% selective
accuracy. A legacy precomputed feature cache was explicitly rejected because
its features were generated with a different encoder state (first-four-row
mean cosine approximately 0.81 versus recomputed features). This correction
keeps the reported calibration aligned with the deployed runtime candidate.

- `reports/intent_confidence_calibration_runtime_ft8_2026-10-11.md`
- `logs/intent_confidence_calibration_ft8_tse_domain_waveform_val_2026-10-11.json`

### E-2026-10-11-presence-threshold-followup

Runtime traces from the safety candidate showed passenger false-wake windows
with presence probabilities around 0.09–0.11 and target wake windows at least
about 0.16 on the same proxy. A separate v2 diagnostic candidate raised only
the pre-wake presence threshold from 0.07 to 0.12, keeping speaker gate 0.22
and intent confidence 0.53. On passenger SNR -10 dB and +10 dB, across 8
scenes / 112 seconds per condition, v2 achieved 100% wake recall, 100% intent
accuracy, 0 false wakes/hour, and 0 false actions/hour; mean wake latency was
2.25 seconds. This is the best current proxy operating point, not a
production result.

The same v2 candidate was also replayed at neutral passenger SNR 0 dB with
the same eight scenes and cabin-noise levels. It retained 100% wake recall and
intent accuracy, 0 false wakes/hour, 0 false actions/hour, and 2.25 seconds
mean wake latency over 112 seconds. This adds a middle proxy condition but
does not add real-driver evidence.

- `configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_safety_diagnostic_candidate_v2.yaml`
- `logs/continuous_full_pipeline_proxy_safety_diagnostic_same_speaker_m10_v2_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_safety_diagnostic_same_speaker_p0_presence012_2026-10-10.json`
- `logs/continuous_full_pipeline_proxy_safety_diagnostic_same_speaker_p10_presence012_2026-10-11.json`

## Production-readiness status

Status: **not production-ready**.

Blocking evidence gaps are the absence of same-driver continuous positive
sessions, lack of a frozen held-out real-driver test, insufficient false-action
evidence under passenger speech, and unresolved target-speech scale/domain
alignment. Further proxy work is diagnostic until those gaps are closed.

### E-2026-10-11-real-cabin-hard-negative-adaptation-and-safety-freeze

The target-speaker wake detector was adapted with permitted AISHELL-5
target-absent cabin speech. Train, validation, and test caches each contain
768 real four-channel cabin windows; Eval2 was kept out of cache construction
and used only as the final continuous negative exposure. The original
synthetic positives and CAVEMOVE/GSC hard negatives were retained, and the
adaptation was initialized from the prior hard-negative target-KWS
checkpoint. The resulting negative-only checkpoint reduced the continuous
AISHELL false-accept rate before speaker gating, but the validation score was
not by itself a release criterion.

A second fine-tune added 768 synthetic target-present AISHELL mixtures per
split. Those examples use speaker-consistent Marvin wake/enrollment clips,
real AISHELL cabin backgrounds, and target SNRs from -5 to +10 dB. Its
validation F1 improved to 0.786, but a .55 threshold rejected the
cross-corpus CAVEMOVE positive proxy; it is retained as an experiment and is
not the selected checkpoint. This result is important: improving an in-domain
validation score did not automatically improve the cross-corpus operating
point.

The selected diagnostic path uses the negative-only AISHELL-adapted
target-KWS checkpoint with the dual-domain Stage-1 ft1 frontend, the
non-wake-trained frame presence head, the existing ft8 intent head, presence
threshold 0.15, KWS threshold 0.50, intent confidence threshold 0.53, and a
split speaker gate: 0.24 before wake detection and 0.22 during command
capture. The split gate is implemented in the runtime state machine so the
higher impostor-rejection threshold does not discard lower-similarity command
frames after a valid wake.

On the full 1,200-second AISHELL-5 Eval2 target-absent exposure (653.6 s from
session 001 and 546.4 s from session 002), the exact runtime-matched setting
(2-window/2-hit smoothing) produced 3.0 full-path smoothed false wakes/hour
(one event in session 002) and 60.0 KWS-only smoothed events/hour. An earlier
0/hour figure came from the evaluator defaulting to 3-window/2-hit smoothing,
which did not match the frozen 2-window/2-hit runtime; that figure is
superseded. The corrected result still shows that the combined
presence/speaker gates remove most wake-head events, but it is not a zero-FAR
claim. This is a negative-only, fixed-enrollment diagnostic: it contains no
driver wake or command positives.

The actual runtime proxy was then run with same-speaker-other enrollment,
CAVEMOVE cabin SNRs of -5/0/+10/+20 dB, and two passenger conditions (-10 dB
and +10 dB). Each condition covered eight 14-second scenes. Both runs
achieved 100% target wake recall, 100% intent accuracy, 0 false wakes/hour,
0 false actions/hour, and approximately 2.5 s mean wake latency. The
enrollment is same-speaker diagnostic audio and the Marvin wake clip is still
not a recorded real-driver sequence, so these values are explicitly not
production evidence.

The 3-of-3 temporal-confirmation alternative was rejected after it reduced
the same proxy to 37.5% wake recall and 12.5% intent accuracy. A target-KWS
ensemble with the prior checkpoint and an ensemble with the negative-only
adapted checkpoint both retained the positive proxy but did not improve the
AISHELL false-accept result; neither is promoted.

Two additional hard-negative-mining retries were then tested. The first
initialized from the negative-only adapted checkpoint and selected the highest
scoring AISHELL negatives; its validation F1 reached 46.6% at threshold 0.62,
but its same-speaker CAVEMOVE proxy reached only 62.5% wake recall and 53.1%
overall intent accuracy at threshold 0.50 (75.0% and 65.6% at threshold 0.40).
The second initialized from the positive AISHELL adaptation; its best
validation F1 was 44.0% at threshold 0.49 and its proxy reached 68.8% wake
recall and 59.4% overall intent accuracy. Both were rejected because neither
preserved the 100% positive-proxy behavior of the reference, even though the
first reduced KWS-only AISHELL events to 48/hour at threshold 0.50. Their
checkpoints remain research artifacts, not release candidates.

A runtime-only speaker-gate narrowing study then tested 0.245 and 0.250 before
wake detection while keeping the 0.22 command gate, p=0.15 presence, KWS=0.50,
and 2-of-2 smoothing fixed. Both thresholds reduced the exact AISHELL Eval2
full-path rate to 0/hour, but the same-speaker harsh-passenger proxy fell to
65.6% wake / 50.0% intent at 0.245 and 62.5% wake / 46.9% intent at 0.250.
The original 0.24 gate therefore remains the reference; the higher gates are
not safe promotions without real-driver tuning data.

During the positive-init retry, a trainer initialization bug was found and
fixed. Target-speaker checkpoints store parameters under `audio.*` and the
relation classifier at the model root; the old code loaded the checkpoint into
`model.audio`, which silently matched no target-head keys. The trainer now
supports explicit full-model warm starts. The corrected full-warm-start v4
run still reached only 62.5% wake / 53.1% intent on the harsh proxy, so the
fix is retained in code but the checkpoint is rejected. A corrected full
transfer from the pre-AISHELL target head and a classifier-only adaptation
produced the same 62.5% / 53.1% proxy result; neither is promoted. This makes
the rejection less dependent on one particular fine-tuning initialization.

Reproducibility artifacts:

- `reports/aishell5_target_kws_adaptation_2026-10-11.md`
- `scripts/data/build_target_kws_aishell5_negative_cache.py`
- `scripts/data/build_target_kws_aishell5_positive_cache.py`
- `scripts/eval/calibrate_target_kws_threshold.py`
- `scripts/eval/eval_real_driver_sessions.py` (now carries the split command speaker gate into the frozen real-driver evaluation)
- `configs/stage3_target_kws_gsc_marvin_hardneg_aishell5.yaml`
- `configs/stage3_target_kws_gsc_marvin_hardneg_aishell5_pos.yaml`
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
- `configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_hardneg_aishell5_neg_safety_splitgate_candidate.yaml`
- `logs/aishell5_continuous_targetkws_hardneg_aishell5_ft1_p015_sg024_kws050_smooth22_1200s_2026-10-11.json`
- `logs/aishell5_continuous_targetkws_hardneg_aishell5_mined_v2_p015_sg024_smooth22_kws_sweep_1200s_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_hardneg_aishell5_mined_v2_p015_sg024_kws050_same_speaker_m10_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_hardneg_aishell5_mined_v2_p015_sg024_kws040_same_speaker_m10_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_hardneg_aishell5_mined_v3_from_pos_p015_sg024_kws049_same_speaker_m10_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_hardneg_aishell5_mined_v4_full_from_pos_p015_sg024_kws050_same_speaker_m10_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_hardneg_aishell5_v2_full_from_target_p015_sg024_kws050_same_speaker_m10_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_hardneg_aishell5_v3_classifier_only_p015_sg024_kws052_same_speaker_m10_2026-10-11.json`
- `logs/aishell5_continuous_targetkws_hardneg_aishell5_ft1_p015_sg0245_kws050_smooth22_1200s_2026-10-11.json`
- `logs/aishell5_continuous_targetkws_hardneg_aishell5_ft1_p015_sg025_kws050_smooth22_1200s_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_hardneg_aishell5_neg_splitgate_p015_wake024_cmd022_same_speaker_m10_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_hardneg_aishell5_neg_splitgate_p015_wake024_cmd022_same_speaker_p10_2026-10-11.json`

The candidate remains blocked from a production label until permitted
same-driver continuous recordings provide tuning and held-out test splits
with driver wake recall, passenger false accepts, false actions/hour,
command accuracy, and end-to-end latency.

### E-2026-10-10-runtime-and-dataset-readiness-followup

The project added two reproducibility tools before attempting another model
promotion. `scripts/eval/eval_full_pipeline_latency.py` loads the exact
runtime configuration and measures `InCarPipeline.forward` rather than only
timing isolated modules. `scripts/reporting/audit_production_readiness.py`
checks the selected checkpoints, the consent/session validator, the frozen
split inputs, and the available diagnostic logs. The snapshot currently
reports `research_engineering_preview`: all six selected checkpoint artifacts
exist, but the real-driver session tree contains zero sessions, so the
same-driver, passenger-rejection, frozen-split, and threshold-freeze gates
remain blocked.

The 20-repeat CUDA measurement for the selected Wav2Vec intent candidate was
approximately 103.5 ms p50 / 125.2 ms p95 for a 3-second wake-stage frame and
120.7 ms p50 / 141.2 ms p95 for the post-wake command branch. One-time
enrollment speaker embedding was approximately 370 ms. The loaded system has
about 106.0M parameters, 424.5 MB of checkpoint files, and 499.5 MB peak CUDA
allocation in this benchmark. A two-repeat CPU check was approximately 522 ms
for wake-stage and 642 ms for command-stage p50. These are compute/resource
measurements on synthetic random tensors, not speech-quality results; the
latest GPU p95 is above the project’s sub-100-ms engineering target.

GPU FP16 autocast was tested as a low-risk optimization. It regressed this
runtime on the current stack to approximately 153 ms wake p50 and 190 ms
command p50, so it was rejected and not enabled in the runtime. A compact
CRNN intent candidate was also evaluated end-to-end. It reduced checkpoint
files to approximately 75.1 MB and total parameters to 18.7M, but its
32-scene same-speaker CAVEMOVE proxy produced 68.75% target-wake recall and
56.25% overall intent accuracy, with approximately 125 ms command p50. The
existing Wav2Vec intent path therefore remains the quality reference; a
future compact model should be obtained by distillation or retraining with a
quality gate, not by swapping in the current CRNN checkpoint.

The online dataset audit found the official ICSRC 2022 intelligent-cockpit
corpus: 20 hours of real in-car Mandarin speech across driving conditions,
with 10-hour Eval and 10-hour Test partitions and passenger/background
conversation categories. Its agreement limits use to non-commercial
education/research/technology development, prohibits redistribution, and
prohibits using Test data for training, so it was not downloaded or mixed into
the English Marvin production path. It also does not provide a continuous
same-driver enrollment -> Marvin -> command sequence. The public CI-AVSR
repository provides Cantonese clean/noisy/out-of-domain in-car command data,
but likewise lacks the required English wake/enrollment continuity and the
processed dataset’s terms must be checked separately. These sources are
documented candidates, not real-driver evidence for this project.

Artifacts from this follow-up:

- `scripts/eval/eval_full_pipeline_latency.py`
- `scripts/reporting/audit_production_readiness.py`
- `configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_crnn_intent_candidate.yaml`
- `logs/full_pipeline_latency_gpu_2026-10-11.json`
- `logs/full_pipeline_latency_cpu_2026-10-11.json`
- `logs/full_pipeline_latency_gpu_fp16_2026-10-11.json`
- `logs/full_pipeline_latency_crnn_intent_gpu_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_crnn_intent_candidate_same_speaker_m10_2026-10-11.json`
- `logs/production_readiness_snapshot_2026-10-11.json`
- `reports/production_readiness_snapshot_2026-10-11.md`

Sources reviewed for the dataset audit: <https://iscslp2022-icsrc.org/> and
<https://github.com/HLTCHKUST/CI-AVSR>.

### E-2026-10-10-real-cabin-negative-long-exposure

To reduce the sampling uncertainty of the shorter AISHELL-5 diagnostic, the
frozen split-gate candidate was replayed over all 18 sessions in Eval2. This
was a target-absent exposure: the fixed LibriSpeech enrollment speaker does
not occur in these sessions, so every full-path event is a false accept for
this diagnostic. The replay used the exact runtime settings: presence 0.15,
pre-wake speaker gate 0.24, command gate 0.22, target-KWS threshold 0.50,
and 2-window/2-hit smoothing with a two-call refractory period.

The exposure was 14,001.9 seconds (3.889 hours), with 80.999% speech
occupancy. It produced 18 full-path smoothed events, or 4.628 full-path
false wakes/hour, and 376 KWS-only smoothed events, or 96.673/hour. The
earlier 1,200-second result of 3.0 full-path/hour remains a valid shorter
slice, but the all-session replay is the stronger aggregate estimate for
future comparisons. This is still cross-corpus negative evidence, not a
real-driver FAR: it has no enrolled-driver wake positives, passenger-wake
labels, command labels, or vehicle-specific playback.

Artifact: `logs/aishell5_continuous_targetkws_hardneg_aishell5_ft1_p015_sg024_kws050_smooth22_eval2_all18_2026-10-10.json`.

### E-2026-10-10-cpu-int8-check

While the long AISHELL target-absent exposure occupied the GPU, dynamic INT8
quantization was tested on the CPU Wav2Vec intent branch. The fp32 pipeline
measured approximately 522.2 ms wake-stage p50 and 641.7 ms post-wake command
p50 in the earlier CPU check; the INT8 replay measured approximately 515.2 ms
and 624.5 ms respectively. This is only a small latency reduction, and no
validation-quality run was used to claim accuracy preservation. It is not
enabled or promoted; it remains a reproducible deployment experiment until a
full intent regression is run.

Artifact: `logs/full_pipeline_latency_cpu_int8_2026-10-11.json`.

### E-2026-10-11-clean-gpu-latency-rerun

After the all-session AISHELL evaluator exited, the selected Wav2Vec pipeline
was rerun with 5 warm-up calls and 20 timed calls on the RTX 4080. This clean
measurement replaced the earlier contention-affected GPU value in the current
runtime report: wake-stage p50/p95 improved to 48.2/51.6 ms and the post-wake
command branch measured 62.6/64.4 ms. Enrollment speaker embedding was 363.6
ms; the loaded pipeline contains 106.0M parameters, stores 424.5 MB of
checkpoints, and used 499.5 MB peak CUDA allocation in this run.

These timings are compute/resource measurements over synthetic random
four-channel tensors, not a speech-quality or real-cabin claim. They show that
the configured fp32 GPU path meets the provisional sub-100-ms compute target
on this hardware, while production readiness remains blocked by the missing
permitted same-driver continuous recordings, frozen tuning/test split, and
real-cabin false-action evaluation.

Artifacts: `logs/full_pipeline_latency_gpu_2026-10-11.json` and
`logs/full_pipeline_latency_fp32_cuda_clean_2026-10-11.json`.

### E-2026-10-11-compact-intent-distillation-and-calibration

The first compact raw-audio CRNN was not competitive with the Wav2Vec intent
reference. Its input path did not apply the per-utterance normalization used
by the reference encoder. An opt-in normalized CRNN was therefore trained as
a separate candidate; it reached 44.3% top-1 on the held-out TSE/cabin split.

A second bounded run initialized that normalized student from its best
checkpoint and distilled the frozen Wav2Vec head logits from the cached
768-dimensional features (temperature 3.0, KD weight 0.5). The best student
reached 56.3% top-1, 87.5% top-5, and 59.2% macro-F1 on the TSE/cabin split;
the clean FSC test was 80.5% top-1 and 98.7% top-5. It is a substantial
improvement over the prior compact candidate (about 41.1% clean and 48.8%
TSE-domain), but it remains below the selected Wav2Vec TSE-domain reference
(about 90.2% top-1).

The full-pipeline proxy exposed an important calibration effect. With the
inherited 0.53 intent-confidence threshold, the student produced 34.4%
overall intent on the 32-scene same-speaker diagnostic. At 0.25 or 0.35 it
produced 59.4% overall and 86.4% conditioned on a detected wake, with zero
proxy false actions. The Wav2Vec reference on the exact same 32-scene matrix
also produced 59.4% overall and 86.4% conditioned-on-wake at its 0.53
threshold, so the compact candidate is a plausible edge fallback, not a
quality replacement. The threshold is exploratory and must be tuned and
frozen on real-driver tuning recordings.

The compact candidate is materially cheaper: 18.68M parameters, 75.1 MB of
stored checkpoints, 165.8 MB peak CUDA allocation, and 38.8 ms wake / 44.8 ms
post-wake command p50 in the synthetic compute benchmark. It was not promoted
as the default because direct TSE-domain intent accuracy is still lower and
the project has no permitted real-driver validation set.

Artifacts:

- `scripts/train/train_intent_student_distill.py`
- `configs/stage4_intent_tse_domain_norm_student.yaml`
- `configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_norm_crnn_candidate.yaml`
- `checkpoints/stage4_intent_tse_domain_distill_t3_a05/best.pt`
- `logs/stage4_intent_tse_domain_distill_t3_a05_val.json`
- `logs/stage4_intent_tse_domain_distill_t3_a05_fsc_test_clean.json`
- `logs/continuous_full_pipeline_proxy_norm_crnn_distill_t3_a05_conf035_same_speaker_m10_2026-10-11.json`
- `logs/full_pipeline_latency_norm_crnn_distill_t3_a05_gpu_2026-10-11.json`

### E-2026-10-11-kws-threshold-sweep-all-session

The exact target-absent AISHELL-5 Eval2 replay was extended with KWS threshold
0.50, 0.55, and 0.60 summaries while keeping presence 0.15, the 0.24/0.22
speaker gate, 2-window/2-hit smoothing, and the same fixed enrollment. Over
3.889 hours and 18 sessions, the aggregate was:

| KWS threshold | Full-path events/hour | KWS-only events/hour | Positive 32-scene wake recall |
| --- | ---: | ---: | ---: |
| 0.50 | 4.628 | 96.673 | 68.75% |
| 0.55 | 4.628 | 60.420 | 62.50% |
| 0.60 | 3.600 | 39.080 | 62.50% |

The stricter thresholds reduce KWS-only activity, but the full-path safety
gain is modest and the positive proxy loses wake recall. The 0.50 operating
point remains the quality reference pending real-driver tuning; the sweep is
an engineering trade-off, not a real-driver false-accept estimate.

Artifact: `logs/aishell5_continuous_targetkws_hardneg_aishell5_ft1_p015_sg024_kws055_sweep_eval2_all18_2026-10-11.json`.

### E-2026-10-11-ft2-frontend-and-downstream-promotion-audit

The next improvement cycle tested whether the standalone Stage 1 extraction
gain would carry through the complete assistant. A separately initialized
Stage 1 ft2 checkpoint improved the fixed dual-domain regression from 6.432
dB to 8.496 dB mean SI-SDR improvement overall. The gain was concentrated in
AISHELL-5 (+4.014 dB) while CAVEMOVE improved only +0.115 dB. This passed the
Stage 1 regression rule, but it did not pass the end-to-end handoff check:
the existing downstream heads fell to 62.5% wake recall and 43.8% overall
intent, and a target-KWS retraining aligned to the ft2 outputs reproduced the
same 56.3% / 37.5% proxy result. The ft2 frontend is therefore retained as a
research branch, not a drop-in production replacement.

The ft2 target-KWS experiment was rebuilt with isolated caches for GSC Marvin,
non-wake hard negatives, AISHELL-5 negative speech, and AISHELL-5 positive
speech. Both a negative-only merge and a positive-enriched merge were trained
and evaluated under the same 32-scene proxy. Neither recovered the ft1-aligned
reference. This rules out a simple KWS-only fix for the current Stage 1 domain
shift.

The frame-presence head was also retrained on bounded ft2 windows. During this
step, the AISHELL manifest builder was corrected so target-absent rows carry a
bounded segment start instead of causing full-session audio loads. The new
head reached 1.000 validation chunk F1 at its selected exploratory threshold,
but its score calibration was incompatible with the existing runtime: at its
default threshold it rejected all proxy wakes, and at 0.05 it retained only
15.6% overall intent. It is not promoted.

Artifacts:

- `logs/stage1_dual_domain_regression_ft2_2026-10-11.json`
- `configs/stage1_spex_dual_domain_rms010_ft2.yaml`
- `checkpoints/stage1_spex_dual_domain_rms010_ft2/best.pt`
- `configs/stage3_target_kws_gsc_marvin_hardneg_aishell5_ft2_stage1.yaml`
- `configs/stage3_target_kws_gsc_marvin_hardneg_aishell5_ft2_stage1_pos.yaml`
- `checkpoints/stage3_target_kws_gsc_marvin_hardneg_aishell5_ft2_stage1/best.pt`
- `checkpoints/stage3_target_kws_gsc_marvin_hardneg_aishell5_ft2_stage1_pos/best.pt`
- `configs/stage2_frame_presence_dual_domain_nonwake_ft2_stage1.yaml`
- `checkpoints/stage2_frame_presence_dual_domain_nonwake_ft2_stage1/best.pt`
- `scripts/data/build_presence_gsc_aishell_manifest.py`
- `logs/continuous_full_pipeline_proxy_ft2_stage1_consistent_negative_same_speaker_m10_32scenes_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_ft2_stage1_consistent_pos_same_speaker_m10_32scenes_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_ft2_stage1_consistent_pos_presence_same_speaker_m10_32scenes_2026-10-11.json`

### E-2026-10-11-positive-enriched-kws-safety-rejection

Before spending another long training run, the existing ft1 positive-enriched
target-KWS candidate was replayed over the complete 32-scene same-speaker
proxy. It improved the proxy to 78.125% target-wake recall, 68.75% overall
intent accuracy, and 88.0% intent accuracy conditioned on a detected wake,
with zero proxy false wakes/actions over 448 seconds. This remains a
cross-corpus diagnostic: the enrollment and command are different utterances
from one FSC speaker, but the Marvin clip is not verified to be that speaker.

The same candidate was then exposed to all 18 AISHELL-5 Eval2 sessions for
3.889 hours with the enrolled target absent. At KWS threshold 0.50 it
produced 36 full-path smoothed events, or 9.256 events/hour, and 573
KWS-only events, or 147.323/hour. A threshold sweep produced zero events in
this target-absent exposure at 0.55 and 0.60, but the existing positive proxy
already fell to zero wake recall at 0.55 on the smaller replay. A full
32-scene replay at 0.55 confirmed the quality collapse: 0.0% wake recall,
0.0% overall intent, and 0.0 false actions/hour. There is no safe, useful
operating point to promote from this cross-corpus evidence.

Decision: reject the positive-enriched candidate for production and retain
the negative-only ft1-aligned reference. The reference is 68.75% wake recall,
59.375% overall intent, 86.364% conditioned intent, and 0 proxy false
actions/hour on the 32-scene matrix; its 3.889-hour target-absent diagnostic
is 4.628 full-path events/hour at KWS 0.50. These are engineering proxies,
not real-driver FAR or production reliability.

Artifacts:

- `logs/continuous_full_pipeline_proxy_ft1_positive_fullmatrix_same_speaker_m10_32scenes_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_ft1_positive_fullmatrix_kws055_same_speaker_m10_32scenes_2026-10-11.json`
- `logs/aishell5_continuous_targetkws_hardneg_aishell5_pos_ft1_p015_sg022_kws050_sweep_eval2_all18_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_wav2vec_reference_same_speaker_m10_32scenes_2026-10-11.json`
- `logs/aishell5_continuous_targetkws_hardneg_aishell5_ft1_p015_sg024_kws050_smooth22_eval2_all18_2026-10-10.json`

The next high-value improvement is not another unvalidated cross-corpus
checkpoint. It is a permitted pilot with speaker- and session-disjoint
tuning/test recordings containing the same driver’s enrollment, Marvin,
command, passenger Marvin, passenger speech, silence, and cabin conditions.
Only then can thresholds, speaker gates, and KWS adaptation be selected
without optimizing a proxy that fails under longer negative exposure.

### E-2026-10-11-real-driver-evidence-contract-hardening

Before collecting the pilot, the real-driver evidence path was audited rather
than assumed correct. The session validator now requires a non-empty consent
record ID, owner, permitted purpose, and license/agreement in addition to
`permission_to_use=true`. It also requires the session ID to match its
directory, a non-empty driver ID, a valid audio contract, all required event
types, and at least one ordered enrollment -> driver wake -> driver command
sequence. This prevents an incomplete or temporally inconsistent recording
from entering threshold tuning or held-out testing.

Four unit tests cover acceptance of a complete consented session, rejection
of incomplete consent metadata, rejection of an enrollment interval that does
not precede a driver wake, and rejection of an unassigned driver. The
repository test suite now reports 18 passed and 1 skipped. The actual pilot
root remains empty, so the validator
still returns invalid and the production gates remain blocked as intended.

The readiness audit now also makes the pilot size explicit: at least three
drivers and three complete sessions, with non-empty tuning and held-out test
driver groups, and every observed driver assigned to exactly one group. This
keeps a one-driver demonstration or an unassigned session from satisfying
the production evidence gate.

Artifacts:

- `scripts/data/validate_real_driver_sessions.py`
- `tests/test_real_driver_session_validation.py`
- `logs/real_driver_validation_latest_2026-10-11.json`

### E-2026-10-11-real-driver-evaluator-gate-fix

The real-driver evaluator itself was audited after the validator hardening.
It previously wrote `production_valid: false` unconditionally and selected
tuning thresholds using a false-action limit without constraining false
accepts. It now records configurable held-out gates for driver-wake recall,
end-to-end command accuracy, conservative unmatched-wake false accepts/hour,
and false actions/hour, and it requires the minimum three-driver/three-session
study structure before a run can be marked production-valid. Threshold search
uses both false-accept and false-action limits. The report preserves both the
conservative unmatched-wake rate and the interval-specific rate for diagnosing
annotation coverage.

This change does not promote the current system: the pilot tree is empty, so
the evaluator still refuses to score it. It makes the future real-driver
decision auditable instead of structurally incapable of passing.

Artifacts:

- `scripts/eval/eval_real_driver_sessions.py`
- `reports/real_driver_pilot_protocol_2026-10-09.md`
- `tests/test_real_driver_session_validation.py`

### E-2026-10-11-preexisting-systems-and-hard-negative-review

The project was compared against primary references for target-speaker
extraction, speaker-conditioned activity detection, personalized keyphrase
detection, compact keyword spotting, and multichannel low-latency enhancement.
The review found that the most relevant production-style pattern is not a
generic wake detector placed after an unrelated separator: the same enrolled
speaker representation is used throughout streaming extraction, personal
activity/presence, speaker verification, and keyphrase decisions. This maps
closely to Personal VAD/TS-VAD and personalized keyphrase-detection work. The
review also identified BC-ResNet and MatchboxNet as compact KWS backbones and
multichannel SpeakerBeam/neural-beamforming systems as references for a future
four-microphone front end.

The University of Rochester Smart Speaker Command Dataset was downloaded
under its CC BY 4.0 license and audited as a clean non-Marvin hard-negative
source. Mycroft Precise Community Data was inspected, but no raw verified
Marvin clip collection was found and it was not added to training. The Smart
Speaker clips were duplicated across the configured channels only to satisfy
the existing evaluator interface; they do not provide spatial or real-cabin
evidence.

The resulting candidate experiments reinforced the data/evaluation priority:
the full hard-negative candidate reduced some negative events but fell to
14.84% wake recall on the synthetic positive proxy; the thin candidate reached
81.25% wake recall and 75.00% overall intent on only 16 scenes but did not
improve the negative diagnostic; the distilled candidate reached 31.25% wake
recall. The retained reference therefore remains unchanged. The thin run at a
0.70 KWS threshold produced zero observed negatives but also zero positive
wakes, demonstrating that a zero-FAR-looking operating point can simply be a
dead detector.

The next architecture experiment recommended by the review is a small causal
speaker-conditioned target/non-target/non-speech activity head, followed by a
speaker-consistent wake detector and, only after matching four-channel data
exists, a spatial/multichannel front end. Quantization and intent-stage
compression are deferred until joint quality is stable.

Artifacts:

- `reports/preexisting_systems_review_2026-10-11.md`
- `reports/smart_speaker_command_dataset_audit_2026-10-11.md`
- `scripts/data/build_target_kws_smart_speaker_negative_cache.py`
- `scripts/eval/eval_smart_speaker_command_negative.py`
- `logs/smart_speaker_command_negative_250_2026-10-11.json`
- `logs/smart_speaker_command_negative_250_candidate_2026-10-11.json`
- `logs/smart_speaker_command_negative_250_thin_candidate_2026-10-11.json`
- `logs/smart_speaker_command_negative_250_thin_candidate_kws070_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_smart_speaker_hardneg_same_speaker_m10_32scenes_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_smart_speaker_thin_same_speaker_m10_8scenes_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_smart_speaker_distill_same_speaker_m10_8scenes_2026-10-11.json`

### E-2026-10-11-negative-event-accounting-fix

The continuous KWS evaluator was counting every positive smoothed window as a
separate event. A sustained wake-like phrase can produce several overlapping
positive windows, so this inflated the reported false-accept rate and made
the Smart Speaker diagnostic harder to compare with a real event-rate gate.
The evaluator now reports both values: positive-window counts for debugging,
and de-duplicated operating events formed by merging positive runs separated
by no more than two seconds. Existing field names remain available, but new
runs interpret `*_smoothed_events` and `*_event_rate_per_hour` as the merged
event values and expose `*_smoothed_positive_windows` separately.

The Smart Speaker evaluator records the same split, including the exact
sequential action sample. This does not turn clean studio audio into cabin
evidence; it makes the negative diagnostic’s unit of measurement auditable.
A regression test covers the case where five positive windows represent two
operating events.

The corrected 250-file replay measured 728.1 merged full-path events/hour and
1,817.2 merged KWS-only events/hour for the retained reference, versus 17.9
and 50.7/hour for the full Smart hard-negative candidate. The exact 14-file
action sample remained 105.9 false actions/hour for the reference and 0.0 for
the candidate. These values should be read together with the positive proxy:
the full candidate only recalled 14.84% of target wakes and was rejected.

The corrected all-18 AISHELL-5 replay is now the preferred long negative
proxy for the readiness audit. It covers 3.889 hours and records 18 merged
full-path events (4.628/hour) and 376 merged KWS-only events (96.673/hour).
The de-duplication changes the accounting semantics, not the AISHELL-5
counts, because its positives were not repeated across adjacent windows.

Artifacts:

- `scripts/eval/eval_aishell5_continuous_kws.py`
- `scripts/eval/eval_smart_speaker_command_negative.py`
- `tests/test_continuous_kws_metrics.py`

### E-2026-10-11-personal-presence-candidate

The first architecture change motivated by the pre-existing-systems review was
implemented as an isolated `PersonalFramePresenceHead`. It keeps the retained
framewise gate available, but modulates log-mel activity features with the
frozen 256-D SpEx+ enrollment embedding. The training script caches both Stage
1 outputs and enrollment embeddings, so the four-epoch experiment remains
reproducible and fast after the initial cache pass.

The internal row-split validation reached chunk F1 1.000 at threshold 0.40,
but this was not treated as evidence of production quality. On the same 16-
scene continuous synthetic proxy, the inherited 0.15/0.10 operating point
reached 81.25% wake recall and 62.50% overall intent but produced 96.4 false
actions/hour. Raising presence to the validation-derived 0.40 removed all
observed false wakes/actions but reduced wake recall to 50.0% and overall
intent to 37.5%. The candidate is rejected and the retained pipeline remains
the reference. This is a useful confirmation that personal activity must be
calibrated jointly on speaker-consistent sessions rather than selected from
internal presence F1.

Artifacts:

- `src/car_tse/models/presence_head.py`
- `src/car_tse/pipeline/in_car_pipeline.py`
- `configs/stage2_personal_presence_dual_domain_nonwake_ft1.yaml`
- `configs/eval_end_to_end_cavemove_personal_presence_dual_domain_stage2_candidate.yaml`
- `checkpoints/stage2_personal_presence_dual_domain_nonwake_ft1/best.pt`
- `logs/stage2_personal_presence_dual_domain_nonwake_ft1/metrics_last.json`
- `logs/continuous_full_pipeline_proxy_personal_presence_same_speaker_m10_8scenes_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_personal_presence_p040_same_speaker_m10_8scenes_2026-10-11.json`
- `reports/personal_presence_candidate_2026-10-11.md`
- `tests/test_personal_presence_head.py`
- `logs/real_driver_pipeline_evaluation_empty_2026-10-11.json`

### E-2026-10-11-enrollment-bank-candidate

The pre-existing personalized wake-word literature also motivated an optional
multi-utterance enrollment representation. The runtime now accepts a
precomputed normalized mean of SpEx embeddings while retaining the historical
single-utterance waveform for Stage 1 extraction. On identical 16-scene
cross-corpus cabin proxies, one short reference reached 81.25% wake recall and
68.75% overall intent with zero observed false actions; the uncalibrated mean
of three FSC utterances reached only 6.25% wake recall and 6.25% overall
intent, also with zero observed false actions. The bank variant is rejected:
it is safe only because it misses almost every target wake, and the wake
speaker is not verified to match the enrollment speaker.

Detailed experiment record: `reports/enrollment_bank_candidate_2026-10-11.md`.

### E-2026-10-11-preexisting-systems-follow-up

The architecture review was extended with newer streaming and personalized-
wake references. SpeakerBeam-SS (2024) uses state-space modeling and a lighter
causal Conv-TasNet-style frontend to reduce real-time extraction cost. A 2026
personalized-KWS preprint combines a phonetic keyword branch with a compact
text-independent speaker-verification branch and lets the speaker branch veto
the keyword branch at late fusion. These references reinforce two changes for
our system: benchmark a causal/streaming Stage 1 only after the data contract
is stable, and make speaker identity an explicit independent wake decision
instead of relying on a single target-KWS score.

No checkpoint was promoted from this literature review. The references use
different datasets, languages, microphone layouts, and operating points; the
proposed dual-branch wake experiment therefore remains gated on a permitted
same-driver pilot with passenger-“Marvin” and long no-command negatives.

References:

- `https://arxiv.org/abs/2407.01857`
- `https://arxiv.org/abs/2606.20106`
- `reports/preexisting_systems_review_2026-10-11.md`.

### E-2026-10-11-late-speaker-fusion-candidate

The research recommendation was made executable as an opt-in runtime
candidate. It maps the SpEx cosine similarity to an explicit speaker score
and fuses it with the target-KWS score before smoothing; the default remains
the historical hard-gate path. On the eight-scene cabin proxy, late `min`
fusion at center 0.24/scale 0.04 retained 100% wake and intent with zero
false actions, but it matched the retained Smart-negative result at 728.1
full-path events/hour. Tightening the center to 0.26 reduced Smart negatives
to 695.3/hour but dropped positive wake/intent to 87.5%. Product fusion at
the same center/scale dropped positive wake/intent to 25.0%. No candidate was
promoted because the raw speaker similarity is not calibrated for this domain.

The result is useful as a design check: an independent speaker branch is
worth keeping, but its operating point must be trained or calibrated from
same-driver target/passenger sessions. Blindly applying a sigmoid to the
existing SpEx embedding does not solve the production false-accept problem.

Detailed experiment record: `reports/late_speaker_fusion_candidate_2026-10-11.md`.

### E-2026-10-11-runtime-domain-verifier-calibration

To test whether the late-fusion failure was only a threshold problem, the
frozen Stage-1 speaker embedding was calibrated on the cached Stage-3
target-KWS cabin mixtures. This used 1,208 train, 260 validation, and 256 test
rows with balanced target-present/target-absent labels. The held-out test AUC
was 0.515. At the validation operating point selected for about 1% FPR, test
TPR was 1.56%. In the end-to-end proxy the calibrated candidate retained zero
observed false events but reduced target wake recall to 56.25% and overall
intent accuracy to 43.75%. No threshold or checkpoint was promoted.

The conclusion is that the extracted embedding itself is not a reliable
speaker-verification feature in this domain. The next improvement must be a
dedicated speaker-conditioned activity/verifier head trained from permitted,
speaker-consistent driver/passenger sessions; more threshold tuning on proxy
data would be misleading.

Artifacts:

- `scripts/eval/calibrate_cached_spex_speaker_verifier.py`
- `reports/speaker_verifier_calibration_2026-10-11.md`
- `logs/speaker_verifier_calibration_gsc_cache_stage1_ft2_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_late_fusion_gsc_cache_calibrated_m10_2026-10-11.json`

### E-2026-10-11-pretrained-speaker-encoder-follow-up

The literature-based next step was tested with Microsoft's pretrained
`microsoft/wavlm-base-plus-sv` speaker-verification encoder. On the full
synthetic CAVEMOVE/GSC-Marvin cabin cache, raw WavLM cosine reached 0.839 test
AUC versus 0.515 for the current SpEx extracted embedding. A compact relation
head trained on frozen WavLM embeddings reached 0.896 test AUC. At the
validation-selected approximately 1% FPR operating point, the relation head
still achieved only 37.5% test TPR and 2.34% test FPR, so it was not promoted.

The resource benchmark measured 25.7 ms p50 / 31.6 ms p95 for one 1-second
forward on the RTX 4080, 101.1M parameters, and 443.8 MB peak CUDA memory. The
model is useful as a teacher/diagnostic reference, but not as an always-on
automotive branch. The next aligned step is to collect permitted same-driver
sessions and distill a small speaker-conditioned activity/verifier model with
thresholds selected on tuning drivers.

Artifacts:

- `scripts/eval/eval_pretrained_wavlm_speaker_verifier.py`
- `scripts/train/train_wavlm_speaker_verifier_head.py`
- `scripts/eval/benchmark_pretrained_wavlm_speaker_verifier.py`
- `reports/wavlm_speaker_verifier_candidate_2026-10-11.md`
- `logs/wavlm_speaker_verifier_cabin_diagnostic_2026-10-11.json`
- `logs/speaker_verifier_wavlm_cabin_ft1_2026-10-11.json`
- `logs/wavlm_speaker_verifier_latency_2026-10-11.json`

### E-2026-10-11-automotive-dataset-sourcing

The follow-up dataset search found the official ANIR automotive noise and
impulse-response corpus. Its free V-class impulse-response archive was
downloaded to `D:\datasets\anir`, SHA-256 verified, and extracted into 682
48-kHz `.fir` files. The complete V-class noise archive was also downloaded,
SHA-256 verified, and extracted into 1,302 48-kHz WAV files across 42
conditions. Both are reserved for future spatial/cabin augmentation and
hard-negative testing; there are no wake/intent labels and the microphone
geometry does not yet match the project's four-channel contract.

iCarB-Voice is a strong research lead for speaker-conditioned in-car modeling,
but the official Zenodo record marks its 121.7-GB files restricted under a
non-commercial EULA. No access was assumed. No public resource found in this
search replaces the missing consented same-driver English pilot.

Detailed sourcing record: `reports/automotive_dataset_sourcing_review_2026-10-11.md`.

### E-2026-10-11-ANIR-noise-only-diagnostic

The complete ANIR V-class noise archive was downloaded and indexed: 1,302
48-kHz WAV files across 42 conditions and 31 microphone IDs. A noise-only
target-absent run using microphones 0-3 and the retained target-KWS reference
path covered 0.35 hours. The full gated path produced 0 observed events/hour;
the KWS-only stream produced 8.57 events/hour, with one event in each of
conditions 04_00, 09_00, and 10_00. This supports ANIR as a hard-negative and
augmentation candidate, but it does not provide speech, passenger, wake, or
command evidence and is not wired into training yet.

Detailed evaluation record: `reports/anir_noise_target_absent_2026-10-11.md`.

### E-2026-10-11-recording-hardware-preflight

The non-mutating recording-device check found a Fifine microphone exposed as
one- and two-channel inputs, but no available four-channel cabin microphone
array. The production configuration expects four microphones and the
real-driver protocol requires the production microphone layout. No recording
was started, because consent and a matching capture setup are prerequisites.
A mono/stereo pilot could be useful as exploratory evidence after explicit
labeling, but it must not be used to satisfy the current production-hardware
gate or be mixed into the four-channel claim.

The readiness audit now treats matching sample rate and configured microphone
count as an explicit gate when sessions exist; the current empty tree and the
hardware preflight therefore cannot pass it.

Artifact: `scripts/data/record_real_driver_session.py --list-devices` was run
read-only; no audio or participant data was created.

Artifact: `reports/capture_hardware_preflight_2026-10-11.md`.

### E-2026-10-11-release-reproducibility-manifest

The readiness audit was extended to generate a release identity: Git revision,
working-tree cleanliness, runtime-config SHA-256, paths-config SHA-256, and
SHA-256 hashes for every selected checkpoint. A new reproducible-release gate
requires the source tree to be clean and all referenced artifacts to be
present and hashed. The current tree is intentionally dirty because this
research work is still in progress, so the gate is blocked rather than being
silently treated as releaseable.

The machine-readable snapshot and Markdown rendering now expose the release
identity, allowing a future sign-off to archive the exact code/config/model
combination used for the held-out test.

Artifact: `scripts/reporting/audit_production_readiness.py`.

### E-2026-10-11-threshold-selection-safety-fix

The held-out real-driver evaluator was re-audited for a subtle selection
failure. It computed both conservative unmatched-wake false accepts/hour and
false actions/hour, but the threshold-search pool filtered only on the
false-action limit. A candidate could therefore be selected because it never
acted while still accepting too many passenger or no-command wake events.

The selector now requires the combined safety predicate before optimizing
wake recall and command accuracy. A regression test constructs exactly this
case and confirms that the higher-safety threshold is selected. The repository
suite now reports 24 passed and 1 skipped.

Artifacts:

- `scripts/eval/eval_real_driver_sessions.py`
- `tests/test_real_driver_session_validation.py`

### E-2026-10-11-ANIR-presence-augmentation

The downloaded ANIR V-class automotive noise archive was converted into 42
matched four-channel 16-kHz noise beds using microphones 0–3. An augmented
personal-presence candidate was trained with 604 additional target-positive
copies and the existing hard negatives. Its internal chunk F1 remained 1.000,
but the same 16-scene cross-corpus cabin proxy was unchanged: 81.25% wake
recall, 62.50% overall intent, 208.93 false wakes/hour, and 96.43 false
actions/hour. The checkpoint was rejected; the ANIR cache is retained as a
future noise-diversity asset, not as driver evidence.

Artifacts:

- `scripts/data/build_anir_multichannel_cache.py`
- `scripts/data/build_anir_augmented_presence_manifest.py`
- `reports/anir_presence_augmentation_candidate_2026-10-11.md`
- `checkpoints/stage2_personal_presence_dual_domain_nonwake_anir_ft1/best.pt`
- `logs/continuous_full_pipeline_proxy_personal_presence_anir_same_speaker_m10_8scenes_2026-10-11.json`

### E-2026-10-11-compact-speaker-verifier-runtime-candidate

The pretrained-speaker result was converted into a small opt-in runtime
candidate rather than silently replacing the baseline. A 176,579-parameter
log-mel relation verifier distilled from the frozen WavLM head reached 0.917
held-out test AUC and ran in 5.14 ms p50 / 5.24 ms p95 for two one-second
waveforms on the RTX 4080, with 10.5 MB peak CUDA allocation. Its validation
selected approximately 1% FPR point transferred poorly: 16.4% test TPR at
2.34% FPR.

The end-to-end proxy confirmed that this is not currently a usable replacement.
At the validation threshold 0.9904 it detected 0/2 target wakes; a deliberately
relaxed 0.80 diagnostic point detected 1/16 target wakes and 0/16 commands,
with zero observed false events in both small runs. The retained baseline on an
8-scene comparison had 100% wake and intent with zero observed false events.
The compact branch is therefore rejected for promotion, but retained as a
resource-efficient implementation point for future same-driver training.

Artifacts:

- `reports/compact_speaker_verifier_runtime_candidate_2026-10-11.md`
- `src/car_tse/models/speaker_verifier.py`
- `src/car_tse/pipeline/in_car_pipeline.py`
- `scripts/eval/eval_continuous_full_pipeline_proxy.py`
- `configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_compact_verifier_candidate.yaml`
- `logs/compact_speaker_verifier_gsc_ft1_2026-10-11.json`
- `logs/compact_speaker_verifier_latency_gpu_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_compact_verifier_operating_diagnostic_2026-10-11.json`

### E-2026-10-12-three-way-personal-activity-head

The literature review identified a concrete weakness in the binary presence
head: passenger speech and silence were both trained as the same negative
class. A new opt-in `PersonalActivityHead` now predicts non-speech, enrolled
target speech, and non-target speech per frame, while the runtime consumes
only the target-speech probability. The existing state machine and default
checkpoint remain unchanged.

The training manifest preserved 1,208 prior GSC-Marvin/AISHELL rows and added
1,200 non-Marvin Speech Commands clips with different-speaker Marvin
enrollment and CAVEMOVE noise. The four-epoch frozen-Stage-1 run used 1,926
training and 482 validation rows; best validation chunk F1 was 0.697 at a
0.40 target threshold.

On the fixed 16-scene cross-corpus proxy, the 0.25 diagnostic operating point
reduced false actions from 96.43/hour for the existing personal head to
48.21/hour, while wake recall changed from 81.25% to 75.00% and overall intent
from 62.50% to 56.25%. A 0.30 point dropped wake recall to 25.00% and a 0.40
point rejected all target scenes. False wakes remained 208.93/hour at the
usable diagnostic points. The candidate is therefore rejected for promotion,
but the architecture is retained for same-driver/passenger training when the
real pilot exists.

The same 0.25 point was replayed on the retained 250-file Smart Speaker
non-Marvin diagnostic: merged full-path events fell modestly from 728.1/hour
to 674.4/hour, while KWS-only events remained 1,817.2/hour and the 14-file
sequential action sample remained 105.9 false actions/hour. This did not meet
the safety-improvement bar.

Artifacts:

- `src/car_tse/models/presence_head.py`
- `src/car_tse/losses/presence.py`
- `scripts/data/build_personal_activity_manifest.py`
- `scripts/train/train_stage2_frame_presence.py`
- `configs/stage2_personal_activity_gsc_cavemove_ft1.yaml`
- `configs/eval_end_to_end_cavemove_personal_activity_candidate.yaml`
- `configs/eval_end_to_end_cavemove_personal_activity_operating_diagnostic.yaml`
- `reports/personal_activity_3way_candidate_2026-10-12.md`
- `logs/continuous_full_pipeline_proxy_personal_activity_operating_p025_same_speaker_m10_8scenes_2026-10-12.json`
- `logs/continuous_full_pipeline_proxy_personal_activity_operating_p030_same_speaker_m10_8scenes_2026-10-12.json`
- `logs/continuous_full_pipeline_proxy_personal_activity_same_speaker_m10_8scenes_2026-10-12.json`
- `logs/smart_speaker_command_negative_250_personal_activity_p025_2026-10-12.json`

### E-2026-10-12-speaker-disjoint-personal-activity-retrain

The first three-way activity candidate used a row split, so its validation
score could be optimistic for repeated public speakers. The manifest builder
now writes a stable enrollment-speaker `group_id`, and the training script
supports `group_by_speaker: true`. The fresh four-epoch run used 1,925 training
rows across 912 groups and 483 validation rows across 225 groups, with zero
group overlap. Its best validation chunk F1 was 0.755 at threshold 0.45 and
its final loss was 0.3361.

The fixed cross-corpus proxy did not improve. At 0.35 the candidate rejected
all 16 scenes; at 0.25 it produced 75.0% wake recall, 56.25% overall intent,
208.93 false wakes/hour, and 96.43 false actions/hour. The earlier row-split
candidate at 0.25 had the same wake/intent and 48.21 false actions/hour, while
the binary personal head had 81.25% / 62.50% and 96.43 actions/hour. No
checkpoint or threshold was promoted. The result strengthens the conclusion
that cross-corpus speaker consistency and real passenger labels, rather than
another proxy threshold, are the limiting evidence.

Artifacts:

- `scripts/data/build_personal_activity_manifest.py`
- `scripts/train/train_stage2_frame_presence.py`
- `tests/test_grouped_presence_split.py`
- `configs/stage2_personal_activity_gsc_cavemove_group_ft1.yaml`
- `configs/eval_end_to_end_cavemove_personal_activity_group_p035.yaml`
- `configs/eval_end_to_end_cavemove_personal_activity_group_p025.yaml`
- `reports/personal_activity_group_split_candidate_2026-10-12.md`
- `logs/stage2_personal_activity_gsc_cavemove_group_ft1/metrics_last.json`
- `logs/continuous_full_pipeline_proxy_personal_activity_group_p035_same_speaker_m10_8scenes_2026-10-12.json`
- `logs/continuous_full_pipeline_proxy_personal_activity_group_p025_same_speaker_m10_8scenes_2026-10-12.json`

### E-2026-10-12-AISHELL-target-absent-personal-activity-evaluation

The AISHELL-5 continuous negative evaluator was extended so the new
three-way `PersonalActivityHead` can be selected directly. A full 3.889-hour
replay of all 18 real four-channel `Eval2` sessions was then run with the
group-split candidate, a fixed LibriSpeech enrollment, the target-speaker KWS
checkpoint, and two-hit smoothing. The fixed enrollment speaker is absent from
AISHELL-5, so this is a target-absent rejection check only.

At the diagnostic operating point (presence 0.25, speaker gate 0.24, KWS 0.50),
the full path produced 4.89 merged events/hour, while the KWS-only stream
produced 96.69/hour. The aggregate KWS threshold sweep was 11.83, 5.66, 4.89,
and 3.09 full-path events/hour at KWS thresholds 0.30, 0.40, 0.50, and 0.60.
Holding KWS at 0.50, the presence sweep produced 9.51, 8.74, 6.94, 4.89,
1.80, and 0.77 full-path events/hour at presence thresholds 0.10 through 0.35.
The KWS-only rate stayed at 96.69/hour, isolating the rejection contribution of
the activity/speaker gate.
The candidate remains unpromoted because the run contains no same-driver wake
or command positives and its thresholds were not frozen on real-driver tuning
sessions.

The evaluator also now stops cleanly when the requested duration budget is
exhausted, avoiding zero-second session rows and meaningless events/hour
values. Aggregate threshold-sweep metrics are stored at the root of each log
for reproducible reporting.

Artifacts:

- `scripts/eval/eval_aishell5_continuous_kws.py`
- `reports/aishell5_personal_activity_target_absent_2026-10-12.md`
- `logs/aishell5_target_absent_personal_activity_group_all18_presence_sweep_2026-10-12.json`

### E-2026-10-12-research-informed-medium-intent-student

The pre-existing-systems review was converted into a bounded edge experiment.
VoiceFilter-Lite and personalized-keyphrase work emphasize a small
speaker-conditioned branch, clean-speech preservation, and an independent
speaker-verification veto. SpeakerBeam-SS suggests that future Stage 1
efficiency work should use a causal/state-space design rather than simply
scaling a recurrent model. Those references motivated a controlled test of
the downstream compact intent branch while leaving the default pipeline
unchanged.

The existing normalized student (hidden 256, two bidirectional LSTM layers,
temperature-3 teacher distillation, KD weight 0.5) was compared with a medium
hidden-384 student trained on the same cached cabin-domain data and teacher
logits. The medium candidate reached 50.4% held-out cabin/TSE validation
accuracy and 73.3% official clean FSC test accuracy, versus 56.3% and 80.5%
for the existing student. On a 40-scene same-speaker diagnostic proxy covering
four cabin SNRs, it reached 75.0% wake recall, 50.0% overall intent accuracy,
66.7% intent given wake, and zero observed false actions. The existing student
reached 68.8%, 59.4%, 86.4%, and zero respectively on its 32-scene comparison
proxy. The medium candidate therefore was rejected for promotion despite a
larger 23.94M-parameter model; its 96.1 MB checkpoint also increased footprint.

The pipeline loader was fixed to read CRNN dimensions from the evaluation
configuration. This makes non-default candidate architectures reproducible and
prevents a checkpoint/config mismatch from silently constructing the default
256-unit network. The retained default checkpoint and production status were
not changed.

Artifacts:

- `reports/research_informed_intent_student_candidate_2026-10-12.md`
- `configs/stage4_intent_tse_domain_norm_student_medium.yaml`
- `configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_norm_crnn_medium_candidate.yaml`
- `checkpoints/stage4_intent_tse_domain_distill_medium_t3_a05/best.pt`
- `logs/continuous_full_pipeline_proxy_medium_student_10scenes_2026-10-12.json`
- `logs/full_pipeline_latency_medium_student_gpu_2026-10-12.json`

### E-2026-10-12-capture-contract-preflight

The host was re-queried before any pilot capture. It exposes only the Fifine
input with one channel through MME/DirectSound/WDM-KS and two channels through
WASAPI; no four-channel cabin array is available. The requested production
contract is 16 kHz, four-channel input, so no participant recording was
started.

The recording utility was hardened so it now queries the selected device and
calls `check_input_settings` before creating a session directory. A four-channel
request against the available one-channel device fails with a clear error and
leaves no empty session folder. Successful captures also record the resolved
device capability in `session_metadata.json`. This prevents a failed hardware
attempt from being mistaken for a permitted or usable pilot session.

Artifacts:

- `scripts/data/record_real_driver_session.py`
- `tests/test_record_real_driver_capture_preflight.py`
- `reports/capture_hardware_preflight_2026-10-11.md`

### E-2026-10-12-safe-threshold-selection

The real-driver evaluator had a safety ambiguity: when no tuning threshold
combination met both the false-accept and false-action limits, it selected the
highest-recall candidate anyway. That could make an unsafe point look like a
frozen operating point in a diagnostic report.

The evaluator now selects only from candidates satisfying both safety limits.
If none exist, it chooses the least-unsafe point only to produce diagnostics,
labels the selection
`no_safe_tuning_point_diagnostic_fallback`, records a zero safe-candidate
count, and forces `production_valid` false. A safe tuning point is now an
explicit production gate. Unit tests cover both the safe and no-safe paths.
The current empty pilot tree still stops before threshold selection, so this
change has not been used to claim a real-driver result.

Artifacts:

- `scripts/eval/eval_real_driver_sessions.py`
- `tests/test_real_driver_threshold_selection.py`
- `logs/real_driver_pipeline_evaluation_2026-10-12.json`

### E-2026-10-12-ReMASC-sample-audit

The public ReMASC repository was reviewed as a possible English in-vehicle
speech resource. Its README describes 50 subjects, 49 vehicle commands, quiet
and moving-vehicle recordings, genuine/source/replayed labels, and microphone
arrays including a 7-channel 16-kHz array. It states that the corpus is free
for academic and commercial use; the complete set is approximately 92.3 GB
and is downloaded through a free IEEE DataPort account.

The publicly linked 10.9 MB sample was downloaded to `D:\datasets\remasc` and
SHA-256 verified. It contains 12 WAV files, `meta.csv`, and `readme.txt`, with
16/44.1-kHz and 1/2/4/6/7-channel examples. The vehicle example is a genuine
2-channel/replayed 6-channel pair, not the project’s four-channel production
layout.

ReMASC is useful for future English vehicle-command, array, and replay-stress
experiments, but it does not provide the continuous same-driver enrollment →
Marvin → command sequence, passenger Marvin negatives, or labeled no-command
intervals required for production acceptance. The sample was therefore not
added to training and no ReMASC score was presented as real-driver pipeline
evidence.

Artifacts:

- `data/sources.json`
- `data/manifests/external/remasc_sample.json`
- `reports/automotive_dataset_sourcing_review_2026-10-11.md`
- `D:\datasets\remasc\sample_v2.zip`

### E-2026-10-12-condition-level-pilot-reporting

The real-driver evaluator previously emitted only whole-study totals. That was
insufficient for the declared pilot protocol, which requires separate results
for road/HVAC, passenger, no-command, and noise-only conditions. The evaluator
now records a `by_condition` breakdown in both each session result and the
aggregate result. It reports driver-wake recall, command accuracy, false
accepts, false actions, and rates normalized by the union of labeled negative
intervals, so overlapping annotations do not inflate exposure hours.

This changes reporting only; it does not alter inference, threshold selection,
or the production gates. The empty permitted-session tree was rechecked and
still correctly stops evaluation before any model score is claimed.

Artifacts:

- `scripts/eval/eval_real_driver_sessions.py`
- `tests/test_real_driver_condition_metrics.py`
- `logs/real_driver_pipeline_evaluation_2026-10-12.json`

### E-2026-10-12-preexisting-systems-remasc-stress

The pre-existing-systems review was extended with primary references for
speaker-conditioned personal VAD, personalized keyphrase detection, adaptive
multichannel noise reduction, and production-oriented streaming KWS. The
common design pattern is a small always-on keyword branch plus a separate
speaker-conditioned activity/verification decision, with explicit streaming
state, long negative exposure, and a frozen operating point. It is not simply
“train a larger intent model.”

To test the most relevant hypothesis without claiming real-driver evidence, a
60-second six-channel ReMASC replay was resampled to 16 kHz and tiled as a
vehicle-only stress bed. The bed was mixed into the fixed synthetic target
scenes at -5/0/5/10 dB cabin SNR. The official ReMASC sample and README were
audited separately; the sample is useful for vehicle/replay stress but does
not provide the continuous same-driver enrollment, Marvin, command,
passenger-Marvin, and no-command protocol.

The retained frame-presence cascade achieved 100% wake recall, 90.625% intent
accuracy, 257.14 diagnostic false wakes/hour, and 64.29 diagnostic false
actions/hour over 32 scenes (448 seconds). The speaker-disjoint three-way
personal-activity candidate at threshold 0.25 preserved 100% wake recall and
raised intent accuracy to 96.875%, but increased diagnostic false actions to
128.57/hour. At threshold 0.35 it rejected all target scenes. An asymmetric
0.25 wake / 0.35 command gate retained wake recall but produced zero correct
commands and 192.86 diagnostic false actions/hour. These are stress-proxy
measurements, not production FAR/FRR.

The decision is to retain the 3-way head as a research implementation but not
promote it. The observed failure is consistent with inadequate target-vs-
passenger calibration and speaker mismatch in the training protocol. The next
experiment should use permitted same-driver continuous sessions to train and
calibrate a target/passenger/non-speech head plus an independent compact
speaker-verification veto; only then should a two-factor wake-to-action rule
be evaluated. Replacing the KWS backbone or increasing intent capacity is not
the highest-value next step.

The runtime was then extended with an opt-in target-minus-passenger activity
margin veto. It uses the same frozen three-way checkpoint and adds no learned
parameters: target and non-target posteriors are aggregated independently over
the top activity frames, and the difference must exceed a configured margin.
The default pipeline remains unchanged. On the ReMASC stress proxy, a margin
of -0.67 gave 90.6% wake recall, 78.1% intent accuracy, 64.3 diagnostic false
wakes/hour, and 0 observed false actions/hour. The neighboring -0.68 point
reintroduced 128.6 false actions/hour, while -0.66 fell to 62.5% wake recall.
Because the threshold was selected from this same diagnostic proxy, it is not
a production operating point.

On the full 18-session, 3.889-hour AISHELL-5 Eval2 target-absent replay, the
same margin reduced full-path smoothed events from 4.89/hour to 3.60/hour,
while KWS-only remained 96.69/hour. This is a modest negative-rejection gain
on real continuous cabin audio; it provides no positive driver recall and does
not unblock the production gates. The correct next experiment remains
speaker-disjoint permitted driver/passenger data and calibration of this
margin jointly with the independent speaker-verification veto.

Artifacts:

- `scripts/data/build_remasc_cabin_stress_bed.py`
- `configs/eval_end_to_end_cavemove_personal_activity_group_p025_cmdp035.yaml`
- `logs/continuous_full_pipeline_proxy_remasc_vehicle_stress_8scenes_2026-10-12.json`
- `logs/continuous_full_pipeline_proxy_remasc_personal_activity_group_p025_8scenes_2026-10-12.json`
- `logs/continuous_full_pipeline_proxy_remasc_personal_activity_group_p035_8scenes_2026-10-12.json`
- `logs/continuous_full_pipeline_proxy_remasc_personal_activity_group_p025_cmdp035_8scenes_2026-10-12.json`
- `configs/eval_end_to_end_cavemove_personal_activity_group_p025_margin_m067.yaml`
- `logs/continuous_full_pipeline_proxy_remasc_personal_activity_group_p025_margin_m0p66_8scenes_2026-10-12.json`
- `logs/continuous_full_pipeline_proxy_remasc_personal_activity_group_p025_margin_m0p67_8scenes_2026-10-12.json`
- `logs/continuous_full_pipeline_proxy_remasc_personal_activity_group_p025_margin_m0p68_8scenes_2026-10-12.json`
- `logs/continuous_full_pipeline_proxy_remasc_personal_activity_group_p025_margin_m0p70_8scenes_2026-10-12.json`
- `logs/aishell5_target_absent_personal_activity_group_all18_margin_m067_2026-10-12.json`

### E-2026-10-12-frame-aligned-activity-margin

The three-way activity margin was audited against the framewise target/non-
target decisions used by personal-VAD style systems. The historical
`independent_top` aggregation compares independently selected target and
passenger frames, so two opt-in frame-aligned modes were implemented:
`target_top` and `framewise_top`. The historical behavior remains the default.

On the same 32-scene / 448-second ReMASC stress proxy, both new modes matched
the permissive no-veto result (100.0% wake, 96.9% intent, 128.6 diagnostic
false actions/hour). With a cutoff of -0.30, `target_top` reached 50.0% wake /
28.1% intent and `framewise_top` reached 68.8% wake / 40.6% intent; each had
64.3 diagnostic false wakes/hour and zero observed false actions/hour. The
target-top -0.30 mode reduced the all-session AISHELL-5 Eval2 target-absent
rate only from 3.60/hour to 3.34/hour, while KWS-only stayed at 96.69/hour.

The frame-aligned modes are therefore retained as reproducible research
options but rejected for promotion. They do not close the calibration gap, and
their thresholds cannot be frozen without permitted same-driver tuning data.

Artifacts:

- `reports/personal_activity_margin_aggregation_2026-10-12.md`
- `src/car_tse/models/presence_head.py`
- `src/car_tse/pipeline/in_car_pipeline.py`
- `scripts/eval/eval_continuous_full_pipeline_proxy.py`
- `scripts/eval/eval_aishell5_continuous_kws.py`
- `logs/continuous_full_pipeline_proxy_remasc_personal_activity_targettop_m030_8scenes_2026-10-12.json`
- `logs/continuous_full_pipeline_proxy_remasc_personal_activity_framewisetop_m030_8scenes_2026-10-12.json`
- `logs/aishell5_target_absent_personal_activity_group_all18_margin_targettop_m030_2026-10-12.json`

### E-2026-10-12-streaming-latency-command-window

The retained one-second-hop pipeline was stress-tested at 500 ms and 250 ms
hops. The dense stream initially reduced intent accuracy because the first
post-wake three-second window still overlapped the wake word. An opt-in
`command_delay_calls` state-machine parameter was added to wait for a clean
command window; the default remains zero for exact historical behavior.

On the 32-scene / 448-second ReMASC stress proxy, the best diagnostic setting
used a 250 ms hop, four delayed command calls, and an eight-call refractory
period. It reached 96.9% wake recall, 81.3% intent accuracy, 64.3 diagnostic
false wakes/hour, zero observed false actions/hour, and 0.968 s mean wake
latency (1.625 s p95). The one-second reference was 90.6% wake, 78.1% intent,
64.3 false wakes/hour, zero false actions/hour, and 1.879 s mean latency.

This is a useful latency/state-machine candidate, not production evidence: the
scene construction is synthetic/cross-corpus and the timing is on an RTX 4080.
The fast setting must be calibrated on real driver tuning sessions and
measured on the intended automotive CPU/DSP before promotion.

The long-negative follow-up rejected it as a current safety improvement. Over
1,200 seconds of AISHELL-5 Eval2 target-absent audio, the 250 ms hop produced
6.0 full-path events/hour with both eight-call and sixteen-call refractory
settings, compared with 3.0/hour for the comparable one-second reference.
The longer refractory reduced KWS-only events but not full-path events. The
one-second configuration therefore remains the reference until real-driver
threshold tuning is available.

Artifacts:

- `reports/streaming_latency_candidate_2026-10-12.md`
- `src/car_tse/pipeline/in_car_pipeline.py`
- `scripts/eval/eval_continuous_full_pipeline_proxy.py`
- `configs/eval_end_to_end_cavemove_personal_activity_group_p025_margin_m067_fast_stream.yaml`
- `logs/continuous_full_pipeline_proxy_remasc_personal_activity_margin_m067_hop025_delay4_8scenes_2026-10-12.json`
- `logs/continuous_full_pipeline_proxy_remasc_personal_activity_margin_m067_fast_stream_hop025_8scenes_2026-10-12.json`
- `scripts/eval/eval_aishell5_continuous_kws.py`
- `logs/aishell5_target_absent_personal_activity_hop025_ref8_1200s_2026-10-12.json`
- `logs/aishell5_target_absent_personal_activity_hop025_ref16_1200s_2026-10-12.json`

### E-2026-10-11-preexisting-systems-research-update

The project was compared against established personalized-keyphrase,
personal-VAD/TS-VAD, streaming targeted-separation, multichannel keyword
spotting, and production-oriented KWS references. The shared design pattern
is a compact streaming phonetic branch plus an independent target-speaker
activity/verification branch, followed by a stateful temporal and action veto.
The references also emphasize clean-speech preservation, spatial/noise
coherence for multichannel inputs, and evaluation over long negative exposure.

The review of ICMC-ASR/AISHELL-5, ICSRC, SPID, CENSREC-3, CI-AVSR, ReMASC,
and restricted/commercial automotive corpora found useful component or noise
data, but no permitted public corpus that supplies the complete English
same-driver enrollment -> Marvin -> command -> passenger-Marvin -> no-command
protocol. Therefore no additional dataset was added to training from this
pass. The current implementation direction is retained: collect the pilot,
train/calibrate the three-way target activity and independent speaker veto on
that pilot, then compare full and ablated cascades using false actions/hour as
the primary safety metric.

Artifact:

- `reports/preexisting_systems_research_update_2026-10-11.md`

The review also covered the recent EvoTSE direction: evolve the enrollment
only from reliability-filtered high-confidence target estimates. This is
retained as a later enhancement for acoustic drift, not enabled now; the
current evidence does not justify self-updating a driver profile before the
speaker/action gates are calibrated on real sessions.

### E-2026-10-11-negative-exposure-rate-audit

The real-driver evaluator was audited for false-accept and false-action
denominators. Its primary aggregate rates had been normalized by total session
duration, while the condition breakdown already used the union of labeled
negative intervals. The evaluator now records both forms, uses
`false_accepts_per_negative_hour` and `false_actions_per_negative_hour` for
threshold selection and held-out production gates, and keeps the historical
total-recording rates for comparison. A compatibility fallback is retained for
older synthetic summaries without exposure metadata.

The empty-session recheck still refuses to score the study, and the full test

The same evaluator now writes 95% Wilson intervals for driver-wake recall and
command accuracy and 95% Poisson-rate intervals for false accepts/actions over
negative exposure. These intervals are reported at session, aggregate, and
condition levels so the eventual pilot cannot present a small-sample point
estimate without its uncertainty.

Artifacts:

- `scripts/eval/eval_real_driver_sessions.py`
- `tests/test_real_driver_session_validation.py`
- `reports/production_readiness_gates.md`
- `logs/real_driver_pipeline_evaluation_ratefix_empty.json`

### E-2026-10-12-smart-speaker-command-negative-audit

The external University of Rochester Smart Speaker Command Dataset was
re-audited as an existing hard-negative resource rather than a merely
potential dataset. It contains approximately 2,110 clean 48-kHz mono clips
from seven speakers, each scripted as `Hey Alexa, <command>`. Because the
current system listens for `Marvin`, these clips are negative for wake-word
evaluation while still exercising continuous wake-like speech followed by a
command. The repository already contains a cache builder and evaluator for
this corpus.

On the same 250-file Speaker1 subset, the current split-gate target-KWS
candidate reduced full-path false wakes from 728.09/hour to 17.90/hour and
reduced KWS-only false wakes from 1,817.23/hour to 50.73/hour. A 14-file,
68.02-second sequential action sample produced two false actions for the
earlier baseline and zero for the candidate. This is a useful safety/debugging
gain, but it is clean single-channel studio audio with duplicated channels,
one source speaker, no cabin noise, and no positive `Marvin` examples; it is
not real-driver or production FAR evidence.

The full 2,110-file all-speaker diagnostic was started separately and is
pending completion. No checkpoint was overwritten and no training was started
for this audit.

Artifacts:

- `reports/smart_speaker_command_dataset_audit_2026-10-11.md`
- `scripts/eval/eval_smart_speaker_command_negative.py`
- `scripts/data/build_target_kws_smart_speaker_negative_cache.py`
- `logs/smart_speaker_command_negative_250_dedup_baseline_2026-10-11.json`
- `logs/smart_speaker_command_negative_250_dedup_candidate_2026-10-11.json`
- `logs/smart_speaker_command_negative_all_2026-10-12.json` (pending)

### E-2026-10-12-compact-cabin-verifier-safety

The retained SpEx speaker gate is nearly uninformative as an identity score
(AUC ~0.5). The cabin-aware target-KWS cache now contains 768 real AISHELL-5
target-absent windows per train/Dev/Eval1 split, which makes it possible to
test an independent identity veto on real cabin data. The existing compact
verifier (176,579 parameters) was evaluated on that cache with restored
enrollment waveforms and the cached Stage 1 ft1 extractions.

Standalone separation is strong on the negative side: AUC 0.9555 (Dev) and
0.9764 (Eval1); the AISHELL windows have mean score 0.059 and a 0.0-5.6%
pass rate over thresholds 0.2-0.9, while synthetic same-speaker target pairs
pass at 84-96% over 0.2-0.5. The earlier cascade failure used a 1%-FPR
threshold (0.9079) that retains only ~52% of synthetic positives; the sweep
provides a usable diagnostic range instead.

The AISHELL evaluator gained an opt-in compact speaker-gate backend (default
behavior unchanged; the SpEx 0.24 regression replay reproduces 3.0/hour and
60.0/hour exactly). Over all 18 Eval2 sessions (3.889 hours), the compact gate
reduced full-path smoothed false accepts from 18 events / 4.628 per hour
(reference) to 7 events / 1.80 per hour at threshold 0.50 (one-sided Poisson
p = 0.003) and 11 events / 2.83 per hour at threshold 0.20 (p = 0.055), with
the KWS-only rate unchanged at 96.69 per hour.

The positive 8-scene cross-corpus proxy drops to 37.5% wake recall at both
thresholds because its Marvin wake clip is not verified to match the
enrollment speaker; a functioning identity gate is expected to reject those
wakes, so the proxy cannot measure this gate's recall. Speaker-consistent
driver recall remains unmeasured and requires the consented pilot. No
checkpoint or default configuration was changed; the compact threshold is not
frozen.

Artifacts:

- `reports/compact_speaker_verifier_cabin_safety_2026-10-12.md`
- `scripts/eval/eval_compact_speaker_verifier_cabin_cache.py`
- `scripts/eval/eval_aishell5_continuous_kws.py`
- `configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_hardneg_aishell5_compact_cabin_safety_candidate_2026-10-12.yaml`
- `configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_hardneg_aishell5_compact_cabin_thr020_candidate_2026-10-12.yaml`
- `logs/compact_speaker_verifier_cabin_cache_diagnostic_2026-10-12.json`
- `logs/aishell5_target_absent_compact_verifier_thr050_all18_2026-10-12.json`
- `logs/aishell5_target_absent_compact_verifier_thr020_all18_2026-10-12.json`
- `logs/aishell5_target_absent_spex_splitgate_regression_1200s_2026-10-12.json`
- `logs/continuous_full_pipeline_proxy_compact_cabin_thr050_same_speaker_p10_2026-10-12.json`
- `logs/continuous_full_pipeline_proxy_compact_cabin_thr020_same_speaker_p10_2026-10-12.json`

### E-2026-10-12-cabin-aware-compact-verifier-retrain

The GSC-trained compact verifier had never seen real cabin negatives, so it
was retrained on a cabin-aware cache built from the same manifests used for
target-KWS adaptation: 3,176/1,328/1,324 train/Dev/Eval1 pairs with 768 real
AISHELL-5 target-absent windows per split, plus the synthetic positive and
non-wake hard-negative rows. Enrollment waves use the source-manifest
enrollment clips with a fixed-enrollment fallback for the AISHELL pseudo-row
manifests; the frozen WavLM relation head provided distillation targets on
fresh WavLM embeddings.

The retrained checkpoint `compact_speaker_verifier_cabin_aware_ft1`
(176,579 parameters) trades a small amount of aggregate AUC for a better
operating region: Eval1 TPR at the validation-selected ~1% FPR point rises
from 51.6% to 66.4%, and the real-cabin pass rate at threshold 0.30 falls
from 3.6% to 2.5% while synthetic target pass moves from 94.5% to 88.3%.
In the full cascade over all 18 AISHELL-5 Eval2 sessions (3.889 hours,
target-absent, KWS-only 96.69/hour in every run), the cabin-aware gate
produced zero smoothed full-path events at both 0.30 and 0.50, versus 7 for
the GSC checkpoint at 0.50 and 18 for the SpEx reference. Per standing rules
this zero count is only a bound (95% Poisson upper limit 0.95/hour), not a
reliable FAR.

The cross-corpus positive proxy stays at 37.5% wake recall for every compact
variant because its Marvin wake clip is not verified to match the enrollment
speaker; that proxy cannot measure this gate's recall. Speaker-consistent
driver evaluation remains blocked on a consented pilot. No existing
checkpoint was modified and no threshold was frozen.

Artifacts:

- `scripts/data/build_cabin_compact_verifier_cache.py`
- `checkpoints/compact_speaker_verifier_cabin_aware_ft1/best.pt`
- `logs/compact_speaker_verifier_cabin_aware_ft1_2026-10-12.json`
- `logs/compact_speaker_verifier_cabin_aware_cache_diagnostic_2026-10-12.json`
- `logs/aishell5_target_absent_compact_cabin_aware_thr030_all18_2026-10-12.json`
- `logs/aishell5_target_absent_compact_cabin_aware_thr050_all18_2026-10-12.json`
- `configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_hardneg_aishell5_compact_cabin_aware_thr030_candidate_2026-10-12.yaml`
- `logs/continuous_full_pipeline_proxy_compact_cabin_aware_thr030_same_speaker_p10_2026-10-12.json`
- `data/cache/compact_speaker_verifier_cabin_aware_2026-10-12/`
- `data/cache/wavlm_speaker_verifier_cabin_aware_2026-10-12/`

### E-2026-10-12-smart-speaker-compact-gate-cross-check

The cabin-aware compact gate was cross-checked on the completed all-file
University of Rochester Smart Speaker Command benchmark (2,110 files, 3.193
hours of clean `Hey Alexa, <command>` speech). The concurrent reference run
used the Smart-Speaker negative-adapted target KWS with the SpEx gate and
recorded 86 full-path false wakes (26.94/hour). Replaying the same
configuration with only the speaker gate swapped to the cabin-aware compact
verifier at 0.30 produced 11 false wakes (3.45/hour), a 7.8x observed
reduction, with the KWS-only rate unchanged at 93.97/hour and zero false
actions in the 14-file action sample. This is a second negative domain for
the same gate, not production evidence: the audio is clean single-channel
studio speech with duplicated channels, and it contains no positive Marvin
examples.

Artifacts:

- `logs/smart_speaker_command_negative_all_compact_cabin_aware_thr030_2026-10-12.json`
- `configs/eval_end_to_end_cavemove_target_kws_smart_speaker_compact_cabin_aware_thr030_2026-10-12.yaml`
