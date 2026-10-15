# Agent handoff context — session 3 (clean-English transfer)

Date: 2026-10-13  
Repository: `D:/Prawin_New/tse/car-tse-main`  
Branch: `full-pipeline`  
Scope: English first; Mandarin AISHELL-5 is auxiliary diagnostic only.

## Current state

The task goal remains active: move the in-car assistant toward documented
production readiness, preserving reproducibility and never presenting synthetic
cross-corpus scenes as real-driver evidence. The 10-minute heartbeat
`continue-in-car-assistant-production-work` is active for this thread.

The current production-readiness audit remains `research_engineering_preview`
with `production_ready=false`. The real-driver directory
`data/raw/custom_incar/sessions` is empty; no consented same-driver recordings
or matching four-channel capture hardware are available in the workspace.

## Clean-English work completed

1. Fixed the clean evaluator configuration to load the Wav2Vec intent head and
   `stage4_intent_wav2vec_ft8_resume_tse_domain/encoder_best.pt`. The previous
   near-uniform intent scores were a CRNN/Wav2Vec harness mismatch.
2. Added `scripts/eval/eval_clean_intent_alignment.py`; standalone FSC command
   alignment is about 0.988 confidence on the expected intent.
3. Built FSC same-speaker clean identity manifests with
   `scripts/data/build_clean_stage1_identity_manifests.py`:
   4,096 train and 1,024 speaker-disjoint validation triplets.
4. Trained clean Stage 1 candidate:
   `checkpoints/stage1_spex_english_clean_identity_ft1/best.pt`.
   Best validation identity SI-SDRi: +8.86 dB.
5. Added clean-aware pipeline timing support:
   `intent_window_offset_seconds` and `command_delay_calls`.
6. Trained a small joint clean+CAVEMOVE transfer candidate:
   `checkpoints/stage1_spex_english_clean_cabin_joint_ft1/best.pt`.
   It preserved the clean control but did not beat the clean candidate on the
   noisy proxy.

## Measured clean proxy results

The end-to-end proxy uses public Speech Commands Marvin clips for wake/enrollment
and FSC clips for commands. These speakers are not verified to be the same
person. Clean scenes contain no passenger, cabin noise, or playback.

On 32 wake identities / 64 scenes / 896 seconds using the clean identity Stage
1 candidate:

| KWS threshold | Wake recall | Intent overall | Intent given wake | False wakes/hour |
|---:|---:|---:|---:|---:|
| 0.25 | 96.9% | 93.8% | 96.8% | 16.1 |
| 0.40 | 93.8% | 90.6% | 96.7% | 8.0 |

These are descriptive proxy numbers only. The false-accept denominator is under
0.25 hours and cannot establish a release FAR.

The retained old Stage 1 at the corrected downstream path scored 75% wake and
0% intent on the earlier 8-scene clean control at threshold 0.45. The clean
identity candidate scored 75% wake and 75% overall intent (100% conditional on
wake) at that same threshold; at diagnostic threshold 0.25 it scored 100%/100%
on that 8-scene control.

## Noisy transfer decision

On the same 8-scene CAVEMOVE proxy at threshold 0.25:

- old Stage 1: 37.5% wake, 12.5% overall intent, 1 false wake in 112 s;
- clean identity Stage 1: 62.5% wake, 50.0% overall intent, 3 false wakes in
  112 s;
- joint clean+CAVEMOVE Stage 1: same 62.5% / 50.0% / 3 result.

A bounded 128-row Stage 1 regression found the clean candidate improved
AISHELL-5 by +3.58 dB SI-SDRi but regressed CAVEMOVE by −2.80 dB versus the
retained dual-domain checkpoint. Do not replace the noisy production candidate
with either clean checkpoint.

The existing target-speaker KWS branch was tested on a small clean ablation. It
raised wake recall to 100% but one late wake caused command-window alignment
errors (87.5% intent); it is not promoted. The fixed-offset command window is
still a timing limitation for late detections.

An opt-in `command_window_strategy=kws_trim` path now stores the accepted wake
frame, trims it after its last local KWS hit, and falls back to the current
frame when no local hit exists. With a 0.8 s endpoint margin it reached 100%
wake / 87.5% intent / 0 false wakes on 64 clean scenes. It remains
experimental; local-KWS trim is not sufficient to claim production endpointing.

## Important files

- Clean study report: `reports/clean_english_stage1_transfer_2026-10-13.md`
- Production plan: `reports/production_plan_2026-10-13.md`
- Journey: `reports/research_journey.md`
- Clean identity config: `configs/eval_end_to_end_clean_english_identity_ft1.yaml`
- Joint transfer config: `configs/eval_end_to_end_clean_english_joint_ft1.yaml`
- Clean proxy logs:
  `logs/clean_full_pipeline_identity_ft1_w2vec_32scenes_kws025_2026-10-13.json`
  and `logs/clean_full_pipeline_identity_ft1_w2vec_32scenes_kws040_2026-10-13.json`
- `logs/clean_full_pipeline_identity_targetkws_endpoint08_32scenes_2026-10-14.json`
- Audit:
  `reports/production_readiness_snapshot_live_2026-10-13_clean_update.md`
  and `logs/production_readiness_audit_clean_update_2026-10-13.json`

## Next work

1. Keep the clean identity candidate isolated; do not overwrite the retained
   noisy checkpoint.
2. Improve command endpointing/window selection for late wake detections, then
   rerun clean 32-scene and noisy regression checks.
3. Calibrate thresholds only on a permitted tuning set. The 0.25/0.40 values
   are proxy operating points, not frozen release thresholds.
4. Prepare/obtain the required consented English same-driver pilot with 4 cabin
   channels plus playback reference. Keep tuning and held-out drivers separate.
5. Once real data exists, run the full evaluator and ablations; only then
   consider production claims or checkpoint promotion.

Validation after this session: `49 passed, 1 skipped`.

## 2026-10-14 continuation update

The endpoint timing work was extended without changing the promoted noisy
configuration. A one-call post-wake delay, even with an eight-call timeout,
reduced clean intent to 50% on 32 scenes; fixed call delays are therefore not
the release solution. The new opt-in
`command_window_strategy=kws_trim_latest` trims the latest accepted post-wake
frame and falls back to the stored wake frame. It passed dedicated unit tests
and the full suite now reports `51 passed, 1 skipped`.

Clean latest-frame proxy result (32 scenes / 448 seconds, cross-corpus,
non-real-driver): 100% wake recall, 87.5% intent overall, 0 false wakes, and
0 false actions. A direct-clean downstream control bypassing Stage 1 reached
81.25% intent on the same scenes, so command timing and the cross-corpus
construction remain limiting factors. The latest-frame path was also checked
on 16 CAVEMOVE scenes at -5 and 0 dB cabin SNR: 75.0% wake recall, 62.5%
intent, and 96.4 false wakes/hour. It is not promoted.

New reproducibility artifacts:

- `configs/eval_end_to_end_clean_english_identity_targetkws_endpoint08_latest.yaml`
- `logs/clean_full_pipeline_identity_targetkws_endpoint08_latest_32scenes_2026-10-14.json`
- `logs/clean_full_pipeline_bypass_stage1_targetkws_endpoint08_32scenes_2026-10-14.json`
- `logs/noisy_full_pipeline_cleanidentity_targetkws_endpoint08_latest_8scenes_2026-10-14.json`
- `logs/clean_full_pipeline_identity_targetkws_endpoint08_delay1_timeout8_32scenes_2026-10-14.json`

The latest-frame configuration was also replayed through the live-style
`StreamingRunner` at a 1x capture cadence for 30 seconds: 0 skipped seconds,
0.2 seconds maximum lag, and 0.293 real-time factor on CUDA. The replay source
is synthetic and the event counts are diagnostic only.

The production-readiness audit was regenerated on the retained dual-domain
configuration. It still correctly reports `research_engineering_preview`,
`production_ready=false`, zero real-driver sessions, and blocked same-driver,
passenger-rejection, study-size, matching-hardware, frozen-split, threshold,
and clean-release-state gates.

## Target-KWS adaptation result

A separate target-KWS cache was built from the clean identity Stage-1 output
with CAVEMOVE augmentation (1,208 train / 260 validation / 256 test rows), and
the target-KWS model was fine-tuned from the retained smart-speaker checkpoint.
Best validation F1 was 0.667. The full regression rejected it: 96.9% wake /
46.9% intent on 32 clean scenes, and 62.5% wake / 37.5% intent / 80.4 false
wakes/hour on 16 noisy scenes. Its checkpoint remains isolated at
`checkpoints/stage3_target_kws_gsc_marvin_cleanidentity_cabin_ft1/best.pt`.
The existing target-KWS checkpoint remains the only candidate for further
runtime work.

Additional negative checks: the older `stage1_spex_gsc_marvin_absent_adapt_ft3`
candidate produced 75.0% wake / 0% intent / 273.2 false wakes/hour and one
false action on 16 noisy proxy scenes. Enabling the compact speaker gate on
the clean branch produced 12.5% wake / 0% intent / 0 false wakes. Both remain
diagnostic only; cross-corpus wake/command clips cannot validate speaker-gate
behavior.

## 2026-10-14 capture hardware preflight

The read-only capture preflight checked all six visible input devices against
the four-channel/16 kHz contract. None passed: the host exposes at most two
input channels, and no real-driver recording was attempted. The required
hardware remains an external dependency; the production plan recommends six
inputs when two playback-reference channels are included for AEC.

New artifacts:

- `scripts/data/preflight_capture_hardware.py`
- `tests/test_capture_hardware_preflight.py`
- `logs/capture_hardware_preflight_2026-10-14.json`
- `reports/production_readiness_snapshot_post_preflight_2026-10-14.md`
- `logs/production_readiness_audit_post_preflight_2026-10-14.json`

Validation after this continuation: `60 passed, 1 skipped`.

The next safe software step is to keep the preflight and session validator in
the capture workflow, then run the frozen tuning/test evaluator as soon as a
permitted, production-equivalent array is attached. Do not treat the current
synthetic clean/noisy proxy metrics as real-driver evidence.

## 2026-10-14 reproducibility manifest

Added `scripts/reporting/build_release_manifest.py` and generated a manifest
for the retained runtime. It hashes 12 present artifacts, including the
runtime/Stage-1 configs, label map, test manifest, and selected checkpoints;
it records `worktree_clean=false`, so the release gate remains blocked. The
retained YAML was corrected from the stale label-map path
`configs/labels_fsc.json` to `meta/intent_label_map.json` and now declares the
`test` split. No model weights or thresholds changed.

Artifacts:

- `tests/test_release_manifest.py`
- `logs/release_manifest_current.json`
- `logs/release_manifest_current_2026-10-14.json`

The clean endpoint path was replayed after the config correction on an
8-scene/224-second smoke subset: 100% wake, 100% intent, and 0 false
wakes/actions. It remains cross-corpus and marked `production_valid=false`.
Artifact: `logs/clean_identity_endpoint08_post_manifest_8scenes_2026-10-14.json`.

The recorder now has an explicit `production6` profile that refuses capture
below six channels or without a playback-reference declaration. The existing
`model4_pilot` profile remains for the current four-channel research evaluator.
This is a capture-contract guard only; no hardware is attached and no session
has been recorded.

The active clean endpoint config now declares `split: test` and the real
`meta/intent_label_map.json` path. A clean component-boundary ablation found
100% direct TSE-to-intent, 87.5% KWS-only wake recall, and 75.0% full-cascade
intent at presence threshold 0.15 (37.5% at 0.45), with no false events in
112 seconds. Presence calibration is the next clean software target; no new
training has been started from this small cross-corpus result.

That isolated clean presence-head experiment is now complete and rejected:
100%/100% on the clean proxy, but 62.5% wake / 62.5% intent / 257.1 false
wakes/hour on the noisy proxy. The checkpoint remains at
`checkpoints/stage2_frame_presence_english_clean_identity_ft1/best.pt` and is
not part of the retained configuration.

The latest audit is
`logs/production_readiness_audit_post_presence_ft1_2026-10-14.json` and
remains `research_engineering_preview` with zero real-driver sessions.

## 2026-10-14 endpoint-buffer continuation

An opt-in `confidence_buffered` command endpoint was implemented and tested
without changing the retained noisy configuration. It delays commitment when
the first post-wake intent candidate is below a 0.90 early-commit confidence,
with a two-candidate maximum; unit tests cover commit and suppression behavior.

Results on the 32-scene clean cross-corpus proxy:

- current `kws_trim_latest`: 100% wake / 87.5% intent / 0 false wakes/hour;
- confidence-buffered: 100% wake / 96.875% intent (31/32) / 0 false wakes/hour;
- three-candidate buffer: unchanged at 31/32;
- dual fixed+trimmed crop: unchanged at 31/32, with a different failure.

Noisy rejection checks:

- clean Stage-1 + buffered: 75% wake / 62.5% intent / 96.4 false wakes/hour
  on 16 CAVEMOVE scenes;
- retained dual-domain + buffered: 50% wake / 0% intent / 0 false wakes/hour
  on the same small proxy.

These are not production numbers. The candidates are isolated, no checkpoint
or release threshold was promoted, and the same-driver/real-cabin evidence
gates remain blocked. The clean gain supports investigating endpoint timing
once the pilot data exists; it does not justify more proxy-only threshold
tuning.

New artifacts:

- `src/car_tse/pipeline/in_car_pipeline.py`
- `scripts/eval/eval_continuous_full_pipeline_proxy.py`
- `tests/test_pipeline_drop.py`
- `reports/clean_endpoint_failure_diagnosis_2026-10-14.md`
- `logs/clean_confidence_buffered_32scenes_2026-10-14.json`
- `logs/clean_confidence_buffered_hybrid_32scenes_2026-10-14.json`
- `logs/noisy_confidence_buffered_8scenes_2026-10-14.json`
- `logs/noisy_confidence_buffered_hybrid_8scenes_2026-10-14.json`
- `logs/noisy_retained_buffered_candidate_8scenes_2026-10-14.json`

## 2026-10-14 ONNX export preflight

The embedded-deployment preflight added an optional `embedded` dependency set
(`onnx`, `onnxruntime`) and a reproducible exporter:
`scripts/export/export_onnx_components.py --component all`. The original
complex-STFT log-mel frontend was not exportable with the legacy ONNX exporter;
an export-only real/imaginary DFT-convolution implementation was added and
unit-tested against the PyTorch frontend. The normal runtime path remains
unchanged.

All six retained components now export and pass ONNX Runtime CPU parity:
Stage 1, KWS, frame presence, target-KWS, compact verifier, and Wav2Vec
intent. Stage 1 relative-L2 parity is 1.93e-4 (maximum sample error 0.108),
and the intent graph is approximately 379 MB. The exports use fixed prototype
input lengths and have not been quantized, compiled for a target SoC, or
substituted into the Python pipeline. Therefore the embedded gate remains
open, but the export/parity artifact is now reproducible.

Artifacts:

- `scripts/export/export_onnx_components.py`
- `src/car_tse/models/kws_model.py`
- `src/car_tse/models/presence_head.py`
- `src/car_tse/models/speaker_verifier.py`
- `tests/test_onnx_export.py`
- `output/onnx/retained_2026-10-14/manifest.json`
- `logs/onnx_export_retained_2026-10-14.stdout.json`

The exported graphs were benchmarked with ONNX Runtime CPU (5 warmups, 20
runs). Stage 1 measured 371/376 ms p50/p95 on a 3-second input, Wav2Vec
intent 91/94 ms p50/p95 on 2 seconds, and each small KWS/presence/verifier
component below 2 ms p95. These are local desktop measurements only; target
SoC, quantization, concurrent load, and 24-hour soak remain open.

Artifact: `logs/onnx_component_cpu_latency_2026-10-14.json`

## 2026-10-14 dynamic-int8 feasibility check

ONNX Runtime's dynamic per-channel int8 quantizer was tested as a diagnostic,
without representative-audio calibration. It reduced model files to about
27--29% of float, but it was slower on this CPU and introduced unacceptable
drift for speaker-conditioned components: Stage 1 relative-L2 6.54% and
target-KWS 12.22%. Stage 1 p95 changed from 380 ms to 5,656 ms; KWS from
0.73 ms to 10.31 ms; presence from 4.04 ms to 11.45 ms. Compact-verifier and
intent conversion failed. The dynamic path is rejected and was not promoted
or connected to runtime. The next deployment experiment requires
representative cabin calibration, static/QAT quantization, and a target-vendor
execution provider.

Artifact: `logs/onnx_dynamic_int8_feasibility_2026-10-14.json`

## 2026-10-14 clean buffered endpoint experiment

The confidence-buffered command path was corrected to retain per-hop history
and select the longest consecutive intent run, with the earliest run winning
ties. This prevents a later trailing window from replacing a stable command.
The focused pipeline tests now report 17 passing tests.

The isolated clean candidate (`configs/eval_end_to_end_clean_english_identity_targetkws_confidence_buffered_hybrid_run3.yaml`)
reached 64/64 target wakes and 64/64 intents with zero false wakes/actions on
the 896-second clean proxy. Mean command completion was 2.22 seconds after
wake end and p95 2.50 seconds, so this is not a production-latency result.
The same policy on 16 noisy proxy scenes reached 50% wake and 0% intent and
was rejected for noisy use; the retained dual-domain runtime was not changed.
The clean replay is still cross-corpus and not same-driver evidence.

Artifacts:

- `src/car_tse/pipeline/in_car_pipeline.py`
- `tests/test_pipeline_drop.py`
- `logs/clean_confidence_buffered_run3_wait_32scenes_2026-10-14.json`
- `logs/noisy_confidence_buffered_run4_wait_8scenes_2026-10-14.json`

The 250 ms-hop follow-up was also rejected: three hops produced 75% clean
intent, while four hops produced 87.5% intent and 128.6 false wakes/hour on
the short proxy. A smaller hop alone is not a production-latency solution.

## 2026-10-14 compact intent transfer candidate

A clean-first 12.7M-parameter CRNN intent model was trained separately from
the retained Wav2Vec path. The AMP attempt produced NaN loss and was stopped;
the FP32 candidate reached 83.7% official FSC test accuracy / 85.6% macro-F1,
87.5% clean end-to-end proxy intent, and ~11 ms CPU PyTorch inference. Fine-
tuning it on the existing Stage-1/TSE cabin cache reached 59.7% cabin-domain
validation accuracy, 86.7% official clean-test accuracy, and 93.75% clean
end-to-end proxy intent. On 16 noisy proxy scenes it reached 50% wake recall
and 6.25% intent, so it was rejected as a noisy replacement.

The transferred compact graph exports to a 51.9 MB float ONNX model with
6.75 ms p95 CPU latency and 3.5e-7 relative-L2 parity. Dynamic int8 reduced
size to 13.2 MB but drifted 6.1% and slowed to 40.2 ms p95; it is diagnostic
only and rejected. The retained runtime/checkpoints are unchanged.

The Stage-4 training script now fails fast on non-finite losses so an AMP
overflow cannot be silently saved as a candidate checkpoint.

Artifacts:

- `scripts/train/train_stage4_intent.py`
- `configs/stage4_intent_clean_compact_v1_fp32.yaml`
- `configs/stage4_intent_compact_clean_to_tse_ft1.yaml`
- `logs/clean_compact_tse_ft1_32scenes_2026-10-14.json`
- `logs/noisy_compact_tse_ft1_8scenes_2026-10-14.json`
- `output/onnx/compact_tse_ft1_2026-10-14/intent.json`
- `logs/onnx_compact_intent_cpu_latency_2026-10-14.json`
- `logs/onnx_compact_intent_dynamic_int8_2026-10-14.json`

Full-pipeline compact-candidate latency was also measured: GPU p95 60.8 ms
for wake compute and 104.3 ms for forced command compute; CPU p95 505.4 ms
and 526.5 ms. The candidate checkpoint set is 98.4 MB. These are desktop
random-input measurements only and do not close the target-SoC/soak gate.

## 2026-10-14 static-int8 compact intent diagnostic

Representative cabin calibration was tested on the compact intent ONNX graph
with 128 calibration rows and a disjoint 128-row evaluation tail. Percentile
calibration first hit an ONNX Runtime/NumPy histogram failure on a
near-constant activation; a MinMax QDQ diagnostic completed. It was rejected:
float versus INT8 accuracy was 55.47% versus 4.69%, relative logit L2 error
was 1.128, cosine similarity was 0.233, size ratio was 0.9785, and CPU p95
latency was 7.13 versus 8.10 ms. The quantized graph is isolated and not
connected to runtime.

Artifacts:

- `scripts/export/quantize_static_onnx_intent.py`
- `logs/onnx_compact_intent_static_int8_2026-10-14.json`
- `output/onnx/compact_tse_ft1_2026-10-14/static_int8/intent.static_int8.onnx`

Full regression after the current source changes: `66 passed, 1 skipped`.

## 2026-10-14 clean latency shortcuts rejected

An initial early-commit replay at confidence 0.90/0.98 used a 1.0-second hop
and reached 87.5% wake with 65.6%/59.4% intent; it is not a threshold-only
comparison. In the matched 0.5-second-hop study, confidence 0.90 reached 100%
wake / 96.875% intent and 0.69/1.0-second mean/p95 command completion, versus
100% / 100% and 1.5/1.5 seconds for the 1.0-threshold baseline. It remains
rejected for release because it loses actions and lacks held-out,
speaker-consistent evidence.

A separately trained 1.5-second, 12.7M-parameter clean CRNN reached 88.4%
official clean-test accuracy / 88.9% macro-F1 but only 87.5% wake and 53.1%
end-to-end intent on the 64-scene clean proxy. Fine-tuning from that checkpoint
using the existing aligned TSE-domain cache reached 88.5% official accuracy /
88.8% macro-F1 but only 40.625% end-to-end intent. Both candidates are
isolated; the retained clean compact transfer and Wav2Vec paths are unchanged.
The next clean latency step is a real streaming intent/endpoint model, not
another fixed-window crop.

Artifacts:

- `configs/stage4_intent_clean_compact_1p5s_v1_fp32.yaml`
- `checkpoints/stage4_intent_clean_compact_1p5s_v1_fp32/best.pt`
- `logs/stage4_intent_clean_compact_1p5s_v1_fp32_metrics_test_clean.json`
- `configs/eval_end_to_end_clean_english_identity_targetkws_compact_1p5s_v1.yaml`
- `logs/clean_compact_1p5s_v1_32scenes_2026-10-14.json`

## 2026-10-14 causal CRNN intent candidate rejected

An opt-in unidirectional LSTM intent head was added for future streaming
accumulation. The clean causal model has 5.79M parameters, 89.0% official FSC
test accuracy / 89.7% macro-F1, a 24.3 MB float ONNX export, and 4.40 ms CPU
p95 with exact parity. TSE-domain fine-tuning raised standalone clean test
accuracy to 89.7% / macro-F1 90.4%, but the full clean cross-corpus proxy stayed
at 53.125% intent and 87.5% wake recall. It is isolated and not connected to
the retained runtime. The current blocker is the mismatched command/speaker
contract and missing streaming same-driver data, not the causal model's size.

Artifacts:

- `src/car_tse/models/intent_classifier.py`
- `configs/stage4_intent_clean_causal_crnn_v1_fp32.yaml`
- `configs/stage4_intent_causal_clean_to_tse_ft1.yaml`
- `logs/clean_causal_tse_ft1_32scenes_2026-10-14.json`
- `output/onnx/causal_crnn_v1_2026-10-14/intent.json`
- `logs/onnx_causal_crnn_cpu_latency_2026-10-14.json`

## 2026-10-14 production capture/audit alignment

The model uses four microphone channels, but the production capture contract
requires six inputs: four cabin microphones plus two playback-reference
channels for AEC. The readiness audit now uses explicit
`capture_expected_channel_count: 6` and
`capture_playback_reference_required: true` fields rather than inferring the
gate from `num_mics: 4`. The session validator rejects six-channel sessions
without an explicit `production6` playback contract; four-channel pilot
sessions remain diagnostic-only. Focused capture/evaluator tests pass (`20 passed`), and
the current audit still correctly reports zero sessions and a blocked hardware
gate.

The production6 preflight was rerun at the actual six-channel/16 kHz contract;
all six visible devices failed with at most two input channels. No recording
was created. Artifact: `logs/capture_hardware_preflight_production6_2026-10-14.json`.
 
## 2026-10-14 production playback-reference routing

The six-channel boundary is now explicit end to end. Production metadata must
declare `microphone_channel_indices` and
`playback_reference_channel_indices`; the real-driver evaluator maps the four
mic channels and averages the playback channels before calling the learned
front-end. `StreamingRunner` now supports a sample-aligned playback-reference
ring and rejects a missing reference when enabled. This fixes a prior silent
failure mode where a six-channel file was truncated to four microphones and
AEC was never exercised. The fix is covered by capture-mapping and streaming
tests, but it does not create real-driver evidence or unblock the hardware
gate.

## 2026-10-14 ONNX deployment-correctness fix

The retained ONNX export initially had a silent checkpoint-loading defect for
the wrapped Stage-1 and compact verifier components. Their checkpoints were
loaded into the wrapper with `strict=False`, so the wrapper appeared to load
while its underlying model remained at initialization. The exporter now maps
those components to the underlying checkpoint target before loading. The new
candidate export is
`output/onnx/retained_ckptfix_2026-10-14/manifest.json`; do not use the older
`retained_2026-10-14` export for deployment.

`logs/onnx_pipeline_golden_parity_ckptfix_cpu_2026-10-14.json` reports all six
trained PyTorch↔ONNX handoffs passing on one deterministic CAVEMOVE cabin-noise
sample. The companion host-CPU benchmark reports about 376 ms for Stage 1 on
3 seconds of audio, 88 ms for intent on 2 seconds, and 0.7–2.2 ms for the
smaller gates. This is export/reproducibility evidence only; the target SoC,
quantization, real-driver continuity, and six-channel capture gates remain
open. The release manifest and readiness audit were refreshed, and release
sign-off remains false.

## 2026-10-14 multi-row ONNX parity promotion

The Stage-1 exporter’s self-check now treats waveform separation outputs
separately from logits: it records the pointwise error but accepts a waveform
when relative L2 is ≤0.1% and cosine similarity is ≥0.99999. Logit and gate
checks remain tight. A fresh candidate was regenerated at
`output/onnx/retained_ckptfix2_2026-10-14/`, and its manifest reports all six
components `ok`.

The trained PyTorch↔ONNX golden harness compared eight consecutive CAVEMOVE
cabin-noise rows spanning -5 through 20 dB conditions. All rows and all six
handoffs passed; worst relative-L2 errors were 8.60e-5 (Stage 1), 9.31e-5
(KWS), 1.35e-5 (presence), 4.18e-5 (target-KWS), 1.63e-4 (compact verifier),
and 1.06e-5 (intent). The release manifest and audit now point to this export.
This remains deployment-parity evidence, not target-SoC, quantization,
accuracy, or real-driver evidence; sign-off remains false.

## 2026-10-14 current ONNX candidate: ckptfix3

The current candidate superseding `ckptfix2` is
`output/onnx/retained_ckptfix3_2026-10-14/manifest.json`. It exports seven
components: Stage 1, SpEx+ speaker embedding, KWS, presence, target-KWS,
compact speaker verification, and intent. The golden harness compares the
trained PyTorch and ONNX paths on eight deterministic CAVEMOVE cabin rows and
feeds the ONNX speaker embeddings into target-KWS; all seven components pass.

The separate variable-length artifact
`logs/onnx_speaker_embedding_variable_length_2026-10-14.json` passes for
1.0→2.0 s, 1.5→3.0 s, and 2.0→4.0 s enrollment/extracted windows. The current
host-CPU p95 references are about 379.3 ms Stage 1, 50.9 ms speaker embedding,
92.9 ms intent, and 0.7–2.1 ms for the small gates. The release manifest has
`all_artifacts_present: true` but `release_signoff_ready: false`; the audit
still reports zero real-driver sessions and the same seven blocked production
gates. Do not describe this as target-SoC, quantized, real-driver, or
production evidence.

## 2026-10-14 dynamic-int8 diagnostic outcome

The current seven-component export was tested with the existing uncalibrated
dynamic-int8 script. It is rejected and must not be wired into the runtime:
Stage 1 shrank to 27.6% of float size but reached 0.121 relative-L2 drift and
about 5.61 s CPU p95; KWS/presence/target-KWS also slowed and showed drift; the
speaker embedding, compact verifier, and wav2vec intent graphs failed
conversion. The float `ckptfix3` graphs remain the retained candidate. A valid
edge optimization now requires representative cabin calibration, QAT or a
vendor compiler, and target-SoC measurements.

Artifact: `logs/onnx_dynamic_int8_feasibility_ckptfix3_2026-10-14.json`.

## 2026-10-14 composed ONNX runtime bridge

Added `src/car_tse/deployment/onnx_runtime.py` and
`scripts/eval/eval_onnx_composed_pipeline_parity.py`. The runtime composes the
seven retained ONNX graphs in production order, enforces four microphone
channels plus a playback reference, feeds the exported speaker embeddings to
target-KWS, and mirrors the validated fixed-window PyTorch state machine.

The eight-row composed parity artifact is
`logs/onnx_composed_pipeline_parity_ckptfix3_cpu_8samples_2026-10-14.json`.
It is `status: ok`: every component and every compared state field matched the
PyTorch reference. This is CPU deterministic proxy parity only; it is not a
real-driver, target-SoC, quantization, streaming-continuity, or production
result. `tests/test_onnx_runtime.py` adds six contract tests, and the release
manifest now has 24 present artifacts including the composed parity report.
The readiness audit remains `research_engineering_preview`, with zero real
driver sessions and the existing seven blocked production gates.

The composed parity run was expanded to 32 CAVEMOVE proxy rows at
`logs/onnx_composed_pipeline_parity_ckptfix3_cpu_32samples_2026-10-14.json`.
All state-machine decisions matched. Five component-row comparisons exceed
the isolated strict numeric rule, but all seven components pass the explicit
integration tolerance (relative L2 ≤0.2% or absolute error ≤0.01); the JSON
keeps both strict and bounded results. The 32-row artifact is the composed
parity report hashed in `logs/release_manifest_current.json`. This remains
CPU proxy evidence only and does not unblock real-driver or target-SoC gates.

## 2026-10-14 ckptfix4 and streaming benchmark

The current ONNX export is
output/onnx/retained_ckptfix4_2026-10-14/manifest.json. The exporter uses
distinct enrollment/extracted symbolic time dimensions, and the composed
runtime disables ORT memory-pattern reuse for the mixed fixed-window contract.
The 8-row golden parity, 32-row composed parity, and all three variable-length
speaker cases pass; the previous shape-mismatch warning is gone.

scripts/eval/benchmark_onnx_streaming_runner.py replayed 10 seconds through
the actual ring-buffer path with four microphones and a playback reference:
29 windows processed, 0 seconds skipped, CPU inference RTF 2.58 on processed
hop audio. This is a useful deployment baseline and a clear latency blocker,
not target-SoC or real-driver evidence. The 25-artifact release manifest now
hashes the corrected export and all associated checks. Readiness remains
research_engineering_preview, with zero real-driver sessions and seven
blocked release gates.

## 2026-10-14 lazy intent runtime optimization

The streaming ONNX path now skips the wav2vec intent graph until the state
machine is awaiting a post-wake command. Full-component mode remains the
default for parity and diagnostics. Focused tests pass 18/18, and the
8-sample composed parity check remains `status: ok`.

With 8 intra-op / 1 inter-op ORT threads, the same 10-second repeated proxy
replay improved from 18.765 s to 16.161 s inference wall time (2.588x to
2.229x RTF) and from 1.878x to 1.618x capture-wall RTF. It processed 29
windows with zero dropped seconds. The stream has no event-positive wake or
command labels, so this is a compute-path benchmark only. Component profiling
still shows Stage 1 at about 375 ms p95 per 3-second window; host CPU Python
ONNX remains outside the target latency budget.

Detailed record: `reports/onnx_runtime_optimization_2026-10-14.md`.

A 500 ms-hop diagnostic is also recorded: 15 windows, 0 dropped seconds,
1.093x inference RTF, and 0.824x capture-wall RTF. This is a host fallback
only; it does not meet the intended latency cadence and the canonical 250 ms
benchmark remains the release artifact.

The release manifest was then rebuilt with both cadence artifacts and now has
26 present hashed artifacts; the readiness audit remains complete but
`production_ready=false` with seven critical gates blocked and zero real-driver
sessions.

## 2026-10-14 clean generalization follow-up

The confidence-buffered clean candidate was expanded from its earlier 32-row
proxy to 128 noise-free rows (64 target wake groups × two evaluator labels).
With a two-call buffer and 0.90 commit confidence it reached 87.5% wake,
76.6% overall intent, and 87.5% intent conditioned on wake, with zero observed
false events. The same policy on matched CAVEMOVE cabin noise reached 45.3%
wake and 25.8% overall intent. A same-command-speaker enrollment diagnostic
did not improve the clean result. These are cross-corpus proxy results; the
buffered policy is not promoted and further clean-only fitting is not justified
until same-driver cabin data exists.

Detailed record: `reports/clean_generalization_and_noise_transfer_2026-10-14.md`.

## 2026-10-14 clean operating-point update

On the same 128-row noise-free cross-corpus proxy, the retained models with
500 ms hop and `kws_trim` endpoint decoding reached 100.0% wake recall and
84.375% intent accuracy, with zero observed false events. This is currently
the strongest clean diagnostic and exceeds the provisional 80% proxy target,
but it is not a production result or frozen configuration because wake and
command speakers are not same-driver recordings and negative exposure is
short. Artifact: `logs/clean_endpoint08_64scenes_clean_hop05_2026-10-14.json`.

## 2026-10-14 clean safety trade-off

The permissive clean endpoint candidate scored 100.0% wake / 84.375% intent
on 128 synthetic positive rows, but on 3.193 hours of non-Marvin Smart
Speaker speech it produced 1,209.38 full-path false wakes/hour and the
14-file sequential action sample produced 419.51 false actions/hour. The
hard-negative/TSE-domain KWS plus SpEx gate reduced this by rejecting the
invalid cross-corpus positive proxy: 4.6875% wake / 0% intent, zero false
events. Do not promote either result; same-driver enrollment/wake/command
recordings are required to calibrate the safe middle ground. Artifacts:
`logs/smart_speaker_command_negative_endpoint08_2026-10-14.json` and
`logs/clean_endpoint08_hardneg_gate_64scenes_clean_hop05_2026-10-14.json`.

## 2026-10-14 causal Stage 1 diagnostic

An opt-in causal 16 kHz SpEx+ path was added and structurally validated, but
it is not a retained model. The existing clean checkpoint scored 41.67 dB
mean validation SI-SDR in its non-causal form versus 4.16 dB when the causal
layers were swapped in without retraining. A separate clean causal fine-tune
was stopped after its first epoch reported NaN train/validation SI-SDR; no
causal checkpoint was promoted. Default `causal: false` behavior and the
release export remain unchanged. See
`reports/causal_stage1_candidate_2026-10-14.md` before attempting another
causal run; first add finite-gradient/AMP guards and a short smoke test.

## 2026-10-14 full open-set hard-negative diagnostic

The complete 2,110-file University of Rochester non-Marvin sweep is now
available at `logs/smart_speaker_command_negative_endpoint08_hardneg_nogate_all_2026-10-14.json`.
With the TSE-domain hard-negative KWS at threshold 0.50 and no explicit
speaker gate, KWS-only smoothing produced 234 events over 3.193 hours
(73.30/hour). Adding the current frame-presence gate reduced this to 155
events (48.55/hour), a 33.8% reduction. Speaker6 was the outlier at
205.27/hour full path. The 14-file sequential action sample had zero false
actions over 68.65 seconds, which is not enough exposure for a safety claim.

This candidate is safer than the clean-only KWS candidate but is not close to
the release target and is not promoted. The clips are clean single-channel
studio negatives duplicated to four channels; no real-driver or real-cabin
gate has changed.
