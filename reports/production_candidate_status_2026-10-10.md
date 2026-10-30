# Production-candidate status — 2026-10-10

## Decision

The project is not production-ready yet. The dual-domain Stage 1 `ft2`
frontend is the strongest extraction candidate on the AISHELL/CAVEMOVE
regression, while the `ft1`-aligned downstream head remains the strongest
full-pipeline proxy candidate. Release evidence is blocked by the absence of
permitted continuous same-driver recordings.

## What improved

| Area | Result | Interpretation |
|---|---:|---|
| AISHELL-5 target-present Stage 1 SI-SDRi | 5.38 dB | Improved from approximately 4.85 dB for the prior dual-domain candidate |
| CAVEMOVE held-out Stage 1 SI-SDRi | 10.88 dB | No meaningful regression from approximately 10.86 dB |
| Clean FSC intent, aligned head | 99.3% top-1 | Clean-domain behavior retained |
| TSE-domain intent validation, aligned ft1 head | 90.3% top-1 | Best of the two aligned head runs |
| CUDA Stage 1 + KWS + intent smoke latency | approximately 55.7 ms | Excludes buffering, I/O, presence, and application overhead |
| AISHELL-5 target-absent full path | 0 smoothed events/hour | 120-second diagnostic stream; not a positive benchmark |

The speaker-gate diagnostic is promising but not yet selectable: on the
same-speaker proxy, 0.15–0.22 gave 100% wake/intent with 0 false actions/hour
at passenger SNR +10 dB, while at passenger SNR -10 dB, 0.22 still gave 32.1
false actions/hour and 0.27 traded down to 75% wake recall for 0 false
actions/hour. On the cross-corpus proxy, the same gate rejects the mismatched
speaker. Threshold selection must therefore wait for real-driver tuning data.

The hard-negative target-KWS follow-up did not improve the current
same-speaker proxy: it matched the original target-KWS checkpoint at 100%
wake/intent and 64.3 false actions/hour ungated, and matched its 0.18–0.22
speaker-gate operating points. The ft2 Stage-1 plus aligned-head follow-up
also matched the ungated result but lost recall at gate 0.20–0.22. The
ft1-aligned downstream candidate therefore remains the current end-to-end
proxy choice; ft2 remains an extraction candidate for the real-driver study.

A new optional post-wake intent-confidence gate was tested as a safety
mechanism. On the harsher same-speaker proxy with passenger SNR -10 dB,
speaker gate 0.22 plus intent confidence 0.53 yielded 100% wake recall, 100%
intent accuracy, and 0 false actions/hour over 112 seconds. This is a useful
diagnostic candidate, not a production metric: the session is short and
cross-corpus. The runtime support is present, but the threshold remains
unset by default until it is tuned on permitted driver recordings.

The exact runtime `ft8` intent encoder/head was calibrated on its matching
waveform cache: 90.49% raw accuracy, ECE 0.023, and at confidence 0.53 it
retains 94.12% coverage at 93.66% selective accuracy. This is validation-domain
evidence for the safety signal, not a release threshold or real-driver result.

Raising only the pre-wake presence threshold to 0.12 produced the strongest
current diagnostic operating point: on both passenger SNR -10 dB and +10 dB
runtime proxies, it retained 100% wake/intent accuracy with 0 false
wakes/hour and 0 false actions/hour over 112 seconds per condition. Mean wake
latency was 2.25 seconds. This remains synthetic/cross-corpus evidence and is
not promoted as a production setting.

The same v2 candidate was also replayed at the neutral passenger SNR of 0 dB:
100% wake recall, 100% intent accuracy, 0 false wakes/hour, 0 false
actions/hour, and 2.25 s mean wake latency over 112 seconds. This fills the
middle condition for the proxy operating-point table but does not change the
real-driver evidence boundary.

The same candidate was also verified through the actual runtime forward path
on the eight-scene same-speaker diagnostic at passenger SNR -10 dB: 100% wake
recall, 100% intent accuracy, 0 false actions/hour, 3 false wake events over
112 seconds, and 2.125 s mean wake latency. This validates implementation
plumbing, not production reliability; the wake/command clips remain
cross-corpus and the exposure is too small for a release claim.

## Full-pipeline evidence boundary

On the eight-scene cross-corpus proxy, where the Marvin wake clip and FSC
command clip are not verified to be from the same speaker:

- full path: 87.5% wake recall, 12.5% intent accuracy, 0 false actions/hour;
- removing presence: 12.5% intent accuracy and approximately 417.9 false
  actions/hour;
- diagnostic same-command-speaker enrollment: 100% intent accuracy and 0
  false actions/hour.

The diagnostic gap shows that the low proxy intent score is dominated by the
cross-corpus speaker mismatch. The same-speaker diagnostic reuses command
audio as enrollment and is therefore not a release result.

## Current blockers

1. No real-driver session currently exists under `data/raw/custom_incar`.
2. No held-out driver/session split exists for threshold freezing.
3. Passenger wake, passenger speech, silence, playback/echo, and no-command
   false-action rates therefore cannot be reported as real-driver metrics.
4. The current model footprint and latency still need measurement on the
   intended automotive target, not only on the development GPU.

## Reproducibility artifacts

- Journey log: `reports/research_journey.md`
- Acceptance gates: `reports/production_readiness_gates.md`
- Guarded recorder: `scripts/data/record_real_driver_session.py`
- Session validator: `scripts/data/validate_real_driver_sessions.py`
- Real-driver evaluator: `scripts/eval/eval_real_driver_sessions.py`
- ft2 Stage 1 config/checkpoint: `configs/stage1_spex_dual_domain_rms010_ft2.yaml`,
  `checkpoints/stage1_spex_dual_domain_rms010_ft2/best.pt`
- ft1-aligned intent candidate: `checkpoints/stage4_intent_wav2vec_dual_domain_head_v1/`
- Latest end-to-end matrix: `logs/continuous_ablation_matrix_dual_domain_intent_v1_2026-10-10.json`
- Wake/Stage-1 follow-up: `reports/wake_model_and_stage1_followup_2026-10-10.md`
- Joint safety-threshold sweep: `reports/joint_safety_threshold_sweep_2026-10-10.md`
- Runtime safety replay: `logs/continuous_full_pipeline_proxy_safety_diagnostic_same_speaker_m10_2026-10-10.json`
- Neutral passenger-SNR v2 replay: `logs/continuous_full_pipeline_proxy_safety_diagnostic_same_speaker_p0_presence012_2026-10-10.json`
- Intent calibration: `reports/intent_confidence_calibration_2026-10-11.md`
- Runtime intent calibration: `reports/intent_confidence_calibration_runtime_ft8_2026-10-11.md`
- Presence-threshold v2: `reports/joint_safety_threshold_sweep_2026-10-10.md`

The 15-minute heartbeat remains active. The next productive iteration is to
record permitted tuning sessions, freeze thresholds on those sessions, and
run the held-out driver evaluation and ablation matrix without changing the
model or thresholds afterward.

## 2026-10-11 update — strongest diagnostic candidate

The wake model now has a real-cabin hard-negative adaptation:

`checkpoints/stage3_target_kws_gsc_marvin_hardneg_aishell5_ft1/best.pt`

It was trained from the prior target-KWS hard-negative checkpoint with 768
AISHELL-5 target-absent windows in each train/validation/test cache. Eval2 was
held out from that construction. A separate positive-domain fine-tune reached
validation F1 0.786 but did not hold its positive score margin on the
CAVEMOVE proxy at the stricter threshold, so it is not the selected path.

The strongest current diagnostic configuration is:

- Stage 1 dual-domain ft1 checkpoint (ft2 remains the extraction-only SI-SDRi
  candidate until same-driver downstream evidence exists)
- dual-domain non-wake frame presence head
- AISHELL-adapted negative-only target KWS
- presence threshold 0.15
- KWS threshold 0.50 with 2-of-2 smoothing
- wake speaker gate 0.24; command speaker gate 0.22
- intent confidence threshold 0.53

Results:

| Test | Exposure | Result |
|---|---:|---|
| AISHELL-5 Eval2, target absent | 1,200 s | 3.0 full-path smoothed false wakes/hour; 60/hour KWS-only under exact 2-of-2 runtime smoothing |
| CAVEMOVE proxy, passenger -10 dB | 112 s / 8 scenes | 100% wake, 100% intent, 0 false wakes/actions/hour, 2.50 s mean latency |
| CAVEMOVE proxy, passenger 0 dB | 112 s / 8 scenes | 100% wake, 100% intent, 0 false wakes/actions/hour, 2.50 s mean latency |
| CAVEMOVE proxy, passenger +10 dB | 112 s / 8 scenes | 100% wake, 100% intent, 0 false wakes/actions/hour, 2.50 s mean latency |

This is a meaningful engineering improvement over the earlier 345/hour
target-KWS-only AISHELL diagnostic, although it does not reduce the exact
runtime-matched full-path rate below the prior 3/hour result. It is still not production-ready: AISHELL is
target-absent, the proxy is synthetic/cross-corpus, and the workspace has no
consented same-driver continuous sessions. The candidate is therefore frozen
as a diagnostic reference, not released as a production model.

Reproduction config and logs:

- `configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_hardneg_aishell5_neg_safety_splitgate_candidate.yaml`
- `logs/aishell5_continuous_targetkws_hardneg_aishell5_ft1_p015_sg024_kws050_smooth22_1200s_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_hardneg_aishell5_neg_splitgate_p015_wake024_cmd022_same_speaker_m10_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_hardneg_aishell5_neg_splitgate_p015_wake024_cmd022_same_speaker_p0_2026-10-10.json`
- `logs/continuous_full_pipeline_proxy_hardneg_aishell5_neg_splitgate_p015_wake024_cmd022_same_speaker_p10_2026-10-11.json`

## 2026-10-11 follow-up - gate narrowing and transfer validation

Two runtime-only wake-gate increases (0.245 and 0.250) reduced the exact
2-of-2 AISHELL Eval2 full-path rate to 0/hour, but reduced the harsh passenger
proxy to 65.6%/50.0% and 62.5%/46.9% wake/intent respectively. They are
rejected; the 0.24/0.22 split gate remains the diagnostic reference at 3/hour
on the exact AISHELL replay.

Hard-negative mining variants were also rejected because they lost positive
proxy recall. A trainer initialization bug was fixed by adding explicit
full-model warm-start support for target-speaker checkpoints; the corrected
transfer run still reached only 62.5% wake recall and 53.1% intent accuracy on
the harsh proxy. No new checkpoint is promoted.

## 2026-10-10 update — all-session AISHELL negative exposure

The frozen diagnostic candidate was replayed over every one of the 18
AISHELL-5 Eval2 sessions rather than only sessions 001/002. The target-absent
exposure was 14,001.9 seconds (3.889 hours), with 80.999% speech occupancy,
using the exact runtime 2-window/2-hit smoothing. It produced 18 full-path
smoothed events, equal to 4.628 full-path false wakes/hour, and 376 KWS-only
smoothed events, equal to 96.673/hour.

This longer result is the current aggregate negative-domain diagnostic. It is
not a real-driver FAR: the enrollment speaker is absent, there are no labeled
driver wake positives or passenger-wake negatives, and the audio is not a
continuous vehicle session for this assistant. No checkpoint or threshold was
promoted from this replay.

Artifact: `logs/aishell5_continuous_targetkws_hardneg_aishell5_ft1_p015_sg024_kws050_smooth22_eval2_all18_2026-10-10.json`.
