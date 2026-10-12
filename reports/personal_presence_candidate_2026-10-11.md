# Personal activity-head candidate

Date: 2026-10-11  
Status: research candidate; not promoted

## Motivation

The retained Stage 2 frame-presence head receives only the Stage 1 extracted
waveform. The literature review identified personal VAD/TS-VAD systems that
condition target activity on the enrolled speaker representation. This
candidate adds that condition without changing the retained `frame` head or
its checkpoint.

## Implementation

`PersonalFramePresenceHead` uses the same 16 kHz log-mel/frame geometry as the
existing frame head, then applies a bounded FiLM-like scale/bias modulation
generated from the frozen 256-D SpEx+ enrollment embedding. It emits one
target-activity logit per frame and uses the same top-fraction aggregation at
runtime.

The candidate is selected with:

- `presence_type: personal_frame`
- 40-mel features, 512-point FFT, 160-sample hop, 64 hidden channels
- frozen Stage 1 extraction and frozen enrollment embeddings
- the existing GSC/AISHELL presence manifest plus target-KWS non-wake hard negatives
- cached Stage 1 outputs for fast epochs
- four epochs, AdamW learning rate `5e-5`

The training split is the repository’s existing balanced row split and is not
a speaker-disjoint real-driver split. Its validation result is therefore an
internal fit check, not a deployment claim.

## Training result

Checkpoint:

`checkpoints/stage2_personal_presence_dual_domain_nonwake_ft1/best.pt`

The four-epoch run used 1,926 training rows and 482 validation rows. The final
validation snapshot reached chunk F1 1.000 at threshold 0.40, with 1.000 chunk
recall and 1.000 chunk specificity on that internal split. This is likely
optimistic because the split is row-based and should not be confused with
held-out driver evidence.

## End-to-end proxy results

The same continuous synthetic proxy was used for both rows: same FSC-speaker
enrollment utterance, cross-corpus Marvin wake, FSC command, passenger Marvin,
passenger speech, CAVEMOVE cabin noise, 8 scenes, two cabin SNRs, and passenger
SNR -10 dB. This remains cross-corpus and is not a real-driver evaluation.

| Personal presence threshold | Wake recall | Overall intent | Conditioned intent | False wakes/hour | False actions/hour | Decision |
|---:|---:|---:|---:|---:|---:|---|
| 0.15 / command 0.10 | 81.25% | 62.50% | 76.92% | 208.9 | 96.4 | Reject: unsafe proxy operating point |
| 0.40 / command 0.10 | 50.00% | 37.50% | 75.00% | 0.0 | 0.0 | Reject: recall collapse |

The 0.40 run demonstrates that the learned head can suppress the observed
synthetic passenger/noise events, but only by rejecting half of the target
wakes. The 0.15 run demonstrates that its high internal validation score does
not transfer into a safe end-to-end threshold. No personal-head checkpoint is
promoted.

## Decision and next experiment

Keep the candidate available for controlled comparison, but retain the
negative-only target-KWS reference pipeline. The next personal-activity study
should use speaker- and session-disjoint real-driver tuning data with explicit
target speech, passenger speech, passenger Marvin, and no-command intervals.
The activity head should then be calibrated jointly with the wake threshold
and command-capture threshold, using false actions/hour as a hard constraint.
Only after that should it be considered for a multichannel spatial-front-end
experiment.

## Reproduction

```powershell
python scripts/train/train_stage2_frame_presence.py `
  --config configs/stage2_personal_presence_dual_domain_nonwake_ft1.yaml

python scripts/eval/eval_continuous_full_pipeline_proxy.py `
  --config configs/eval_end_to_end_cavemove_personal_presence_dual_domain_stage2_candidate.yaml `
  --scenes 8 --cabin-snr-db -5 10 --passenger-snr-db -10 `
  --hop-seconds 0.5 --enrollment-source same_speaker_other `
  --output logs/continuous_full_pipeline_proxy_personal_presence_same_speaker_m10_8scenes_2026-10-11.json

python scripts/eval/eval_continuous_full_pipeline_proxy.py `
  --config configs/eval_end_to_end_cavemove_personal_presence_dual_domain_stage2_candidate.yaml `
  --scenes 8 --cabin-snr-db -5 10 --passenger-snr-db -10 `
  --hop-seconds 0.5 --presence-threshold 0.40 `
  --enrollment-source same_speaker_other `
  --output logs/continuous_full_pipeline_proxy_personal_presence_p040_same_speaker_m10_8scenes_2026-10-11.json
```
