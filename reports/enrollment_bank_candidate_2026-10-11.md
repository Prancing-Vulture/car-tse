# Enrollment-bank candidate

Date: 2026-10-11  
Status: research candidate; not promoted

## Motivation

Personalized wake-word and target-speaker systems often stabilize the speaker
reference with more than one short enrollment utterance. The optional runtime
path in `InCarPipeline` accepts a precomputed normalized mean of SpEx speaker
embeddings and reuses it for the speaker gate, target-KWS conditioning, and
personal activity head. The first waveform still conditions Stage 1 extraction;
the bank changes only the reference representation used downstream.

The default single-enrollment path is unchanged when no bank embedding is
provided.

## Controlled proxy

Both runs used the same retained reference checkpoints, 8 synthetic scenes,
two CAVEMOVE cabin SNRs (-5 and +10 dB), passenger SNR -10 dB, 0.5-second
stream hops, and a different FSC utterance from the command speaker as the
first enrollment reference. The Marvin wake clip remains cross-corpus and is
not verified to be spoken by that speaker, so this is not real-driver evidence.

| Reference | Scenes | Wake recall | Overall intent | Intent conditioned on wake | False wakes/hour | False actions/hour | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| One short utterance | 16 | 81.25% | 68.75% | 84.62% | 0.0 | 0.0 | Retain as reference |
| Mean of three utterances | 16 | 6.25% | 6.25% | 100.00% | 0.0 | 0.0 | Reject |

The three-utterance bank was safe on this short proxy only because it rejected
nearly every target wake. Its lower recall is not an acceptable tradeoff, and
the experiment does not establish that a bank is intrinsically harmful. The
selected utterances were cross-corpus FSC references and the wake speaker was
not verified; a useful bank needs same-driver positive wake recordings,
speaker/session-disjoint tuning, and calibration of the mean or an attention/
quality-weighted aggregation.

## Implementation and reproduction

```powershell
python scripts/eval/eval_continuous_full_pipeline_proxy.py `
  --config configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_hardneg_aishell5_neg_safety_splitgate_candidate.yaml `
  --scenes 8 --cabin-snr-db -5 10 --passenger-snr-db -10 --hop-seconds 0.5 `
  --enrollment-source same_speaker_other --enrollment-bank-size 1 `
  --output logs/continuous_full_pipeline_proxy_enrollment_bank1_m10_8scenes_2026-10-11.json

python scripts/eval/eval_continuous_full_pipeline_proxy.py `
  --config configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_hardneg_aishell5_neg_safety_splitgate_candidate.yaml `
  --scenes 8 --cabin-snr-db -5 10 --passenger-snr-db -10 --hop-seconds 0.5 `
  --enrollment-source same_speaker_other --enrollment-bank-size 3 `
  --output logs/continuous_full_pipeline_proxy_enrollment_bank3_m10_8scenes_2026-10-11.json
```

The implementation is in `src/car_tse/pipeline/in_car_pipeline.py` and the
proxy evaluator records `enrollment_bank_size` and the selected source files
in each result.

## Decision

Do not promote the three-utterance bank. Keep the optional mechanism available
for a future same-driver pilot, where bank aggregation can be selected on a
tuning split under an explicit false-action/hour constraint.
