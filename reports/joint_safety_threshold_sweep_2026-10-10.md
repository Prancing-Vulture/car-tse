# Joint safety-threshold sweep — 2026-10-10

## Objective

Test whether the full pipeline can reject passenger-triggered actions after a
target-speaker wake without retraining. The sweep varied the post-wake
presence threshold, the Stage-1 speaker-similarity gate, and a new optional
minimum intent-confidence gate. The intent gate suppresses an action when the
post-wake classifier is too uncertain.

## Dataset and evidence boundary

This is a controlled cross-corpus engineering proxy: two FSC command speakers
were given a different utterance from the same command speaker as enrollment,
then mixed with a Marvin wake clip, passenger wake/non-wake speech, and
CAVEMOVE cabin noise. Passenger speech was set to -10 dB relative to the
target. There were 8 scenes, 112 seconds total, and no real-driver recording.
The wake clip and command speaker are still not verified to be the same person.
Therefore this result is suitable for choosing follow-up experiments, not for
claiming real-cabin reliability.

## Results

| Operating point | Wake recall | Intent accuracy | False actions/hour | Interpretation |
|---|---:|---:|---:|---|
| no speaker gate / no confidence gate | 100% | 100% | 192.9 | unsafe baseline at passenger -10 dB |
| speaker gate 0.22 / no confidence gate | 100% | 100% | 32.1 | speaker gate helps, but residual action remains |
| speaker gate 0.22 / confidence 0.50 | 100% | 100% | 32.1 | too low to suppress the residual |
| speaker gate 0.22 / confidence **0.53** | **100%** | **100%** | **0.0** | best observed proxy operating point |
| speaker gate 0.22 / confidence 0.55 | 100% | 87.5% | 0.0 | safe but begins suppressing target commands |
| speaker gate 0.22 / confidence 0.60 | 100% | 87.5% | 0.0 | no added benefit over 0.55 |

For the post-wake presence sweep, gate 0.22 with command-presence threshold
0.15 also reached 0 false actions/hour, but intent accuracy fell to 87.5%.
Threshold 0.10 retained 100% intent and left 32.1 false actions/hour. The
confidence gate is therefore the less destructive safety control on this
proxy.

As a condition check, the same 0.22/0.53 pair was replayed at passenger SNR
+10 dB. It retained 100% wake and intent accuracy with 0 false actions/hour;
the ungated speaker-gate baseline was also 0 false actions/hour in that
condition. This confirms the confidence gate did not introduce a measurable
positive-condition regression in this small proxy.

The safety candidate was also run through the actual `InCarPipeline.forward`
runtime path rather than only the replay evaluator. With same-speaker-other
enrollment and passenger SNR -10 dB, it produced 100% wake recall, 100% intent
accuracy, 0 false actions/hour, 3 false wake events over 112 seconds (96.4
false-wake events/hour equivalent), and 2.125 s mean wake latency. This is a
stronger implementation check, but it remains a short cross-corpus proxy and
is not a real-driver result.

## Implementation

The runtime now supports an optional `intent_confidence_threshold` in
`InCarPipeline`. If the post-wake classifier confidence is below the selected
threshold, the state is reset and no action is emitted; the output marks
`command_suppressed=true`. The real-driver evaluator records this field. The
default remains unset, so existing checkpoints and behavior are unchanged.

The diagnostic candidate config is:

`configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_safety_diagnostic_candidate.yaml`

It contains speaker gate 0.22 and confidence gate 0.53 for repeatable A/B
testing, but those values must be recalibrated on tuning drivers and frozen
before any held-out test.

## Reproduction

```powershell
python scripts/eval/eval_continuous_ablation_matrix.py `
  --config configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_operating_candidate.yaml `
  --output logs/continuous_ablation_matrix_stage2_intent_confidence_gate_passenger_m10_2026-10-10.json `
  --scenes 2 --cabin-snr-db -5 0 10 20 --passenger-snr-db -10 `
  --hop-seconds 1.0 --enrollment-source same_speaker_other `
  --speaker-gate-thresholds 0.22 `
  --intent-confidence-thresholds 0.50 0.53 0.55 0.60

python scripts/eval/eval_continuous_ablation_matrix.py `
  --config configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_operating_candidate.yaml `
  --output logs/continuous_ablation_matrix_stage2_intent_confidence_gate_passenger_p10_2026-10-10.json `
  --scenes 2 --cabin-snr-db -5 0 10 20 --passenger-snr-db 10 `
  --hop-seconds 1.0 --enrollment-source same_speaker_other `
  --speaker-gate-thresholds 0.22 --intent-confidence-thresholds 0.53

python scripts/eval/eval_continuous_full_pipeline_proxy.py `
  --config configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_safety_diagnostic_candidate.yaml `
  --output logs/continuous_full_pipeline_proxy_safety_diagnostic_same_speaker_m10_2026-10-10.json `
  --artifact-dir reports/artifacts/continuous_full_pipeline_proxy_safety_diagnostic_same_speaker_m10_2026-10-10 `
  --scenes 2 --cabin-snr-db -5 0 10 20 --passenger-snr-db -10 `
  --hop-seconds 1.0 --enrollment-source same_speaker_other
```

## Presence-threshold follow-up — 2026-10-11

The runtime traces showed that passenger false-wake windows had presence
probabilities around 0.09–0.11, while target wake windows were at least about
0.16 in the same proxy. Raising only the pre-wake presence threshold from
0.07 to 0.12 was therefore tested in a separate v2 candidate:

| Candidate | Passenger SNR | Wake recall | Intent accuracy | False wakes/hour | False actions/hour | Mean latency |
|---|---:|---:|---:|---:|---:|---:|
| v1: presence 0.07, gate 0.22, confidence 0.53 | -10 dB | 100% | 100% | 96.4 | 0.0 | 2.125 s |
| v2: presence 0.12, gate 0.22, confidence 0.53 | -10 dB | 100% | 100% | **0.0** | **0.0** | 2.25 s |
| v2: presence 0.12, gate 0.22, confidence 0.53 | 0 dB | 100% | 100% | **0.0** | **0.0** | 2.25 s |
| v2: presence 0.12, gate 0.22, confidence 0.53 | +10 dB | 100% | 100% | **0.0** | **0.0** | 2.25 s |

Each row is 8 scenes / 112 seconds. v2 is the strongest current diagnostic
operating point, but it is not promoted or called production-ready until the
same threshold is selected on real-driver tuning sessions and evaluated on a
held-out driver split.

Artifacts:

- `configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_safety_diagnostic_candidate_v2.yaml`
- `logs/continuous_full_pipeline_proxy_safety_diagnostic_same_speaker_m10_v2_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_safety_diagnostic_same_speaker_p0_presence012_2026-10-10.json`
- `logs/continuous_full_pipeline_proxy_safety_diagnostic_same_speaker_p10_presence012_2026-10-11.json`

## Decision

Keep the confidence gate as an optional production-safety mechanism and test
it first on permitted real-driver tuning sessions. Do not call 0 false
actions/hour a production result: the sample is short, cross-corpus, and
synthetic.

## Real-cabin hard-negative adaptation and split speaker gate — 2026-10-11

The next iteration used permitted AISHELL-5 target-absent speech as a real
cabin hard-negative domain for the enrollment-conditioned target-KWS model.
Eval2 was held out from cache construction. A negative-only fine-tune was
selected for this diagnostic because the later positive-domain fine-tune,
although it reached validation F1 0.786, rejected the cross-corpus positive
proxy at KWS threshold 0.55. The selected checkpoint is:

`checkpoints/stage3_target_kws_gsc_marvin_hardneg_aishell5_ft1/best.pt`

The runtime candidate adds a split speaker gate: 0.24 before wake detection
and 0.22 after a wake while capturing the command. This avoids raising the
command gate to the point where lower-similarity but valid command frames are
discarded.

| Evaluation | Wake recall | Intent accuracy | Full-path false wakes/hour | False actions/hour | Mean latency |
|---|---:|---:|---:|---:|---:|
| AISHELL-5 Eval2, 1,200 s target-absent, p=.15 / wake=.24 / KWS=.50 | n/a | n/a | **3.0** | n/a | n/a |
| CAVEMOVE proxy, passenger -10 dB, 8 scenes / 112 s | 100% | 100% | **0.0** | **0.0** | 2.50 s |
| CAVEMOVE proxy, passenger +10 dB, 8 scenes / 112 s | 100% | 100% | **0.0** | **0.0** | 2.50 s |

The AISHELL run is target-absent relative to a fixed LibriSpeech enrollment,
not a driver-positive test. Under the exact runtime 2-window/2-hit smoothing,
its KWS-only smoothed rate was 60/hour and its full-path rate was 3/hour. An
earlier 0/hour replay used the evaluator's 3-window/2-hit default and is not
the frozen runtime result. This remains a combined-cascade diagnostic, not a
claim that the acoustic detector alone has zero false accepts.
The proxy uses same-speaker-other enrollment and synthetic wake/command
placement, so it is diagnostic rather than production evidence.

Artifacts:

- `configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_hardneg_aishell5_neg_safety_splitgate_candidate.yaml`
- `logs/aishell5_continuous_targetkws_hardneg_aishell5_ft1_p015_sg024_kws050_smooth22_1200s_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_hardneg_aishell5_neg_splitgate_p015_wake024_cmd022_same_speaker_m10_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_hardneg_aishell5_neg_splitgate_p015_wake024_cmd022_same_speaker_p10_2026-10-11.json`

Decision: retain this as the strongest current diagnostic operating point,
but keep the release gate blocked until the same settings are tuned and
frozen on permitted same-driver recordings and then evaluated on disjoint
held-out drivers.
