# Wake-model and Stage-1 follow-up — 2026-10-10

## Purpose

This follow-up tested two plausible improvements against the same eight-scene
continuous proxy used by the earlier ablations:

1. replace the generic target-speaker wake checkpoint with the
   hard-negative-trained target-KWS checkpoint; and
2. replace the retained ft1 dual-domain Stage-1 frontend plus aligned intent
   head with the additional low-rate ft2 frontend plus its separately aligned
   intent head.

The evaluation used the same two FSC command speakers, different utterance
enrollment from the command speaker (`same_speaker_other`), CAVEMOVE cabin
noise at -5/0/+10/+20 dB, passenger speech at +10 dB, and the same runtime
ablation matrix. These are controlled engineering proxies, not real-driver
evidence: the Marvin wake clip is still not verified to be from the FSC
command speaker.

## Results

| Candidate | Wake recall | Intent accuracy | False actions/hour | Gate observations |
|---|---:|---:|---:|---|
| ft1 aligned head + original target KWS, no speaker gate | 100% | 100% | 64.3 | baseline |
| ft1 aligned head + hard-negative target KWS, no speaker gate | 100% | 100% | 64.3 | no end-to-end improvement |
| ft1 + original target KWS, speaker gate 0.22 | 100% | 100% | 0.0 | 1 false wake event; mean latency 2.125 s |
| ft1 + hard-negative target KWS, speaker gate 0.22 | 100% | 100% | 0.0 | 1 false wake event; mean latency 2.125 s |
| ft2 aligned head + original target KWS, no speaker gate | 100% | 100% | 64.3 | no improvement |
| ft2 aligned head, speaker gate 0.18 | 100% | 100% | 0.0 | same safety point as ft1 at 0.18 |
| ft2 aligned head, speaker gate 0.20 | 100% | 87.5% | 0.0 | recall/intent regression |
| ft2 aligned head, speaker gate 0.22 | 87.5% | 87.5% | 0.0 | too strict on this proxy |

Each row is eight scenes / 112 seconds. The rates are therefore diagnostic
and have wide uncertainty; they must not be read as production estimates.

## Decision

Neither tested model replacement is promoted as the default:

- the hard-negative target-KWS checkpoint is retained as a reproducible
  alternative, but it produced the same full-pipeline operating results as
  the original target-KWS checkpoint;
- the ft2 Stage-1 checkpoint remains valuable for AISHELL-5 extraction
  (+0.53 dB in the latest bounded comparison) but its separately aligned
  downstream head did not improve the full-pipeline proxy and was less
  tolerant of the stricter speaker-gate settings;
- the ft1-aligned downstream candidate remains the current proxy champion.

The strongest current engineering recipe on this proxy is therefore the
existing ft1-aligned head with an approximately 0.18–0.22 speaker-similarity
gate, but the threshold cannot be frozen from this proxy. In the harsher
passenger -10 dB diagnostic, 0.22 still produced 32.1 false actions/hour and
0.27 reduced false actions to zero only by dropping wake recall to 75%.

## Reproduction

```powershell
python scripts/eval/eval_continuous_ablation_matrix.py `
  --config configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_hardneg_candidate.yaml `
  --output logs/continuous_ablation_matrix_stage2_hardneg_same_speaker_gate_sweep_2026-10-10.json `
  --scenes 2 --cabin-snr-db -5 0 10 20 --passenger-snr-db 10 `
  --hop-seconds 1.0 --enrollment-source same_speaker_other `
  --speaker-gate-thresholds 0.18 0.20 0.22 0.25

python scripts/eval/eval_continuous_ablation_matrix.py `
  --config configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_ft2_head_operating_candidate.yaml `
  --output logs/continuous_ablation_matrix_stage2_ft2_head_same_speaker_gate_sweep_2026-10-10.json `
  --scenes 2 --cabin-snr-db -5 0 10 20 --passenger-snr-db 10 `
  --hop-seconds 1.0 --enrollment-source same_speaker_other `
  --speaker-gate-thresholds 0.15 0.18 0.20 0.22 0.25
```

## Next action

The next model-facing experiment is threshold calibration on permitted,
speaker-consistent tuning sessions, followed by a frozen held-out evaluation.
Without those recordings, further proxy tuning can identify failure modes but
cannot close the production-readiness gates.
