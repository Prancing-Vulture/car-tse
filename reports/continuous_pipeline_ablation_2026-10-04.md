# Continuous pipeline ablation — 2026-10-04

## Question

This run separates the stages on the same continuous scenes:

```text
A. TSE → intent directly
B. TSE → KWS without the presence gate
C. TSE → presence → KWS → intent
```

The latest candidate checkpoints were used:

- cabin-adapted SpEx+ TSE;
- CAVEMOVE presence checkpoint;
- hybrid TSE-domain `marvin` KWS;
- Wav2Vec2 TSE-domain intent model.

The test used four 14-second cross-corpus proxy scenes: two target wake/command pairs at cabin SNR `0 dB` and `-5 dB`, with passenger speech at `0 dB` relative to the target. The target wake and target FSC command are not verified to be the same speaker, so this is diagnostic and not a production accuracy estimate.

## Results

| Path | Result |
|---|---:|
| TSE → intent directly | 25% intent accuracy |
| TSE → KWS without presence | 25% target wake recall |
| TSE → KWS without presence | 1 false event, 64.3 false actions/hour equivalent |
| Full pipeline, presence threshold 0.50 | 25% target wake recall; 0% final intent success |
| Full pipeline, presence threshold 0.90 | 0% target wake recall; 0% final intent success |

### Direct intent predictions

| Condition | Expected | Predicted |
|---|---|---|
| Scene 0, 0 dB | `activate\|lamp` | `activate\|lamp` |
| Scene 0, -5 dB | `activate\|lamp` | `deactivate\|lamp` |
| Scene 1, 0 dB | `activate\|lights` | `change language` |
| Scene 1, -5 dB | `activate\|lights` | `change language\|Korean` |

## What this isolates

1. The full pipeline is not failing solely because of the presence gate. Removing the gate still produced only 25% wake recall in this proxy.
2. A `0.50` presence threshold is too permissive: it passed essentially all target and passenger-region windows in this run, so it does not protect against false activations.
3. A `0.90` threshold is too strict for these target windows: it passed none of them.
4. The intent classifier can produce a correct result on one TSE-conditioned command, but its predictions are unstable under these cross-corpus noisy conditions.

The principal confound is speaker mismatch. The enrollment and `marvin` clip come from Speech Commands, while the command comes from Fluent Speech Commands. A target-speaker extractor is expected to suppress a command spoken by a different voice, so these results cannot be interpreted as the driver’s true end-to-end accuracy.

## Reproduction

```powershell
python scripts/eval/eval_continuous_pipeline_ablation.py `
  --scenes 2 `
  --cabin-snr-db 0 -5 `
  --passenger-snr-db 0 `
  --presence-thresholds 0.50 0.90
```

Raw output: `logs/continuous_pipeline_ablation.json`.

Evaluator: `scripts/eval/eval_continuous_pipeline_ablation.py`.

## Conclusion

The evidence does not support saying “the full model is simply 0% accurate.” It does support saying that the current cascade has not yet demonstrated robust continuous performance. The next valid experiment must use same-driver enrollment, wake word, and command recordings; otherwise the TSE stage is deliberately being tested with the wrong speaker identity.
