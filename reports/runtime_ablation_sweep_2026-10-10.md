# Runtime ablation sweep

Date: 2026-10-10  
Study output: `logs/runtime_ablation_sweep_2026-10-10.json`

## Scope

This study replayed the fixed checkpoints on the existing eight-scene
continuous proxy: two target wake speakers, four CAVEMOVE cabin-noise levels
(-5/0/10/20 dB), a passenger wake word, passenger speech, and a target
command. Each scene is 14 seconds. The wake clip and command clip are drawn
from different public corpora and are not verified to be the same speaker.
Therefore these results are runtime diagnostics, not real-driver evidence.

No training was started. The study ran on CUDA and used 56 cached Stage-1
feature collections.

## Temporal settings

The base enrollment was 1.5 seconds. The table reports target wake recall,
target intent accuracy, false actions/hour, and mean wake latency.

| Hop | Temporal rule | Wake recall | Intent accuracy | False actions/hour | Mean latency |
|---:|---|---:|---:|---:|---:|
| 0.5 s | one hit | 100% | 0% | 32.1 | 0.875 s |
| 0.5 s | two of two | 100% | 0% | 32.1 | 1.375 s |
| 0.5 s | two of three + refractory | 25% | 12.5% | 0 | 1.500 s |
| 0.5 s | three of five + refractory | 0% | 0% | 0 | — |
| 1.0 s | one hit | 100% | 0% | 0 | 1.250 s |
| 1.0 s | two of two | 25% | 12.5% | 0 | 1.500 s |
| 1.0 s | two of three + refractory | 12.5% | 12.5% | 0 | 2.500 s |
| 1.0 s | three of five + refractory | 0% | 0% | 0 | — |

Interpretation: stricter temporal confirmation suppresses false actions but
also suppresses the target wake. On this proxy, the practical operating point
is not established; threshold and temporal settings must be selected on a
same-driver tuning split and frozen before a separate test split.

## Enrollment-duration sweep

These values are proxy durations; the source Marvin clips are short, so the
1.5-second enrollment can include zero-padding for some clips.

| Enrollment | Wake recall | Intent accuracy | False actions/hour | Mean latency |
|---:|---:|---:|---:|---:|
| 0.5 s | 25% | 0% | 0 | 1.500 s |
| 1.0 s | 12.5% | 0% | 0 | 1.500 s |
| 1.5 s | 25% | 12.5% | 0 | 1.500 s |

The 1.5-second result is marginally best in this small proxy, but the result
is too small and cross-corpus to justify a deployment choice.

## Microphone/front-end channel sweep

The four-channel input was replaced by simulated all-channel, mono-repeated,
front-pair-repeated, and rear-pair-repeated inputs. All used the same
1.5-second enrollment and 1.0-second hop.

| Front-end | Wake recall | Intent accuracy | False actions/hour | Target presence pass |
|---|---:|---:|---:|---:|
| All 4 channels | 25% | 12.5% | 0 | 37.5% |
| Mono repeated to 4 | 25% | 12.5% | 0 | 34.4% |
| Front pair repeated | 25% | 12.5% | 0 | 34.4% |
| Rear pair repeated | 25% | 12.5% | 0 | 34.4% |

No useful separation appeared in this simulated channel study. It is not a
substitute for recordings from different microphone placements.

## Speaker gate and command timeout

The speaker-gate replay used cosine similarity from the Stage-1 speaker
embeddings. The mean similarity across all windows was only 0.031 on this
proxy. A threshold of 0.0 passed 72.3% of windows and changed no outcome;
thresholds of 0.2 or higher rejected 100% of windows and produced 0% target
wake recall. The gate therefore cannot be calibrated responsibly from this
proxy.

| Speaker-gate threshold | Wake recall | Intent accuracy | False actions/hour | Gate pass rate |
|---:|---:|---:|---:|---:|
| Disabled | 25% | 12.5% | 0 | — |
| -0.2 | 25% | 12.5% | 0 | 100% |
| 0.0 | 25% | 12.5% | 0 | 72.3% |
| 0.2 | 0% | 0% | 0 | 0% |
| 0.4 | 0% | 0% | 0 | 0% |

Command timeout values of 1, 2, 4, and 6 streaming calls produced the same
aggregate result in this proxy: 25% wake recall, 12.5% intent accuracy, and
0 false actions/hour. The scenes do not contain a sufficiently varied pause
structure to make this setting informative.

## Conclusion

The additional ablations do not show a safe path to a large accuracy gain by
runtime tuning alone. The strongest evidence is instead diagnostic:

1. temporal smoothing controls the precision/recall trade-off;
2. channel reduction is not the dominant failure on this proxy;
3. the current speaker-similarity gate is uncalibrated for this input domain;
4. the next decisive experiment must be the real-driver pilot with the same
   driver's enrollment, wake word, and command, plus passenger and no-command
   segments split into tuning and held-out test sessions.

## Reproduction

```powershell
python scripts/eval/eval_runtime_ablation_sweep.py `
  --config configs/eval_end_to_end_cavemove_target_kws_temporal2_candidate.yaml `
  --paths configs/paths.yaml --scenes 2 --cabin-snr-db -5 0 10 20 `
  --passenger-snr-db 0 `
  --hops 0.5 1.0 --enrollment-seconds 0.5 1.0 1.5 `
  --output logs/runtime_ablation_sweep_2026-10-10.json
```
