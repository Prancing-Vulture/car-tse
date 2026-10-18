# Intent-confidence calibration — 2026-10-11

## Objective

Measure whether the optional post-wake intent-confidence gate is meaningful on
the full frozen TSE-domain validation cache, rather than only on the short
continuous proxy that motivated threshold 0.53.

## Evaluation

- Frozen checkpoint: `stage4_intent_wav2vec_dual_domain_head_v1`
- Split: dual-domain TSE validation
- Examples: 2,314
- Device: CUDA
- No weights or thresholds were changed.

| Confidence threshold | Coverage | Selective accuracy | Suppressed examples |
|---:|---:|---:|---:|
| 0.30 | 98.14% | 91.55% | 43 |
| 0.40 | 96.11% | 92.81% | 90 |
| 0.50 | 93.09% | 94.24% | 160 |
| **0.53** | **92.13%** | **94.75%** | **182** |
| 0.55 | 91.57% | 95.09% | 195 |
| 0.60 | 90.10% | 95.78% | 229 |
| 0.70 | 86.13% | 96.89% | 321 |
| 0.80 | 82.54% | 97.80% | 404 |
| 0.90 | 76.06% | 99.03% | 554 |

Overall top-1 accuracy is 90.32%, expected calibration error is 0.0240 using
15 bins, and the multiclass Brier score is 0.1393. The confidence values are
therefore reasonably calibrated on this validation domain, and threshold 0.53
gives a moderate safety/coverage tradeoff.

## Decision

Keep `intent_confidence_threshold` as an optional runtime safety gate. Retain
0.53 as a diagnostic starting point for real-driver tuning, not as a frozen
release threshold. The final threshold must be selected using only the
speaker/session-disjoint tuning split and then evaluated unchanged on held-out
drivers.

## Reproduction

```powershell
python scripts/eval/eval_intent_confidence_calibration.py `
  --config configs/stage4_intent_tse_dual_domain.yaml `
  --head-checkpoint checkpoints/stage4_intent_wav2vec_dual_domain_head_v1/best.pt `
  --encoder-checkpoint checkpoints/stage4_intent_wav2vec_dual_domain_head_v1/encoder_best.pt `
  --batch-size 32 `
  --output logs/intent_confidence_calibration_dual_domain_ft1_val_2026-10-11.json
```
