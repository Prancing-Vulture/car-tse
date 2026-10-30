# Runtime intent-confidence calibration — 2026-10-11

## Scope

This is the calibration result for the exact intent encoder/head pair used by
the current safety diagnostic runtime candidate:

- encoder: `checkpoints/stage4_intent_wav2vec_ft8_resume_tse_domain/encoder_best.pt`
- head: `checkpoints/stage4_intent_wav2vec_ft8_resume_tse_domain/best.pt`
- waveform cache: `data/cache/stage4_intent_tse_domain_16k_v4`
- validation examples: 2,314

## Results

| Confidence threshold | Coverage | Selective accuracy | Suppressed examples |
|---:|---:|---:|---:|
| 0.30 | 98.70% | 91.29% | 30 |
| 0.40 | 96.89% | 92.42% | 72 |
| 0.50 | 95.12% | 93.28% | 113 |
| **0.53** | **94.12%** | **93.66%** | **136** |
| 0.55 | 93.56% | 93.95% | 149 |
| 0.60 | 92.05% | 94.79% | 184 |
| 0.70 | 89.15% | 95.98% | 251 |
| 0.80 | 85.65% | 97.28% | 332 |
| 0.90 | 80.86% | 98.56% | 443 |

Raw top-1 accuracy is 90.49%, 15-bin ECE is 0.0230, and the multiclass
Brier score is 0.1353. The runtime model is therefore reasonably calibrated
on its declared TSE-domain validation cache; 0.53 is a defensible starting
point for real-driver tuning, but is not a frozen release threshold.

## Reproducibility correction

An initial feature-cache calibration attempt was rejected. The legacy
`data/cache/stage4_intent_wav2vec_tse_domain/val_features.npy` was generated
with a different encoder state than the current `ft8` checkpoint; the first
four feature rows had mean cosine similarity approximately 0.81 against
features recomputed from the current encoder. The feature-cache result was
therefore not used. The numbers above use the exact waveform cache and
encoder/head pair used by runtime evaluation.

## Reproduction

```powershell
python scripts/eval/eval_intent_confidence_calibration.py `
  --config configs/stage4_intent_tse_domain.yaml `
  --head-checkpoint checkpoints/stage4_intent_wav2vec_ft8_resume_tse_domain/best.pt `
  --encoder-checkpoint checkpoints/stage4_intent_wav2vec_ft8_resume_tse_domain/encoder_best.pt `
  --batch-size 32 `
  --output logs/intent_confidence_calibration_ft8_tse_domain_waveform_val_2026-10-11.json
```
