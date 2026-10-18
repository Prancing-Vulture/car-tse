# Additional ablation results

Date: 2026-10-09  
Scope: passenger loudness, threshold operating points, and presence-head architecture

All results below use the same 8-scene continuous cross-corpus proxy design as
the earlier ablation report: two target wake clips, four cabin SNRs (-5, 0,
10, 20 dB), 14 seconds per scene, and CAVEMOVE cabin noise. The target wake
and command are not verified to be from the same speaker, so these are
engineering diagnostics only.

## Passenger loudness sweep

The passenger level is relative to the target speech. Negative values mean the
passenger is louder; positive values mean the passenger is quieter.

| Passenger SNR | Variant | Wake recall | Intent accuracy | False actions/hour |
|---:|---|---:|---:|---:|
| -10 dB | Full | 25% | 12.5% | 96.4 |
| -10 dB | No TSE | 50% | 50% | 160.7 |
| -10 dB | No presence | 100% | 12.5% | 450.0 |
| -10 dB | No wake smoothing | 100% | 0% | 192.9 |
| +0 dB | Full | 25% | 12.5% | 0.0 |
| +0 dB | No TSE | 50% | 50% | 64.3 |
| +0 dB | No presence | 100% | 12.5% | 417.9 |
| +0 dB | No wake smoothing | 100% | 0% | 0.0 |
| +10 dB | Full | 25% | 12.5% | 0.0 |
| +10 dB | No TSE | 50% | 50% | 32.1 |
| +10 dB | No presence | 100% | 12.5% | 289.3 |
| +10 dB | No wake smoothing | 100% | 0% | 0.0 |

The main signal is consistent: disabling presence produces high wake recall
but a large false-action rate. A louder passenger increases false actions in
the full path and especially in the no-presence path. The no-TSE row is not a
recommended design; its raw-mixture presence and intent inputs are outside
their training distribution.

Raw outputs:

- `logs/continuous_ablation_matrix_passenger_m10_2026-10-09.json`
- `logs/continuous_ablation_matrix_passenger_p10_2026-10-09.json`
- `logs/continuous_ablation_matrix_2026-10-09.json` (0 dB reference)

## Presence/KWS threshold sweep

The grid used presence thresholds 0.30/0.50/0.70/0.90 and KWS thresholds
0.30/0.40/0.50/0.60. The full candidate path at the current operating point
(presence 0.70, KWS 0.50) produced:

- wake recall: **25%**;
- intent accuracy: **12.5%**;
- false actions: **0/hour**.

The best full-path intent result in this proxy was at presence 0.50 (KWS
threshold 0.30--0.50):

- wake recall: **100%**;
- intent accuracy: **37.5%**;
- false actions: **128.6/hour**.

At presence 0.50 with no smoothing, intent accuracy stayed at 0% and false
actions increased to 257.1/hour. This indicates the dominant tradeoff is
presence operating point and post-wake state quality, not merely the KWS
threshold.

Raw output: `logs/threshold_ablation_sweep_2026-10-09.json`.

## Scalar versus frame presence

I ran the same target-conditioned KWS matrix with the presence head changed
from the candidate framewise head to the scalar cabin-trained head:

| Presence head | Wake recall | Intent accuracy | False actions/hour | Wake latency mean / p95 |
|---|---:|---:|---:|---:|
| Framewise | 25% | 12.5% | 0.0 | 1.50 / 1.50 s |
| Scalar | 100% | 25% | 32.1 | 1.75 / 2.80 s |

This is promising for recall, but the scalar head admits false actions and has
slower/more variable wake timing in this small proxy. It should not replace
the framewise head until the real-driver tuning split confirms the tradeoff.

Raw output: `logs/continuous_ablation_matrix_scalar_presence_2026-10-09.json`.

## Reproduction commands

Passenger runs:

```powershell
python scripts/eval/eval_continuous_ablation_matrix.py `
  --config configs/eval_end_to_end_cavemove_target_kws_temporal2_candidate.yaml `
  --paths configs/paths.yaml --scenes 2 --cabin-snr-db -5 0 10 20 `
  --passenger-snr-db -10 --output logs/continuous_ablation_matrix_passenger_m10_2026-10-09.json

python scripts/eval/eval_continuous_ablation_matrix.py `
  --config configs/eval_end_to_end_cavemove_target_kws_temporal2_candidate.yaml `
  --paths configs/paths.yaml --scenes 2 --cabin-snr-db -5 0 10 20 `
  --passenger-snr-db 10 --output logs/continuous_ablation_matrix_passenger_p10_2026-10-09.json
```

Threshold run:

```powershell
python scripts/eval/eval_threshold_ablation_sweep.py `
  --config configs/eval_end_to_end_cavemove_target_kws_temporal2_candidate.yaml `
  --paths configs/paths.yaml --scenes 2 --cabin-snr-db -5 0 10 20 `
  --passenger-snr-db 0 `
  --presence-thresholds 0.30 0.50 0.70 0.90 `
  --kws-thresholds 0.30 0.40 0.50 0.60 `
  --variants full no_speaker_matching no_wake_smoothing `
  --output logs/threshold_ablation_sweep_2026-10-09.json
```

Scalar presence run:

```powershell
python scripts/eval/eval_continuous_ablation_matrix.py `
  --config configs/eval_end_to_end_cavemove_target_kws_scalar_candidate.yaml `
  --paths configs/paths.yaml --scenes 2 --cabin-snr-db -5 0 10 20 `
  --passenger-snr-db 0 `
  --output logs/continuous_ablation_matrix_scalar_presence_2026-10-09.json
```
