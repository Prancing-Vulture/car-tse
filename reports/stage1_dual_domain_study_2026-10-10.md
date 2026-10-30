# Bounded dual-domain Stage 1 study

Date: 2026-10-10  
Checkpoint: `checkpoints/stage1_spex_dual_domain_rms010_ft1/best.pt`  
Pilot regression output: `logs/stage1_dual_domain_regression_2026-10-10.json`  
Full regression output: `logs/stage1_dual_domain_regression_full_2026-10-10.json`

## Objective

Test a small input-conditioning adaptation using existing permitted AISHELL-5
target-present rows and CAVEMOVE rows. The SpEx encoder was kept frozen; only
the cabin frontend was optimized for two epochs with an input RMS target of
0.1. The study used 512 training rows (256 per domain) and 288 validation
rows (144 per domain), with no new model architecture and no data from the
real-driver pilot.

## Pilot same-row regression check

The first inference-only comparison used 64 AISHELL-5 and 64 CAVEMOVE
validation rows for each checkpoint. The accepted CAVEMOVE cabin checkpoint
was compared with the bounded dual-domain checkpoint under identical
deterministic crops and cabin-noise settings.

| Domain | Baseline SI-SDRi | Dual-domain SI-SDRi | Delta |
|---|---:|---:|---:|
| AISHELL-5 | 2.46 dB | 6.05 dB | **+3.59 dB** |
| CAVEMOVE | 11.14 dB | 11.12 dB | **-0.02 dB** |
| Overall | 6.80 dB | 8.58 dB | +1.78 dB |

The adaptation materially improves the AISHELL-5 proxy and is effectively
neutral on this 64-row CAVEMOVE sample. This pilot was intentionally not used
as the final promotion decision.

## Full validation regression check

The follow-up comparison used all 144 AISHELL-5 and all 144 CAVEMOVE rows in
the bounded validation manifests. The deterministic crop and mixture settings
were held fixed between checkpoints.

| Domain | Baseline SI-SDRi | Dual-domain SI-SDRi | Delta | Baseline negative rows | Candidate negative rows |
|---|---:|---:|---:|---:|---:|
| AISHELL-5 | 2.385 dB | 6.243 dB | **+3.858 dB** | 59 | 42 |
| CAVEMOVE | 10.478 dB | 10.592 dB | **+0.114 dB** | — | — |
| Overall | 6.432 dB | 8.417 dB | **+1.986 dB** | — | — |

On the declared two-domain no-regression rule, the candidate passes this
bounded Stage 1 regression: it improves AISHELL-5 and does not regress
CAVEMOVE on the full validation sample. The candidate is therefore retained
for downstream proxy replay. This is still not a real-driver result and does
not by itself justify replacing the accepted production checkpoint.

## Decision

The checkpoint **passes the bounded Stage 1 no-regression gate** and is used in
the downstream fixed-checkpoint proxy comparison. It is **not promoted as the
project default yet**: the end-to-end handoff, real-cabin false-action, and
same-driver evaluation gates are still open.

This study does not establish real-driver reliability. The workspace still
has zero permitted continuous same-driver sessions.

## Reproduction

```powershell
python scripts/data/build_stage1_dual_domain_manifest.py `
  --manifest-root manifests --output-name stage1_dual_domain_rms010.jsonl `
  --aishell-train-count 256 --cavemove-train-count 256 `
  --aishell-val-count 144 --cavemove-val-count 144

python scripts/eval/eval_stage1_dual_domain_regression.py `
  --max-per-domain 64 `
  --output logs/stage1_dual_domain_regression_2026-10-10.json

python scripts/eval/eval_stage1_dual_domain_regression.py `
  --max-per-domain 144 `
  --output logs/stage1_dual_domain_regression_full_2026-10-10.json
```
