# In-car voice assistant pilot evaluation status

Date: 2026-10-09  
Scope: driver enrollment → `Marvin` → command, rejection of passenger/background speech, and component ablations

## Bottom line

The repository is ready to accept and score a small consented real-driver
pilot, but it does not contain that pilot yet. The custom recording directory
has **0 sessions, 0 drivers, and 0 recording hours**. The existing public data
cannot be relabeled as the requested same-driver evidence.

I therefore ran two permitted diagnostics:

1. a continuous cross-corpus proxy using the current checkpoints and cabin-noise
   mixtures; and
2. a real AISHELL-5 target-absent check using the current fixed enrollment.

The proxy is useful for finding component tradeoffs, but its wake and command
clips are not verified to be the same speaker. It is not a real-driver score.

## Data audit

`data/raw/custom_incar` contains only its README. There are no WAV sessions,
consent records, or event annotations to split into tuning and test sets.

AISHELL-5 is real multi-channel in-car speech and is released under CC BY-SA
4.0 by its official source ([OpenSLR 159](https://us.openslr.org/159/)), but
the local sessions do not provide the required same-driver enrollment/wake/
command sequence. It is used here only as a target-absent cabin diagnostic.
CAVEMOVE is used as cabin/noise material. Commercial or sign-in datasets were
not used as evidence without project permission.

The collection schema, permission requirements, and speaker-disjoint split
are documented in:

- `reports/real_driver_pilot_audit_2026-10-09.md`
- `reports/real_driver_pilot_protocol_2026-10-09.md`
- `reports/real_driver_pilot_session_template.json`
- `reports/real_driver_pilot_annotations_template.jsonl`
- `scripts/data/validate_real_driver_sessions.py`

The validator was run and correctly reported no session directories:
`logs/real_driver_validation_2026-10-09.json`.

## Continuous full-pipeline proxy

Settings: 8 scenes (2 target clips × 4 cabin SNRs: -5, 0, +10, +20 dB),
passenger speech at 0 dB relative level, 14 seconds per scene, 112 seconds
total. This used the repository's normal scalar-presence full pipeline and
its current checkpoints; thresholds were not re-tuned on these scenes.

| Metric | Result |
|---|---:|
| Target wake recall | 0/8 (0.0%) |
| Target command accuracy | 0/8 (0.0%) |
| False wake events | 1 |
| False wake rate | 32.1/hour |
| False actions | 0 |
| Wake latency | not available; no target wake was detected |

Raw output: `logs/full_pipeline_proxy_study_2026-10-09.json`.

The older threshold sweep on the same proxy scenes shows why this is a
calibration tradeoff rather than a single “accuracy” number:

| Presence setting | Target wake recall | Target intent accuracy | False-action rate |
|---|---:|---:|---:|
| threshold 0.50 | 3/8 (37.5%) | 1/8 (12.5%) | 64.3/hour |
| threshold 0.90 | 0/8 (0.0%) | 0/8 (0.0%) | 32.1/hour |
| Direct TSE → intent, no wake gate | — | 1/8 (12.5%) | not comparable |

Raw output: `logs/continuous_pipeline_ablation_study_2026-10-09.json`.

## Same-scene ablation matrix

This run used the target-speaker KWS candidate configuration so speaker
matching could be enabled and disabled while keeping the scene set fixed.
“False actions” are post-wake command events outside the labeled target
command region in the proxy; they are not a real human action-rate estimate.

| Variant | TSE | Presence | Speaker matching | Smoothing | Wake recall | Intent accuracy | False actions/hour | Wake latency mean / p95 |
|---|---|---|---|---|---:|---:|---:|---:|
| Full | yes | on | on | on | 2/8 (25%) | 1/8 (12.5%) | 0.0 | 1.50 / 1.50 s |
| No TSE | no | on | off* | on | 4/8 (50%) | 4/8 (50%) | 64.3 | 1.75 / 2.35 s |
| No presence | yes | off | on | on | 8/8 (100%) | 1/8 (12.5%) | 417.9 | 1.50 / 1.50 s |
| No speaker matching | yes | on | off | on | 2/8 (25%) | 1/8 (12.5%) | 0.0 | 1.50 / 1.50 s |
| No wake smoothing | yes | on | on | off | 8/8 (100%) | 0/8 (0%) | 0.0 | 1.25 / 1.50 s |
| Direct command oracle | yes | off | off | off | — | 1/8 (12.5%) | not scored | not scored |

\*The raw-mixture no-TSE variant cannot use the target-speaker embedding in a
meaningful way, so matching is disabled for that row. Its presence model is
also operating outside its training distribution; this row is a diagnostic,
not a recommended architecture.

Interpretation: the presence gate is suppressing many proxy events, but at the
current operating point it also suppresses the target. Removing it exposes a
large false-action problem. One-hit wake detection improves recall but does
not yield correct commands. Speaker matching did not change the aggregate
proxy result in this small run; it needs a speaker-consistent test before a
conclusion can be drawn.

Raw output: `logs/continuous_ablation_matrix_2026-10-09.json`.

The exact commands used for the reproducible runs are saved below (run from
the repository root):

```powershell
python scripts/data/validate_real_driver_sessions.py `
  --output logs/real_driver_validation_2026-10-09.json

python scripts/eval/eval_continuous_full_pipeline_proxy.py `
  --config configs/eval_end_to_end_cavemove_wav2vec_ft8_kws_hybrid_presence90_norm_smooth.yaml `
  --paths configs/paths.yaml --scenes 2 --cabin-snr-db -5 0 10 20 `
  --passenger-snr-db 0 --hop-seconds 1.0 `
  --output logs/full_pipeline_proxy_study_2026-10-09.json `
  --artifact-dir reports/artifacts/full_pipeline_proxy_study_2026-10-09

python scripts/eval/eval_continuous_ablation_matrix.py `
  --config configs/eval_end_to_end_cavemove_target_kws_temporal2_candidate.yaml `
  --paths configs/paths.yaml --scenes 2 --cabin-snr-db -5 0 10 20 `
  --passenger-snr-db 0 --hop-seconds 1.0 `
  --output logs/continuous_ablation_matrix_2026-10-09.json

python scripts/eval/eval_continuous_pipeline_ablation.py `
  --config configs/eval_end_to_end_cavemove_wav2vec_ft8_kws_hybrid_presence90_norm_smooth.yaml `
  --paths configs/paths.yaml --scenes 2 --cabin-snr-db -5 0 10 20 `
  --passenger-snr-db 0 --presence-thresholds 0.50 0.90 `
  --kws-threshold 0.29 --hop-seconds 1.0 `
  --output logs/continuous_pipeline_ablation_study_2026-10-09.json

python scripts/eval/eval_aishell5_continuous_kws.py `
  --paths configs/paths.yaml --split Eval2 --session-ids 001 --max-seconds 120 `
  --stage1-config configs/stage1_spex_transfer_mc_cavemove.yaml `
  --stage1-checkpoint checkpoints/stage1_spex_transfer_mc_cavemove_cabin/best.pt `
  --kws-config configs/stage3_kws_tse_hybrid.yaml `
  --kws-checkpoint checkpoints/stage3_kws_tse_hybrid_ft1/best.pt `
  --presence-checkpoint checkpoints/stage2_presence_cavemove_cabin/best.pt `
  --presence-type scalar `
  --enrollment-wav data/raw/librispeech/extracted/LibriSpeech/test-clean/1089/134686/1089-134686-0014.flac `
  --output logs/aishell5_target_absent_study_2026-10-09.json `
  --presence-threshold 0.90 --kws-threshold 0.29 `
  --smooth-window 3 --required-hits 2 --refractory-calls 2 --batch-size 16
```

## Real-cabin target-absent diagnostic

AISHELL-5 Eval2 session 001 was evaluated for 120 seconds with a fixed
LibriSpeech enrollment. The enrolled target is absent from this recording, so
this measures rejection only, not driver recall or command understanding.

| Metric | Result |
|---|---:|
| Duration | 120 s (0.0333 h) |
| Speech occupancy | 80.6% |
| Full path smoothed false events | 0.0/hour |
| KWS-only smoothed false events | 0.0/hour |

Raw output: `logs/aishell5_target_absent_study_2026-10-09.json`.

Additional passenger, threshold, and scalar-versus-frame presence ablations are
summarized in `reports/additional_ablation_results_2026-10-09.md`.

## Code and reproducibility changes

- Added a consented real-driver pilot audit/protocol and machine-readable
  session/annotation templates.
- Added `scripts/data/validate_real_driver_sessions.py` to enforce metadata,
  consent, required event types, and speaker-disjoint tuning/test assignment.
- Added `scripts/eval/eval_continuous_ablation_matrix.py` for the fixed-scene
  component comparison.
- Fixed an existing pipeline checkpoint-loader bug where scalar presence
  configurations received unused frame-presence arguments and crashed before
  inference.
- Verification: **12 tests passed, 1 skipped** (`python -m pytest -q`). No
  training was started.

## What is still required for the requested study

Collect the pilot described above, preferably with at least two tuning drivers
and at least one held-out test driver (more held-out drivers are preferable).
Then use only the tuning sessions to select presence/KWS thresholds, freeze
them, and run the held-out sessions once. The final report must state drivers,
sessions, hours, conditions, event counts, wake recall, command accuracy,
false actions/hour, and response latency with denominators. Until that happens,
the project has diagnostic evidence, not real-driver reliability evidence.
