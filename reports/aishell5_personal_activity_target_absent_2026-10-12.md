# AISHELL-5 target-absent evaluation of the three-way personal-activity candidate

Date: 2026-10-12  
Status: diagnostic evidence; not a driver wake-recall benchmark

## Purpose

The new `PersonalActivityHead` was evaluated on continuous, real four-channel
AISHELL-5 cabin recordings to test rejection when the fixed enrollment speaker
is absent. AISHELL-5 does not contain the project wake word or the same-driver
enrollment-to-command protocol, so no positive driver accuracy is inferred.

## Fixed setup

- Split: AISHELL-5 `Eval2`, all 18 sessions (`001`–`018`)
- Duration: 14,000 seconds / 3.889 hours
- Speech occupancy: 0.8101
- Channels: `DX01C01.wav` through `DX04C01.wav`
- Enrollment: fixed LibriSpeech test-clean utterance
- Stage 1: `stage1_spex_dual_domain_rms010_ft1/best.pt`
- Presence: `stage2_personal_activity_gsc_cavemove_group_ft1/best.pt`
- Presence type: three-way `personal_activity`
- Presence threshold: 0.25
- Speaker gate threshold: 0.24
- Target KWS: `stage3_target_kws_gsc_marvin_hardneg_aishell5_ft1/best.pt`
- KWS threshold: 0.50
- Smoothing: two windows, two required hits, two-call refractory period

## Results

| KWS threshold | Full-path events/hour | KWS-only events/hour |
|---:|---:|---:|
| 0.30 | 11.83 | 1,200.60 |
| 0.40 | 5.66 | 341.49 |
| 0.50 | **4.89** | 96.69 |
| 0.60 | 3.09 | 39.09 |

With KWS fixed at 0.50, the presence-threshold sweep was:

| Presence threshold | Full-path events/hour |
|---:|---:|
| 0.10 | 9.51 |
| 0.15 | 8.74 |
| 0.20 | 6.94 |
| 0.25 | 4.89 |
| 0.30 | 1.80 |
| 0.35 | 0.77 |

At the selected diagnostic point (0.50), the full path produced 19 merged
events over 3.889 hours, or 4.89/hour. The speaker-conditioned gates therefore
removed most KWS-only triggers, but the result is still a target-absent
negative diagnostic rather than evidence that the system accepts the correct
driver.

## Interpretation

This is useful evidence that the three-way activity head plus the existing
speaker gate can reject a large amount of real multichannel cabin speech. It
does not establish a safe operating threshold: the threshold was selected on
synthetic proxy data, not on permitted driver/passenger tuning sessions, and
the fixed enrollment speaker is absent from every AISHELL recording.

The presence sweep separates the gate effect from the KWS effect: the
KWS-only rate is unchanged when only the presence threshold moves, while the
full-path rate falls monotonically. This is a useful rejection diagnostic, not
evidence that the higher threshold preserves driver recall.

The result supports keeping the candidate as a research baseline while
blocking production promotion until the same-driver pilot exists. The next
decisive measurement is a speaker-disjoint real-cabin test containing driver
wake/command positives, passenger speech and passenger “Marvin,” no-command
intervals, and long negative audio under the same four-channel capture setup.

## Margin-veto follow-up

The runtime now supports an opt-in target-minus-passenger activity margin.
Using the same frozen checkpoint, a margin of `-0.67` was replayed on all 18
sessions. Full-path smoothed events fell to **3.60/hour** (14 merged events)
from 4.89/hour, while KWS-only remained **96.69/hour**. The fixed enrollment
speaker is absent, so this remains target-absent rejection evidence only. The
threshold was selected on the synthetic ReMASC stress proxy and is not frozen
for deployment.

The machine-readable follow-up is:
`logs/aishell5_target_absent_personal_activity_group_all18_margin_m067_2026-10-12.json`.
The runtime implementation and reproducible candidate configuration are in
`src/car_tse/pipeline/in_car_pipeline.py`,
`scripts/eval/eval_aishell5_continuous_kws.py`, and
`configs/eval_end_to_end_cavemove_personal_activity_group_p025_margin_m067.yaml`.

## Reproduction

```powershell
python scripts/eval/eval_aishell5_continuous_kws.py `
  --split Eval2 `
  --session-ids 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 `
  --max-seconds 14000 `
  --stage1-config configs/stage1_spex_dual_domain_rms010_ft1.yaml `
  --stage1-checkpoint checkpoints/stage1_spex_dual_domain_rms010_ft1/best.pt `
  --kws-config configs/stage3_kws_gsc_marvin_tse_ft1.yaml `
  --kws-checkpoint checkpoints/stage3_kws_gsc_marvin_tse_ft2/best.pt `
  --target-kws-config configs/stage3_target_kws_gsc_marvin.yaml `
  --target-kws-checkpoint checkpoints/stage3_target_kws_gsc_marvin_hardneg_aishell5_ft1/best.pt `
  --presence-type personal_activity `
  --presence-checkpoint checkpoints/stage2_personal_activity_gsc_cavemove_group_ft1/best.pt `
  --presence-threshold 0.25 `
  --presence-thresholds 0.10 0.15 0.20 0.25 0.30 0.35 `
  --kws-threshold 0.50 `
  --kws-thresholds 0.30 0.40 0.50 0.60 `
  --speaker-gate-threshold 0.24 `
  --smooth-window 2 --required-hits 2 --refractory-calls 2 `
  --batch-size 32 `
  --output logs/aishell5_target_absent_personal_activity_group_all18_presence_sweep_2026-10-12.json
```

Machine-readable result: `logs/aishell5_target_absent_personal_activity_group_all18_presence_sweep_2026-10-12.json`.

## Frame-aligned margin follow-up

To test whether the margin should compare target and passenger evidence on the
same frames, two opt-in aggregators were added. The `target_top` mode at a
proxy-selected `-0.30` cutoff produced **3.34 full-path events/hour** over the
same 3.889-hour Eval2 exposure, compared with 3.60/hour for the independent-
top `-0.67` diagnostic and 4.89/hour without a margin. KWS-only remained
96.69/hour. This is a small target-absent reduction with no positive recall
measurement; it is not a production FAR and was rejected for promotion after
losing positive proxy recall.

Result: `logs/aishell5_target_absent_personal_activity_group_all18_margin_targettop_m030_2026-10-12.json`.
