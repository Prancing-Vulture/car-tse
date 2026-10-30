# Speaker-disjoint personal-activity candidate

Date: 2026-10-12  
Status: diagnostic candidate; not promoted

## Purpose

The first three-way personal-activity experiment used a row split. This
follow-up adds a stable `group_id` derived from the enrollment speaker and
trains/validates with disjoint enrollment groups. The purpose is to prevent
speaker leakage from inflating the activity-head validation score.

## Split and training

- Manifest: 2,408 rows; target/non-target/non-speech class contract unchanged.
- Training rows: 1,925 across 912 groups.
- Validation rows: 483 across 225 groups.
- Group overlap: 0.
- Four-epoch final loss: 0.3361.
- Best validation chunk F1: 0.755 at threshold 0.45.
- Checkpoint: `checkpoints/stage2_personal_activity_gsc_cavemove_group_ft1/best.pt`.

The higher F1 is more credible than the earlier row-split score, but it is
still public-corpus activity validation, not real-driver evidence.

## Full-pipeline proxy

The same 16-scene cross-corpus proxy was used. It still has an unverified
Marvin-speaker/command-speaker relationship.

| Operating point | Wake | Intent | False wakes/h | False actions/h |
|---|---:|---:|---:|---:|
| Group-disjoint, 0.35 | 0.0% | 0.0% | 0.0 | 0.0 |
| Group-disjoint, 0.25 | 75.0% | 56.25% | 208.93 | 96.43 |
| Earlier row-split, 0.25 | 75.0% | 56.25% | 208.93 | 48.21 |
| Existing binary personal head, 0.15 | 81.25% | 62.50% | 208.93 | 96.43 |

The group-disjoint checkpoint does not improve the fixed end-to-end proxy. At
0.35 it rejects every proxy target because the transferred target probability
peaks around 0.36; at 0.25 it matches the earlier candidate’s recall but loses
the earlier candidate’s observed action reduction. This is calibration and
domain/speaker mismatch evidence, not a reason to claim the head is useless.

## Decision

Reject for runtime promotion. Keep the group-split implementation and
checkpoint as the more honest training baseline. The next training run should
use the real pilot’s same-driver enrollment, driver wake/command, passenger
speech, passenger “Marvin,” and no-command labels; thresholds must be selected
on tuning drivers and frozen before held-out scoring.

## Reproduction

```powershell
python scripts/data/build_personal_activity_manifest.py `
  --base-manifest manifests/train/stage2_presence_gsc_aishell.jsonl `
  --speech-commands-root data/raw/speech_commands_v2/extracted `
  --output manifests/train/stage2_personal_activity_gsc_cavemove.jsonl `
  --max-nontarget 1200

python scripts/train/train_stage2_frame_presence.py `
  --config configs/stage2_personal_activity_gsc_cavemove_group_ft1.yaml `
  --paths configs/paths.yaml
```
