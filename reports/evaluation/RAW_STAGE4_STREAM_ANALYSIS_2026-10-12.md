# Raw-buffer Stage 4 stream replay — 2026-10-12

## Purpose

Repeat the matched 16-scene continuous proxy while keeping SpEx output for Stage 2/3 wake detection, but feeding Stage 4 the original microphone downmix recovered from the timestamped buffer. This isolates the effect of sending the TSE output versus the original buffered track to command recognition.

## Results

| Stage 2/3 gate | Stage 4 input | Target wakes | Correct end-to-end commands | Correct given wake | False wakes | False actions |
|---|---|---:|---:|---:|---:|---:|
| Standard thresholds (0.50 / 0.29) | SpEx output | 14/16 | 2/16 (12.5%) | 2/14 (14.3%) | 15 | 11 |
| Standard thresholds (0.50 / 0.29) | Original buffered track | 14/16 | 9/16 (56.3%) | 9/14 (64.3%) | 15 | 11 |
| Validation-tuned thresholds (0.70 / 0.45) | SpEx output | 7/16 | 0/16 | 0/7 | 4 | 4 |
| Validation-tuned thresholds (0.70 / 0.45) | Original buffered track | 7/16 | 6/16 (37.5%) | 6/7 (85.7%) | 4 | 4 |
| Learned Stage 2/3 fusion (0.60) | SpEx output | 5/16 | 0/16 | 0/5 | 3 | 3 |
| Learned Stage 2/3 fusion (0.60) | Original buffered track | 5/16 | 5/16 (31.3%) | 5/5 (100%) | 3 | 3 |

The scenes, seed, Stage 1/2/3 checkpoints, Stage 4 checkpoint, noise levels, and temporal settings match the previous replay. The command presence threshold is 0.35. The only change is Stage 4's audio source. The original input is the mono downmix used by the stream evaluator, sliced from the Stage 4 capture's source-time start and padded to the same buffer length when needed.

## What this shows

For these scenes, routing the original buffered audio to Stage 4 improved command accuracy for every gate setting. Stage 2/3 wake counts and false-trigger counts stayed unchanged. In the standard-threshold condition, correct commands rose from 2 to 9 of 16. With the learned gate, all five detected target wakes led to the expected command when Stage 4 received the original buffered track.

This is direct evidence that SpEx output was hurting Stage 4 command recognition in this proxy. It does **not** establish that TSE should be removed from the real system: all wake/enrollment speakers came from Speech Commands, while the commands came from Fluent Speech Commands speakers. The extractor was enrolled on a different speaker than the command speaker. A same-speaker continuous test is still needed before selecting the production audio route.

## Artifacts

The three JSON reports include per-scene traces and checkpoint provenance; corresponding CSVs provide row-level summaries.

- Standard thresholds: `continuous_pipeline_heldout_proxy_8scene_rawstage4_hardgate050_20261012.json` / `.csv`
- Validation-tuned gate: `continuous_pipeline_heldout_proxy_8scene_rawstage4_hardgate_valselected_20261012.json` / `.csv`
- Learned fusion: `continuous_pipeline_heldout_proxy_8scene_rawstage4_fusion060_20261012.json` / `.csv`

W&B saved local offline runs: standard `1vwz6685`, tuned `g4arhd5q`, fusion `hjxosdf9`. This environment still has no W&B API login configured, so the runs have not synced to the cloud.

No SpEx code or weights were changed. The input-route substitution was confined to a temporary evaluation copy of the stream evaluator.
