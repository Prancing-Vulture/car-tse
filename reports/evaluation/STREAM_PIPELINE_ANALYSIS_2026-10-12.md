# Continuous full-pipeline stream analysis — 2026-10-12

## Result

This test does **not** show a higher end-to-end score from the learned Stage 2/3 fusion gate. On the fixed 16-scene proxy, it reduced false triggers but also lost target wakes, and produced no correct Stage 4 command decisions. The threshold baseline had two correct end-to-end scenes.

| Operating point | Target wakes detected | Correct end-to-end commands | False wake events | False action events |
|---|---:|---:|---:|---:|
| Standard hard gate: Stage 2 0.50, Stage 3 0.29 | 14/16 (87.5%) | 2/16 (12.5%) | 15 | 11 |
| Validation-selected hard gate: Stage 2 0.70, Stage 3 0.45 | 7/16 (43.8%) | 0/16 | 4 | 4 |
| Learned Stage 2/3 fusion: threshold 0.60 | 5/16 (31.3%) | 0/16 | 3 | 3 |

All rows used the same Stage 1 SpEx, Stage 2 presence, Stage 3 wake, and Stage 4 intent checkpoints, the same random seed and 16 generated scenes, 3-second Stage 1 windows, 0.5-second hop, and cabin SNRs -5 dB and 0 dB. The command-capture presence threshold was 0.35 in all three runs. The fusion and validation-selected thresholds were selected using the separate Stage 2/3 validation sweep.

## What the counts mean

The stream contained eight selected wake recordings, each replayed at two cabin noise levels, for 16 target-wake opportunities and 224 seconds total. Fusion identified five of those opportunities. None yielded the correct final command. The ordinary threshold setup identified 14 opportunities and got two commands right, but triggered 15 false wakes and 11 false actions elsewhere in the scenes. The tuned hard gate and learned fusion reduced false events to four and three respectively, at the cost of lower wake recall and no correct commands.

These are **event counts over a short synthetic proxy**, not reliable per-hour rates. Do not extrapolate the counts to production traffic.

## Important limitation

This is not a same-speaker stream test. Wake and enrollment audio comes from Speech Commands speakers; the command clips come from Fluent Speech Commands speakers. The TSE is enrolled on the wake speaker, then asked to extract a command from a different speaker. Because Stage 1 is designed to isolate the enrolled speaker, that speaker mismatch can suppress the command before Stage 4. Therefore these results can compare gate behavior on these constructed scenes, but cannot establish real user end-to-end accuracy or prove that Stage 4 itself is the main failure.

The report marks `speaker_consistent: false` and `production_valid: false`. No SpEx source or weights were modified.

## Configuration correction during evaluation

The first fusion replay accidentally inherited a 0.90 command-audio threshold from the candidate config. It yielded no Stage 4 decisions and was excluded from the comparison. The final fusion run explicitly used command presence 0.35, matching the hard-gate runs.

## Reproduction and artifacts

Evaluation used `scripts/eval/eval_continuous_full_pipeline_proxy.py` in the `audioml_base` environment on CUDA. Each JSON report includes checkpoint hashes, scene details, source manifest provenance, measured metrics, and evaluator hash. A matching CSV is provided for each run.

- Fusion: `continuous_pipeline_heldout_proxy_8scene_fusion060_cmd035_20261012.json` and `.csv`
- Standard hard gate: `continuous_pipeline_heldout_proxy_8scene_hardgate_matched_20261012.json` and `.csv`
- Validation-selected hard gate: `continuous_pipeline_heldout_proxy_8scene_hardgate_valselected_20261012.json` and `.csv`

W&B generated local offline runs because the `audioml_base` environment had no W&B API key or saved login configured. Run IDs: `gi5u5tgk` (fusion), `034xti7s` (standard hard gate), `hc6khirs` (validation-selected hard gate). The three runs can be synced later with `wandb sync <offline-run-directory>` after logging into W&B in this environment.

## Conclusion

The learned gate currently trades recall for fewer false triggers; it has not improved the full pipeline score in this proxy. The largest unresolved measurement issue is speaker mismatch between wake/enrollment and command. The next decisive evaluation should construct continuous wake-plus-command clips from the same speaker, then replay the same frozen scenes through each gate setting and inspect the exact Stage 4 input and prediction. Until then, do not claim production-level gains or identify TSE versus Stage 4 as the proven limiter.
