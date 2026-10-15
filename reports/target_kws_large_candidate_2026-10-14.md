# Larger target-KWS candidate

Date: 2026-10-14

## Motivation

The retained Smart Speaker/AISHELL hard-negative target-KWS model uses
`[32, 64, 128]` audio channels and had cached validation F1 of 0.474. A larger
`[48, 96, 192, 256]` model was trained on the same speaker-disjoint cache,
initialized from the clean large generic KWS acoustic branch. The retained
checkpoint was not overwritten.

## Training result

The candidate trained for 12 epochs with a balanced sampler and reached:

- best validation F1: **0.667** at the default argmax;
- validation threshold sweep best F1: **0.761** at threshold 0.690;
- held-out test threshold sweep best F1: **0.830** at threshold 0.725.

These are cached 1-second target-KWS metrics, not continuous in-car results.
The val-selected threshold was used for the proxy check; the test sweep was not
used for tuning.

## End-to-end result

The candidate did not transfer to the current cross-corpus clean proxy:

| Threshold | Scenes | Wake recall | Intent overall | False wakes | False actions |
|---:|---:|---:|---:|---:|---:|
| 0.690 (validation-selected) | 64 | 1.6% | 1.6% | 0 | 0 |
| 0.400 diagnostic | 16 | 12.5% | 12.5% | 0 | 0 |

The proxy's Marvin wake audio and FSC command/enrollment audio are not
speaker-consistent, so this is a domain/identity transfer failure rather than
proof that the larger head is intrinsically worse. It is nevertheless not a
valid replacement for the current cascade. The candidate is retained for a
future same-driver evaluation, not promoted.

On the 250-file Smart Speaker negative subset (0.335 hours), the same
validation-selected threshold produced 29 full-path false wakes, or **86.5
false wakes/hour**, with zero observed false actions in the 14-file sequential
action sample. This is worse than the retained hard-negative candidate on its
matched diagnostic and reinforces the decision not to promote the larger head.

## Engineering fix discovered

The proxy evaluator did not pass target-KWS architecture fields to the pipeline
loader, so non-default target-KWS channel configurations failed with a state
dict shape mismatch. The evaluator now forwards target-KWS mel/channel,
embedding, and hidden-size settings explicitly. Existing default-model runs are
unchanged, and the full test suite remains the promotion gate.

## Artifacts

- Config: `configs/stage3_target_kws_gsc_marvin_hardneg_aishell5_smart_speaker_large_candidate.yaml`
- Checkpoint: `checkpoints/stage3_target_kws_gsc_marvin_hardneg_aishell5_smart_speaker_large_candidate/best.pt`
- Validation calibration: `logs/stage3_target_kws_gsc_marvin_hardneg_aishell5_smart_speaker_large_candidate/threshold_calibration_val_2026-10-14.json`
- Test calibration: `logs/stage3_target_kws_gsc_marvin_hardneg_aishell5_smart_speaker_large_candidate/threshold_calibration_test_2026-10-14.json`
- Clean proxy: `logs/clean_targetkws_large_endpoint08_64scenes_threshold069_v2_2026-10-14.json`
- Smart Speaker negative subset: `logs/smart_speaker_command_negative_large_targetkws_250_threshold069_2026-10-14.json`

## Evaluation-policy correction

The original negative-subset comparison above used an evaluator that scored a
zero-padded final partial window. That policy has been removed from all
continuous evaluators. The large candidate's 86.5/hour subset number should be
treated as a historical pre-fix diagnostic, not compared directly with current
full-context results. The retained hard-negative checkpoint was rerun on all
2,110 Smart Speaker negatives under the corrected policy and produced 76 false
wake events over 3.193 hours (23.81/hour), with zero false actions in the
14-file sequential action sample:
`logs/smart_speaker_command_negative_endpoint08_hardneg_nogate_fullwindows_v2_2026-10-14.json`.

This correction removes an evaluation artifact but does not make the current
target-KWS path production-safe; 23.81/hour remains far above the provisional
0.1/hour target, and the large candidate still has not transferred to the
cross-corpus positive proxy.
