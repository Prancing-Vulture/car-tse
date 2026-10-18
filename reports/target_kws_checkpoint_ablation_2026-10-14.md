# Target-KWS checkpoint ablation

Date: 2026-10-14

## Question

The clean endpoint candidate uses the target-speaker KWS checkpoint trained with
AISHELL-5 and Smart Speaker hard negatives. The older GSC-only target-KWS model
has a better cached validation F1, so this ablation checks whether it is a better
end-to-end choice.

## Matched clean proxy

Both runs use the same clean English identity Stage 1, frame-presence head,
clean generic KWS, Wav2Vec intent head, endpoint trimming, 500 ms hop,
same-speaker FSC enrollment diagnostic, 3.5-second proxy command placement, and
3.25-second intent input. The wake and command clips are still not verified to
be from the same real driver.

| Target-KWS checkpoint | Scenes | Wake recall | Intent overall | False wakes | False actions |
|---|---:|---:|---:|---:|---:|
| AISHELL-5/Smart Speaker hard-negative candidate | 64 | 100.0% | 100.0% | 1 | 0 |
| Older GSC-only target-KWS model | 64 | 100.0% | 89.1% | 384 | 0 |

The GSC-only model produced six repeated false wake events in every clean proxy
scene. Its higher cached validation F1 therefore does not translate into a safe
continuous cascade. The hard-negative checkpoint remains the better candidate.

The GSC-only result is an ablation, not a production estimate: the 896-second
proxy is synthetic and cross-corpus. It should not be used to claim a measured
false-accept rate for a deployed system.

## Additional candidates

Warm-starting the Smart Speaker hard-negative dataset from the positive-heavy
target-KWS checkpoint was also tried with `init_full_model: true`. Its best
validation F1 was 0.402, below the retained Smart Speaker hard-negative model's
0.474, so the new checkpoint was not evaluated in the end-to-end path or
promoted.

An ensemble of the retained hard-negative model with the positive-heavy model
using `target_kws_fusion_mode: min` produced the same 100.0% / 100.0% clean
proxy result and the same 75.0% / 70.3% noisy transfer result as the hard-
negative model alone. It adds complexity without measured benefit and is not
promoted.

## Decision

Do not replace the hard-negative target-KWS checkpoint with the GSC-only model.
The next improvement should add same-driver English cabin data and tune the
target-speaker gate and wake decoder on a tuning split. More positive-only clean
KWS tuning would likely reproduce the high false-accept trade-off shown here.

## Artifacts

- `logs/clean_interval_overlap_endpoint_trim_fullcommand_64scenes_v1_2026-10-14.json`
- `logs/clean_base_targetkws_endpoint08_fullcommand_64scenes_v1_2026-10-14.json`
- `logs/clean_targetkws_hardneg_pos_min_endpoint08_64scenes_v1_2026-10-14.json`
- `logs/noisy_targetkws_hardneg_pos_min_endpoint08_32scenes_v1_2026-10-14.json`
- `logs/stage3_target_kws_gsc_marvin_hardneg_aishell5_smart_speaker_from_pos_candidate/metrics_last.json`
- `configs/eval_end_to_end_clean_english_identity_targetkws_base_endpoint08_candidate.yaml`
