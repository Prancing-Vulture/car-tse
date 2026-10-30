# Personal-activity margin aggregation study

Date: 2026-10-12  
Status: diagnostic architecture study; no production promotion

## Motivation

The original three-way activity margin computed the top target probability
and top non-target probability independently, then subtracted them. Those
top frames can come from different portions of a three-second window. Personal
VAD and target-speaker VAD systems make framewise target/non-target decisions,
so two frame-aligned alternatives were added as opt-in modes:

- `target_top`: select the strongest target-evidence frames, then measure the
  target-minus-passenger margin on those same frames.
- `framewise_top`: compute a margin per frame, then aggregate the strongest
  framewise margins.

The historical `independent_top` mode remains the default so existing reports
and checkpoints are reproducible.

## Implementation

- `PersonalActivityHead.aggregate_margin()` implements all four modes:
  `independent_top`, `target_top`, `framewise_top`, and `mean`.
- `InCarPipeline` accepts `presence_activity_margin_mode` and records the
  selected mode in the runtime output.
- Both continuous evaluators expose the mode through configuration/CLI and
  store it in their JSON outputs.
- No learned weights were changed.

## ReMASC vehicle-stress proxy

The same 32-scene / 448-second ReMASC stress proxy was used for all modes.
It contains real vehicle/replay noise but synthetic, cross-corpus target and
passenger clips, so it is not real-driver evidence.

| Mode and threshold | Wake recall | Intent accuracy | False wakes/hour | False actions/hour |
|---|---:|---:|---:|---:|
| `independent_top`, -0.67 | 90.6% | 78.1% | 64.3 | 0.0 |
| `target_top`, no margin veto | 100.0% | 96.9% | 257.1 | 128.6 |
| `target_top`, -0.30 | 50.0% | 28.1% | 64.3 | 0.0 |
| `framewise_top`, no margin veto | 100.0% | 96.9% | 257.1 | 128.6 |
| `framewise_top`, -0.30 | 68.8% | 40.6% | 64.3 | 0.0 |

The frame-aligned modes have similar score distributions. At the stricter
cutoff they remove observed false actions, but the positive proxy recall loss
is worse than the retained independent-top operating diagnostic.

## AISHELL-5 target-absent check

The `target_top` mode with threshold `-0.30` was replayed over all 18 Eval2
sessions (3.889 hours). Full-path smoothed events were **3.34/hour**, versus
**3.60/hour** for the independent-top `-0.67` diagnostic and **4.89/hour**
for the no-margin baseline. KWS-only remained **96.69/hour**. Since AISHELL-5
contains no positive driver wake/command examples for this enrollment, this
does not show recall or a safe production FAR.

## Decision

Do not promote either frame-aligned mode. The small negative-domain reduction
does not offset the proxy recall loss, and the threshold is still selected on
synthetic proxy data. Retain the implementation as a reproducible research
option and select the aggregation mode, margin, and speaker gate jointly on
permitted same-driver tuning sessions.

## Artifacts

- `src/car_tse/models/presence_head.py`
- `src/car_tse/pipeline/in_car_pipeline.py`
- `scripts/eval/eval_continuous_full_pipeline_proxy.py`
- `scripts/eval/eval_aishell5_continuous_kws.py`
- `configs/eval_end_to_end_cavemove_personal_activity_group_p025_margin_targettop_observe.yaml`
- `configs/eval_end_to_end_cavemove_personal_activity_group_p025_margin_framewise_observe.yaml`
- `logs/continuous_full_pipeline_proxy_remasc_personal_activity_targettop_observe_8scenes_2026-10-12.json`
- `logs/continuous_full_pipeline_proxy_remasc_personal_activity_targettop_m030_8scenes_2026-10-12.json`
- `logs/continuous_full_pipeline_proxy_remasc_personal_activity_framewisetop_observe_8scenes_2026-10-12.json`
- `logs/continuous_full_pipeline_proxy_remasc_personal_activity_framewisetop_m030_8scenes_2026-10-12.json`
- `logs/aishell5_target_absent_personal_activity_group_all18_margin_targettop_m030_2026-10-12.json`
