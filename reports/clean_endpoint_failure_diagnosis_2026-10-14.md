# Clean endpoint failure diagnosis

Date: 2026-10-14  
Status: diagnostic only; not a production or real-driver result

## Scope

This report inspects the latest clean English endpoint candidate:

- Config: `configs/eval_end_to_end_clean_english_identity_targetkws_endpoint08_latest.yaml`
- Log: `logs/clean_full_pipeline_identity_targetkws_endpoint08_latest_32scenes_2026-10-14.json`
- 32 scenes / 448 seconds
- `kws_trim_latest`, 0.8 s endpoint margin, KWS threshold 0.45

Aggregate result: 100% target wake recall, 87.5% intent accuracy, 0 observed
false wakes, and 0 observed false actions. The scenes are synthetic,
cross-corpus proxies; their wake/enrollment and command clips are not verified
to be the same speaker.

## Four failed commands

| Scene | Expected | Pipeline output | Diagnosis |
|---:|---|---|---|
| 17 | `deactivate|lights|bedroom` | `deactivate|lights|kitchen` | Direct intent confusion; expected class was not present in the emitted command candidates |
| 21 | `decrease|heat` | `increase|heat` | Direct intent confusion; expected class was not present in the emitted command candidates |
| 27 | `increase|heat|bedroom` | no intent | Endpoint/window failure; the raw command clip is classified correctly by the intent head, but no command event was emitted |
| 31 | `activate|lamp` | `deactivate|lamp` | Endpoint/window selection failure; a later candidate classified the command correctly, but the earlier candidate was selected |

The raw command check used the same intent checkpoint and the corresponding
FSC command files. It classified scenes 27 and 31 correctly, confirming that
those two misses are downstream temporal-selection failures. Scenes 17 and 21
remain intent-head errors on the raw command clip. This gives 30/32 correct
raw command classifications versus 28/32 correct full-pipeline actions on
this small proxy.

## Engineering implication

The next clean-path software experiment should be a buffered, multi-frame
command decoder: retain post-wake candidate windows until a short endpoint or
timeout, score the candidate set, and emit only after a stable decision. It
must be compared against the current first-frame behavior on a fresh clean
split and then on the noisy regression before any promotion. Separately, the
two direct FSC confusions require either a stronger/distilled English intent
model or additional speaker/channel-matched training data; changing the TSE
checkpoint alone is unlikely to fix them.

No threshold was frozen and no checkpoint was overwritten from this analysis.

## Component-boundary ablation

An independent 8-scene / 112-second replay at effectively silent cabin and
passenger levels separated the component boundaries:

| Path | Result |
|---|---:|
| Direct TSE → intent | 100% intent |
| TSE → KWS without presence | 87.5% wake recall |
| Full cascade, presence threshold 0.15 | 87.5% wake recall / 75.0% intent |
| Full cascade, presence threshold 0.45 | 50.0% wake recall / 37.5% intent |

No false events were observed in this short proxy. This identifies presence
threshold/model calibration as a clean-path bottleneck in this harness: the
intent head and Stage-1 output are not the limiting components on these eight
commands. The thresholds are not release values because the wake and command
speakers remain cross-corpus.

Artifact: `logs/clean_component_boundary_ablation_8scenes_2026-10-14.json`

## Clean presence-head adaptation (rejected)

Because the boundary ablation implicated presence calibration, an isolated
Stage-2 head was fine-tuned on 2,048 balanced clean FSC pairs (1,024
target-present and 1,024 non-target, 77 speakers), initialized from the
retained dual-domain presence checkpoint. Frozen Stage-1 features were cached
to keep the run fast.

The candidate reached 100% wake recall and 100% intent on the 16-scene clean
proxy, but the cabin-noise regression fell to 62.5% wake / 62.5% intent and
257.1 false wakes/hour on 16 CAVEMOVE scenes. Its validation balanced chunk
false-accept rate was 57.4%. The checkpoint is rejected and was not promoted.

Artifacts:

- `scripts/data/build_clean_presence_manifest.py`
- `configs/stage2_frame_presence_english_clean_identity_ft1.yaml`
- `checkpoints/stage2_frame_presence_english_clean_identity_ft1/best.pt`
- `logs/stage2_frame_presence_english_clean_identity_ft1/metrics_last.json`
- `configs/eval_end_to_end_clean_english_identity_targetkws_endpoint08_cleanpresence_ft1.yaml`
- `logs/clean_full_pipeline_cleanpresence_ft1_8scenes_2026-10-14.json`
- `logs/noisy_full_pipeline_cleanpresence_ft1_8scenes_2026-10-14.json`

## Buffered endpoint experiment (diagnostic; not promoted)

The first-frame endpoint was replaced by an opt-in confidence-buffered
strategy. It evaluates the first post-wake command crop and waits for one
additional candidate unless the first confidence is at least 0.90. A second
diagnostic variant scores both the fixed and KWS-trimmed crops. This prevents a
partial overlapping window from being committed immediately, while preserving
the low-latency path for high-confidence candidates.

| Variant | Proxy | Wake recall | Intent accuracy | False wakes/h |
|---|---|---:|---:|---:|
| Existing `kws_trim_latest` | 32 clean scenes | 100.0% | 87.5% | 0.0 |
| Confidence-buffered, 2 candidates | 32 clean scenes | 100.0% | 96.875% (31/32) | 0.0 |
| Confidence-buffered, 3 candidates | 32 clean scenes | 100.0% | 96.875% (31/32) | 0.0 |
| Confidence-buffered dual-crop | 32 clean scenes | 100.0% | 96.875% (31/32) | 0.0 |
| Confidence-buffered, clean Stage-1 | 16 noisy scenes | 75.0% | 62.5% | 96.4 |
| Confidence-buffered dual-crop, clean Stage-1 | 16 noisy scenes | 75.0% | 62.5% | 96.4 |
| Confidence-buffered, retained dual-domain | 16 noisy scenes | 50.0% | 0.0% | 0.0 |

The clean gain is useful evidence that endpoint timing contributes to the
proxy failures, but it is not a release result: all wake and command clips
remain cross-corpus and the noisy regressions do not improve. The retained
dual-domain configuration and all promoted checkpoints are unchanged. The
buffered strategies remain isolated candidates until same-driver data and a
real negative exposure set are available.

Artifacts:

- `src/car_tse/pipeline/in_car_pipeline.py`
- `scripts/eval/eval_continuous_full_pipeline_proxy.py`
- `configs/eval_end_to_end_clean_english_identity_targetkws_confidence_buffered_candidate.yaml`
- `configs/eval_end_to_end_clean_english_identity_targetkws_confidence_buffered_hybrid_candidate.yaml`
- `configs/eval_end_to_end_cavemove_target_kws_smart_speaker_compact_cabin_smart_buffered_candidate_2026-10-14.yaml`
- `logs/clean_confidence_buffered_32scenes_2026-10-14.json`
- `logs/clean_confidence_buffered3_32scenes_2026-10-14.json`
- `logs/clean_confidence_buffered_hybrid_32scenes_2026-10-14.json`
- `logs/noisy_confidence_buffered_8scenes_2026-10-14.json`
- `logs/noisy_confidence_buffered_hybrid_8scenes_2026-10-14.json`
- `logs/noisy_retained_buffered_candidate_8scenes_2026-10-14.json`
