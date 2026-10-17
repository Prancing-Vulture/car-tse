# Multi-speaker clean-data iteration — 2026-10-17

## Scope

This iteration targeted the clean-data synthetic proxy only.  The target and
passenger are speech-only; no CAVEMOVE, road, HVAC, or other cabin-noise bed is
present.  Each scene uses matched target wake/command speakers and a distinct
passenger control, but the scenes are assembled clips rather than continuous
real-driver recordings.  Stage 1 SpEx+ was frozen throughout.

The evaluation uses 128 matched-speaker scenes for each condition, full-length
FSC commands, a 0.5 s streaming hop, the same presence/KWS/safety settings,
and the same post-command refractory setting unless noted.  The target metric
is fully correct scenes; safety metrics are reported separately.

## Main finding: the post-wake speaker gate was the immediate proxy bottleneck

The default command speaker-gate threshold was 0.15.  In valid target-command
windows, the frozen extractor sometimes produced a similarity below that
threshold.  The three-hop command buffer then had fewer than three accepted
candidates and timed out, even when the intent model had a high-confidence
correct candidate.  Lowering only this post-wake threshold to 0.0 recovered
those target commands in the current proxy.

This is a diagnostic operating point, not a production calibration.  It must
be replaced by a learned/calibrated gate on held-out driver/passenger data;
the synthetic proxy does not establish a safe false-accept rate.

| Stack | Passenger-only fully correct | Overlap fully correct | Target wake | False actions | Wrong executed intents |
|---|---:|---:|---:|---:|---:|
| FT3, threshold 0.15 | 78.91% | 78.91% | 100% | 0 | 4 / 12 |
| FT3, threshold 0.0 | **96.88%** | 87.50% | 100% | 0 | 4 / 14 |
| FT4, threshold 0.0 | not run | 89.06% | 100% | 0 | 12 |
| FT6, threshold 0.0 | **96.88%** | **89.84%** | 100% | 0 | 4 / 11 |

The FT3 gate-0 passenger run had 12.05 strict false commands/hour and
46.21 strict false wakes/hour in this proxy.  The FT6 overlap run had zero
strict false commands/hour, 54.24 strict false wakes/hour, and zero
passenger-only false actions.  These rates are calculated from only 0.498 h of
synthetic scenes and are not release evidence.

## Intent training candidates

### FT5 — temporal-jitter fine-tune from FT3

The new cache contains 16,384 training and 8,192 validation waveforms,
balanced between clean and passenger-overlap TSE outputs and between command
offsets of 0.1 s and 0.5 s.  FT5 was initialized from FT3 and trained for
five epochs.  Best validation performance was 88.7% top-1, 88.4% macro-F1,
and 97.0% top-5.  It reached 78.91% passenger-only and 79.69% overlap on the
128-scene end-to-end test with the original 0.15 gate.  It was not promoted.

### FT6 — temporal-jitter fine-tune from overlap-specialized FT4

FT6 used the same cache but initialized from FT4 and unfroze four upper
wav2vec layers.  Best validation performance was 89.1% top-1, 88.9% macro-F1,
and 97.0% top-5.  With the calibrated proxy gate at 0.0, it reached 96.88%
passenger-only and 89.84% overlap.  It is the strongest current intent
candidate, but it is still below the 90% overlap target on the 128-scene
screen and is not production-ready.

### FT7 — factorized auxiliary-loss experiment

FT7 added training-only action/object/location auxiliary heads while saving the
same flat 31-way deployed head.  It did not improve validation (best 88.5%
top-1 / 88.4% macro-F1) and was rejected before end-to-end promotion.

## Rejected runtime variants

On FT6 with the gate at 0.0:

| Variant | Overlap fully correct | Wrong executed intents | Strict false commands/hour | Decision |
|---|---:|---:|---:|---|
| 3-hop confidence-margin buffer | 89.84% | 11 | 0.0 | current best |
| 4-hop confidence-margin buffer | 89.84% | 11 | 28.13 | reject |
| 3-hop mean-logit buffer | 89.06% | 7 | 0.0 | reject |
| FT6/FT4 75/25 logit ensemble | 89.84% | 11 | 0.0 | reject/no gain |

The remaining overlap failures are primarily genuine intent confusions after
TSE: activate/deactivate, increase/decrease, and item/location substitutions.
They are not passenger-only false actions in this proxy.

## Reproducibility artifacts

- Temporal-jitter cache builder: `scripts/data/build_joint_tse_overlap_intent_cache.py`
- FT5 training plot: `reports/figures/stage4_intent_runtime_jitter_clean_overlap_ft5_2026-10-17.png` and `.pdf`
- FT6 training plot: `reports/figures/stage4_intent_runtime_jitter_clean_overlap_ft6_fromft4_2026-10-17.png` and `.pdf`
- FT7 training plot: `reports/figures/stage4_intent_hierarchical_jitter_ft7_2026-10-17.png` and `.pdf`
- Current candidate checkpoint: `checkpoints/stage4_intent_runtime_jitter_clean_overlap_ft6_fromft4_2026-10-17/best.pt`
- Current candidate encoder: `checkpoints/stage4_intent_runtime_jitter_clean_overlap_ft6_fromft4_2026-10-17/encoder_best.pt`
- Passenger FT6 run: `logs/multispeaker_passenger_ft6_gate000_postcommand08_128_2026-10-17.json`
- Overlap FT6 run: `logs/multispeaker_overlap_ft6_gate000_postcommand08_128_2026-10-17.json`
- FT6 configuration: `configs/eval_end_to_end_fscpassenger_identity_runtimealignedintent_ft6_jitter_prewakebuffer_postcommand08_2026-10-17.yaml`

## Next engineering step

Do not promote the 0.0 gate as a release setting.  The next experiment should
use a held-out calibration split to learn a post-wake target-speaker gate or a
soft gate that preserves the valid low-similarity target windows while
rejecting passenger windows.  In parallel, the FT6 candidate should be
evaluated on a larger frozen proxy (at least 1,000 scenes) before claiming a
90% result.  Real-driver continuous recordings with playback and a fixed
microphone array remain necessary for a production or paper claim.
