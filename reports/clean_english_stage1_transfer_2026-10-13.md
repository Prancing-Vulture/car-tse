# Clean-English Stage 1 transfer study

Date: 2026-10-13  
Scope: English clean-data path first; cabin-noise transfer is diagnostic only.  
Status: research engineering; not production evidence.

## Question

The official clean intent model is strong, but the previous Stage 1 TSE output
could make a clean command unusable by the intent head. This study tests whether
an English clean-domain identity/preservation adaptation fixes the TSE-to-intent
handoff without overwriting the retained noisy-domain checkpoint.

## Data and evaluation contract

- Clean adaptation data: Fluent Speech Commands (FSC), 4,096 train and 1,024
  speaker-disjoint validation triplets.
- Each adaptation triplet uses one FSC utterance as mixture and target, and a
  different utterance from the same speaker as enrollment. Four repeated
  microphone paths preserve the Stage 1 multichannel input contract.
- Clean end-to-end control: 8 synthetic 14-second scenes (112 seconds total),
  four public Marvin clips and four FSC command clips, no passenger speech,
  cabin noise, or playback.
- The Marvin wake and FSC command speakers are from different corpora and are
  not verified to be the same person. Therefore this is not a same-driver or
  real-cabin evaluation.
- The command window uses a 0.1-second post-wake offset and a 0.5-second
  evaluation hop to avoid truncating the first command frame.

## Important harness correction

The initial clean-control YAML did not specify the Wav2Vec intent type and
encoder checkpoint. The loader consequently used the default CRNN intent head
against a Wav2Vec checkpoint, producing nearly uniform confidence (about
0.037). The configuration now explicitly sets:

```yaml
intent_type: wav2vec
intent_encoder: facebook/wav2vec2-base-960h
intent_encoder_checkpoint: stage4_intent_wav2vec_ft8_resume_tse_domain/encoder_best.pt
```

The standalone command-alignment probe now gives about 0.988 confidence on
the expected FSC intent. The earlier zero-intent result must not be used as a
model result.

## Results

| System / operating point | Wake recall | Intent overall | Intent given wake | False wakes/actions |
|---|---:|---:|---:|---:|
| Retained Stage 1 + corrected Wav2Vec, threshold 0.45 | 75.0% | 0.0% | 0.0% | 0 / 0 |
| Clean identity Stage 1 `stage1_spex_english_clean_identity_ft1`, threshold 0.45 | 75.0% | 75.0% | 100.0% | 0 / 0 |
| Clean identity Stage 1, diagnostic threshold 0.25 | 100.0% | 100.0% | 100.0% | 0 / 0 |
| Clean+cabin joint Stage 1 `stage1_spex_english_clean_cabin_joint_ft1`, threshold 0.25 | 100.0% | 100.0% | 100.0% | 0 / 0 |

The clean identity adaptation therefore removed the clean TSE-to-intent
failure in this proxy. The 0.25 threshold is not frozen for release; it is a
clean-control operating point. The clean KWS test sweep previously found its
best test F1 near 0.29 and a higher-precision validation operating point near
0.40, so threshold selection must be redone on a held-out, speaker-consistent
set.

On a larger 32-speaker clean proxy (64 scenes, 896 seconds), the same
checkpoint produced this threshold tradeoff:

| KWS threshold | Wake recall | Intent overall | Intent given wake | False wakes/hour |
|---:|---:|---:|---:|---:|
| 0.25 | 96.9% | 93.8% | 96.8% | 16.1 |
| 0.40 | 93.8% | 90.6% | 96.7% | 8.0 |

These rates are only descriptive because the scenes are synthetic and short;
the false-accept estimate is especially unstable at under 0.25 hours.

## Noisy transfer check

On the same 8-scene CAVEMOVE proxy at threshold 0.25:

| Stage 1 | Wake recall | Intent overall | Intent given wake | False wakes in 112 s |
|---|---:|---:|---:|---:|
| Retained old Stage 1 | 37.5% | 12.5% | 33.3% | 1 |
| Clean identity Stage 1 | 62.5% | 50.0% | 80.0% | 3 |
| Clean+cabin joint Stage 1 | 62.5% | 50.0% | 80.0% | 3 |

This is an improvement in this small synthetic proxy, but the absolute false
wake rate is not meaningful at 112 seconds and the corpus speakers are not
consistent. It cannot support a production or real-driver claim.

The bounded Stage 1 regression check also shows why the clean identity
checkpoint is not a universal replacement: versus the retained dual-domain
checkpoint it improved AISHELL-5 by +3.58 dB SI-SDRi but regressed CAVEMOVE by
−2.80 dB on 128 selected rows. The clean checkpoint is retained as a clean
branch, not promoted over the noisy baseline.

As a control, the evaluator was corrected to honor a Stage 1 configuration
with no learned cabin front end. That mean-pool/no-front-end variant still
produced 75% wake but 0% intent on the 8-scene clean control, so removing the
front end is not a clean-path replacement.

## Wake-verifier and timing ablation

The existing target-speaker KWS branch was also evaluated on 16 clean scenes.
At threshold 0.45 it reached 100% wake recall and 0 false wakes, but intent
accuracy was 87.5% because a late wake detection shifted the fixed post-wake
command window into the wrong part of the command. Changing the one-call delay
did not remove this timing sensitivity. The target-KWS branch is therefore not
promoted as-is. The opt-in `kws_trim` endpoint-aware window, with a 0.8-second
margin, preserved 100% wake recall and 0 false wakes/actions on 64 clean scenes
(896 seconds), but intent accuracy remained 87.5%. It is useful infrastructure,
but the remaining errors are command-window/data-contract errors rather than a
reason to promote the verifier or tune the intent model again. A streaming
ring buffer plus endpointing on same-driver recordings is still required.

## 2026-10-14 timing follow-up

Two additional controls were run on the same clean synthetic contract. A
one-call post-wake delay reduced intent to 50% on 32 scenes; increasing the
call-count timeout from 4 to 8 did not recover it. This shows that a fixed
call delay is not a safe substitute for endpointing when the hop changes.

The new opt-in `kws_trim_latest` strategy trims the most recent accepted
post-wake Stage-1 frame, with fallback to the original wake frame. On 32 clean
scenes / 448 seconds it produced 100% wake recall, 87.5% intent accuracy,
0 false wakes, and 0 false actions. This matches the previous 0.8-second
`kws_trim` aggregate, but removes the frozen-at-wake-frame assumption and is
therefore the better implementation candidate for future same-driver tests.

The direct-clean downstream control, which bypasses Stage 1 while retaining
the rest of the cascade, reached 81.25% intent on the same 32 scenes. This
confirms that the remaining clean-proxy errors are not solely TSE distortion:
the synthetic wake/command timing and the cross-corpus command contract are
also limiting factors.

The latest-frame strategy was then checked on 16 CAVEMOVE scenes (8 identities
at -5 and 0 dB cabin SNR). It reached 75.0% wake recall, 62.5% overall intent,
and 96.4 false wakes/hour. It is not promoted; the result is a noisy-domain
diagnostic and reinforces the need for speaker-consistent cabin tuning.

## Target-KWS cabin-cache adaptation (rejected)

To test whether the noisy score shift came from a mismatch between the
target-KWS cache and the clean Stage-1 extraction distribution, a new cache was
built from the clean identity Stage-1 checkpoint with the existing CAVEMOVE
cabin augmentation. It contained 1,208 balanced train rows, 260 validation
rows, and 256 test rows. The target-speaker KWS model was fine-tuned from the
retained smart-speaker checkpoint in a separate run. Its best validation F1
was 0.667 and it early-stopped after epoch 5.

The end-to-end regression rejected this candidate. On 32 clean scenes it gave
96.9% wake recall, 46.9% intent accuracy, and 3.05 s mean wake latency,
compared with 100% / 87.5% / 1.72 s for the existing target-KWS branch. On 16
CAVEMOVE scenes at -5 and 0 dB it gave 62.5% wake recall, 37.5% intent, and
80.4 false wakes/hour. The checkpoint is retained only as a documented
negative experiment; the existing target-KWS checkpoint remains the candidate.

## Existing cabin Stage-1 and compact-gate checks

An older `stage1_spex_gsc_marvin_absent_adapt_ft3` checkpoint was also replayed
with the current target-KWS and latest-frame endpoint. On 16 CAVEMOVE scenes at
-5 and 0 dB it produced 75.0% wake recall, 0% intent accuracy, 273.2 false
wakes/hour, and one false action. It is rejected as a replacement.

Enabling the compact speaker verifier from the retained production
configuration on the clean-identity branch produced the opposite tradeoff on
the same noisy proxy: 12.5% wake recall, 0% intent, and 0 false wakes. The
result is not a fair same-driver speaker-verification measurement because the
proxy uses cross-corpus wake/command clips and has no real enrollment-to-drive
continuity. It is retained as a safety/domain-mismatch diagnostic, not as a
release operating point.

## Checkpoints and reproducibility

- Clean adaptation config: `configs/stage1_spex_english_clean_identity_ft1.yaml`
- Clean adaptation checkpoint: `checkpoints/stage1_spex_english_clean_identity_ft1/best.pt`
- Joint transfer config: `configs/stage1_spex_english_clean_cabin_joint_ft1.yaml`
- Joint transfer checkpoint: `checkpoints/stage1_spex_english_clean_cabin_joint_ft1/best.pt`
- Clean end-to-end config: `configs/eval_end_to_end_clean_english_identity_ft1.yaml`
- Clean manifest builder: `scripts/data/build_clean_stage1_identity_manifests.py`
- Joint manifest builder: `scripts/data/build_stage1_clean_noisy_joint_manifest.py`
- End-to-end evaluator: `scripts/eval/eval_continuous_full_pipeline_proxy.py`
- Endpoint-aware target-KWS config:
  `configs/eval_end_to_end_clean_english_identity_targetkws_endpoint08.yaml`
- 32-scene clean threshold runs:
  `logs/clean_full_pipeline_identity_ft1_w2vec_32scenes_kws025_2026-10-13.json`
  and `logs/clean_full_pipeline_identity_ft1_w2vec_32scenes_kws040_2026-10-13.json`
- Endpoint-aware target-KWS run:
  `logs/clean_full_pipeline_identity_targetkws_endpoint08_32scenes_2026-10-14.json`
- Latest-frame endpoint config:
  `configs/eval_end_to_end_clean_english_identity_targetkws_endpoint08_latest.yaml`
- Latest-frame endpoint run:
  `logs/clean_full_pipeline_identity_targetkws_endpoint08_latest_32scenes_2026-10-14.json`
- Direct-clean downstream control:
  `logs/clean_full_pipeline_bypass_stage1_targetkws_endpoint08_32scenes_2026-10-14.json`
- Noisy latest-frame diagnostic:
  `logs/noisy_full_pipeline_cleanidentity_targetkws_endpoint08_latest_8scenes_2026-10-14.json`
- Real-time streaming-runner replay:
  `logs/streaming_runner_latest_endpoint08_30s_realtime_2026-10-14.json`
- Target-KWS cabin-cache adaptation config:
  `configs/stage3_target_kws_gsc_marvin_cleanidentity_cabin_ft1.yaml`
- Target-KWS cabin-cache adaptation checkpoint:
  `checkpoints/stage3_target_kws_gsc_marvin_cleanidentity_cabin_ft1/best.pt`
- Target-KWS cabin-cache adaptation clean regression:
  `logs/clean_full_pipeline_identity_targetkws_cleanidentity_cabin_ft1_latest_32scenes_2026-10-14.json`
- Target-KWS cabin-cache adaptation noisy regression:
  `logs/noisy_full_pipeline_identity_targetkws_cleanidentity_cabin_ft1_latest_8scenes_2026-10-14.json`
- Existing cabin Stage-1 candidate regression:
  `logs/noisy_full_pipeline_gsc_absent_targetkws_endpoint08_latest_8scenes_2026-10-14.json`
- Compact speaker-gate diagnostic:
  `logs/noisy_full_pipeline_cleanidentity_targetkws_endpoint08_latest_compact_8scenes_2026-10-14.json`

## Decision and next step

The clean identity checkpoint is the current clean-path candidate. The joint
checkpoint is not promoted because it did not improve the measured noisy proxy.
The next meaningful step is not more clean intent training: it is a
speaker-consistent English recording/evaluation set and a domain-balanced
Stage 1 transfer using real cabin multichannel data, followed by threshold
freezing on tuning sessions and a held-out passenger/no-command set.
