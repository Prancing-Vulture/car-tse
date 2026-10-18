# Speaker-verifier calibration audit

Date: 2026-10-11  
Status: diagnostic rejection; no runtime threshold promoted

## Why this experiment was run

The pre-existing-systems review recommends keeping keyword evidence and
speaker identity as separate signals. Before collecting the real-driver pilot,
we tested whether the current frozen SpEx embeddings could support that second
signal when calibrated on the closest data already available.

The calibration used the cached Stage-3 target-KWS preparation data rather
than clean isolated speech. Each row contains an enrollment embedding, an
embedding computed from the extracted mixture, and a target-present or
target-absent label. The source is the existing synthetic CAVEMOVE/GSC-Marvin
cabin-mixture split; it is not a real-driver recording and it does not prove
passenger rejection.

## Method

- Stage 1 remained frozen; no model training was performed.
- Cosine similarity was computed after L2-normalizing both cached embeddings.
- A one-dimensional positive-slope logistic calibrator was fit on the existing
  train split only.
- The validation split selected the operating point at a target 1% verifier
  FPR; the test split was read only for final reporting.
- The candidate was then exercised in the existing end-to-end CAVEMOVE proxy
  with late `min` fusion.

## Verification result

| Split | Rows | Positive / negative | AUC | Mean positive cosine | Mean negative cosine |
|---|---:|---:|---:|---:|---:|
| Train | 1,208 | 604 / 604 | not used for decision | 0.2728 | 0.2581 |
| Validation | 260 | 130 / 130 | 0.5354 | 0.2232 | 0.2062 |
| Test | 256 | 128 / 128 | **0.5152** | 0.2390 | 0.2295 |

The fitted mapping was `sigmoid((cosine - 0.2661) / 2.7197)`. At the selected
validation operating point, verifier TPR was 0.0% at 0.77% FPR. On the held-out
test split, TPR was 1.56% at 0.78% FPR. This is effectively chance-level
separation and is not a usable speaker veto.

## End-to-end check

The calibrated candidate was evaluated on 16 proxy conditions (8 scenes at
two cabin-noise levels, 224 seconds total) using same-speaker-other diagnostic
enrollment. It produced:

- target wake recall: 56.25%;
- overall intent accuracy: 43.75%;
- intent accuracy conditioned on wake: 77.78%;
- false wake rate: 0 observed/hour;
- false action rate: 0 observed/hour;
- mean wake latency: 2.39 s; p95: 2.50 s.

The zero-negative rate is not a success by itself: the verifier suppressed
too many target events and the proxy contains no real passenger/session
continuity. The candidate is therefore rejected for promotion.

## Interpretation and next step

The failure is upstream of threshold selection. The current extracted-mixture
embedding does not preserve enough target identity for a generic SpEx cosine
score to distinguish the cached positive and negative mixtures. This is
consistent with the literature’s use of an explicitly trained speaker
verification or personal-activity branch, rather than assuming that an
extraction embedding is already calibrated for passenger rejection.

The next valid upgrade is a small speaker-conditioned activity/verifier head
trained on permitted same-driver sessions: enrollment, driver wake/command,
passenger speech, passenger “Marvin,” and no-command intervals. The head should
be trained and calibrated on tuning drivers, then frozen and evaluated on
held-out drivers/sessions. The current proxy data can be used for debugging,
but not as evidence that this requirement has been satisfied.

## Reproduction artifacts

- `scripts/eval/calibrate_cached_spex_speaker_verifier.py`
- `data/cache/stage3_target_kws_gsc_marvin_ft2_stage1_16k/`
- `manifests/{train,val,test}/stage3_target_kws_gsc_marvin_ft2_stage1_{split}.jsonl`
- `logs/speaker_verifier_calibration_gsc_cache_stage1_ft2_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_late_fusion_gsc_cache_calibrated_m10_2026-10-11.json`

