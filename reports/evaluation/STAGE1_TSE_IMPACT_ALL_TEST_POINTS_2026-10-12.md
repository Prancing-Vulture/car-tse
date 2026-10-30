# Stage 1 TSE impact on wake and command tracks

**Run date:** 2026-10-12  
**Purpose:** Check whether Stage 1 extraction changes the wake-word signal seen by Stage 3 or the command signal seen by Stage 4.  
**Result type:** Diagnostic cross-corpus proxy, not a production-accuracy benchmark.

## What was tested

- Took every positive wake recording in the held-out Stage 3 test manifest: **195 recordings from 150 speakers**.
- Paired each wake recording with one of **31 representative Fluent Speech Commands (FSC) test utterances**, then made two noisy scenes per wake recording, at **−5 dB and 0 dB** cabin-noise SNR: **390 scenes total**.
- Used the same-speaker wake recording for enrollment. The command recordings have unrelated FSC speaker IDs; they are **not the enrolled wake speaker**.
- Compared wake audio before extraction, the clean target reference, and after SpEx extraction. Compared command recognition on the raw mixture, clean command reference, and Stage 1 extracted command. Also recorded the actual end-to-end runtime wake decision, Stage 4 prediction, and the exact audio files sent to Stage 4.
- No model weights were changed or trained for this test. The changes were to the evaluation script so these per-scene comparisons and audio files could be recorded.

## Findings

| Measurement | Result | What it says |
|---|---:|---|
| Stage 3 runtime wake detections | **289 / 390 (74.1%)** | The full wake path missed 101 target scenes. |
| Mean Stage 3 wake score, clean target → Stage 1 output | **0.939 → 0.661** | The extracted wake track scores lower in most scenes. |
| Stage 1 wake score lower than clean target | **389 / 390** | Extraction usually reduces the wake detector's score, sometimes substantially. |
| Stage 1 wake score fell by more than 0.10 | **283 / 390** | The reduction is large enough to matter in many scenes. |
| Stage 1 output wake score above configured 0.29 cutoff | **353 / 390** | A single score above the cutoff is not sufficient for a runtime detection; the runtime also applies its time/gate/confirmation logic. |
| Stage 4 correct on raw-mixture command crop | **390 / 390 (100%)** | The intent model recognized these isolated command examples before extraction. |
| Stage 4 correct on clean command reference | **390 / 390 (100%)** | Confirms those command examples are recognizable in isolation in this diagnostic. |
| Stage 4 correct on Stage 1 extracted command crop | **69 / 390 (17.7%)** | The extracted track often no longer supports the expected intent. |
| Actual full pipeline correct end-to-end | **49 / 390 (12.6%)** | The real pipeline has additional wake/gate/timing decisions beyond the isolated crop check. |

### By noise level

| Cabin SNR | Runtime wake detections | Correct actual end-to-end intents | Correct Stage 4 extracted-command crops | Mean wake score: clean → extracted |
|---:|---:|---:|---:|---:|
| −5 dB | 146 / 195 (74.9%) | 24 / 195 (12.3%) | 37 / 195 (19.0%) | 0.939 → 0.651 |
| 0 dB | 143 / 195 (73.3%) | 25 / 195 (12.8%) | 32 / 195 (16.4%) | 0.939 → 0.672 |

## What the evidence supports

1. **Stage 1 changes the wake track in a way that hurts Stage 3 scores.** The clean reference scored above the configured 0.29 cutoff in all 390 scenes; the Stage 1 output did so in 353, and its score was lower than the clean reference in 389. The runtime still detected only 289 wakes, so there is also a runtime timing/gating/confirmation gap after frame scoring. This test does not isolate which of those runtime rules accounts for each missed wake.
2. **Stage 1 extraction strongly changes the command audio and Stage 4 result in these scenes.** Stage 4 got all 390 raw and clean command crops right but got 69 Stage 1 extracted crops right. Since the FSC command speaker differs from the enrolled wake speaker, this is consistent with TSE suppressing a non-enrolled speaker. It does **not** prove that TSE is incorrectly suppressing a real driver's command when that driver matches the enrolled voice.
3. **The end-to-end score is not a fair estimate of real same-speaker performance.** The test intentionally combines separate datasets to inspect the signal path; the speaker mismatch makes low command accuracy expected. The 12.6% result must not be presented as production accuracy.

## Per-scene evidence and audio

- [CSV: all 390 datapoints, decisions, scores, expected/predicted intents, and WAV paths](stage1_tse_impact_all_test_points_20261012.csv)
- [Full evaluator JSON, including per-scene details and run provenance](stage1_tse_impact_all_test_points_20261012.json)
- Audio folder: `reports/artifacts/stage1_tse_impact_all_test_points_20261012/stage1_tracks/`
- Runtime Stage 4 input folder: `reports/artifacts/stage1_tse_impact_all_test_points_20261012/`

Each CSV row identifies its wake recording, enrollment recording, command recording and expected intent. It includes separate WAV paths for each **wake** and **command** event: mono input context, multichannel input context, clean target reference, raw SpEx output, and normalized SpEx output. It also lists every exact WAV passed to Stage 4 at runtime. **All 5,405 WAV files referenced by the 390 CSV rows were checked and exist.**

For example, datapoint 0 at −5 dB has the wake tracks under `stage1_tracks/point_0000_snr_-05.0/` and command tracks in the same folder. The row's `stage4_runtime_input_files` lists the runtime Stage 4 event audio for that scene.

## Reproducibility

- Pipeline config: `tmp/eval_stage3_spex_aligned_v2.yaml`
- Test wake manifest: `E:/car-tse-data/manifests/test/stage3_kws_cabin_test.jsonl`
- Enrollment/passenger manifest: `E:/car-tse-data/manifests/train/stage3_kws_cabin_train.jsonl`
- FSC intent manifest: `E:/car-tse-data/manifests/official/stage4_intent_fsc_test.jsonl`
- W&B run: [stage1-tse-impact-all-test-points-20261012](https://wandb.ai/narainharreesh-vit/car-tse/runs/vc4g0bz4)

The run used CUDA and logged the aggregate pipeline metrics and run settings to W&B. The row-level table and WAV evidence are local files linked above; do not assume the W&B run contains all WAVs.

## Next test needed for the real question

Repeat the same capture with a **same-person wake phrase and command** in one continuous recording, with held-out speakers and noise. For each scene, compare the command before and after TSE and Stage 4's result. That test can distinguish harmful command loss from expected removal of a different speaker. Until those same-speaker recordings exist, this run shows exactly what happened to the available cross-dataset audio, but cannot settle whether Stage 1 harms a real enrolled user's command.
