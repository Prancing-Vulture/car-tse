# Stage 2 + Stage 3 joint wake decision

## What changed

The streaming pipeline can now optionally run Stage 3 on every pre-wake audio
window, even when Stage 2 gives a low speech-presence score. When a trained
fusion checkpoint is loaded, a small four-input classifier combines:

1. Stage 2 speech-presence probability.
2. Stage 3 wake-word probability.
3. Similarity between the enrolled speaker and the extracted audio speaker.
4. A flag saying whether speaker similarity was available.

Its output is used by the existing multi-window confirmation logic. After a
wake is confirmed, command capture continues through the existing timestamped
buffer path. The ordinary pipeline behavior remains the default when no
fusion checkpoint is configured.

## Why this change

Previously, Stage 2 could return early before Stage 3 scored a window. This
created a hard dependency: one low presence score erased the accumulated wake
evidence. Joint mode removes that early stop for the pre-wake decision. It
does not change SpEx+, the command buffer, or Stage 4.

Running Stage 3 on more windows costs additional inference time. The fusion
head itself has only four inputs and one small hidden layer; the added cost is
mainly evaluating Stage 3 on windows previously skipped by Stage 2. Runtime
latency still needs to be measured on the target device before deployment.

## Files

- `src/car_tse/models/stage23_fusion.py`: small fusion classifier.
- `src/car_tse/pipeline/in_car_pipeline.py`: optional joint scoring and
  checkpoint loading. The added `wake_model_probability` output preserves the
  Stage 3 score separately from the fused decision score.
- `configs/eval_stage23_score_collection.yaml`: scores every window to prepare
  train and validation examples; this is a data-collection mode, not production.
- `scripts/data/build_stage23_fusion_scores.py`: labels score windows from
  wake timing in continuous evaluation reports and checks target speakers do
  not cross train/validation splits.
- `scripts/train/train_stage23_fusion.py`: trains, saves, and logs the fusion
  head. W&B logs epoch loss, validation F1, recall, precision, and false
  positive rate. It defaults to online logging; pass `--wandb-mode offline`
  only when you intentionally want a local run. Each epoch also reports the
  validation F1-maximizing score cutoff; this is a development operating point
  and still needs continuous pipeline validation before use.
- `configs/eval_stage23_fusion_candidate.yaml`: loads the trained fusion
  checkpoint for candidate evaluation.

## How to train it

Run the score-collection config on development scenes that contain labeled
target wakes, passenger wakes, ordinary passenger speech, and vehicle noise.
Build a separate held-out-speaker validation set. Do not use the final test
set to fit the fusion head or choose its threshold.

```powershell
conda run -n audioml_base python scripts/eval/eval_continuous_full_pipeline_proxy.py `
  --config configs/eval_stage23_score_collection.yaml `
  --paths configs/paths.e_research.yaml --wandb
```

Use the resulting development reports as inputs to the builder:

```powershell
conda run -n audioml_base python scripts/data/build_stage23_fusion_scores.py `
  --train-report <train-scene-report.json> `
  --val-report <heldout-speaker-validation-report.json> `
  --output <score-rows.csv>
```

Then train the score combiner:

```powershell
conda run -n audioml_base python scripts/train/train_stage23_fusion.py `
  --scores-csv <score-rows.csv> `
  --output-dir <checkpoint-root>/stage23_fusion `
  --run-name stage23-fusion-v1
```

Copy the resulting `best.pt` under the configured checkpoint root at
`stage23_fusion/best.pt`, then run the fusion candidate config on validation
scenes to select a safe decision threshold. Keep the final test speakers and
recordings untouched until the design and threshold are frozen.

The score builder marks a window positive when at least half of its KWS crop
overlaps the labeled wake interval. It rejects a target speaker appearing in
both train and validation reports. The report collection must include both
positive and negative windows in each split.

## What this can and cannot solve

This change can address a wake missed because Stage 2 blocked Stage 3 from
scoring. It may also improve passenger/noise rejection if those examples are
represented in training. It cannot fix slow SpEx inference, incorrect command
audio timing, a Stage 4 intent mistake, or real-time deadline misses caused by
the combined compute. Those need separate measurements. No new model has been
trained and no accuracy gain is claimed yet; a candidate score and same-speaker
continuous evaluation are still required.

## Verified implementation references

No third-party model code was copied into this project. The fusion head uses
the PyTorch modules already used throughout this repository. The speaker
similarity input follows the cosine-score pattern documented in SpeechBrain's
official verification implementation:

- [SpeechBrain cosine speaker verification recipe](https://github.com/speechbrain/speechbrain/blob/develop/recipes/VoxCeleb/SpeakerRec/speaker_verification_cosine.py)
- [SpeechBrain speaker verification inference API](https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/inference/speaker.py)
- The wake and presence models being fused are the existing project models:
  `src/car_tse/models/kws_model.py` and `src/car_tse/models/presence_head.py`.
