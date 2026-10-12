# Continuous full-pipeline evaluation — 2026-10-04

## Purpose

This is the first continuous evaluation of the actual cascade:

```text
4-channel cabin input → Stage 1 TSE → presence → temporal Marvin KWS → intent
```

Each 14-second synthetic scene contains:

- a target `marvin` clip followed by a labelled Fluent Speech Commands utterance;
- a passenger `marvin` clip that should not trigger an action;
- passenger non-wake speech;
- multichannel CAVEMOVE cabin noise;
- overlapping one-second KWS decisions driven by the production pipeline state.

The scene is a cross-corpus proxy. Speech Commands and Fluent Speech Commands do not provide a natural recording of the same person saying both `marvin` and an FSC command. Consequently, `speaker_consistent=false` and this result is not a real-cabin production benchmark.

## Checkpoint path

- Stage 1: `stage1_spex_transfer_mc_cavemove_cabin/best.pt`
- Stage 2: `stage2_presence_cavemove_cabin/best.pt`
- Stage 3: `stage3_kws_tse_hybrid_ft1/best.pt`
- Stage 4: `stage4_intent_wav2vec_ft8_resume_tse_domain/best.pt` plus `encoder_best.pt`

GPU: NVIDIA GeForce RTX 4080.

## Results

Four scene/condition trials were run: two target scenes at cabin SNR `0 dB` and `-5 dB`, with passenger speech at `0 dB` relative to the target.

| Presence threshold | Target wake recall | End-to-end intent success | False wake events | False actions/hour |
|---:|---:|---:|---:|---:|
| 0.90 | 0/4 (0%) | 0/4 (0%) | 1 | 64.3 |
| 0.50 | 1/4 (25%) | 0/4 (0%) | 1 | 64.3 |

At threshold `0.50`, the one detected target event reached the intent model, but the prediction was:

```text
Expected:  activate|lamp
Predicted: deactivate|lamp
```

The detected wake latency was approximately `2.5 s` from the target wake start in this proxy setup.

## Interpretation

The `0.90` gate is too strict for this cross-corpus target: the target windows produced presence probabilities around `0.68–0.72`, so the cascade stopped before KWS and intent. Lowering the gate allowed one target event through, but did not produce a correct final intent and did not eliminate the false activation.

This does not show that the clean FSC intent model has 0% accuracy. It shows that clean isolated intent accuracy does not transfer automatically to a continuous TSE-conditioned interaction, especially when the target wake and command are from different corpora and speakers.

## Reproduction

```powershell
python scripts/eval/eval_continuous_full_pipeline_proxy.py `
  --scenes 2 `
  --cabin-snr-db 0 -5 `
  --passenger-snr-db 0 `
  --presence-threshold 0.50 `
  --output logs/continuous_full_pipeline_proxy_presence50.json
```

Evaluator: `scripts/eval/eval_continuous_full_pipeline_proxy.py`.

Audio artifacts for the first scene are in `reports/artifacts/continuous_full_pipeline_proxy_presence50/`.

## Next required benchmark

The next valid step is to collect or receive continuous in-car recordings where the same driver supplies enrollment, `marvin`, and the command, with passenger and vehicle-noise annotations. The evaluator can then be reused without the cross-corpus caveat.
