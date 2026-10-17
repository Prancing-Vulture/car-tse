# Runtime-aligned intent iteration

Date: 2026-10-17  
Scope: improve multi-speaker end-to-end accuracy while keeping Stage 1 TSE frozen.

## Executive result

Fine-tuning only the final two wav2vec layers of the intent model produced the strongest current proxy result without changing TSE:

| Condition | Scenes | Fully correct | Strict action-correct | Target wake recall | False actions | Wrong executed actions |
|---|---:|---:|---:|---:|---:|---:|
| Passenger, non-overlap | 1,000 | **80.30%** | **78.00%** | 99.70% | 15 / 3.86 h⁻¹ | 35 / 3.50% |
| Passenger, overlap | 1,000 | **79.40%** | **78.00%** | 100.00% | 0 / 0 h⁻¹ | 56 / 5.60% |

These are synthetic proxy scenes, not real-driver recordings. The pipeline reports `production_valid: false` by design. Each scene contains a speaker-consistent target wake and command, a separate passenger source, 4-channel synthetic mixing, and no cabin-noise track; the passenger SNR is 0 dB. They are useful for controlled comparisons, but they do not establish real-cabin reliability.

## Model change

- Stage 1 SpEx+ TSE: frozen; no TSE training was performed in this iteration.
- Stage 2 presence, Stage 3 KWS, compact speaker verifier: unchanged.
- Stage 4: wav2vec2 intent encoder initialized from the mixed-overlap checkpoint; the final two encoder layers were unfrozen with a separate low learning rate, while the classification head was fine-tuned.
- Training cache: runtime-aligned 3-second TSE command windows, 16,384 training rows and 4,096 validation rows.
- Best checkpoint: `checkpoints/stage4_intent_runtime_aligned_frozen_small_3s_ft3_unfreeze2_2026-10-17/best.pt`.
- Best validation point: epoch 4, 88.8% accuracy, 88.9% macro-F1, 97.7% top-5 accuracy.
- Training plot: `reports/figures/stage4_intent_runtime_aligned_frozen_small_3s_ft3_unfreeze2_2026-10-17.png` and `.pdf`.
- End-to-end comparison plots: `reports/figures/runtime_aligned_intent_ft3_overlap_vs_baseline_2026-10-17.png` and `reports/figures/runtime_aligned_intent_ft3_passenger_vs_baseline_2026-10-17.png` (PDF counterparts are beside them).

## Why the first run was rejected

The first frozen-feature 3-second run used a stale feature cache whose row order did not match the current manifest. It showed about 9% training accuracy while validation was much higher, which is an alignment failure rather than useful learning. The feature cache was rebuilt from the current manifests before FT2 and FT3. The stale candidate is retained for audit only and is not promoted.

## End-to-end comparison

The closest prior 128-scene configuration used the mixed-overlap intent head with the same pre-wake buffer and command-window strategy:

| Condition | Prior mixed head, 128 | FT3, 128 | FT3, 1,000 |
|---|---:|---:|---:|
| Passenger, non-overlap fully correct | 78.13% | 78.91% | **80.30%** |
| Passenger, overlap fully correct | 76.56% | 78.91% | **79.40%** |

The 1,000-scene overlap result has zero false actions but 56 wrong target actions. The non-overlap result has 15 false actions and 35 wrong target actions. Therefore the intent adaptation improves task completion, but the dominant remaining product risk is still action authorization: the system can accept a wake and produce a plausible command while the target/passenger attribution is not sufficiently reliable.

## Safety and latency observations

- Target wake latency: mean 2.15 s overlap and 2.13 s non-overlap; p95 3.5 s for both 1,000-scene runs.
- Strict false-wake rate: 63.0 h⁻¹ overlap and 119.6 h⁻¹ non-overlap.
- Strict false-command rate: 4.37 h⁻¹ overlap and 11.06 h⁻¹ non-overlap.
- These rates are proxy-scene rates and should not be confused with a production FAR estimate; they are dominated by the evaluator's passenger event construction and tolerance rules.

## Interpretation

The fine-tuned intent head is a valid candidate and is the current best intent checkpoint for the controlled multi-speaker proxy. It does not solve the main bottleneck by itself. The next high-value improvement is a target-speaker authorization gate or seat-zone cue calibrated on held-out speakers, followed by true endpointing. More intent-only training without fixing that authorization path is likely to improve recognition while leaving wrong actions unchanged.

## Reproducibility artifacts

- Config: `configs/eval_end_to_end_fscpassenger_identity_runtimealignedintent_ft3_prewakebuffer_2026-10-17.yaml`
- Overlap JSON: `logs/multispeaker_overlap_runtimealignedintent_ft3_prewakebuffer_1000_2026-10-17.json`
- Non-overlap JSON: `logs/multispeaker_passenger_runtimealignedintent_ft3_prewakebuffer_1000_2026-10-17.json`
- 128-scene JSONs: same basename with `_128_`.
- Training log: `logs/stage4_intent_runtime_aligned_frozen_small_3s_ft3_unfreeze2_2026-10-17/`.
- The complete JSON scene records include effective thresholds, per-window probabilities, expected/predicted intents, action events, and input-source identities.

