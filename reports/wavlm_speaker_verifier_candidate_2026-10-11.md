# Pretrained WavLM speaker-verifier candidate

Date: 2026-10-11  
Status: research candidate; not integrated into the default pipeline

## Motivation

The pre-existing-systems review recommended an independent, text-independent
speaker-verification branch instead of treating the SpEx extraction embedding
as a calibrated identity score. Microsoft publishes `microsoft/wavlm-base-plus-
sv`, a WavLM model with an X-vector speaker-verification head. The model card
specifies 16-kHz audio and cosine similarity for verification. It was tested
only as an external diagnostic; the production checkpoint and default runtime
were unchanged.

## Data and protocol

- Existing synthetic CAVEMOVE/GSC-Marvin target-KWS cache.
- 1,208 train, 260 validation, and 256 test rows.
- Each row pairs the original enrollment clip with the frozen Stage-1
  extracted waveform and has a target-present/target-absent label.
- AUC is reported on the held-out test split.
- The validation split selects the threshold at approximately 1% FPR; the test
  split is not used for selection.
- This is not real-driver or real-passenger evidence.

## Results

| Verifier | Test AUC | Test TPR at validation-selected point | Test FPR |
|---|---:|---:|---:|
| Existing SpEx extracted embedding | 0.515 | 1.56% | 0.78% |
| WavLM cosine | **0.839** | 52.34% | 7.03% |
| Frozen WavLM + compact relation head | **0.896** | 37.50% | 2.34% |

The compact relation head uses normalized enrollment/extracted embeddings,
absolute difference, elementwise product, and cosine similarity. It contains
approximately 0.26 million trainable parameters; WavLM remains frozen.

The candidate improves representation quality substantially over the current
SpEx similarity, but it still does not meet a safe operating point. At the
selected low-FPR threshold, recall is too low, and the test FPR is higher than
the validation estimate. This distribution shift reinforces the need for
speaker-consistent tuning and held-out driver sessions.

## Resource result

On the RTX 4080, one 1-second, 16-kHz WavLM forward measured:

- p50: 25.7 ms;
- p95: 31.6 ms;
- 101.1 million parameters;
- 443.8 MB peak CUDA allocation.

The downloaded model file is approximately 405 MB before the duplicated
Windows cache copy. This is far too heavy to promote as an always-on automotive
branch without distillation, quantization, and a target-device benchmark.

## Decision

Do not replace the current default verifier or add WavLM to the production
pipeline. Keep the result as evidence that the identity representation is the
right improvement direction. The next production-oriented experiment is to
collect permitted same-driver cabin data, train or distill a much smaller
speaker-conditioned activity/verifier branch, and select thresholds on tuning
drivers before evaluating held-out drivers.

## Reproduction artifacts

- `scripts/eval/eval_pretrained_wavlm_speaker_verifier.py`
- `scripts/train/train_wavlm_speaker_verifier_head.py`
- `scripts/eval/benchmark_pretrained_wavlm_speaker_verifier.py`
- `data/cache/wavlm_speaker_verifier_gsc_marvin_2026-10-11/`
- `logs/wavlm_speaker_verifier_cabin_diagnostic_2026-10-11.json`
- `logs/speaker_verifier_wavlm_cabin_ft1_2026-10-11.json`
- `logs/wavlm_speaker_verifier_latency_2026-10-11.json`

