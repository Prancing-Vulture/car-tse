# Causal Stage 1 candidate — 2026-10-14

## Purpose

The active 16 kHz SpEx+ transfer model uses symmetric TCN padding and a
global time-and-channel normalization. That is unsuitable for a strict
streaming deployment budget. The repository already had an opt-in causal
path in the older 8 kHz vendor model, so this experiment added a backward-
compatible 16 kHz research path with left-only depthwise convolutions and
per-frame channel normalization. The default (`causal: false`) behavior was
left unchanged.

## Implementation checks

- `CausalDepthwiseConv1d` passed a prefix-invariance check: changing future
  samples did not change earlier outputs.
- A small causal SpEx+ model produced the expected three output waveforms and
  finite values.
- The existing 16 kHz transfer checkpoint loaded into the causal model with
  matching parameter names and shapes.
- The regression contract is covered by
  `tests/test_spex_adapter.py::test_spex_plus_16k_causal_shape_and_prefix_invariance`.

## Quality and training result

The same clean validation split and the same existing Stage 1 checkpoint were
used for the swap-in diagnostic:

| Variant | Checkpoint | Clean validation mean SI-SDR |
|---|---|---:|
| Accepted non-causal baseline | `stage1_spex_english_clean_identity_ft1/best.pt` | **41.67 dB** |
| Causal layers swapped in at inference | same checkpoint, `causal: true` | **4.16 dB** |

The large drop is an initialization mismatch: the per-frame causal
normalization changes the distribution expected by the pretrained TCN masks.
It is not evidence that a trained causal model is impossible.

A separate clean causal fine-tune was started under
`stage1_spex_english_clean_identity_causal_ft1`. The first epoch reported
`NaN` train and validation SI-SDR, so the run was stopped and no causal
checkpoint was promoted or used in the full pipeline. A first-batch diagnostic
was finite in FP32, while mixed-precision gradients showed NaN/Inf before the
AMP scaler reduced its scale; the current candidate therefore needs numerical
hardening and a controlled warm-start strategy before another long run.

## Decision

The causal path is **diagnostic only**. It does not change the retained
non-causal checkpoint, the ONNX release export, or any reported production
metric. The clean baseline remains the active research baseline. A future
causal attempt must first use finite-gradient guards, lower-risk initialization
or normalization, and a short smoke fine-tune before a full clean run.

This experiment also reinforces the production plan: latency cannot be fixed
by swapping a non-causal architecture at inference. It needs a model trained
under the streaming contract and then verified on the target hardware.
