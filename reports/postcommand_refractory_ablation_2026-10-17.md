# Post-command-cycle refractory ablation

Date: 2026-10-17  
Scope: reduce passenger-triggered false actions while keeping Stage 1 TSE and
the FT3 intent checkpoint frozen.

## Finding

The previous state machine only armed a post-action lockout after a command was
committed. If a target wake timed out or was suppressed by the intent-confidence
gate, a later passenger wake could start another command cycle. The new
`post_command_refractory_calls` option arms the wake lockout after those failed
cycles as well. It is opt-in and defaults to zero, preserving old behavior.

## Evaluation

Both runs use the same 1,000 speaker-consistent synthetic scenes, held-out FSC
intent rows, separate passenger sources, 0 dB passenger mixing, full-length
commands, a 0.5 s hop, the frozen TSE, and FT3 intent checkpoint. They are
controlled proxy tests, not real-cabin evidence.

| Condition | Setting | Wake recall | Fully correct | False actions | Wrong executed | Strict FA/h | Strict FC/h |
|---|---|---:|---:|---:|---:|---:|---:|
| Sequential | FT3 baseline | 99.7% | 80.3% | 15 | 35 | 119.6 | 11.1 |
| Sequential | + 8 s failed-cycle lockout | 99.7% | 80.3% | **0** | **17** | **48.1** | **6.4** |
| Overlap | FT3 baseline | 100.0% | 79.4% | 0 | 56 | 63.0 | 4.37 |
| Overlap | + 8 s failed-cycle lockout | 100.0% | 79.4% | 0 | **55** | **61.7** | **4.11** |

Target wake latency is unchanged: sequential mean/p95 2.13/3.5 s and overlap
2.15/3.5 s. The main effect is therefore safety filtering, not faster
recognition.

## Interpretation and limitation

This is the strongest current improvement to the multi-speaker proxy: it removes
all counted sequential false actions without reducing target wake recall or
fully-correct scene rate. It does not solve the remaining wrong target actions,
which are primarily intent/window/authorization errors. The 8-second value must
be tuned on real continuous sessions with deliberate back-to-back driver
commands before it can be frozen for a product.

## Files

- Config: `configs/eval_end_to_end_fscpassenger_identity_runtimealignedintent_ft3_prewakebuffer_postcommand16_2026-10-17.yaml`
- Sequential result: `logs/multispeaker_passenger_ft3_postcommand16_1000_2026-10-17.json`
- Overlap result: `logs/multispeaker_overlap_ft3_postcommand16_1000_2026-10-17.json`
- Figures: `reports/figures/postcommand_refractory_passenger_2026-10-17.{png,pdf}` and `reports/figures/postcommand_refractory_overlap_2026-10-17.{png,pdf}`
- Code: `src/car_tse/pipeline/in_car_pipeline.py`
- Retained ONNX parity: `src/car_tse/deployment/onnx_runtime.py` and `scripts/eval/eval_onnx_composed_pipeline_parity.py`
- Regression test: `tests/test_pipeline_drop.py`
