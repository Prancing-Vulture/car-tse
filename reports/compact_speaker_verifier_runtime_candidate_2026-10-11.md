# Compact speaker-verifier runtime candidate

Date: 2026-10-11  
Status: research-only; rejected for promotion

## Why this candidate was added

The pre-existing-systems review points toward a separate, speaker-conditioned
decision branch: a small phonetic wake detector should not be the only evidence
for acting. The first WavLM experiment showed that a pretrained speaker
representation is better than the current SpEx cosine for this proxy, but its
backbone is too large for an always-on automotive path. The compact candidate
distills that relation decision into a 176,579-parameter log-mel CNN verifier.

The runtime integration is opt-in through
`speaker_gate_backend: compact`; the default SpEx path is unchanged. The
compact score is exposed as `speaker_similarity` for compatibility, but it is
a sigmoid verifier probability, not a SpEx cosine. The operating threshold is
therefore explicit in the candidate config.

## Standalone quality and resource result

Training used the existing synthetic CAVEMOVE/GSC-Marvin target-present versus
target-absent cache and a frozen WavLM relation head as teacher:

| Measurement | Result |
|---|---:|
| Train/validation/test rows | 1,208 / 260 / 256 |
| Best validation AUC | 0.913 |
| Held-out test AUC | **0.917** |
| Test EER | 15.6% |
| Test TPR at validation-selected ~1% FPR point | 16.4% |
| Test FPR at that point | 2.34% |
| Parameters | 176,579 |
| RTX 4080, two 1-second waveforms, p50 | 5.14 ms |
| RTX 4080, two 1-second waveforms, p95 | 5.24 ms |
| RTX 4080 peak CUDA allocation | 10.5 MB |

The AUC is encouraging, but the low-FPR operating point has inadequate recall.
This is not a threshold-only success: the validation point did not transfer
cleanly to the held-out test split.

## Full-cascade result

The same continuous cross-corpus proxy used for the retained baseline was run
through Stage 1 → presence → compact speaker veto → target KWS → post-wake
intent. It remains a synthetic diagnostic: the “Marvin” clip is not verified to
be spoken by the same person as the FSC enrollment/command speaker.

| Pipeline/configuration | Scenes | Target wake recall | Overall intent | False wake/h | False action/h |
|---|---:|---:|---:|---:|---:|
| Retained late-fusion baseline, c=0.24 | 8 | 100.0% | 100.0% | 0.0 | 0.0 |
| Compact verifier, validation ~1% FPR threshold 0.9904 | 2 | 0.0% | 0.0% | 0.0 | 0.0 |
| Compact verifier, relaxed diagnostic threshold 0.80 | 16 | 6.25% | 0.0% | 0.0 | 0.0 |

The relaxed point restored only one wake and no correct action. The candidate
was therefore rejected. Its low resource cost is useful, but its current input
contract and training domain do not transfer to the complete streaming scene.
It should not replace SpEx or be described as a production improvement.

## Interpretation and next use

This experiment sharpens the improvement plan from the literature:

1. collect same-driver enrollment, wake, command, passenger-wake, and
   no-command sessions first;
2. train a causal target/non-target/non-speech activity head and a separate
   speaker verifier on those sessions;
3. use tuning drivers to calibrate the verifier jointly with wake smoothing;
4. keep the compact architecture only if it preserves target recall at the
   false-action/hour constraint on held-out drivers.

The result supports researching speaker-conditioned activity detection, but it
does not justify more synthetic threshold hunting or default integration.

## Reproduction artifacts

- `src/car_tse/models/speaker_verifier.py`
- `src/car_tse/pipeline/in_car_pipeline.py`
- `scripts/eval/eval_continuous_full_pipeline_proxy.py`
- `scripts/train/train_compact_speaker_verifier_distill.py`
- `scripts/eval/benchmark_compact_speaker_verifier.py`
- `configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_compact_verifier_candidate.yaml`
- `configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_compact_verifier_operating_diagnostic.yaml`
- `logs/compact_speaker_verifier_gsc_ft1_2026-10-11.json`
- `logs/compact_speaker_verifier_latency_gpu_2026-10-11.json`
- `logs/compact_speaker_verifier_latency_cpu_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_compact_verifier_smoke_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_compact_verifier_operating_diagnostic_2026-10-11.json`
