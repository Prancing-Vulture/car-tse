# Research-informed compact intent candidate — 2026-10-12

## Question

Can the compact intent branch be made more suitable for edge deployment by
adding capacity to the existing normalized, teacher-distilled CRNN?

## Research basis

The review of pre-existing systems suggests a strict separation of concerns:

- VoiceFilter-Lite uses a small speaker-conditioned feature-mask branch and
  emphasizes preserving clean speech while suppressing overlap; its reported
  on-device model is 2.2 MB after quantization.
- Personalized keyphrase detection combines phonetic detection with an
  independent speaker-verification veto, plus separation/noise cancellation.
- SpeakerBeam-SS reduces streaming extraction cost with state-space modeling,
  but this is a Stage 1 efficiency study and does not replace the need for
  same-driver cabin evaluation.
- Recent personalized-KWS work also uses late fusion so the keyword branch and
  speaker branch can independently reject an unsafe trigger.

These references support retaining the current independent speaker/presence
vetoes and testing compact downstream models only as candidates. They do not
justify promoting a larger CRNN without evidence that it preserves domain
accuracy.

## Candidate

The candidate kept the same training data, normalization, cached Wav2Vec
teacher logits, temperature 3.0, and KD weight 0.5 as the retained compact
student, but changed the CRNN from hidden 256 to hidden 384. The pipeline
loader was made configuration-aware so non-default CRNN dimensions load
reproducibly instead of silently constructing the 256-unit model.

| Measurement | Existing normalized student | Medium candidate |
|---|---:|---:|
| Held-out TSE/cabin validation accuracy | 56.3% | 50.4% |
| Official FSC clean test accuracy | 80.5% | 73.3% |
| Full proxy overall intent accuracy | 59.4% | 50.0% |
| Full proxy intent accuracy given wake | 86.4% | 66.7% |
| Full proxy wake recall | 68.8% | 75.0% |
| Full proxy false actions | 0 observed | 0 observed |
| Parameters | 18.68M | 23.94M |
| Checkpoint size | 75.1 MB | 96.1 MB |
| GPU wake p50, 3 s context | 42.1 ms | 42.4 ms |
| GPU post-wake command p50 | 65.1 ms | 50.0 ms |

The latency measurements use synthetic random input and are engineering
measurements, not quality claims. The proxy uses 10 same-speaker diagnostic
scenes at cabin SNRs -5, 0, 5, and 10 dB; it is not a real-driver test.

## Decision

Rejected for promotion. Increasing CRNN capacity alone made both clean and
cabin-domain intent accuracy worse. The current normalized student remains the
edge fallback, while the Wav2Vec path remains the quality reference/default.
The next model improvement should be data/conditioning-aligned: a
speaker-conditioned streaming feature-mask or command encoder trained on
speaker-consistent cabin recordings, with asymmetric clean-speech-preservation
loss and an independent speaker veto. That requires the real-driver pilot
before its safety or production value can be established.

## Reproduction

```powershell
python scripts/train/train_intent_student_distill.py `
  --config configs/stage4_intent_tse_domain_norm_student_medium.yaml `
  --run-name stage4_intent_tse_domain_distill_medium_t3_a05

python scripts/eval/eval_stage4_intent.py `
  --config configs/stage4_intent_tse_domain_norm_student_medium.yaml `
  --split test --clean `
  --manifest manifests/official/stage4_intent_fsc_test.jsonl `
  --checkpoint checkpoints/stage4_intent_tse_domain_distill_medium_t3_a05/best.pt

python scripts/eval/eval_continuous_full_pipeline_proxy.py `
  --config configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_norm_crnn_medium_candidate.yaml `
  --scenes 10 --cabin-snr-db -5 0 5 10 `
  --enrollment-source same_speaker_other
```

## Artifacts

- `configs/stage4_intent_tse_domain_norm_student_medium.yaml`
- `configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_norm_crnn_medium_candidate.yaml`
- `checkpoints/stage4_intent_tse_domain_distill_medium_t3_a05/best.pt`
- `logs/stage4_intent_tse_domain_distill_medium_t3_a05/metrics_val_clean.json`
- `logs/stage4_intent_tse_domain_distill_medium_t3_a05/metrics_val_cabin.json`
- `logs/stage4_intent_tse_domain_distill_medium_t3_a05/metrics_test_clean.json`
- `logs/continuous_full_pipeline_proxy_medium_student_10scenes_2026-10-12.json`
- `logs/full_pipeline_latency_medium_student_gpu_2026-10-12.json`
- `src/car_tse/pipeline/in_car_pipeline.py`
- `scripts/eval/eval_continuous_full_pipeline_proxy.py`
