# Collaboration issue 002: simultaneous overlap collapses command accuracy

Status: confirmed on a synthetic, speaker-consistent proxy; no real-driver or
spatial-array claim is made.

## Reproduction

```powershell
python scripts/eval/eval_continuous_full_pipeline_proxy.py `
  --config configs/eval_end_to_end_clean_english_identity_targetkws_confidence_buffered_hybrid_run3.yaml `
  --wake-manifest manifests/clean_e2e_v2/wake_knnvc_same_speaker_20x10_test_2026-10-14.jsonl `
  --passenger-manifest manifests/test/stage3_kws_cabin_test.jsonl `
  --scene-sampling matched_speaker --scenes 1000 `
  --overlap --cabin-snr-db 0 --passenger-snr-db 0 `
  --enrollment-source same_speaker_other --hop-seconds 0.5 `
  --output logs/multispeaker_overlap_speech_only_e2e_v2_knnvc_matched_buffer4_1000_2026-10-15.json `
  --artifact-dir reports/artifacts/multispeaker_overlap_speech_only_e2e_v2_knnvc_matched_buffer4_1000_2026-10-15
```

The target wake is at 1.0 s, the target command starts at 2.1 s, passenger
`Marvin` starts at 2.4 s, and passenger non-wake speech starts at 3.2 s. The
passenger and target tracks overlap. All microphone channels receive the same
summed speech, so no spatial cue is available.

## Observed result

- 1,000 scenes / 3.889 synthetic hours.
- Target wake recall: 100.0%.
- Overall command accuracy: 45.3%.
- Legacy outside-target counts: 0 false wakes and 0 false actions.
- Strict exact-interval association: 802 wake windows outside the target wake
  interval (206.23/hour), plus 766 target/passenger-overlap command windows.
- Ambiguous target/passenger command-window supports: 766, or 196.97/hour.
- Stage-1 bypass control: 45.31% command accuracy on 128 scenes.

The zero legacy false-action count must not be interpreted as passenger
rejection: the passenger speech is deliberately inside the target command
interval, so an emitted action cannot be assigned to one speaker from the
window timestamp alone. The broad three-second association tolerance also
absorbs many wake windows after the target wake. The evaluator now reports
strict interval, contaminated, and ambiguous support windows separately.

## First mitigation

An append-only Wav2Vec intent candidate was initialized from the retained
checkpoint and trained on a balanced clean/overlap cache. It improved the
128-scene overlap pilot to 57.03% (+11.72 points) while keeping clean pilot
accuracy at 94.53% and target wake recall at 100%. The candidate is not yet
promoted; the 1,000-scene regression is required.

## Next engineering action

Use actual multichannel mixtures with different spatial responses and a
target-conditioned Stage-1/Stage-2 training target. Then train and evaluate a
speaker-consistent command endpoint and intent head on held-out overlap data.
Keep clean regression, long negative streams, and real-driver recording gates
in the release checklist.
