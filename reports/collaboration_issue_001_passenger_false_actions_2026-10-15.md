# Collaboration issue 001: passenger speech causes false actions

Status: confirmed by synthetic multi-speaker speech-only evaluation; not yet a
real-driver result.

## Reproduction

From the repository root:

```powershell
python scripts/eval/eval_continuous_full_pipeline_proxy.py `
  --config configs/eval_end_to_end_clean_english_identity_targetkws_confidence_buffered_hybrid_run3.yaml `
  --wake-manifest manifests/clean_e2e_v2/wake_knnvc_same_speaker_20x10_test_2026-10-14.jsonl `
  --passenger-manifest manifests/test/stage3_kws_cabin_test.jsonl `
  --scene-sampling matched_speaker --scenes 1000 `
  --passenger-only --cabin-snr-db 0 --passenger-snr-db 0 `
  --enrollment-source same_speaker_other --hop-seconds 0.5 `
  --output logs/multispeaker_speech_only_e2e_v2_knnvc_matched_buffer4_1000_2026-10-15.json `
  --artifact-dir reports/artifacts/multispeaker_speech_only_e2e_v2_knnvc_matched_buffer4_1000_2026-10-15
```

## Observed result

- 1,000 scenes / 3.889 synthetic hours; no cabin/road noise.
- Target wake recall: 99.9%.
- Target command accuracy: 90.2% overall, 90.29% conditioned on wake.
- False wake events: 1,102, or 283.37/hour.
- False action events: 99, or 25.46/hour.
- Target-wake latency: 1.664 s mean, 2.0 s p95.

## Why it matters

The retained clean target-only benchmark passes 90%, but this follow-up shows
that target-only accuracy hides passenger-triggered actions. The system does
not currently meet the safety objective in a multi-speaker proxy, even with no
background noise. The current scenes are sequential, so a second reproduction
with simultaneous target/passenger overlap is required before attributing the
entire gap to TSE separation.

## Proposed acceptance criteria for the fix

1. Preserve at least 98% target wake recall on the same held-out clean target
   scenes.
2. Reduce false actions by at least one order of magnitude on this proxy.
3. Add an overlap variant and report target/passenger events separately.
4. Re-run the full test suite and the long target-absent negative streams.
5. Do not promote the change until thresholds are frozen on a separate tuning
   split and then checked on permitted real-driver recordings.

This report is the local, versioned issue record. A GitHub issue should mirror
it when authenticated issue creation is available.
