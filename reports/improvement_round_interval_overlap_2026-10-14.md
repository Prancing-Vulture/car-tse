# Improvement round: reproducible scoring and clean-path command capture

Date: 2026-10-14  
Status: diagnostic improvement; no retained checkpoint was overwritten.

## Changes made

1. Continuous proxy scoring now associates wake and command detections using
   overlap between the model window and the labelled interval. The old
   centre-time rule could miss short events near a window boundary. Each
   output window now records `window_start_seconds`, `window_end_seconds`, and
   `window_center_seconds`, and the report identifies the method as
   `interval_overlap_v2`.
2. Stage 2 evaluation now fails when the requested manifest is missing. The
   old generated validation split remains available only through the explicit
   `--allow-generated-val` flag and is recorded as
   `manifest_mode=generated_from_training_manifest`.
3. Wav2vec intent evaluation accepts `--split test` and records the actual
   manifest path, preventing an official FSC test result from being labelled
   as validation data.
4. Stage 2 and Stage 3 training now write immutable per-run metadata and
   per-epoch JSONL history alongside the existing latest-metrics files.

## Verification

The repository test suite passes:

```
90 passed, 1 skipped
```

The corrected official FSC test evaluation is stored at
`logs/stage4_intent_wav2vec_ft8_resume_tse_domain/metrics_official_test_v2.json`.
It contains 3,793 examples and reports 99.31% top-1 accuracy, 99.89% top-5
accuracy, and 99.24% macro F1 with `split=test` and the official test manifest.

## Full-pipeline diagnostic results

All rows below use the current runtime and the new interval-overlap scorer.
They are cross-corpus proxies, not real-driver evidence.

| Configuration | Rows | Wake recall | Intent overall | Intent given wake | False wakes/hour | False actions/hour |
|---|---:|---:|---:|---:|---:|---:|
| Clean, buffered command capture | 16 | 100.0% | 31.25% | 31.25% | 0.0 | 0.0 |
| Clean, endpoint-trim command capture | 16 | 100.0% | 100.0% | 100.0% | 0.0 | 0.0 |
| Clean, endpoint-trim command capture | 64 | 100.0% | 87.5% | 87.5% | 0.0 | 0.0 |
| CAVEMOVE noise, endpoint-trim | 64 | 64.06% | 51.56% | 80.49% | 84.38 | 0.0 |
| CAVEMOVE smart-speaker safety candidate | 32 | 0.0% | 0.0% | 0.0% | 0.0 | 0.0 |

The clean comparison isolates a real runtime issue: the buffered command
strategy often commits an unsuitable command window, while endpoint trimming
recovers the correct command on the same clean scenes. The larger clean run
still has 8 intent failures out of 64 scenes, so it is not a production claim.

The noisy endpoint candidate retains a substantial domain gap and produces
84.38 diagnostic false wakes/hour. The smart-speaker safety candidate rejects
the entire cross-corpus positive proxy because its speaker-conditioned gate is
too conservative for mismatched public-corpus speakers. This is safer than
acting on passengers, but it is not usable until the same driver's enrollment,
wake phrase, and command are recorded together.

## Decision

The endpoint-trim policy is the strongest clean-path candidate and should be
used for further clean tuning. It should not be promoted as the final noisy
runtime policy. The next model improvement should use speaker-consistent
English cabin recordings and hard negatives, then freeze thresholds on a
driver-disjoint tuning split before evaluating the held-out split.

The updated readiness audit remains `research_engineering_preview` and
`production_ready=false`: there are still zero real-driver sessions, no
six-channel microphone/playback capture, and no frozen real-cabin test set.

## Reproduction artifacts

- `scripts/eval/eval_continuous_full_pipeline_proxy.py`
- `tests/test_proxy_event_association.py`
- `logs/clean_interval_overlap_endpoint_trim_64scenes_v2_2026-10-14.json`
- `logs/noisy_interval_overlap_endpoint_trim_32scenes_v2_2026-10-14.json`
- `logs/continuous_full_pipeline_proxy_interval_overlap_v2_2026-10-14.json`
- `logs/production_readiness_snapshot_post_overlap_2026-10-14.json`
- `reports/production_readiness_snapshot_post_overlap_2026-10-14.md`

## Later correction: incomplete-tail scoring

The proxy runs listed above also used a final partial model window padded with
zeros. The shared continuous-window policy now excludes that tail. The
authoritative full-command rerun keeps the 75.0% wake / 70.3% overall intent /
93.8% intent-given-wake result but changes the observed noisy false wakes from
21 (84.38/hour) to 0 over 896 seconds. The old hourly value is therefore
superseded. See
`logs/noisy_interval_overlap_endpoint_trim_fullcommand_32scenes_v3_fullwindows_2026-10-14.json`
and `src/car_tse/pipeline/windowing.py`.
