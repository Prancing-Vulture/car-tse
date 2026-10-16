# Multi-speaker improvement iteration — 2026-10-16

## Executive result

This iteration tested a new Stage-1 target-speaker extraction adaptation
against the corrected multi-speaker proxy benchmark. The new training did
produce a small, reproducible overlap improvement, but it did not improve the
complete system in both conditions:

| 1,000-scene condition | Reference clean Stage-1 + command gate | New large passenger-mixture Stage-1 + command gate | Change |
|---|---:|---:|---:|
| Sequential target intent | 93.2% | 69.1% | -24.1 pp |
| Sequential fully correct | 84.9% | 67.2% | -17.7 pp |
| Sequential false actions | 27 (27 passenger-only) | 12 (10 passenger-only) | safer, but less accurate |
| Overlap target intent | 63.2% | 65.0% | +1.8 pp |
| Overlap fully correct | 61.2% | 64.2% | +3.0 pp |
| Overlap false actions | 0 | 0 | unchanged |

The correct conclusion is therefore conditional: the new Stage-1 model is a
useful overlap candidate and reduces erroneous sequential actions, but it is
not a replacement for the clean Stage-1 model on sequential commands. No
model from this iteration is promoted as production-ready.

## Scope and validity

The benchmark is a reproducible proxy, not a real-driver study. It uses the
same-speaker matched wake/enrollment/command construction, train/test speaker
separation as encoded by the manifests, and synthetic passenger speech. It
does not contain a consented continuous car recording, a real microphone-array
seat cue, playback-reference channels, or measured Jetson latency. The
evaluator marks these runs `production_valid=false`.

Two conditions were evaluated:

- **Sequential/non-overlap:** `--passenger-only`, target command and passenger
  events are temporally separated.
- **Simultaneous overlap:** `--overlap`, target and passenger speech are
  co-channel in the stress-test construction.

Both use full-length commands, 0 dB target/passenger proxy mixing where the
condition applies, a 0.5 s streaming hop, and the same 1,000 matched-speaker
scene construction. The current pipeline uses a 3-call confidence buffer,
post-wake compact speaker gate threshold 0.15, and intent confidence 0.90 for
the conservative final candidate.

## What changed

### Data and Stage-1 adaptation

`scripts/data/build_stage1_fsc_passenger_mixtures.py` creates train-only
target-command/passenger mixtures. The four input channels are intentionally
identical; this isolates co-channel robustness and does not claim spatial
separation. Passenger audio comes from the train-side cabin/KWS pool and is
never taken from the held-out proxy scene construction.

Two candidate adaptations were trained from the dual-domain Stage-1 model:

| Candidate | Train/validation scenes | Epochs | Best validation SI-SDRi |
|---|---:|---:|---:|
| Small | 768 / 256 | 4 | 8.43 dB |
| Large | 2,048 / 512 | 5 | 8.29 dB |

SpEx+ and its speaker encoder were frozen; only the cabin frontend was
updated at 1e-5. The large run is deliberately preserved even though its
validation SI-SDRi is lower than the small run, because its larger sample
count makes the end-to-end comparison more informative.

Training curves are in:

- `reports/figures/stage1_spex_fsc_passenger_adaptation_comparison_2026-10-16.png`
- `reports/figures/stage1_spex_fsc_passenger_adaptation_comparison_2026-10-16.pdf`

### Runtime/evaluation fixes

- The pipeline now consumes target-KWS-only constructor fields even when the
  target-KWS ablation is disabled, so regular-KWS ablations do not fail on
  unexpected configuration keys.
- `--command-speaker-gate-threshold` changes only the post-wake command gate;
  it no longer silently changes the pre-wake speaker gate.
- The multi-speaker plotter reports direct evaluator rates when present,
  including strict false wakes and strict false commands per hour.
- `plot_stage1_history.py` renders Stage-1 SI-SDR/SI-SDRi curves and supports
  comparing multiple training runs.

## Final 1,000-scene metrics

The primary safety metrics are shown before legacy wake counters. “Fully
correct” means the intended target action was correctly recognized and no
wrong or passenger action was executed in the scene.

| Run | Target wake recall | Target intent | Fully correct | False actions | Passenger-only false actions | Strict false wakes/h | Strict false commands/h |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sequential reference clean Stage-1 + gate, conf 0.80 | 100.0% | 93.2% | 84.9% | 27 | 27 | 529.97 | 17.23 |
| Sequential large passenger-mixture Stage-1 + gate, conf 0.90 | 99.3% | 69.1% | 67.2% | 12 | 10 | 494.23 | 5.91 |
| Overlap reference clean Stage-1 + gate, conf 0.80 | 100.0% | 63.2% | 61.2% | 0 | 0 | 199.03 | 1.29 |
| Overlap large passenger-mixture Stage-1 + gate, conf 0.90 | 100.0% | 65.0% | 64.2% | 0 | 0 | 174.60 | 0.77 |

The final comparison figure is:

- `reports/figures/multispeaker_final_1000_comparison_2026-10-16.png`
- `reports/figures/multispeaker_final_1000_comparison_2026-10-16.pdf`

Raw JSON logs:

- `logs/multispeaker_speech_only_commandgate015_conf080_1000_2026-10-16.json`
- `logs/multispeaker_passenger_fscpassenger_large_commandgate015_conf090_1000_2026-10-16.json`
- `logs/multispeaker_overlap_commandgate015_conf080_1000_2026-10-16.json`
- `logs/multispeaker_overlap_fscpassenger_large_commandgate015_conf090_1000_2026-10-16.json`

## Tuning-slice results

The 64-scene tuning slice was useful for selecting safety operating points but
was not used as the final claim. On that slice, the small passenger-mixture
Stage-1 model reached 82.8% target intent/fully-correct overlap with zero
false actions. The larger model plus the compact gate reached 78.1% fully
correct with zero false actions in both conditions at confidence 0.80. A
confidence threshold of 0.90 preserved 78.1% fully correct and removed the
remaining wrong executed action on the sequential slice.

The 1,000-scene results above supersede these tuning-slice numbers for the
headline comparison. The difference is evidence that the small slice is too
optimistic for model promotion.

## Interpretation and bottleneck

Wake recall is not the dominant failure in these runs. The primary sequential
failure is passenger rejection/action attribution after a target wake: the
system can detect a target wake but later open a command window or commit a
command attributable to passenger material. A post-wake compact speaker gate
substantially reduces this risk, at the cost of rejected or delayed target
commands.

Overlap remains primarily an intent/separation problem. The new passenger
mixture adaptation helps it modestly, but all channels are identical in this
proxy, so the experiment cannot measure the benefit of a real driver-seat
spatial cue. The next model step should use a mixed clean-plus-passenger
adaptation that preserves the clean sequential domain, then evaluate against
the same 1,000-scene frozen protocol. A real cabin pilot remains necessary
before any production claim.

## Reproduction commands

From the repository root:

```powershell
python scripts/eval/eval_continuous_full_pipeline_proxy.py `
  --config configs/eval_end_to_end_fscpassenger_large_identity_targetkws_overlap_trainsplit_commandgate_conf090_2026-10-16.yaml `
  --wake-manifest manifests/clean_e2e_v2/wake_knnvc_same_speaker_20x10_test_2026-10-14.jsonl `
  --passenger-manifest manifests/test/stage3_kws_cabin_test.jsonl `
  --scene-sampling matched_speaker --scenes 1000 --passenger-only `
  --full-length-commands --cabin-snr-db 0 --passenger-snr-db 0 `
  --enrollment-source same_speaker_other --hop-seconds 0.5 `
  --intent-confidence-threshold 0.90 --command-speaker-gate-threshold 0.15 `
  --output logs/multispeaker_passenger_fscpassenger_large_commandgate015_conf090_1000_2026-10-16.json `
  --artifact-dir reports/artifacts/multispeaker_passenger_fscpassenger_large_commandgate015_conf090_1000_2026-10-16
```

Change `--passenger-only` to `--overlap` and use the overlap output path to
reproduce the second final row.

## Model/package handoff

The Jetson transfer archive is:

`car_tse_jetson_multispeaker_models_2026-10-16_final.zip`

It contains the large/small/dual-domain Stage-1 checkpoints, shared presence,
KWS, target-KWS, compact verifier, wav2vec intent head and encoder, the
previous ONNX runtime archive, selected configs, and SHA-256 sums. It is kept
out of GitHub because it is approximately 881 MB; the committed manifest is
`reports/jetson_model_package_manifest_2026-10-16.md`.

Outer archive SHA-256:

`B5AEA8EB32820DBC06EF49E8C40D8B9AB2EAEB46217E38C6168EB49B18F52753`

The wav2vec path still requires the local Hugging Face
`facebook/wav2vec2-base-960h` assets or an equivalent local cache on Jetson.
