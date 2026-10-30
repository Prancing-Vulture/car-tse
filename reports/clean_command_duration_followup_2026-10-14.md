# Clean command-duration follow-up

Date: 2026-10-14

## Purpose

The first clean end-to-end proxy run used a fixed two-second command placement and
two-second intent input. Several Fluent Speech Commands examples are longer than
two seconds, especially the bedroom heat/light commands. This follow-up checks
whether the apparent clean-pipeline errors were caused by truncation rather than
by the intent model.

This is still a synthetic cross-corpus engineering diagnostic. It is not a
real-driver, real-cabin, or production-readiness evaluation.

## Change

`scripts/eval/eval_continuous_full_pipeline_proxy.py` now exposes:

- `--proxy-command-seconds`: duration reserved for the command in the proxy;
- `--intent-input-seconds`: fixed input duration supplied to the intent stage.

The follow-up used `--proxy-command-seconds 3.5` and
`--intent-input-seconds 3.25`, which preserves the longest selected FSC source
utterances instead of clipping them at two seconds. The existing endpoint-trim
command-window strategy and all model checkpoints were kept unchanged.

## Clean comparison

| Run | Scenes | Command / intent duration | Wake recall | Intent overall | Intent given wake | False wakes | False actions |
|---|---:|---:|---:|---:|---:|---:|---:|
| Previous clean proxy | 16 | 2.0 / 2.0 s | 100.0% | 100.0% | 100.0% | 0 | 0 |
| Previous clean proxy, 64-scene extension | 64 | 2.0 / 2.0 s | 100.0% | 87.5% | 87.5% | 0 | 0 |
| Full-command follow-up | 16 | 3.5 / 3.25 s | 100.0% | 100.0% | 100.0% | 0 | 0 |
| Full-command follow-up | 64 | 3.5 / 3.25 s | 100.0% | 100.0% | 100.0% | 1 | 0 |

The 64-scene full-command run covered 896 seconds (0.249 hours). Mean target
wake latency was 1.914 seconds and p95 was 3.0 seconds. The single false wake is
an event-level proxy error; its short duration means its hourly rate is not a
reliable release FAR estimate.

## Noisy transfer diagnostic

The same full-command setting was tested with CAVEMOVE cabin noise at -5 dB and
0 dB cabin SNR, 32 scene/SNR rows and 896 seconds total:

| Setting | Wake recall | Intent overall | Intent given wake | False wakes/hour | False actions/hour |
|---|---:|---:|---:|---:|---:|
| Earlier two-second command setting (pre-correction) | 64.1% | 51.6% | 80.5% | stale | stale |
| Full-command v3, full-context windows | 75.0% | 70.3% | 93.8% | **0.0** | 0.0 |

The longer input materially improves command recognition after a wake and overall
completion. The full-context correction removes the apparent noisy false wakes
from this short proxy, but that zero is not a production FAR estimate. The old
threshold check (0.45/0.50/0.55) used the partial-tail scorer and must be
replayed under the corrected policy before any threshold is frozen.

The intermediate-threshold replay at 0.48 is also historical because it used
the partial-tail scorer; its false-wake rate must not be cited.

## Interpretation and next action

The clean-path errors were primarily a command-window/data-placement issue, not
evidence that the clean FSC intent classifier needs retraining. The next
high-value improvement is therefore not another clean-intent training run. The
system needs a tuning split with the real wake phrase, same-driver enrollment,
passenger speech, playback/echo, and representative cabin noise so the KWS,
presence, and event-state thresholds can be calibrated jointly. Thresholds must
then be frozen and evaluated on a separate held-out split.

The noisy proxy still has a large wake-stage gap and is not a valid basis for a
production claim. The artifacts also retain the proxy's known limitation: the
Marvin wake clip is not verified to be the same speaker as the FSC command
speaker, even though enrollment uses a same-speaker FSC utterance.

## Reproduction

```powershell
python scripts/eval/eval_continuous_full_pipeline_proxy.py `
  --config configs/eval_end_to_end_clean_english_identity_targetkws_endpoint08.yaml `
  --clean --scenes 64 --cabin-snr-db 0 --passenger-snr-db 0 `
  --enrollment-source same_speaker_other --hop-seconds 0.5 `
  --intent-input-seconds 3.25 --proxy-command-seconds 3.5 `
  --output logs/clean_interval_overlap_endpoint_trim_fullcommand_64scenes_v1_2026-10-14.json `
  --artifact-dir reports/artifacts/clean_interval_overlap_endpoint_trim_fullcommand_64scenes_v1_2026-10-14

python scripts/eval/eval_continuous_full_pipeline_proxy.py `
  --config configs/eval_end_to_end_clean_english_identity_targetkws_endpoint08.yaml `
  --scenes 32 --cabin-snr-db -5 0 --passenger-snr-db 0 `
  --enrollment-source same_speaker_other --hop-seconds 0.5 `
  --intent-input-seconds 3.25 --proxy-command-seconds 3.5 `
  --output logs/noisy_interval_overlap_endpoint_trim_fullcommand_32scenes_v1_2026-10-14.json `
  --artifact-dir reports/artifacts/noisy_interval_overlap_endpoint_trim_fullcommand_32scenes_v1_2026-10-14
```

Raw outputs:

- `logs/clean_interval_overlap_endpoint_trim_fullcommand_64scenes_v1_2026-10-14.json`
- `logs/noisy_interval_overlap_endpoint_trim_fullcommand_32scenes_v1_2026-10-14.json`
- `logs/noisy_fullcommand_threshold050_16scenes_v1_2026-10-14.json`
- `logs/noisy_fullcommand_threshold055_16scenes_v1_2026-10-14.json`
- `logs/noisy_hardneg_threshold048_fullcommand_32scenes_v1_2026-10-14.json`

## Post-provenance authoritative reruns

After the evaluator metadata correction, the full proxy runs were regenerated
without changing checkpoints or thresholds:

| Artifact | Scenes / rows | Wake recall | Intent overall | Intent given wake | False wakes/hour | False actions/hour | Speaker-consistent |
|---|---:|---:|---:|---:|---:|---:|---|
| `clean_interval_overlap_endpoint_trim_fullcommand_64scenes_v2_provenance_2026-10-14.json` | 64 / 896 s | 100.0% | 100.0% | 100.0% | 4.02* | 0.0 | **false** |
| `noisy_interval_overlap_endpoint_trim_fullcommand_32scenes_v2_provenance_2026-10-14.json` | 64 / 896 s | 75.0% | 70.3% | 93.8% | 84.4 | 0.0 | **false** |

\*The clean false-wake rate is shown only as the mechanical short-exposure
normalization (one event / 0.249 hours); it is not a release FAR estimate.
Both artifacts explicitly mark the public cross-corpus wake/command pairing as
not speaker-consistent and keep `production_valid=false`.

## Full-context window correction

The v1/v2 noisy proxy artifacts processed a final 3-second model window at
`13.0 s` even though the 14-second scene ended at `14.0 s`; the missing two
seconds were zeros. Every one of the 21 previously reported noisy false wakes
was in that invalid tail window. All continuous evaluators now share
`full_context_window_starts`, which excludes incomplete tails while retaining a
single padded window for clips shorter than one context.

The authoritative v3 reruns are therefore:

| Artifact | Wake recall | Intent overall | Intent given wake | False wakes/hour | False actions/hour |
|---|---:|---:|---:|---:|---:|
| Clean, 64 scenes | 100.0% | 100.0% | 100.0% | 4.02* | 0.0 |
| CAVEMOVE, 64 scene/SNR rows | 75.0% | 70.3% | 93.8% | **0.0** | 0.0 |

The clean one-event rate remains a short-exposure diagnostic, not a release FAR
estimate. The earlier 84.4/hour noisy value is superseded and must not be used
in the paper or readiness decision. The Smart Speaker negative corpus still
contains non-tail false accepts; its corrected full-corpus result is documented
in the stale audit.

## Corrected threshold sweep

The threshold replay was repeated under the full-context policy on 16 scenes at
both `-5 dB` and `0 dB` cabin SNR (32 rows, 448 seconds per setting):

| KWS threshold | Wake recall | Intent overall | Intent given wake | False wakes/hour | False actions/hour |
|---:|---:|---:|---:|---:|---:|
| 0.45 | 78.1% | 75.0% | 96.0% | 0.0 | 0.0 |
| 0.48 | 75.0% | 62.5% | 83.3% | 0.0 | 0.0 |
| 0.50 | 21.9% | 21.9% | 100.0% | 0.0 | 0.0 |
| 0.55 | 0.0% | 0.0% | — | 0.0 | 0.0 |

The 0.45 point is the best diagnostic point in this corrected small proxy, but
it is not a release threshold: the proxy has no verified same-driver wake,
passenger, playback, or long-duration negative study. Raw outputs are
`logs/noisy_fullcontext_threshold045_16scenes_2026-10-14.json` through
`logs/noisy_fullcontext_threshold055_16scenes_2026-10-14.json`.

## Corrected component latency benchmark

The per-component CUDA FP32 benchmark was rerun after fixing a measurement
bug: it previously hard-coded a two-second intent tensor even when the selected
configuration used a 3.25-second intent window. The benchmark now reads
`intent_input_seconds` from the loaded pipeline and pads the synthetic buffer
when the configured window exceeds the three-second Stage-1 context. This is a
measurement correction only; no checkpoint was changed.

Command:

```powershell
python scripts/eval/eval_component_latency.py `
  --config configs/eval_end_to_end_clean_english_identity_targetkws_endpoint08_longcommand_candidate.yaml `
  --device cuda --precision fp32 --warmup 3 --repeats 10 `
  --output logs/component_latency_clean_longcommand_3p25_2026-10-14.json
```

| Component | Mean ms | P95 ms | Parameters |
|---|---:|---:|---:|
| Stage 1 extraction, 3 s / 4 channels | 42.20 | 48.69 | 10,727,881 |
| Enrollment speaker embedding | 2.81 | 2.84 | 10,727,881 |
| Extracted speaker embedding | 2.81 | 2.84 | 10,727,881 |
| Presence head | 1.16 | 1.17 | 33,729 |
| Target-speaker KWS | 1.74 | 1.77 | 389,316 |
| Generic KWS | 1.40 | 1.43 | 652,450 |
| Wav2Vec intent, 3.25 s | 10.92 | 11.00 | 94,782,879 |
| Composite wake stage | 45.90 | 46.13 | — |
| Composite command stage | 58.30 | 80.88 | — |

These are model-forward measurements on random tensors using the workstation
GPU. They exclude capture, resampling, ring-buffer scheduling, host/device
transfer, and action execution, so they are not a Jetson or end-to-end latency
claim. The corrected raw result is
`logs/component_latency_clean_longcommand_3p25_2026-10-14.json`.

The regression suite remains green after the correction: **95 passed, 1
skipped**.

The follow-up open-set threshold replay adds a complete negative-corpus
calibration point: 23.81 full-path events/hour at KWS 0.50, and zero observed
events at 0.55/0.60 over 3.193 hours of Smart Speaker audio. The noisy proxy
loses all target wakes at 0.55, so this is a score-separation diagnosis, not a
threshold-freeze recommendation. See
`reports/smart_speaker_threshold_sweep_2026-10-14.md`.
