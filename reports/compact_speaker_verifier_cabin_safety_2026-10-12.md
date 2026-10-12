# Cabin-aware compact speaker-verifier veto: AISHELL-5 safety result

Date: 2026-10-12  
Status: research candidate; not a production operating point; no default
checkpoint or configuration changed.

## Motivation

The retained split-gate path uses the SpEx extracted-embedding cosine as the
speaker gate. That score is nearly uninformative as an identity probability
(AUC ~0.5 on the synthetic verifier pairs). The compact log-mel speaker
verifier (`compact_speaker_verifier_gsc_ft1`, 176,579 parameters) was previously
rejected because its validation-selected threshold did not transfer into the
continuous cascade.

The cabin-aware target-KWS cache
(`stage3_target_kws_gsc_marvin_hardneg_aishell5`) now contains 768 real
AISHELL-5 four-channel target-absent windows in each of train/Dev/Eval1.
This study re-evaluates the compact verifier on that cabin-aware data,
selects a diagnostic threshold, and replays the full long target-absent
benchmark with the compact backend. It remains target-absent, cross-corpus
evidence; Eval2 has no driver wake positives.

## Standalone cabin-cache diagnostic

Script: `scripts/eval/eval_compact_speaker_verifier_cabin_cache.py`  
Log: `logs/compact_speaker_verifier_cabin_cache_diagnostic_2026-10-12.json`

The checkpoint used is the existing `compact_speaker_verifier_gsc_ft1/best.pt`
(trained on synthetic CAVEMOVE/GSC-Marvin pairs; AISHELL-5 was never in its
training data). Enrollment waves are reconstructed from the source manifests
(fixed LibriSpeech clip for AISHELL rows); extracted waves are the cached
Stage 1 ft1 outputs.

| Split | AUC | Selected threshold (1% FPR) | Test TPR | Test FPR |
|---|---:|---:|---:|---:|
| Dev | 0.9555 | 0.9079 | 0.516 | 0.0092 |
| Eval1 | 0.9764 | — | (same threshold) | — |

Threshold sweep (Eval1 test split), fraction of rows above threshold:

| Threshold | Synthetic target present | AISHELL-5 cabin negative | Non-wake hard negative |
|---:|---:|---:|---:|
| 0.20 | 96.1% | 5.6% | 20.7% |
| 0.30 | 94.5% | 3.6% | 15.7% |
| 0.50 | 84.4% | 1.7% | 8.0% |
| 0.70 | 72.7% | 0.8% | 4.7% |
| 0.9079 | 51.6% | 0.0% | 1.7% |

The verifier separates real AISHELL-5 cabin negatives almost perfectly
(mean score 0.059) while retaining most synthetic same-speaker target pairs
at thresholds of 0.2–0.5. The 1%-FPR threshold selected by the earlier cache
protocol is too strict for the positive side, which is why the earlier
cascade attempt failed; the sweep provides a usable diagnostic range instead.

## Long target-absent AISHELL-5 replay

Evaluator: `scripts/eval/eval_aishell5_continuous_kws.py`, extended with an
opt-in `--speaker-gate-backend compact`
(`--compact-speaker-verifier-checkpoint`). Default behavior is unchanged; the
reference SpEx split-gate replay reproduces the historical values exactly
(`logs/aishell5_target_absent_spex_splitgate_regression_1200s_2026-10-12.json`:
3.0 full-path events/hour and 60.0 KWS-only events/hour over 1,200 s).

Fixed setup for all rows: AISHELL-5 Eval2, all 18 sessions, 14,001.9 s
(3.889 h), 80.999% speech occupancy, Stage 1 ft1, frame presence 0.15,
target-KWS 0.50 with 2-of-2 smoothing and two-call refractory.

| Speaker gate | Full-path smoothed events | Full-path events/hour | 95% Poisson CI | KWS-only events/hour |
|---|---:|---:|---:|---:|
| SpEx cosine 0.24 (reference) | 18 | 4.63 | 2.74 – 7.00 | 96.69 |
| Compact (GSC) 0.20 | 11 | 2.83 | 1.41 – 4.73 | 96.69 |
| Compact (GSC) 0.50 | 7 | 1.80 | 0.72 – 3.36 | 96.69 |
| Compact cabin-aware 0.30 | **0** | **0.00** | 0.00 – 0.95 (one-sided upper) | 96.69 |
| Compact cabin-aware 0.50 | **0** | **0.00** | 0.00 – 0.95 (one-sided upper) | 96.69 |

The GSC-trained compact 0.50 result is a 2.6× reduction in observed real-cabin
full-path false accepts against the reference (one-sided Poisson p = 0.003).
The cabin-aware retrain (below) reaches **zero observed full-path events over
the full 3.889-hour exposure** (one-sided p = 1.5e-8 against the reference
rate). Zero events over this exposure only bounds the rate (95% upper bound
0.95/hour); it is not a license to claim reliable field performance. The
KWS-only rate is identical in every row, so the change is attributable to the
identity gate alone, not to threshold drift elsewhere.

Logs:

- `logs/aishell5_target_absent_compact_verifier_thr050_all18_2026-10-12.json`
- `logs/aishell5_target_absent_compact_verifier_thr020_all18_2026-10-12.json`
- `logs/aishell5_target_absent_compact_verifier_thr050_1200s_2026-10-12.json`
- `logs/aishell5_target_absent_compact_cabin_aware_thr030_all18_2026-10-12.json`
- `logs/aishell5_target_absent_compact_cabin_aware_thr050_all18_2026-10-12.json`

## Cabin-aware retrain (new checkpoint)

The GSC-trained verifier had never seen real cabin negatives. A cabin-aware
retrain uses the same split structure and keeps Eval2 untouched:

- Cache builder: `scripts/data/build_cabin_compact_verifier_cache.py`
  (fixed-enrollment fallback for the AISHELL pseudo-manifest rows);
- Wave cache: `data/cache/compact_speaker_verifier_cabin_aware_2026-10-12`
  (3,176 / 1,328 / 1,324 train/Dev/Eval1 pairs);
- WavLM teacher embeddings: `data/cache/wavlm_speaker_verifier_cabin_aware_2026-10-12`
  (512-dim, from the frozen `speaker_verifier_wavlm_cabin_ft1` relation head);
- Trainer: `scripts/train/train_compact_speaker_verifier_distill.py` with
  distillation weight 0.25; early stop at epoch 22;
- Checkpoint: `checkpoints/compact_speaker_verifier_cabin_aware_ft1/best.pt`
  (176,579 parameters; the original checkpoint is untouched);
- Log: `logs/compact_speaker_verifier_cabin_aware_ft1_2026-10-12.json`.

| Measure (Dev selected / Eval1 evaluated) | GSC compact | Cabin-aware compact |
|---|---:|---:|
| Dev AUC | 0.9555 | 0.9155 |
| Eval1 AUC | 0.9764 | 0.9584 |
| TPR at validation-selected ~1% FPR | 0.516 | **0.664** |
| FPR at that point | 0.0092 | 0.0067 |
| Synthetic target pass rate @ 0.30 (Eval1) | 0.945 | 0.883 |
| AISHELL-5 cabin pass rate @ 0.30 (Eval1) | 0.036 | **0.025** |
| Full-path events, 3.889 h @ 0.50 | 7 | **0** |
| Full-path events, 3.889 h @ 0.30 | not run | **0** |

The retrain lowers the aggregate AUC while improving the operating region that
matters (low-FPR recall and real-cabin rejection). This is reported as a
candidate; the threshold is not frozen.

## Smart Speaker cross-domain negative check

The University of Rochester Smart Speaker Command corpus (2,110 clean
`Hey Alexa, <command>` files, 3.193 hours) is a second negative domain with
wake-like speech. The completed all-file reference used the Smart-Speaker
negative-adapted target KWS and the SpEx gate. To isolate the identity-gate
effect, the same configuration was replayed with only the speaker gate
changed to the cabin-aware compact verifier at 0.30:

| Configuration | Full-path false wakes | False wakes/hour | KWS-only wakes/hour | 14-file action sample |
|---|---:|---:|---:|---:|
| SpEx 0.24 (reference, concurrent run) | 86 | 26.94 | 93.97 | 0 false actions |
| Cabin-aware compact 0.30 | **11** | **3.45** | 93.97 | 0 false actions |

The cabin-aware gate reduces observed full-path false wakes by 7.8× on a
second, acoustically different negative domain with the KWS-only rate
unchanged. This remains clean single-channel studio audio duplicated across
four interface channels, one file per scripted `Hey Alexa` utterance; it is
not real-cabin or production FAR evidence.

Logs:

- `logs/smart_speaker_command_negative_all_2026-10-12.json` (reference, concurrent)
- `logs/smart_speaker_command_negative_all_compact_cabin_aware_thr030_2026-10-12.json`
- Config: `configs/eval_end_to_end_cavemove_target_kws_smart_speaker_compact_cabin_aware_thr030_2026-10-12.yaml`

## Iteration 2: Smart-Speaker-augmented verifier

The cabin-aware verifier was extended with the Smart Speaker negatives in its
training data (merged manifest
`stage3_target_kws_gsc_marvin_hardneg_aishell5_smart_speaker`):
4,684 / 1,629 / 1,625 train/Dev/Eval1 pairs, 604 positives, 768 AISHELL and
1,508 Smart Speaker negatives in train. New artifacts keep the previous
checkpoint untouched:

- Caches: `data/cache/compact_speaker_verifier_cabin_smart_2026-10-12/`,
  `data/cache/wavlm_speaker_verifier_cabin_smart_2026-10-12/`
- Checkpoint: `checkpoints/compact_speaker_verifier_cabin_smart_ft1/best.pt`
- Training log: `logs/compact_speaker_verifier_cabin_smart_ft1_2026-10-12.json`
  (early stop epoch 27, distillation weight 0.25)
- Diagnostic: `logs/compact_speaker_verifier_cabin_smart_smartcache_diagnostic_2026-10-12.json`

| Measure | Cabin-aware | Cabin+Smart |
|---|---:|---:|
| Dev AUC | 0.9061 | 0.9380 |
| Eval1 AUC | 0.9547 | 0.9706 |
| TPR at validation ~1% FPR | 0.680 | **0.797** |
| FPR at that point | 0.0053 | 0.0053 |

Expanded-cache pass rates (Eval1):

| Threshold | Synthetic target present | AISHELL-5 | Smart Speaker | Non-wake |
|---:|---:|---:|---:|---:|
| 0.30 | 93.0% | 2.0% | 0.0% | 5.7% |
| 0.50 | 87.5% | 0.9% | 0.0% | 0.3% |

Cascade results with the new checkpoint at threshold 0.30:

| Benchmark | Exposure | Full-path events | Rate | KWS-only | Verdict |
|---|---:|---:|---:|---:|---|
| AISHELL-5 Eval2, target absent | 3.889 h | 0 | 0.00/h (95% UB 0.95) | 96.69/h | Held |
| Smart Speaker all-file | 3.193 h | 11 | 3.45/h | 93.97/h | Same as cabin-only, 0 false actions |

The Smart Speaker @0.50 replay was not completed (interrupted, no partial log);
the @0.30 result already reproduces the cabin-only safety level, so this was
left as an optional check. The new checkpoint improves positive-side pass
rates at every threshold without sacrificing the safety result, so it
supersedes the cabin-only candidate as the recommended research verifier.

### Cost of the compact gate (RTX 4080 / same config)

| Stage | Without compact | With compact |
|---|---:|---:|
| GPU FP16 wake stage, mean | 48.55 ms | 56.99 ms |
| GPU FP16 post-wake command stage, mean | 62.34 ms | 76.21 ms |
| CPU FP32 wake stage, mean | 522.22 ms | 561.1 ms |
| CPU FP32 post-wake command stage, mean | 641.67 ms | 679.6 ms |
| Complete checkpoint size | 424.49 MB | 425.24 MB |

The verifier itself is 176,579 parameters (~5 ms for two 1-second waveforms on
GPU) and adds ~0.75 MB. The gate is only evaluated on presence-passing windows.
These remain development-GPU measurements, not an automotive target benchmark.

### Readiness audit refresh

`reports/production_readiness_snapshot_compact_cabin_smart_2026-10-12.md`
(machine-readable:
`logs/production_readiness_snapshot_compact_cabin_smart_2026-10-12.json`)
records the new candidate. Status remains `research_engineering_preview`,
`production_ready=false`: same-driver continuity, passenger rejection, study
size, capture-hardware match, frozen split, threshold freeze, and reproducible
release state remain blocked. A capture preflight on 2026-10-12 still finds
only a single-channel interface, so the pilot cannot be recorded on this host.

## Positive-proxy caveat

The retained 8-scene same-speaker proxy
(`--enrollment-source same_speaker_other`) uses a `Marvin` wake clip whose
speaker is not verified to match the enrollment/command speaker. A functioning
identity gate is therefore expected to reject those wakes; the proxy cannot
measure this gate's recall.

| Configuration | Wake recall | Intent overall | False actions/hour |
|---|---:|---:|---:|
| SpEx split-gate reference (historical) | 100.0% | 100.0% | 0.0 |
| Compact (GSC) 0.50 | 37.5% | 0.0% | 0.0 |
| Compact (GSC) 0.20 | 37.5% | 0.0% | 0.0 |
| Compact cabin-aware 0.30 | 37.5% | 0.0% | 0.0 |

The result is identical across thresholds and checkpoints: five of eight scenes
score the wake pair below 0.20 and are rejected regardless of the gate. This is
consistent with a genuine speaker mismatch, not a threshold artifact. The
command-side behavior on this proxy is confounded and is not interpreted.
Speaker-consistent driver recall remains unmeasured and requires the consented
pilot.

Logs:

- `logs/continuous_full_pipeline_proxy_compact_cabin_thr050_same_speaker_p10_2026-10-12.json`
- `logs/continuous_full_pipeline_proxy_compact_cabin_thr020_same_speaker_p10_2026-10-12.json`
- `logs/continuous_full_pipeline_proxy_compact_cabin_aware_thr030_same_speaker_p10_2026-10-12.json`

## Interpretation and limits

- This is the first independent identity gate in the project that measurably
  reduces the longest real-cabin negative benchmark, without touching KWS,
  presence, or intent thresholds. The cabin-aware retrain reduces observed
  full-path events from 18 to 0 over the same exposure.
- The zero-event result only bounds the rate: 95% upper bound 0.95/hour over
  3.889 hours. It must not be presented as a reliable field FAR.
- It is still target-absent evidence. There are no real driver wake positives,
  no passenger-wake negatives, and no speaker-disjoint tuning/test sessions.
- The operating threshold (0.30 versus 0.50) must not be frozen from this
  study; it has to be selected on permitted tuning sessions and evaluated on
  held-out sessions.
- The compact verifier is cheap (176,579 parameters, ~5 ms for two 1-second
  waveforms on the RTX 4080) but its input contract is 1-second fixed windows;
  a streaming integration cost audit is still required.
- AISHELL-5 Train/Dev/Eval1 were in the retrain data; Eval2 stayed out. The
  AISHELL sessions are target-absent, so no positive side benefited from
  Eval2 either way.
- The next decisive verification is unchanged: permitted same-driver sessions
  with speaker-disjoint tuning/test splits.

## Artifacts

- `scripts/eval/eval_compact_speaker_verifier_cabin_cache.py` (new diagnostic)
- `scripts/eval/eval_aishell5_continuous_kws.py` (opt-in compact backend)
- `scripts/data/build_cabin_compact_verifier_cache.py` (wave + WavLM teacher cache builder)
- `scripts/train/train_compact_speaker_verifier_distill.py` (reused trainer)
- `configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_hardneg_aishell5_compact_cabin_safety_candidate_2026-10-12.yaml`
- `configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_hardneg_aishell5_compact_cabin_thr020_candidate_2026-10-12.yaml`
- `configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_hardneg_aishell5_compact_cabin_aware_thr030_candidate_2026-10-12.yaml`
- `checkpoints/compact_speaker_verifier_gsc_ft1/best.pt` (unchanged)
- `checkpoints/compact_speaker_verifier_cabin_aware_ft1/best.pt` (new, 176,579 parameters)
- `data/cache/compact_speaker_verifier_cabin_aware_2026-10-12/`
- `data/cache/wavlm_speaker_verifier_cabin_aware_2026-10-12/`
- `logs/compact_speaker_verifier_cabin_cache_diagnostic_2026-10-12.json`
- `logs/compact_speaker_verifier_cabin_aware_cache_diagnostic_2026-10-12.json`
- `checkpoints/compact_speaker_verifier_cabin_smart_ft1/best.pt` (iteration 2)
- `logs/compact_speaker_verifier_cabin_smart_ft1_2026-10-12.json`
- `logs/compact_speaker_verifier_cabin_smart_smartcache_diagnostic_2026-10-12.json`
- `logs/aishell5_target_absent_compact_cabin_smart_thr030_all18_2026-10-12.json`
- `logs/smart_speaker_command_negative_all_compact_cabin_smart_thr030_2026-10-12.json`
- `logs/full_pipeline_latency_compact_cabin_smart_gpu_2026-10-12.json`
- `logs/full_pipeline_latency_compact_cabin_smart_cpu_2026-10-12.json`
- `reports/production_readiness_snapshot_compact_cabin_smart_2026-10-12.md`
- `configs/eval_end_to_end_cavemove_target_kws_smart_speaker_compact_cabin_smart_thr030_2026-10-12.yaml`

## Reproduction

```powershell
python scripts/eval/eval_compact_speaker_verifier_cabin_cache.py `
  --output logs/compact_speaker_verifier_cabin_cache_diagnostic_2026-10-12.json

python scripts/eval/eval_aishell5_continuous_kws.py `
  --split Eval2 --session-ids 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 `
  --max-seconds 14000 `
  --stage1-config configs/stage1_spex_dual_domain_rms010_ft1.yaml `
  --stage1-checkpoint checkpoints/stage1_spex_dual_domain_rms010_ft1/best.pt `
  --kws-config configs/stage3_kws_gsc_marvin_tse_ft1.yaml `
  --kws-checkpoint checkpoints/stage3_kws_gsc_marvin_tse_ft2/best.pt `
  --target-kws-config configs/stage3_target_kws_gsc_marvin.yaml `
  --target-kws-checkpoint checkpoints/stage3_target_kws_gsc_marvin_hardneg_aishell5_ft1/best.pt `
  --presence-type frame `
  --presence-checkpoint checkpoints/stage2_frame_presence_dual_domain_nonwake_ft1/best.pt `
  --presence-threshold 0.15 --kws-threshold 0.50 `
  --speaker-gate-threshold 0.50 --speaker-gate-backend compact `
  --compact-speaker-verifier-checkpoint checkpoints/compact_speaker_verifier_gsc_ft1/best.pt `
  --smooth-window 2 --required-hits 2 --refractory-calls 2 `
  --batch-size 32 `
  --output logs/aishell5_target_absent_compact_verifier_thr050_all18_2026-10-12.json
```
