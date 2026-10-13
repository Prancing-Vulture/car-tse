# In-car voice assistant — accuracy and latency report

Date: 2026-10-12  
System status: `research_engineering_preview` (`production_ready=false`)  
Runtime identity: `configs/eval_end_to_end_cavemove_target_kws_smart_speaker_compact_cabin_smart_thr030_2026-10-12.yaml`

Promoted checkpoint roles:

| Stage | Checkpoint | Parameters |
|---|---|---:|
| Stage 1 extraction (dual-domain ft1) | `checkpoints/stage1_spex_dual_domain_rms010_ft1/best.pt` | 10,727,881 |
| Stage 2 frame presence (dual-domain non-wake ft1) | `checkpoints/stage2_frame_presence_dual_domain_nonwake_ft1/best.pt` | 33,729 |
| Stage 3 general KWS (GSC-Marvin TSE ft2) | `checkpoints/stage3_kws_gsc_marvin_tse_ft2/best.pt` | 93,378 |
| Stage 3 target KWS (hard-neg AISHELL-5 ft1) | `checkpoints/stage3_target_kws_gsc_marvin_hardneg_aishell5_ft1/best.pt` | 389,316 |
| Stage 3b compact speaker verifier (cabin+Smart) | `checkpoints/compact_speaker_verifier_cabin_smart_ft1/best.pt` | 176,579 |
| Stage 4 intent (wav2vec2-base-960h ft8 TSE-domain) | `checkpoints/stage4_intent_wav2vec_ft8_resume_tse_domain/{best.pt,encoder_best.pt}` | 94,782,879 |

Total: **106,203,762 parameters / 425.24 MB** of checkpoints. The intent branch is
89% of all parameters.

## 1. Accuracy by stage

### Stage 1 — target-speaker extraction (SpEx+ dual-domain ft1)

Matched dual-domain regression, 288 examples (144 AISHELL-5 + 144 CAVEMOVE
validation), `logs/stage1_dual_domain_regression_full_2026-10-10.json`:

| Domain | Baseline (transfer mc) | Promoted ft1 | Change |
|---|---:|---:|---:|
| AISHELL-5 SI-SDRi | 2.39 dB | **6.24 dB** | +3.86 dB |
| CAVEMOVE SI-SDRi | 10.48 dB | **10.59 dB** | +0.11 dB |
| Overall | 6.43 dB | **8.42 dB** | +1.99 dB |

The original CAVEMOVE held-out benchmark reported +10.98 dB on a different
400-example protocol and is not directly comparable. `ft2` is extraction-only
(kept diagnostic; not the downstream handoff).

### Stage 2 — frame presence / activity

Validation on the dual-domain non-wake cache (482 validation chunks),
`logs/stage2_frame_presence_dual_domain_nonwake_ft1/metrics_last.json`:

| Operating point | Accuracy | Precision | Recall | F1 | FAR | FRR |
|---|---:|---:|---:|---:|---:|---:|
| Frame @0.5 | 98.84% | 86.71% | 94.51% | 90.44% | 0.90% | 5.49% |
| Chunk @0.5 | 99.79% | 99.18% | 100.0% | 99.59% | 0.28% | 0.00% |

In the cascade the presence threshold is 0.15 (wake) / 0.10 (command), chosen
for rejection margin, not for standalone F1.

### Stage 3 — keyword spotting

| Head | Split / threshold | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| General KWS (TSE ft2) | val, 0.50 | 69.2% | 72.3% | 62.3% | 66.9% |
| Target KWS (hard-neg AISHELL-5 ft1) | val 1328 rows, 0.50 | 80.7% | 30.7% | 76.9% | 43.9% |

Both heads have weak standalone scores because their validation caches are
negative-heavy and cross-domain; the operative behavior is the continuous
cascade in Section 2. The negative-only adaptation was selected precisely
because its higher-F1 positive-adapted sibling did not hold its proxy margin.

### Stage 3b — compact speaker verifier (cabin+Smart)

Expanded cabin-aware cache (3,176/1,629/1,625 train/Dev/Eval1 pairs; 604
positives), `logs/compact_speaker_verifier_cabin_smart_smartcache_diagnostic_2026-10-12.json`:

| Measure | Value |
|---|---:|
| Dev AUC | 0.9380 |
| Eval1 AUC | 0.9706 |
| TPR at validation-selected ~1% FPR | **79.7%** |
| FPR at that point | 0.53% |

Pass rates on the Eval1 cache:

| Threshold | Synthetic target present | AISHELL-5 cabin | Smart Speaker | Non-wake |
|---:|---:|---:|---:|---:|
| 0.30 | 93.0% | 2.0% | 0.0% | 5.7% |
| 0.50 | 87.5% | 0.9% | 0.0% | 0.3% |

### Stage 4 — intent classification (wav2vec 2.0)

| Evaluation | Examples | Top-1 | Top-5 | Macro-F1 |
|---|---:|---:|---:|---:|
| Clean official FSC test | 3,793 | **99.31%** | 99.89% | 99.24% |
| TSE-domain validation | 2,314 | **90.49%** | 98.01% | 90.13% |

Confidence calibration on the runtime waveform cache (2,314 examples): raw
accuracy 90.49%, ECE 0.023, Brier 0.135, mean confidence 0.913. At confidence
threshold 0.53 the calibrated run retained 94.12% coverage at 93.66%
selective accuracy (validation-domain evidence, not a frozen threshold).

## 2. Full-system accuracy

All rows are synthetic/cross-corpus diagnostics unless marked "real cabin".

### 2.1 Long target-absent real-cabin negative benchmark

AISHELL-5 Eval2, all 18 sessions, 3.889 h, 80.999% speech occupancy, fixed
LibriSpeech enrollment speaker absent; frame presence 0.15, target KWS 0.50,
2-of-2 smoothing, two-call refractory:

| Speaker gate | Full-path events | Events/hour | 95% Poisson CI | KWS-only/hour |
|---|---:|---:|---:|---:|
| SpEx cosine 0.24 (previous reference) | 18 | 4.63 | 2.74 – 7.00 | 96.69 |
| Compact (GSC-trained) 0.50 | 7 | 1.80 | 0.72 – 3.36 | 96.69 |
| Compact (GSC-trained) 0.20 | 11 | 2.83 | 1.41 – 4.73 | 96.69 |
| Compact cabin-aware 0.30 / 0.50 | **0** | **0.00** | 0 – 0.95 (one-sided) | 96.69 |
| Compact cabin+Smart 0.30 | **0** | **0.00** | 0 – 0.95 (one-sided) | 96.69 |

### 2.2 Clean wake-like negative benchmark

Rochester Smart Speaker Command corpus, all 2,110 files, 3.193 h, 0.25 s hop,
Smart-Speaker-adapted target KWS:

| Speaker gate | Full-path false wakes | Per hour | KWS-only/hour | Action sample |
|---|---:|---:|---:|---:|
| SpEx 0.24 (reference) | 86 | 26.94 | 93.97 | 0 false actions |
| Compact cabin-aware 0.30 | **11** | **3.45** | 93.97 | 0 false actions |
| Compact cabin+Smart 0.30 | **11** | **3.45** | 93.97 | 0 false actions |

### 2.3 Positive continuous proxies (cross-corpus)

| Benchmark | Exposure | Wake recall | Intent overall | Intent given wake | False wakes/h | False actions/h | Latency mean/p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Expanded same-speaker proxy (splitgate) | 448 s | 78.1% (25/32) | 68.8% (22/32) | 88.0% | 0 | 0 | 2.1 / 2.5 s |
| Selected 8-scene safety proxy (splitgate) | 112 s | 100% | 100% | 100% | 0 | 0 | 2.5 s |
| ReMASC vehicle-stress fast stream (margin −0.67) | 448 s | 96.9% | 81.3% | — | 64.3 diagnostic | 0 | 0.968 / 1.625 s |

### 2.4 15-minute continuous streams (new)

One stateful pass, 0.25 s hop, two-call command delay, 0–15 dB cabin SNR,
noise bed + silence + sparse random events:

| Stream | Target events | Wake recall | Latency | False wakes | False actions | Intent |
|---|---:|---:|---:|---:|---:|---:|
| A: speaker-consistent wake/enrollment | 12 | **12/12 (100%)** | 1.49 s | 2 (8.0/h) | 0 | n/a (command speaker differs) |
| B: command-speaker enrollment, cross-corpus wake | 12 | 0/12 | — | 1 (4.0/h) | 0 | n/a |

Stream A is the strongest positive-stream evidence available with public
corpora; one passenger "Marvin" slipped through the gate at 2.9 dB SNR. Stream
B demonstrates that a cross-corpus enrollment suppresses the different
speaker's wake under arbitrary alignment.

## 3. Latency by component

Method: `scripts/eval/eval_component_latency.py`, 30 repeats / 5 warm-up on GPU
(RTX 4080) and 15 repeats / 3 warm-up on CPU, random 3 s four-channel input,
2 s intent input. Model-forward times only; microphone I/O, resampling,
buffering, and action execution excluded.

| Component (params) | GPU FP16 mean / p50 / p95 | CPU FP32 mean / p50 / p95 |
|---|---:|---:|
| Stage 1 extraction, 3 s × 4 ch (10.73 M) | 66.46 / 62.78 / 96.51 ms | 423.04 / 420.79 / 439.47 ms |
| Enrollment speaker embedding, 3 s (10.73 M) | 4.84 / 4.85 / 4.97 ms | 23.16 / 23.96 / 24.90 ms |
| Extracted speaker embedding, 3 s (10.73 M) | 4.90 / 4.87 / 5.05 ms | 23.40 / 23.58 / 25.00 ms |
| Frame presence, 3 s (0.034 M) | 1.57 / 1.57 / 1.60 ms | 1.13 / 1.13 / 1.15 ms |
| Target KWS, 1 s (0.39 M) | 2.20 / 2.20 / 2.28 ms | 2.11 / 2.02 / 2.29 ms |
| General KWS, 1 s (0.09 M) | 1.47 / 1.45 / 1.56 ms | 1.97 / 1.95 / 2.10 ms |
| Compact speaker verifier, 2 × 1 s (0.18 M) | 5.04 / 5.02 / 5.17 ms | 3.04 / 2.96 / 3.39 ms |
| Intent wav2vec2, 2 s (94.78 M) | 17.41 / 15.66 / 21.50 ms | 94.93 / 94.73 / 99.64 ms |

Composite stages from `eval_full_pipeline_latency.py` (same methodology as the
historical benchmark). The compact-gate column includes the verifier:

| Measurement | Without compact gate | With compact cabin+Smart gate |
|---|---:|---:|
| GPU FP16 wake stage, mean / p50 / p95 | 48.55 / 48.21 / 51.56 ms | 56.99 / 56.94 / 60.84 ms |
| GPU FP16 post-wake command stage | 62.34 / 62.57 / 64.44 ms | 76.21 / 76.11 / 80.27 ms |
| CPU FP32 wake stage | 522.22 ms | 561.15 / 562.82 / 576.69 ms |
| CPU FP32 post-wake command stage | 641.67 ms | 679.57 / 677.89 / 715.99 ms |
| FP32 CUDA wake / command (older run) | 100.81 / 115.22 ms | — |
| Peak CUDA allocation (wake benchmark) | 499 MB (fp32), 486 MB (fp16 compact) | — |

One-time enrollment: first call measured 604 ms (GPU fp16, includes warm-up) /
466 ms (CPU); steady-state repeated calls measure ~5 ms GPU / ~23 ms CPU.

### What the costs mean per hop

| Hop | GPU FP16 wake stage | CPU FP32 wake stage |
|---:|---:|---:|
| 0.25 s | 23% of budget (57/250) | **180–225%** (over budget) |
| 0.50 s | 11% | 90–112% (borderline) |
| 1.00 s | 5.7% | 45–56% |

Streaming-runner measurements (`scripts/eval/streaming_runner_simulation.py`,
separate capture thread, real-time cadence):

| Run | Wall | RTF | Skipped | Peak lag |
|---|---:|---:|---:|---:|
| 60 s GPU, 0.25 s hop | 60.0 s | 0.28 | 0.0 s | 0.2 s |
| 30 s CPU, 0.25 s hop | 34.6 s | 1.73 | 9.0 s | 4.85 s |

CPU cannot sustain the 0.25 s hop; the runner skips and reports the skipped
seconds instead of growing latency or silently losing audio. Strictly bounded
latency additionally needs chunked inference because one call may process
several buffered windows.

## 4. Footprint

| Item | Value |
|---|---:|
| Total parameters | 106,203,762 |
| Intent branch share | 94.78 M (89%) |
| Checkpoint bytes | 425.24 MB |
| Peak CUDA workspace (fp16 wake benchmark) | 486 MB |
| Older fp32 CUDA run peak | up to 9.9 GB (cuDNN workspace; not an edge-safe default) |

## 5. Evidence boundary (read before quoting numbers)

- All positive results are synthetic/cross-corpus: enrollment, `Marvin`, and
  command come from different public corpora or different utterances; the wake
  speaker is never verified to match the driver.
- AISHELL-5 and Smart Speaker runs are negative-only: they bound false
  accepts, not recall. The `0.00/h` results are rate bounds (95% upper bound
  0.95/h over 3.889 h), not proof of zero misses.
- Latency numbers are model-forward measurements on a development host; they
  exclude microphone I/O, resampling, scheduling, and the action layer, and are
  not an automotive CPU/DSP benchmark.
- No real-driver session exists; the pilot (4-channel capture, speaker-disjoint
  tuning/test, passenger wake, no-command intervals) remains the decisive
  missing evidence. The current host exposes only a single-channel input.

## 6. Source artifacts

- Stage 1: `logs/stage1_dual_domain_regression_full_2026-10-10.json`
- Stage 2: `logs/stage2_frame_presence_dual_domain_nonwake_ft1/metrics_last.json`
- Stage 3: `logs/stage3_kws_gsc_marvin_tse_ft2/metrics_last.json`,
  `logs/stage3_target_kws_gsc_marvin_hardneg_aishell5_ft1/metrics_last.json`
- Stage 3b: `logs/compact_speaker_verifier_cabin_smart_smartcache_diagnostic_2026-10-12.json`
- Stage 4: `logs/stage4_intent_wav2vec_ft8_resume_tse_domain/metrics_official_test.json`,
  `.../metrics_tse_domain.json`,
  `logs/intent_confidence_calibration_ft8_tse_domain_waveform_val_2026-10-11.json`
- Full pipeline negative/positive pools and streams: see
  `reports/compact_speaker_verifier_cabin_safety_2026-10-12.md` and
  `reports/research_journey.md`
- Latency: `logs/component_latency_cuda_fp16_2026-10-12.json`,
  `logs/component_latency_cpu_fp32_2026-10-12.json`,
  `logs/full_pipeline_latency_compact_cabin_smart_{gpu,cpu}_2026-10-12.json`,
  `logs/full_pipeline_latency_{gpu,cpu}_2026-10-11.json`,
  `logs/streaming_runner_simulation_thread_{gpu,cpu}_30s_2026-10-12.json`
