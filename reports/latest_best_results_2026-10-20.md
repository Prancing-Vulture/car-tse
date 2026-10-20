# Latest best-results report — in-car target-speaker voice assistant

**Date:** 2026-10-20  
**Repository:** `Prancing-Vulture/car-tse`  
**Branch:** `multi-speaker-eval`  
**Source revision for the figure pack:** `610f261`  
**Evidence status:** research-engineering preview; not production-ready

## 1. Executive summary

The strongest current stack is the **FT6 downstream system**: the frozen
four-channel SpEx+-style TSE front end, frame-presence gate, general wake-word
classifier, target-conditioned wake classifier, compact speaker verifier, and
FT6 runtime-jitter intent model. The current recommended runtime uses the
pre-wake-buffered three-call command buffer and a soft command speaker gate.

The best corrected synthetic multi-speaker results are:

| Condition | Scenes | Fully correct | Target wake recall | False-action events | Strict false wakes/hour |
|---|---:|---:|---:|---:|---:|
| Clean target-only | 128 | **94.53%** (121/128) | 100.00% | 0 | 16.07 |
| Sequential passenger, no overlap | 1,000 | **96.20%** (962/1,000) | 99.70% | 3 | 49.37 |
| Simultaneous passenger overlap | 1,000 | **91.70%** (917/1,000) | 100.00% | 0* | 61.71 |

The component picture is stronger than the full-pipeline safety picture:

- Clean FSC intent classification: **99.31% top-1**, **99.89% top-5**,
  **99.24% macro-F1**.
- TSE SI-SDR improvement: **+3.858 dB** on AISHELL-5 and **+0.114 dB** on
  CAVEMOVE in the current dual-domain regression audit.
- General wake-word classifier: **94.21% F1** on its component test.
- Compact speaker verifier: **0.9706 AUC**, with **79.69% TPR** at **0.534%
  FPR** at the stored operating point.
- Current stored target-conditioned KWS operating point: **43.14% F1**. This
  is the main target-selectivity warning; an older narrow held-out candidate
  reached 97.40% F1 but still had approximately 28.8 false accepts/hour and
  is not the current continuous-safety result.

The main remaining bottleneck is therefore not clean intent classification or
basic TSE training. It is continuous authorization: target-speaker gating,
false-wake suppression, command endpointing, and the lack of a same-driver
real-cabin test. The system must not be described as production-ready or
end-to-end SOTA from these proxy results.

## 2. Current best architecture and checkpoints

Runtime flow:

```text
4-channel cabin audio + short enrollment
        -> Stage 1 target-speaker extraction
        -> frame/chunk presence gate
        -> general "Marvin" KWS + smoothing
        -> target-conditioned KWS + compact speaker verifier
        -> pre-wake buffer / three-call command window
        -> wav2vec2-base + 31-class FSC intent head
        -> guarded vehicle action
```

Recommended soft-gate configuration:

```text
configs/eval_end_to_end_fscpassenger_identity_runtimealignedintent_ft6_jitter_prewakebuffer_gate015_rescue085_postcommand08_2026-10-17.yaml
```

Important runtime settings include a 0.5-second hop, KWS threshold 0.45,
two-hit KWS smoothing, command speaker-gate threshold 0.15 with 0.85 rescue
confidence, a three-call command buffer, 0.80-second endpoint margin, and
post-command refractory behavior. The current synthetic evaluation has
`cabin_noise: false`; it is not a measured vehicle recording.

| Component | Current checkpoint | Role | SHA-256 prefix |
|---|---|---|---|
| Stage 1 TSE | `stage1_spex_fsc_passenger_cochannel_ft1_2026-10-16/best.pt` | Four-channel target-speaker extraction | `13ec7c58...edda3ac` |
| Stage 2 presence | `stage2_frame_presence_dual_domain_nonwake_ft1/best.pt` | Frame/chunk speech-presence gate | `d56b6c9d...e4331550` |
| General KWS | `stage3_kws_clean_control_large_ft1/best.pt` | General “Marvin” detector | `c6593722...7287f1c3f` |
| Target KWS | `stage3_target_kws_gsc_marvin_hardneg_aishell5_smart_speaker_ft1/best.pt` | Target-conditioned wake check | `7331991e...4d125af1` |
| Speaker verifier | `compact_speaker_verifier_cabin_smart_ft1/best.pt` | Enrollment/command speaker match | `1767d672...50466ecea` |
| Intent head | `stage4_intent_runtime_jitter_clean_overlap_ft6_fromft4_2026-10-17/best.pt` | 31-class FSC intent classifier | `de6689d3...d599813c` |
| Intent encoder | `stage4_intent_runtime_jitter_clean_overlap_ft6_fromft4_2026-10-17/encoder_best.pt` | wav2vec2 encoder | `a1b8fa00...e9245f643` |

Full hashes, configs, and source logs are preserved in
`reports/model_architecture_metrics_latency_sota_2026-10-17.md` and
`reports/jetson_inference_bundle_2026-10-19.md`.

## 3. Full-pipeline results

### 3.1 Clean target-only proxy

Source: `logs/clean_e2e_v2_knnvc_matched_postaction16_128_2026-10-16.json`

| Metric | Result |
|---|---:|
| Scenes | 128 |
| Fully correct scenes | **121/128 = 94.53%** |
| Target wake recall | 100% |
| False-action events | 0 |
| Passenger-only false actions | 0 |
| Executed non-target actions | 0 |
| Strict fully correct action scenes | 120/128 |
| Strict false wake rate | 16.071/hour |
| Strict false command rate | 2.009/hour |
| Mean wake latency | 1.699 s |
| P95 wake latency | 2.000 s |

This is a clean upper-bound-like diagnostic, not a real-driver result.

### 3.2 Corrected sequential passenger proxy

Source: `logs/multispeaker_passenger_ft6_p0_dynamic_full_1000_2026-10-17.json`

The test contains dynamically selected, speaker-disjoint passenger speech and
full-length commands, but no vehicle/road noise.

| Metric | Result |
|---|---:|
| Scenes | 1,000 |
| Fully correct scenes | **962/1,000 = 96.20%** |
| Target wake recall | **997/1,000 = 99.70%** |
| Intent accuracy conditioned on target wake | 96.39% |
| False-action events | 3 = 0.771/hour |
| Passenger-only false actions | 1 = 0.257/hour |
| Executed non-target actions | 3 = 0.771/hour |
| Wrong-intent action events | 28 |
| Strict fully correct action scenes | 914/1,000 = 91.40% |
| Strict false wake rate | **49.371/hour** |
| Strict false command rate | **14.143/hour** |
| Mean target-wake latency | 2.125 s |
| P95 target-wake latency | 3.500 s |

### 3.3 Corrected simultaneous-overlap proxy

Source: `logs/multispeaker_overlap_ft6_p0_dynamic_full_1000_2026-10-17.json`

The overlap condition sums target and passenger speech into a co-channel
mixture. It has no spatial seat cue and contains no cabin/road noise.

| Metric | Result |
|---|---:|
| Scenes | 1,000 |
| Fully correct scenes | **917/1,000 = 91.70%** |
| Target wake recall | **1,000/1,000 = 100%** |
| Intent accuracy conditioned on target wake | 91.70% |
| False-action events | 0 |
| Passenger-only false actions | 0* |
| Executed non-target actions | 0 |
| Wrong-intent action events | 60 |
| Strict fully correct action scenes | 890/1,000 = 89.00% |
| Strict false wake rate | **61.714/hour** |
| Strict false command rate | **7.714/hour** |
| Mean target-wake latency | 2.147 s |
| P95 target-wake latency | 3.500 s |

\* Passenger-only attribution is ambiguous because 999/1,000 scenes contain
passenger-wake overlap. The zero value is not evidence of perfect passenger
rejection; wrong-intent actions and strict source-attributed wake rates are the
meaningful overlap diagnostics.

## 4. Component results

| Component | Best/current result | Interpretation |
|---|---|---|
| TSE, AISHELL-5 | Baseline 2.385 dB -> candidate 6.243 dB; **+3.858 dB SI-SDRi** | Useful domain improvement |
| TSE, CAVEMOVE | 10.478 dB -> 10.592 dB; **+0.114 dB SI-SDRi** | Preserves the existing cabin domain |
| TSE overall | 6.432 dB -> 8.417 dB; **+1.985 dB SI-SDRi** | Regression audit, not command accuracy |
| Presence, frame | 98.84% accuracy; 90.44% F1 | Frame operating point has a precision/recall trade-off |
| Presence, chunk | 99.79% accuracy; 99.59% F1 | Strong isolated component result |
| General KWS | 97.74% accuracy; 94.21% F1 | Component test only; not long-stream FAR |
| Target KWS | 82.20% accuracy; 43.14% F1 at stored point | Current target-selectivity weakness |
| Compact verifier | 0.9706 AUC; 79.69% TPR; 0.534% FPR | Strong ranking, but operating point needs real-driver tuning |
| Intent, clean FSC | **99.31% top-1; 99.89% top-5; 99.24% macro-F1** over 3,793 | Strongest individual component |
| Intent, TSE-domain | 90.49% top-1; 98.01% top-5; 90.13% macro-F1 over 2,314 | Extraction-domain degradation |
| Intent, FT6 runtime validation | 89.09% accuracy; 96.81% top-5; 88.90% macro-F1 over 8,192 | Candidate adaptation checkpoint, not clean-test replacement |

These metrics use different splits and operating points; they must not be
multiplied together or presented as one joint accuracy.

## 5. External reference comparisons

These are component-level comparisons, not full-pipeline replications.

| Reference track | Our result | Reported reference | Difference |
|---|---:|---:|---:|
| Hussein-style Speech Commands track | **96.75% top-1**, 96.37% macro-F1, 99.25% top-5 | 93% accuracy | +3.75 points |
| Paranjape-style clean FSC track | **99.31% top-1**, 99.24% macro-F1, 99.89% top-5 | 93.67% intent accuracy | +5.64 points |
| SLURP WavLM direct-audio action baseline, official all-real test | **81.11% accuracy**, 74.20% macro-F1, 95.22% top-5 | Not an exact paper replication | Separate common benchmark |
| SLURP WavLM, distant microphones | 79.40% accuracy; 72.19% macro-F1 | — | Acoustic-distance diagnostic |
| SLURP WavLM, close-talk microphones | 83.01% accuracy; 76.24% macro-F1 | — | Acoustic-distance diagnostic |

Hussein and Paranjape use different datasets, class inventories, splits, and
pipeline definitions. The results support only the limited claim that the
clean command/intent components are strong; they do not show that the complete
TSE -> speaker authorization -> wake -> intent system beats either paper under
the same interaction conditions.

## 6. Model size, latency, and compute

| Quantity | Current value |
|---|---:|
| Total parameters | **106.76M** |
| Raw PyTorch checkpoint bytes | **427.48 MB** / 407.68 MiB |
| wav2vec2 + intent parameters | 94.78M |
| RTX 4080 FP16 Stage 1 mean / P95 | 63.47 / 100.39 ms |
| RTX 4080 FP16 intent mean / P95 | 25.70 / 32.00 ms |
| RTX 4080 FP16 composite wake mean / P95 | 74.85 / 108.12 ms |
| RTX 4080 FP16 composite command mean / P95 | **103.31 / 123.16 ms** |
| Older CPU full-command reference | 679.57 ms mean / 715.995 ms P95 |
| Stage 1 estimated compute | 103.945 GFLOPs per 3-second call |
| Intent estimated compute | 25.431 GFLOPs per 3-second call |

The latency numbers are model-forward measurements on an RTX 4080. They exclude
audio I/O, buffering, scheduling, endpointing, action execution, and device
transfers. Jetson latency, thermal behavior, audio dropouts, and TensorRT
compatibility remain unmeasured. INT8 figures are not production claims.

## 7. Training plots and paper figures

The complete tracked archive is in `reports/figures/`. The curated paper pack
contains PNG/PDF versions of the current/best system, all stages, training
curves, ablations, resource plots, and external benchmark figures:

- `reports/paper_figures_latest_2026-10-20/README.md`
- `reports/figures/ft6_current_system_e2e_2026-10-20.{png,pdf}`
- `reports/figures/ft6_corrected_multispeaker_1000_2026-10-20.{png,pdf}`
- `reports/figures/ft6_resource_summary_2026-10-20.{png,pdf}`
- `reports/figures/stage1_spex_fsc_passenger_cochannel_ft1_training_2026-10-16.{png,pdf}`
- `reports/figures/stage2_presence_training_comparison_2026-10-20.{png,pdf}`
- `reports/figures/stage3_target_kws_negative_calibrated_comparison_2026-10-16.{png,pdf}`
- `reports/figures/stage4_intent_runtime_jitter_clean_overlap_ft6_fromft4_2026-10-17.{png,pdf}`
- `reports/figures/compact_speaker_verifier_cabin_smart_nodistill_h256_training_2026-10-16.{png,pdf}`

The current Stage-2 dual-domain FT1 log contains only a final summary, not an
epoch history. Its plotted comparison therefore uses older retained Stage-2
runs with complete histories; this limitation is documented in the figure-pack
README.

## 8. Evidence boundary and production status

The system is **not production-ready**. The main blockers are:

1. No continuous same-driver recording currently contains enrollment, “Marvin,”
   command, passenger speech, no-command intervals, and vehicle conditions.
2. The corrected synthetic proxy still reports tens of strict false wakes per
   hour, far above a plausible production target.
3. The overlap test has ambiguous source attribution and no spatial seat cue.
4. Thresholds are not frozen on a real-driver tuning split.
5. Command endpointing and the wake-tail/command-window interaction remain
   unresolved.
6. Jetson/automotive-device latency and long-duration audio stability have not
   been measured.
7. The current model is dominated by wav2vec2 and is too large for an assumed
   small edge footprint without distillation/streaming/quantization work.

The results are suitable for a research-engineering progress report and for a
paper describing the architecture, component evaluations, synthetic proxy
study, ablations, and limitations. They are not sufficient for a claim of
real-world driver reliability or production deployment.

## 9. Next highest-value work

1. Record a consented real-driver pilot with the same driver providing
   enrollment, wake word, and command; include passenger wakes and negatives.
2. Split sessions by driver/vehicle into tuning and held-out test sets.
3. Tune target-KWS/verifier/presence thresholds only on tuning data, then freeze
   them and report wake recall, intent success, passenger rejection, false
   actions/hour, and latency by condition.
4. Replace the fixed command window with reliable target-presence endpointing,
   retaining the pre-wake buffer.
5. Re-run one-component-at-a-time ablations on the held-out real-driver set.
6. Transfer the exact FT6 bundle to the Jetson and measure end-to-end latency,
   memory, thermal stability, and audio-drop rate.

No new model training was started for this report; the update consolidates the
latest completed FT6 training/evaluation artifacts and adds the current-best
figures.
