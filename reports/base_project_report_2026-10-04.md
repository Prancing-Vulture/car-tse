# Reliable In-Car Voice Assistance with Target Speaker Extraction

## Base-project status report

Date: 2026-10-04  
Workspace: `D:/Prawin_New/tse/car-tse-main`

## Executive conclusion

This project is a strong research prototype for an in-car voice-assistant cascade. The individual components are useful, especially the target-speaker extractor and the clean Fluent Speech Commands intent model. The complete driver-wake-to-command-to-action system is not production-ready yet.

The central issue is now well isolated:

> The main remaining failure is target-speaker selectivity during continuous speech. Stationary cabin noise is largely controlled, but passenger or non-wake speech can still leak through the extractor and trigger downstream command actions.

The most defensible current description is:

**“A staged, GPU-capable research prototype with promising synthetic cabin TSE performance, excellent clean intent classification, and incomplete real-cabin end-to-end validation.”**

It should not yet be described as production-ready, SOTA, or validated at 80% end-to-end wake-to-action accuracy.

## 1. Requirements from the in-car brief

The supplied brief defines four practical requirements:

| Requirement | What it means for this project | Current status |
|---|---|---|
| Ultra-low latency | Streaming/chunked operation, ideally under 100 ms and eventually much lower for full duplex | GPU model-forward sum is about 70 ms, but this is not a complete streaming measurement; CPU is about 275 ms; Stage 1 is non-causal and uses a 3-second context |
| Edge footprint | Compact speaker encoder and deployable DSP/edge footprint | Not met: current complete checkpoint footprint is about 424 MB, dominated by wav2vec |
| Short enrollment | A stable target-speaker profile from no more than roughly 3 seconds | The runtime accepts a short enrollment proxy, but robustness with one short clip in real cabin conditions is not established |
| Passenger/noise rejection | Suppress passenger speech, road/HVAC/wind noise, echo and playback | Cabin-only noise is controlled in current tests; passenger/non-wake speech remains the main failure |

The brief also mentions zone-based spatial TSE, evolving enrollment, target-speaker VAD, and compact commercial engines. The current project contains partial analogues of these ideas, but not full implementations:

- It has a learned four-channel cabin frontend and a target-conditioned SpEx+-style extractor, but not a validated seat-zone beamformer or driver/passenger spatial benchmark.
- It has a fixed enrollment representation and optional speaker gating, but not reliability-filtered EvoTSE-style enrollment updates.
- It has a framewise target-presence head and temporal KWS, but not a trained A-TSVAD or a real continuous target-speaker VAD benchmark.
- It has a playback-reference input in the frontend, but real echo cancellation performance has not been measured.

## 2. Implemented architecture

The current production-shaped path is:

```text
4-channel cabin input + optional playback reference
        ↓
learned cabin frontend: echo-reference branch + spatial channel collapse
        ↓
16-kHz SpEx+-style target-speaker extractor
        ↓
framewise target-presence gate
        ↓
TSE-domain wake-word detector: “marvin”
        ↓
streaming state machine: wake arms command capture; wake frame is not decoded as intent
        ↓
fixed 2-second extracted command window
        ↓
wav2vec 2.0 intent classifier, 31 FSC intents
        ↓
vehicle action + optional spoken reply
```

Relevant implementation files:

- Stage 1 and cabin frontend: `src/car_tse/models/stage1_system.py`, `src/car_tse/frontend/cabin_frontend.py`
- Presence: `src/car_tse/models/presence_head.py`
- KWS: `src/car_tse/models/kws_model.py`
- Pipeline state machine: `src/car_tse/pipeline/in_car_pipeline.py`
- Intent: `src/car_tse/models/wav2vec_intent.py`
- Continuous proxy evaluator: `scripts/eval/eval_continuous_full_pipeline_proxy.py`

The state-machine correction is important: a wake frame no longer gets sent directly to the intent head. The next accepted target-speech frame is captured as the command. A timeout prevents the system from remaining armed indefinitely.

The current code also contains an experimental enrollment-conditioned KWS head. It is preserved for research, but it has not improved the continuous passenger proxy enough to become the default path.

## 3. Training data and role of each dataset

| Stage | Training data and role | Current use |
|---|---|---|
| Stage 1 TSE | LibriSpeech/Libri2Mix clean speaker mixtures, with CAVEMOVE multichannel cabin noise | Target extraction pretraining/transfer and cabin frontend adaptation |
| Stage 1 diagnostics | AISHELL-5 real four-channel far-field/near-field speech and cabin noise | Real-cabin transfer diagnostic and negative-stream evaluation; not a complete target-speaker wake corpus |
| Stage 2 presence | CAVEMOVE mixtures, GSC `marvin` target-present examples, AISHELL-5 target-absent/noise examples, and hard negatives | Decide whether the extracted signal contains the enrolled target speech |
| Stage 3 KWS | Google Speech Commands `marvin` positives and non-wake words, Stage 1 TSE caches, CAVEMOVE cabin augmentation, AISHELL-5 speech-rich negatives | Detect the target wake word after extraction |
| Stage 4 intent | Fluent Speech Commands, 31 intent classes | Convert the post-wake command into an intent/action |
| Vehicle noise | CAVEMOVE and Vehicle Interior Sound Dataset material | Cabin/road/HVAC noise augmentation; not speech supervision |
| Custom in-car data | Manifest builder exists, but a complete real driver/passenger continuous corpus is not yet present | Required next milestone |

The project did not redistribute the commercial Nexdata 531-hour corpus. The MagicHub/ASR-CabNois source also has access/licensing constraints. The local system therefore relies on public speech corpora, CAVEMOVE, AISHELL-5, and generated mixtures.

## 4. Current measured results

### Stage 1: target-speaker extraction

Fresh held-out CAVEMOVE evaluation:

| Metric | Input mixture | Extracted output | Improvement |
|---|---:|---:|---:|
| Mean SI-SDR | -2.41 dB | 8.57 dB | **+10.98 dB** |
| Median SI-SDR | -2.37 dB | 10.49 dB | **+12.77 dB** |
| 10th percentile | -8.32 dB | 0.22 dB | **+4.06 dB** |
| 90th percentile | 3.30 dB | 15.07 dB | **+16.88 dB** |

This is a meaningful synthetic cabin result. It does not by itself prove that the output contains only the driver, preserves command intelligibility, or rejects a passenger.

AISHELL-5 diagnostic results show the domain gap:

- Current Stage 1 output: approximately **-31.22 dB mean SI-SDR** on the evaluated target-present windows.
- Far-field input baseline: approximately **-37.31 dB**.
- A bounded frontend adaptation improved the AISHELL diagnostic by only about **0.65 dB** but regressed held-out CAVEMOVE validation.

The transfer model is therefore promising on synthetic CAVEMOVE, but real-cabin target-speaker quality is not yet solved.

### Stage 2: target-presence gate

The original scalar presence calibration is highly threshold-sensitive:

- Default threshold `0.50`: 91.25% accuracy, **96.25% FAR**, 0% FRR on the imbalanced validation split.
- Calibrated threshold near `0.90`: approximately **17.5% FAR / 8.9% FRR** on that split.
- In the continuous target-positive proxy, the high threshold can suppress target speech entirely; lowering it restores recall but exposes speech-triggered false accepts.

The framewise head is a better runtime design than the original whole-chunk scalar decision, but its operating point still needs a genuinely joint target-present/target-absent calibration set.

### Stage 3: wake-word detection

The wake word is `marvin`.

Representative offline results from the project benchmark:

| KWS candidate | F1 | Precision | Recall | Notes |
|---|---:|---:|---:|---|
| Original cabin KWS | 63.87% | 52.29% | 82.05% | Narrow cabin training distribution |
| TSE-domain KWS | 96.43% on its held-out set | 95.94% | 96.92% | Strongest narrow TSE-domain regression baseline |
| Selected high-precision TSE-domain operating point | **97.40%** | **98.94%** | 95.90% | Approx. 28.8 independent-window FA/hour estimate |
| Corrected post-TSE hybrid | 79.67% | 84.48% | 75.38% | Larger hard-negative test; more realistic stress test |
| AISHELL-augmented KWS | 80.74% | 83.15% | 78.46% | Better coverage but threshold/stream calibration mismatch |

The important distinction is between isolated-window F1 and continuous false accepts/hour. The latter is the deployment metric.

### Stage 4: intent classification

The current intent model is a wav2vec 2.0 encoder with a 31-class head.

| Evaluation condition | Top-1 | Top-5 |
|---|---:|---:|
| Official Fluent Speech Commands validation | **96.60%** | 99.36% |
| Official Fluent Speech Commands test | **99.31%** | **99.89%** |
| Local raw CAVEMOVE validation | about 90.71% | about 97.84% |
| Local TSE-domain validation | about 90.49% | about 98.01% |

The official clean test result is close to the reported 99.7% clean result, but it is not the same architecture or training recipe as the published 99.7% system. It should be presented as a strong clean benchmark, not as a reproduction of that paper result.

### Latency and model size

Measured model-forward timing for a one-second input:

| Device | Stage 1 + KWS + intent |
|---|---:|
| GPU | **about 69.95 ms** |
| CPU | **about 275.34 ms** |

This does not include microphone I/O, resampling, buffering, stream scheduling, overlap/look-ahead, or action execution. The Stage 1 model is non-causal and consumes a 3-second context, so the number is not a complete end-to-end streaming latency claim.

Approximate checkpoint footprint:

| Component | Size |
|---|---:|
| Stage 1 TSE + cabin frontend | 43.1 MB |
| Presence head | 1.0 MB |
| KWS | 0.4 MB |
| wav2vec encoder | 377.6 MB |
| Intent head | 1.6 MB |
| **Total** | **about 423.7 MB** |

The wav2vec encoder is the main obstacle to the stated compact automotive-edge target.

## 5. Full-pipeline evidence and what it means

### Independent AISHELL negative stream

The current full path produces **0 full-pipeline smoothed events/hour** on the evaluated AISHELL-5 negative stream. This is useful evidence that the presence gate can suppress a real speech-rich target-absent stream.

It is not a wake recall result: the stream contains no enrolled-target wake word and no command from the target driver.

### Continuous CAVEMOVE cross-corpus proxy

The proxy contains target wake, a cross-corpus command, passenger wake/non-wake speech, and multichannel CAVEMOVE noise. Its limitation is explicit: the target wake and command are not a natural same-speaker pair.

Representative passenger-only stress case, with passenger speech at -5 dB relative to target-level normalization and cabin noise disabled:

| Configuration | Target wake recall | False actions | Rate |
|---|---:|---:|---:|
| Existing downstream path | 100% | 7 in 56 s | 450/hour |
| Same-domain KWS fine-tune | 100% | 7 in 56 s | 450/hour |
| Enrollment-conditioned KWS | 100% | 6 in 56 s | 385.7/hour |
| Extra non-wake hard-negative KWS fine-tune | 100% | 6 in 56 s | 385.7/hour |
| 2-of-2 temporal confirmation | 100% | 8 in 56 s | 514.3/hour |

These are engineering proxies, not production metrics, but they identify the problem clearly: false actions are caused by speech leakage and target-speaker selectivity, not simply by static cabin noise.

### Recent experiments that were rejected

The following experiments remain in uniquely named directories for reproducibility but are not promoted:

- Same-domain KWS fine-tune: offline validation F1 around 0.69 in the last epoch and no passenger-proxy improvement.
- Enrollment-conditioned KWS: validation F1 around 0.817, but only a small false-action reduction in the proxy.
- Hard-negative-conditioned KWS: validation F1 around 0.547 on its imbalanced merged validation set and no additional proxy improvement.
- 2-of-2 KWS confirmation: increased false actions in the current proxy.
- Stage 1 non-wake hard-negative fine-tune: first validation had SI-SDR improvement **-0.44 dB**, so training was stopped and the checkpoint was not promoted.
- A new presence-head cache run was started but stopped before training after the request to pause; it has no result and is not part of the benchmark.

## 6. What is working well

1. The codebase has a coherent staged architecture matching the intended product flow.
2. The TSE frontend gives approximately +10.98 dB SI-SDR improvement on held-out synthetic cabin mixtures.
3. The clean intent model is strong: 99.31% top-1 on official FSC test and 96.60% on official validation.
4. KWS has a strong narrow TSE-domain baseline, with approximately 97.4% F1 at a high-precision operating point on its selected held-out set.
5. Static CAVEMOVE noise alone can be suppressed to zero smoothed wake events/hour in the long negative check.
6. The state machine now separates wake detection from command capture and prevents intent decoding from the wake-word frame.
7. Training experiments use separate checkpoint names, so the established baseline was not overwritten.
8. Tests currently pass: **7 passed, 1 skipped**. The skipped test requires an unavailable vendor checkpoint.

## 7. What is not yet proven

The project does not yet have evidence for:

- Real same-driver wake word followed by a same-driver command.
- Passenger speech in the same continuous recording with the target driver.
- Real spatial separation by seat/zone.
- Robust target recall at a fixed low false-accept rate.
- False actions/hour on a real, long-duration, labeled cabin stream.
- Echo/playback rejection with a measured playback reference.
- Window-open wind, reverberation, seat movement, and microphone failure robustness.
- Intent accuracy conditioned on a correctly detected target wake in real cabin audio.
- ASR WER or action execution accuracy after TSE.
- Automotive-DSP latency, memory, quantization, thermal behavior, or power consumption.
- Sub-10-ms hands-free/full-duplex operation.

## 8. Novelty and SOTA positioning

The overall staged combination—target-speaker extraction, target presence, wake-word detection, and intent/action—is a sensible system design for reliable in-car voice assistance. The project also has a useful engineering contribution in connecting these stages and evaluating failure propagation.

However, the current results do not establish SOTA performance or scientific novelty of the individual components:

- SpEx+ transfer learning, log-mel KWS, wav2vec intent classification, and small presence heads are established techniques.
- A complete end-to-end evaluation with speaker-consistent real-cabin recordings is still missing.
- Novelty claims should focus on the complete target-speaker-conditioned in-car cascade and its evaluation methodology only after the missing benchmark exists.

## 9. Recommended next milestone

The highest-value next step is data/evaluation, not another blind head fine-tune.

Build a small but rigorous speaker-consistent cabin corpus containing:

1. The same enrolled driver’s short reference, wake word, and command.
2. Passenger wake words and non-wake speech from different speakers.
3. Road, HVAC, wind, reverberation, and playback/echo conditions.
4. Four-channel microphone recordings with known seat/zone geometry.
5. Continuous timestamps for driver speech, passenger speech, wake, command, silence, and action labels.
6. Speaker-disjoint train/validation/test splits.

Then select operating points jointly on:

- target wake recall;
- false wakes/hour;
- false actions/hour;
- command intent accuracy conditioned on correct target capture;
- target miss rate during pauses and overlap;
- SI-SDR/STOI or intelligibility;
- p50/p95 streaming latency;
- memory and power after quantization.

After that benchmark exists, the technical sequence should be:

1. Re-train a target-speaker VAD/presence head on the real continuous labels.
2. Add a compact speaker-verification gate or speaker-conditioned KWS trained on the same driver/passenger mixtures.
3. Revisit Stage 1 with a dual-domain selection metric so real-cabin improvement cannot silently damage CAVEMOVE performance.
4. Distill or replace wav2vec intent with a compact streaming encoder and quantize the complete cascade.
5. Add conservative action-level rejection: no vehicle action unless target presence, wake, and command confidence all pass calibrated thresholds.

## 10. Current recommended baseline checkpoints

These are the safest checkpoints to use for the base project status:

- Stage 1: `checkpoints/stage1_spex_transfer_mc_cavemove_cabin/best.pt`
- Stage 2: `checkpoints/stage2_frame_presence_gsc_aishell_ft1/best.pt`
- Stage 3 KWS: `checkpoints/stage3_kws_tse_morepos_aishell_ft1/best.pt` for the broader experimental path, with `stage3_kws_tse_domain_binary`/the narrow TSE-domain candidate retained as regression baselines.
- Stage 4 intent: `checkpoints/stage4_intent_wav2vec_ft8_resume_tse_domain/best.pt` plus its `encoder_best.pt`.

The recent same-domain target-KWS and Stage 1 hard-negative checkpoints should remain experimental and should not replace the established baseline without a new speaker-consistent benchmark.

## 11. Data/storage inventory

Measured workspace inventory from the project audit:

| Location | Approximate size |
|---|---:|
| `data/raw` | 298.10 GiB |
| `data/processed` | 3.15 GiB |
| `data/cache` | 19.29 GiB |
| `checkpoints` | 0.75 GiB |
| `logs` | about 2 MiB |

Largest local raw corpora:

| Dataset | Approximate local size |
|---|---:|
| AISHELL-5 | 247.41 GiB |
| LibriSpeech | 27.10 GiB |
| CAVEMOVE | 8.61 GiB |
| Speech Commands v2 | 5.37 GiB |
| Fluent Speech Commands | 5.00 GiB |
| Vehicle Interior Sound | 4.45 GiB |

The large AISHELL-5 footprint is useful for real-cabin negative/transfer work, but the project still needs target-speaker wake/command positives in the same acoustic domain.

## Final status

**Base project: functional research prototype; not production-ready.**

The strongest verified result is the combination of:

- approximately +10.98 dB SI-SDR improvement on held-out synthetic cabin mixtures;
- 99.31% official clean FSC intent top-1;
- approximately 70 ms GPU model-forward latency;
- zero full-path events/hour on the current AISHELL target-absent stream.

The decisive missing result is a real, speaker-consistent continuous driver/passenger benchmark. Until that exists, the system’s main risk remains passenger speech leaking into the TSE/KWS path and producing false vehicle actions.
