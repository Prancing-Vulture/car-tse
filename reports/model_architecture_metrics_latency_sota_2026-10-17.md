# In-car target-speaker voice assistant

## Architecture, parameter count, latency, accuracy, compute, and SOTA context

**Report date:** 2026-10-17
**Repository:** car-tse, branch multi-speaker-eval
**Scope:** current research-engineering system and its clean/synthetic-cabin evaluations
**Training status for this report:** no training was started; this is an audit of existing checkpoints, logs, and evaluation artifacts.

## Executive summary

The system is a cascaded, target-speaker voice assistant:

1. A four-microphone cabin front end and SpEx+-style target-speaker extractor produce an estimate of the enrolled driver's speech.
2. A frame/chunk presence head decides whether speech is present.
3. A general wake-word classifier detects the keyword "Marvin."
4. A target-conditioned wake classifier and compact speaker verifier check that the accepted speech resembles the enrolled speaker.
5. A wav2vec 2.0 intent classifier maps the command to one of 31 Fluent Speech Commands intents.
6. A guarded state machine buffers audio, applies wake/command thresholds and refractory periods, and emits at most one action per accepted command window.

The strongest currently documented end-to-end numbers are from 128-scene synthetic evaluations. They are useful for debugging, but they are **not real-driver evidence**: the repository still lacks a continuous recording in which the same consenting driver provides enrollment, wake word, and command while passenger speech and vehicle conditions vary.

| Evaluation | Fully correct | Target wake recall | Wrong/false actions | Main interpretation |
|---|---:|---:|---:|---|
| Clean target-only, 128 scenes | **94.53%** (121/128) | 100% | 0 | Upper-bound-like clean pipeline result |
| Synthetic passenger, non-overlap, FT6 | **96.09%** (123/128) | 100% | 0 | Good proxy result; 4 wrong intents remain |
| Synthetic passenger, overlap, FT6 | **89.84%** (115/128) | 100% | 0 | Overlap and command separation are the current end-to-end weakness |

The component picture is different from the end-to-end picture. Clean FSC intent classification is **99.31% top-1**, while the held-out TSE-domain intent result is **90.49% top-1**. General wake-word F1 is **94.21%** on its component test, but the target-conditioned KWS F1 is only **43.14%** at its stored test operating point. The compact speaker verifier has **0.9706 test AUC**, with **0.7969 TPR** and **0.00534 FPR** at its selected threshold. These values are not directly interchangeable because they use different splits, labels, operating points, and task definitions.

The current measured GPU path is practical for a desktop GPU but not yet a production-device result: the main 3-second command path is **103.31 ms mean / 123.16 ms p95** on an RTX 4080 using FP16, while the Stage-1 extractor alone is **63.47 ms mean / 100.39 ms p95**. The CPU reference is much slower (old configuration: roughly **679.6 ms mean** command latency). Jetson/device latency remains to be measured on the target hardware.

The most important conclusion is that this project should not claim SOTA or production readiness yet. The architecture is a credible research prototype, and the synthetic clean/passenger results are encouraging, but the missing same-driver real-cabin study, high false-wake rates in long negatives, command endpointing, and hardware-specific measurements remain release blockers.

## 1. System definition and data flow

### 1.1 Runtime data flow

The runtime path is:

~~~text
4-channel cabin audio
        |
        v
Cabin echo/spatial front end (optional playback reference)
        |
        v
Stage 1: SpEx+-style target-speaker extraction
        |                         ^
        |                         |
        |                    enrollment utterance
        v
estimated target speech
        |
        +--> Stage 2 frame/chunk presence gate
        |
        +--> general KWS: "Marvin"
        |
        +--> target-conditioned KWS + compact speaker verifier
                                  |
                                  v
                         command buffer / endpointing
                                  |
                                  v
                    wav2vec2-base + 31-class intent head
                                  |
                                  v
                         one guarded vehicle action
~~~

The enrollment utterance is converted to a speaker embedding by the Stage-1 speaker encoder. The extractor uses that embedding to condition the masks applied to the mixture representation. The general KWS, target KWS, and verifier operate on candidate speech windows. The intent model is not a text decoder: it classifies a 3-second audio window directly into the FSC intent inventory.

### 1.2 What the current evaluation does and does not represent

The clean and passenger evaluations are synthetic proxy scenes built from public speech/noise sources. They are not equivalent to a real in-car session. In particular:

- The same-driver continuity and microphone geometry are not established by the proxy.
- The current passenger proxy is non-overlap or constructed overlap, not a measured vehicle recording.
- Current proxy configs use four microphones and cabin_noise=false; no car-speaker playback reference is supplied to the echo canceller.
- The fixed/three-step command buffer and endpoint heuristics can include wake-word tail or cut off command speech.
- The reported 128-scene percentages have wide uncertainty; a single scene changes the score by 0.781 percentage points.

Consequently, "fully correct" below means correct on the defined proxy scene and action metric, not "production reliability."

## 2. Exact model architecture

### 2.1 Stage 1 — target-speaker extraction

The current Stage-1 backend is a 16-kHz transfer SpEx+ implementation, initialized from:

~~~text
E:/prawin/snn/runs/spexplus_no_jepa_30/best.pt
~~~

The frontend accepts four microphone channels. It contains a lightweight echo/spatial preprocessing path followed by tied reference and mixture speech encoders, a speaker encoder, a multi-scale TCN extractor, and three mask decoders. At runtime the enrollment branch produces a speaker representation; the mixture branch produces the target estimate.

The trainable parameter count is **10,727,881**. For the promoted FT6 configuration Stage 1 is frozen.

| Stage-1 block | Parameters | Architecture detail |
|---|---:|---|
| Cabin/echo/spatial frontend | 9,861 | EC hidden 128; spatial hidden 64; four-channel input |
| Short speech encoder | 5,376 | 1-D convolution, approximately 2.5-ms receptive field |
| Middle speech encoder | 20,736 | 1-D convolution, approximately 10-ms receptive field |
| Long speech encoder | 41,216 | 1-D convolution, approximately 20-ms receptive field |
| Speech encoders subtotal | 67,328 | Three scales; weights tied between reference and mixture branches |
| Speaker encoder normalization | 1,536 | Normalization parameters |
| Speaker encoder conv1 | 196,864 | Initial 1-D convolution |
| Speaker encoder, 3 ResNet blocks | 791,046 | 3 x 263,682 |
| Speaker encoder conv2 | 65,792 | Projection convolution |
| Speaker classifier | 64,507 | Training-time speaker CE head |
| Speaker encoder subtotal | 1,119,745 | Classifier is not needed for inference |
| Extractor normalization + conv1 | 198,400 | Conditioning input projection |
| Extractor, four 8-TCN stacks | 9,068,608 | 32 TCN blocks total |
| Three mask/output projections | 197,376 | One output path per scale |
| Extractor subtotal | 9,464,384 | Produces three masks |
| Three mask decoders | 66,563 | Short 5,121; middle 20,481; long 40,961 |
| **Stage-1 total** | **10,727,881** | Exact state-dict/module count |

The TCN stack has eight blocks per stack and four repeated stacks. Each stack has one conditioning-aware first block and seven subsequent TCN blocks:

| TCN unit | Parameters |
|---|---:|
| First block in a stack | 398,082 |
| Each of the other seven blocks | 267,010 |
| One eight-block stack | 2,267,152 |
| Four stacks | 9,068,608 |

The Stage-1 architecture is materially smaller than the full system, but it dominates the runtime FLOPs because the four TCN stacks are evaluated over long 3-second feature sequences. The current report keeps Stage 1 unchanged, consistent with the project decision to improve the downstream gates and command handling before retraining TSE.

### 2.2 Stage 2 — frame/chunk target-presence gate

FramePresenceHead operates on log-mel features:

- sample rate: 16 kHz;
- FFT size: 512;
- hop: 160 samples (10 ms);
- mel bins: 40;
- hidden channels: 64;
- convolutional stack: 40→64, then a dilated 64→64 layer;
- scalar presence output: 64→1.

The exact trainable parameter count is **33,729**:

| Layer | Parameters |
|---|---:|
| Conv1d 40→64, kernel 5 | 12,864 |
| GroupNorm(64) | 128 |
| Dilated Conv1d 64→64, kernel 5 | 20,544 |
| GroupNorm(64) | 128 |
| Output Conv1d 64→1 | 65 |
| **Total** | **33,729** |

This is a generic speech-presence detector, not a fully target-conditioned personal VAD. The stored chunk operating point is strong on its own test: **99.79% accuracy, 99.59% F1, 0.277% FAR, 0% FRR**. The frame operating point is less balanced: **90.44% F1, 0.897% FAR, 5.487% FRR**.

### 2.3 Stage 3a — general wake-word KWS

The current FT6 general KWS head uses four temporal convolution blocks with channels [48, 96, 192, 256], batch normalization, and a two-class output:

| Layer | Parameters |
|---|---:|
| Conv1d 1→48, kernel 10 | 480 |
| BatchNorm1d(48) | 96 |
| Conv1d 48→96, kernel 9 | 41,568 |
| BatchNorm1d(96) | 192 |
| Conv1d 96→192, kernel 9 | 166,080 |
| BatchNorm1d(192) | 384 |
| Conv1d 192→256, kernel 9 | 442,624 |
| BatchNorm1d(256) | 512 |
| Final linear classifier 256→2 | 514 |
| **Total** | **652,450** |

Its stored component result is **97.74% accuracy, 96.76% precision, 91.79% recall, and 94.21% F1** over 975 examples. This is not a long-term false-accept-rate result; the continuous negative-stream test is a separate metric and currently reports high hourly false-wake rates.

### 2.4 Stage 3b — target-conditioned KWS

TargetSpeakerKWSModel uses a three-block audio CNN with channels [32, 64, 128], then compares the audio representation with enrollment/extracted speaker embeddings:

~~~text
relation = [enrollment, extracted, |enrollment-extracted|,
            enrollment*extracted, cosine(enrollment, extracted)]
dimension = 256 + 256 + 256 + 256 + 1 = 1025
MLP: 1153 -> 256 -> 2
~~~

The exact trainable parameter count is **389,316**:

| Layer/group | Parameters |
|---|---:|
| Audio branch | 93,378 |
| Relation MLP 1, 1153→256 | 295,424 |
| Relation MLP 2, 256→2 | 514 |
| **Total** | **389,316** |

Stored component metrics are **82.20% accuracy, 28.95% precision, 84.62% recall, and 43.14% F1** on 1,629 examples. This is the weakest isolated classifier by F1 and needs threshold/label/split analysis before it can be treated as a reliable safety gate.

### 2.5 Stage 3c — compact speaker verifier

CompactSpeakerVerifier uses a small CNN embedding encoder followed by a relation head:

~~~text
audio -> compact CNN/BN encoder -> 128-dimensional embedding
enrollment/extracted relation -> LayerNorm(513) -> 513->128->1
~~~

Exact trainable parameters: **176,579**.

| Layer/group | Parameters |
|---|---:|
| Compact audio encoder | 109,632 |
| Relation LayerNorm(513) | 1,026 |
| Relation Linear 513→128 | 65,792 |
| Relation Linear 128→1 | 129 |
| **Total** | **176,579** |

The diagnostic result is **test AUC 0.97059**, with the selected threshold giving **TPR 79.69%, FPR 0.534%, precision 92.73%**. AUC is threshold independent; TPR/FPR/precision are not. The verifier is therefore promising as a filter, but the operating point must be chosen against a fixed false-action budget and evaluated on held-out same-driver/passenger sessions.

### 2.6 Stage 4 — intent classifier

The intent model is a wav2vec 2.0 Base encoder initialized from the 16-kHz wav2vec2-base-960h family, followed by a 31-class FSC head:

~~~text
16-kHz target estimate, 3-second window
  -> convolutional feature encoder
  -> feature projection and positional convolution
  -> 12 Transformer encoder blocks, hidden size 768
  -> mean/pooled representation
  -> LayerNorm(768)
  -> Linear 768->512
  -> Linear 512->31 FSC intents
~~~

Exact encoder parameters: **94,371,712**. Exact intent-head parameters: **411,167**. Exact intent subsystem total: **94,782,879**.

| Intent layer/group | Parameters |
|---|---:|
| Masked spectrogram embedding | 768 |
| Feature extractor conv0 | 6,144 |
| Feature extractor conv1 | 786,432 |
| Feature extractor conv2 | 786,432 |
| Feature extractor conv3 | 786,432 |
| Feature extractor conv4 | 786,432 |
| Feature extractor conv5 | 524,288 |
| Feature extractor conv6 | 524,288 |
| Feature extractor subtotal | 4,200,448 |
| Feature projection | 395,008 |
| Positional convolution | 4,719,488 |
| Encoder final layer norm | 1,536 |
| 12 Transformer blocks | 85,054,464 |
| **wav2vec2 encoder total** | **94,371,712** |
| Intent LayerNorm | 1,536 |
| Linear 768→512 | 393,728 |
| Linear 512→31 | 15,903 |
| **Intent head** | **411,167** |
| **Intent subsystem total** | **94,782,879** |

Each wav2vec2 Transformer block has the same exact parameter structure:

| Transformer sublayer | Parameters per block |
|---|---:|
| Attention query projection | 590,592 |
| Attention key projection | 590,592 |
| Attention value projection | 590,592 |
| Attention output projection | 590,592 |
| Two layer norms | 3,072 |
| Feed-forward intermediate projection | 2,362,368 |
| Feed-forward output projection | 2,360,064 |
| **One Transformer block** | **7,087,872** |
| **12 blocks** | **85,054,464** |

The convolutional and Transformer rows sum exactly to the authoritative state-dict totals above. Biases are included in the listed parameter counts. Exact counts should still be regenerated from the checkpoint/module state dict after any architecture change.

The current clean official FSC result is **99.31% top-1, 99.89% top-5, and 99.24% macro-F1** over 3,793 examples. The TSE-domain result is **90.49% top-1, 98.01% top-5, and 90.13% macro-F1** over 2,314 examples. The current FT6 runtime-jitter validation checkpoint reports **89.09% accuracy, 96.81% top-5, and 88.90% macro-F1** over 8,192 validation examples; it is a candidate adaptation checkpoint, not a replacement for the clean FSC result.

## 3. Parameter count, storage, and precision

### 3.1 Current FT6 runtime system

The current FT6 composition uses the wider general KWS head above:

| Component | Parameters | FP32 parameter MB | FP16 parameter MB | INT8 parameter MB |
|---|---:|---:|---:|---:|
| Stage 1 TSE | 10,727,881 | 42.9115 | 21.4558 | 10.7279 |
| Stage 2 presence | 33,729 | 0.1349 | 0.0675 | 0.0337 |
| General KWS | 652,450 | 2.6098 | 1.3049 | 0.6525 |
| Target KWS | 389,316 | 1.5573 | 0.7786 | 0.3893 |
| Speaker verifier | 176,579 | 0.7063 | 0.3532 | 0.1766 |
| wav2vec2 + intent head | 94,782,879 | 379.1315 | 189.5658 | 94.7829 |
| **Total** | **106,762,834** | **427.0513** | **213.5257** | **106.7628** |

The MB figures above are decimal parameter storage only (bytes divided by 1,000,000), excluding optimizer state, activation memory, framework overhead, lookup tables, padding, and serialized metadata. The raw current checkpoint files occupy **427,483,384 bytes (427.48 MB; 407.68 MiB)**. The older 106.20M-parameter/425.24-MB figure in prior reports refers to an earlier baseline KWS composition and must not be mixed with FT6.

### 3.2 Bit-width interpretation

The deployed precision has not yet been validated on a Jetson or automotive target. The report therefore distinguishes:

- **FP32:** exact current parameter representation for most stored PyTorch checkpoints.
- **FP16:** estimated parameter-only storage and current RTX 4080 latency mode for the GPU benchmark.
- **INT8:** storage estimate only; no end-to-end accuracy/latency claim is made until quantization-aware or post-training quantization is run and validated.

For production, the wav2vec2 encoder is the dominant footprint and compute cost. Distillation, streaming encoders, layer dropping, mixed precision, and accelerator-specific INT8/FP16 export should be evaluated before claiming an edge-ready footprint.

## 4. Latency and compute

### 4.1 Measured RTX 4080 FP16 component latency

Benchmark conditions: RTX 4080, CUDA 13 / PyTorch 2.14.0+cu130, FP16, five warmups, twenty timed repeats, 3-second intent input. Values are milliseconds per invocation.

| Component | Mean ms | P50 ms | P95 ms | 2x-MAC estimate | Effective TFLOP/s* |
|---|---:|---:|---:|---:|---:|
| Stage 1 TSE | 63.471 | 56.674 | 100.387 | 103.945 GFLOPs | 1.638 |
| Enrollment embedding | 5.164 | 3.637 | 7.469 | not isolated | — |
| Extracted embedding | 3.728 | 3.509 | 4.002 | not isolated | — |
| Presence head | 1.276 | 1.271 | 1.321 | 0.020 GFLOPs | 0.016 |
| Target KWS | 4.651 | 4.679 | 4.704 | 0.152 GFLOPs | 0.033 |
| General KWS | 3.439 | 3.527 | 3.591 | 0.449 GFLOPs | 0.131 |
| Compact verifier | 4.048 | 3.689 | 6.636 | 0.152 GFLOPs | 0.038 |
| wav2vec2 intent, 3 s | 25.705 | 30.165 | 32.001 | 25.431 GFLOPs** | 0.989 |
| Composite wake | 74.853 | 53.689 | 108.121 | 104.719 GFLOPs | 1.399 |
| Composite command | 103.314 | 100.330 | 123.156 | 130.150 GFLOPs | 1.260 |

*Effective TFLOP/s is analytical estimated FLOPs divided by measured mean seconds. It is not the GPU's vendor peak TFLOPS and includes neither a rigorous kernel-level FLOP count nor power/thermal constraints.
**The wav2vec2 profiler's FLOP value is a torch.profiler estimate and can undercount attention, normalization, activation, and other operations. Treat it as an engineering estimate.

### 4.2 CPU reference

The latest full CPU log is for an older compact configuration with a 2-second intent window, so it is not an apples-to-apples replacement for the current 3-second FT6 GPU run:

| Path | Mean ms | P50 ms | P95 ms | Configuration note |
|---|---:|---:|---:|---|
| Stage 1 | 423.044 | 420.793 | 439.474 | CPU FP32 reference |
| Full pipeline wake | 561.149 | 562.818 | 576.689 | compact cabin-smart CPU log |
| Full pipeline command | 679.566 | 677.889 | 715.995 | 2-second intent window |
| Current direct 3-second intent probe | 148.162 | 148.033 | 149.626 | ad-hoc, five repeats; not a saved benchmark artifact |

The CPU results show why a target-device benchmark is mandatory. A desktop GPU timing should not be used as a Jetson or automotive-DSP claim. Also, model execution time after audio is available is only one latency term: buffering, audio capture, endpointing, scheduling, and device transfers add system delay.

### 4.3 FLOP/TFLOP methodology

For convolutional and linear layers, the report uses the conventional approximation:

~~~text
FLOPs approximately equal 2 x MACs
~~~

The main analytical totals are:

| Component | Estimated compute per 3-second call |
|---|---:|
| Stage 1 | 103.945 GFLOPs |
| Presence | 0.020 GFLOPs |
| General KWS | 0.449 GFLOPs |
| Target KWS | 0.152 GFLOPs |
| Compact verifier | 0.152 GFLOPs |
| Intent wav2vec2 | 25.431 GFLOPs (profiler estimate) |

The Stage-1 breakdown is approximately:

| Stage-1 group | GFLOPs |
|---|---:|
| Cabin/echo/spatial frontend | 0.928 |
| Twin speech encoders | 2.555 |
| Speaker encoder | 5.544 |
| Extractor input projection | 1.887 |
| Four 8-TCN stacks | 86.018 |
| Three mask/output projections | 1.887 |
| Mask decoders | 6.901 |
| **Total** | **103.945** |

These numbers answer "how many operations does this implementation execute?" rather than "what is the peak throughput of the GPU?" The latter is a hardware specification and is not inferred from this benchmark.

### 4.4 Layer-level latency boundary

The saved benchmark measures complete modules and composite stages, not every
individual layer. A valid per-layer CUDA measurement requires a separate
instrumented run with synchronization around each operation; CUDA kernel
fusion, memory transfers, cuDNN algorithm selection, and asynchronous execution
make it invalid to divide a module time by its number of layers. Therefore:

- the parameter tables above are layer/sub-layer exact;
- the latency table is component/module exact for the recorded benchmark;
- no per-layer latency number is claimed for the TSE TCN blocks, Transformer
  blocks, KWS convolutions, or presence layers;
- the Stage-1 TCN and wav2vec2 Transformer groups are the compute hotspots
  according to the FLOP analysis, but hotspot status is not the same as a
  measured wall-clock share.

A future layer-profile artifact should report synchronized CUDA time,
input/output shapes, kernel count, peak memory, and precision for each layer.
It should be run on the Jetson target as well as the development RTX 4080.

### 4.5 Existing ONNX and quantization diagnostics

Older ONNX exports provide useful bit-width evidence, but they are not a
replacement for a current FT6 Jetson benchmark:

| Artifact/precision | Result | Interpretation |
|---|---|---|
| Retained ONNX FP32 Stage 1, CPU | 368.0 ms mean / 376.4 ms p95 | Exported 3-second Stage 1 reference |
| Retained ONNX FP32 intent, CPU | 91.1 ms mean / 93.6 ms p95 | Older 2-second intent export |
| Compact intent dynamic INT8 | 25.35% of FP32 file size; cosine 0.9987 | Weight-only diagnostic; CPU mean became 36.4 ms vs 7.0 ms float |
| Retained Stage-1 dynamic INT8 | 27.64% of FP32 file size; cosine 0.9979 | CPU mean became 5,570 ms vs 375 ms float; not deployable |
| Static INT8 compact intent | 97.85% of FP32 file size; accuracy 4.69% on 128 rows | Failed calibration/quantization candidate |

The dynamic INT8 values are diagnostic weight-only quantization results and
show that file-size reduction does not guarantee speedup on this CPU provider.
The static INT8 result is a failed candidate, not a production result. No
INT8 accuracy or latency claim should be made for the current full FT6 system
until all stages are quantized/exported together and evaluated on the target
accelerator.

## 5. Component accuracy and diagnostics

### 5.1 TSE

Held-out dual-domain regression, 288 examples (144 AISHELL-5 and 144 CAVEMOVE):

| Domain | Baseline SI-SDR | Candidate SI-SDR | Improvement |
|---|---:|---:|---:|
| AISHELL-5 | 2.385 dB | 6.243 dB | +3.858 dB |
| CAVEMOVE | 10.478 dB | 10.592 dB | +0.114 dB |
| Overall | 6.432 dB | 8.417 dB | +1.985 dB |

This verifies that Stage 1 improves the difficult AISHELL domain while preserving the CAVEMOVE domain. It does not establish intelligibility, command accuracy, or driver/passenger rejection by itself.

### 5.2 Presence, KWS, verifier, and intent

| Component | Accuracy | Precision | Recall | F1/AUC | Examples |
|---|---:|---:|---:|---:|---:|
| Presence, frame threshold .5 | 98.84% | 86.71% | 94.51% | F1 90.44% | component test |
| Presence, chunk threshold .5 | 99.79% | 99.18% | 100% | F1 99.59% | component test |
| General KWS | 97.74% | 96.76% | 91.79% | F1 94.21% | 975 |
| Target KWS | 82.20% | 28.95% | 84.62% | F1 43.14% | 1,629 |
| Compact verifier | — | 92.73% | 79.69% | AUC 97.06% | stored diagnostic |
| Intent, clean FSC | 99.31% | — | — | macro-F1 99.24% | 3,793 |
| Intent, TSE-domain | 90.49% | — | — | macro-F1 90.13% | 2,314 |
| Intent FT6 runtime-jitter validation | 89.09% | — | — | macro-F1 88.90% | 8,192 |

The target KWS F1 and verifier operating point should be investigated before further intent fine-tuning. A high general-KWS score does not imply a low false-accept rate in continuous multi-speaker audio.

## 6. End-to-end results

### 6.1 Clean target-only

Artifact: logs/clean_e2e_v2_knnvc_matched_postaction16_128_2026-10-16.json

| Metric | Result |
|---|---:|
| Scenes | 128 |
| Fully correct scenes | 121/128 = **94.53%** |
| Target wake recall | 100% |
| False action events | 0 |
| Passenger-only false actions | 0 |
| Executed non-target actions | 0 |
| Wrong executed action events | 0 |
| Strict fully correct action scenes | 120/128 |
| Strict false command rate | 2.009 per hour |
| Strict false wake rate | 16.071 per hour |
| Mean wake latency | 1.699 s |
| P95 wake latency | 2.000 s |

### 6.2 Synthetic passenger, non-overlap

Artifact: logs/multispeaker_passenger_ft6_gate015_rescue085_postcommand08_128_2026-10-17.json

| Metric | Result |
|---|---:|
| Scenes | 128 |
| Fully correct scenes | 123/128 = **96.09%** |
| Target wake recall | 100% |
| False action events | 0 |
| Passenger-only false actions | 0 |
| Executed non-target actions | 0 |
| Wrong intent events | 4 |
| Strict fully correct action scenes | 118/128 |
| Strict false command rate | 10.045 per hour |
| Strict false wake rate | 46.205 per hour |
| Mean wake latency | 2.133 s |
| P95 wake latency | 3.500 s |

### 6.3 Synthetic passenger, overlap

Artifact: logs/multispeaker_overlap_ft6_gate015_rescue085_postcommand08_128_2026-10-17.json

| Metric | Result |
|---|---:|
| Scenes | 128 |
| Fully correct scenes | 115/128 = **89.84%** |
| Target wake recall | 100% |
| False action events | 0 |
| Passenger-only false actions | 0 |
| Executed non-target actions | 0 |
| Wrong intent events | 10 |
| Strict fully correct action scenes | 115/128 |
| Strict false command rate | 0 per hour |
| Strict false wake rate | 54.241 per hour |
| Mean wake latency | 2.117 s |
| P95 wake latency | 3.000 s |

The contrast is important: the current proxy's visible action safety is good because the state machine suppresses many candidate events, but the strict false-wake rates and wrong-intent counts show that the system is not yet production-ready. The overlap case loses roughly 6.25 percentage points relative to non-overlap, mostly through command separation/intent errors rather than target-wake recall.

## 7. SOTA and prior-work context

This section is a positioning review, not a claim that the current system beats any cited work. Direct comparisons are invalid unless datasets, labels, operating points, latency hardware, and definitions are matched.

### 7.1 Target-speaker extraction

SpEx+ is the closest architectural reference for the current Stage 1. It uses weight-shared reference/mixture speech encoders, a speaker encoder, and multi-scale TCN extraction. Its reported improvements are on speech separation/extraction metrics, not wake-to-intent action success. The current system follows that general design while adding a cabin frontend and downstream gates.

WeSep and later target-speaker extraction systems show that the field is moving toward stronger separation backbones, directional/spatial cues, and more robust multi-speaker evaluation. Our current Stage-1 regression is promising (+3.858 dB AISHELL SI-SDR improvement and preserved CAVEMOVE performance), but it is not a SOTA comparison because the evaluation is small and domain-specific.

Primary references:

- [SpEx+](https://arxiv.org/abs/2005.04686)
- [WeSep: a speech separation toolkit](https://arxiv.org/abs/2409.15799)

### 7.2 Presence and personal VAD

Personal VAD work uses a target-speaker embedding to distinguish target speech from non-target speech and non-speech. The current Stage 2 head is much smaller and generic: it predicts speech presence but is not fully target-conditioned. This makes it a useful fast gate, not a direct replacement for a personal VAD or target-speaker voice-activity detector.

Primary references:

- [Personal VAD](https://arxiv.org/abs/1908.04284)
- [Personal VAD 2.0](https://arxiv.org/abs/2204.03793)

### 7.3 Wake-word detection

Compact KWS models such as BC-ResNet demonstrate that very small networks can achieve high Speech Commands accuracy. That benchmark is a short isolated-word classification task; it does not measure passenger rejection, continuous false accepts per hour, or wake-to-command action correctness. The current general KWS component is strong on its stored component split, but its continuous false-wake results remain much weaker than a production KWS target.

Primary references:

- [BC-ResNet](https://arxiv.org/abs/2106.04140)
- [Speech Commands](https://arxiv.org/abs/1804.03209)

### 7.4 Speech-to-intent and SLU

The current wav2vec2 plus linear head is a strong clean-data classifier and reaches 99.31% top-1 on the official FSC evaluation. However, it is a batch 3-second classifier, not a streaming end-to-end SLU model. Streaming end-to-end SLU work reports approximately 97% on FSC multi-intent settings with a different model/task formulation, and those numbers should not be compared directly to our single-window 31-class score.

Primary references:

- [wav2vec 2.0](https://arxiv.org/abs/2006.11477)
- [Streaming end-to-end SLU](https://arxiv.org/abs/2105.10042)
- [SLURP](https://arxiv.org/abs/2011.13205)

### 7.5 End-to-end comparison: Paranjape et al.

Paranjape et al. describe a RASA-based voice assistant study using noisy conditions and a task-level assistant evaluation. Their reported result is **281/300 successful interactions (93.67%)**. The paper also reports approximately **0.08 s intent identification**, **2.4 s input/confirmation**, and **4.6 s API/output** response components.

This is the closest cited end-to-end assistant comparison, but it is still not apples-to-apples:

- their task/intent inventory and interaction protocol differ;
- the current system requires target-speaker extraction, wake acceptance, speaker gating, command buffering, and a 31-class FSC intent;
- our current clean/passenger results are 128-scene synthetic proxy evaluations;
- their latency includes an application interaction protocol, while our model timing is mostly neural inference after the audio window is available.

Reference: [Paranjape et al., Voice Assistant for Noisy Environments](https://arxiv.org/abs/2312.01642).

### 7.6 End-to-end comparison: Hussein et al.

Hussein et al. report approximately **93% accuracy** on a small Speech Commands-style experiment with **687 English clips**, a 70/30 train/test split, and approximately **0.75 s response** in their described setup. This is a command-classification comparison, not a target-speaker, passenger-rejection, continuous wake-word, or vehicle-action benchmark.

The comparison is therefore useful only as a lower-complexity speech-command baseline. It is not evidence that a 93% Speech Commands classifier solves the current in-car problem. The current clean FSC intent result is higher, but the complete target-speaker assistant has extra failure modes that the Hussein task does not measure.

Reference: [Hussein et al., Speech Command Recognition Using Deep Learning, Electronics 14(3), 496](https://doi.org/10.3390/electronics14030496).

### 7.7 Positioning summary

| System/component | What its headline number measures | Why it is not directly comparable to full pipeline |
|---|---|---|
| Current clean FSC intent | 31-class isolated command classification | No wake, passenger, or action-state requirement |
| Current TSE-domain intent | Intent after extracted/degraded audio | Still not a continuous cabin interaction |
| Current synthetic full pipeline | Driver wake + command + passenger proxy | 128 scenes; no real-driver continuous recordings |
| Paranjape et al. | Task-level noisy RASA assistant success | Different task, intents, interaction and latency definitions |
| Hussein et al. | Small Speech Commands classifier | No TSE, wake-word, passenger rejection, or vehicle action |

## 8. Limitations and production gates

The current evidence supports a research prototype, not a production release. The main blockers are:

1. **No real-driver same-session test set.** Need enrollment, "Marvin," command, passenger speech, no-command intervals, consent, and exact timestamps in a held-out cabin study.
2. **False-wake rate.** Current strict proxy values are tens of false wakes/hour, far above a plausible production target such as 0.1/hour. At least 30 hours of held-out negative audio is needed to make a zero-event claim meaningful.
3. **Target speaker gate.** Target-KWS F1 is low at its stored threshold; verifier AUC is strong but TPR/FPR need an operating-point study.
4. **Endpointing and command windows.** Fixed windows can include wake tail or truncate the command, especially under overlap.
5. **Hardware measurement.** Jetson/automotive target latency, peak memory, thermal behavior, audio I/O, and dropouts are unmeasured.
6. **Model size.** wav2vec2 dominates both parameters and storage; a distilled/streaming intent model is likely needed for edge deployment.
7. **Safety.** Risky actions need confirmation and vehicle-state interlocks; false actions must be reported separately from classification accuracy.
8. **Data split discipline.** Driver/session/vehicle separation and threshold freezing must precede a final claim.

## 9. Reproducibility and artifact manifest

### 9.1 Key configurations

- Canonical FT6 runtime: configs/eval_end_to_end_fscpassenger_identity_runtimealignedintent_ft6_jitter_prewakebuffer_postcommand08_2026-10-17.yaml
- Recommended soft-gate FT6 runtime: configs/eval_end_to_end_fscpassenger_identity_runtimealignedintent_ft6_jitter_prewakebuffer_gate015_rescue085_postcommand08_2026-10-17.yaml
- Stage-4 runtime-jitter training config: configs/stage4_intent_runtime_jitter_clean_overlap_full_2026-10-17.yaml

### 9.2 Checkpoint SHA-256

| Checkpoint | SHA-256 |
|---|---|
| Stage 1 best.pt | 13EC7C58870FEB975EFB266E4064BF58575DBC6120F787A77A49C3BC4EDDA3AC |
| Stage 2 best.pt | D56B6C9D8F9EFCFB211D50BA6C6EF9F6D0F628C0F417C51837582207E4331550 |
| General KWS best.pt | C659372280C40B328AB724A8614A11C829DCC25A94EF2003FA480223287F1C3F |
| Target KWS best.pt | 7331991E12E6802AB1BEB17827BD707D3CC407691C53AB4AA55CF12D4D125AF1 |
| Compact verifier best.pt | 1767D672E5B1963827CA43B93976049B8E5FB828BB8F631A44D63FE50466ECEA |
| Intent head best.pt | DE6689D3A61664F16DAB0F5E4EC6F9132877CDA3D76707306760CD76D599813C |
| Intent encoder best.pt | A1B8FA00C1D9C3031A7BA586721C6B37A1D30B1062AC0270AD8A673E9245F643 |
| Canonical config | E4F4F1A881273C78FA46291DBFD0BC96E409460FE67BAA2A33B2B55D135233EF |
| Soft-gate config | 7FD443B5EC9F130AC64E1DBD28E9E471DA4A47B8EE628D446BF7AF8C30FD5FE5 |

### 9.3 Source logs

- Stage 1: logs/stage1_dual_domain_regression_full_2026-10-10.json
- Stage 2: logs/stage2_frame_presence_dual_domain_nonwake_ft1/metrics_last.json
- General KWS: logs/stage3_kws_clean_control_large_ft1/metrics_last.json
- Target KWS: logs/stage3_target_kws_gsc_marvin_hardneg_aishell5_smart_speaker_ft1/metrics_last.json
- Verifier: logs/compact_speaker_verifier_cabin_smart_smartcache_diagnostic_2026-10-12.json
- Intent FT6: logs/stage4_intent_runtime_jitter_clean_overlap_ft6_fromft4_2026-10-17/metrics_last.json
- Clean E2E: logs/clean_e2e_v2_knnvc_matched_postaction16_128_2026-10-16.json
- Non-overlap E2E: logs/multispeaker_passenger_ft6_gate015_rescue085_postcommand08_128_2026-10-17.json
- Overlap E2E: logs/multispeaker_overlap_ft6_gate015_rescue085_postcommand08_128_2026-10-17.json
- Current GPU latency: logs/component_latency_ft6_current_cuda_fp16_2026-10-17.json

## 10. Recommended next experiments

The highest-value sequence is:

1. Freeze a real-driver tuning/test split and record at least a pilot with several consenting speakers, passenger wakes, no-command intervals, and playback reference.
2. Tune the target-speaker verifier and KWS thresholds on tuning sessions only.
3. Replace the fixed command window with target-presence endpointing and retain the pre-wake buffer.
4. Compare the current voiceprint gates against a classic driver-seat beamforming/zone baseline at matched false-accept rates.
5. Re-run overlap/non-overlap ablations with different passengers per scene and no test-data reuse.
6. Export the reduced/streaming intent path to ONNX/TensorRT and measure on the Jetson target.
7. Quantize only after a frozen accuracy baseline; report FP32, FP16, and INT8 accuracy/latency side by side.
8. Build 30+ hours of held-out negatives before publishing a false-accept-rate claim.

## Bottom line

The current system is a strong research prototype with a credible SpEx+-style TSE front end, a compact presence/KWS/verifier cascade, and a high-performing clean FSC intent classifier. The best proxy full-pipeline result is **96.09% non-overlap** and **89.84% overlap**, while the clean target-only proxy is **94.53%**. The major engineering weaknesses are not the clean intent head alone: they are target-speaker gating, command endpointing, continuous false wakes, and the lack of real-driver/device evidence.

Do not label the system "production-ready" or "SOTA" from these results. The proper next claim is: **a reproducible research prototype with promising synthetic-cabin performance and a clearly defined real-cabin validation plan.**
