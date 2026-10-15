# Car-TSE end-to-end processing latency and system specification

**Measurement date:** 13 October 2026 (Asia/Calcutta)  
**Purpose:** Measure wall-clock processing time for one runtime window through the repository's `InCarPipeline.forward` path.

## Result

| Processing case | Median | Mean | Range | RTF for 3 s input |
|---|---:|---:|---:|---:|
| One normal processing call: Stage 1 extraction + Stage 2 presence + Stage 3 KWS | **440.9 ms** | 441.8 ms | 389.0–505.7 ms | **0.147** |
| One armed command-completion call: Stage 1 + presence + Stage 4 intent inference | **453.4 ms** | 447.3 ms | 398.1–487.4 ms | **0.151** |

Each result summarizes 10 calls after 2 warm-up calls. The p95 estimate is the maximum of these 10 observations (505.7 ms and 487.4 ms, respectively), so it should not be interpreted as a stable tail-latency guarantee.

The input to each call was 3.0 seconds of 16 kHz four-channel audio plus a 2.0-second enrollment waveform. Both calls processed the full 3-second context. The command-completion case began with the pipeline already armed and a 2-second command crop available, so it times command inference on that call. These numbers measure computation after audio is available; they exclude microphone/audio acquisition, buffering until a 3-second context is available, wake-word-to-command speaking time, OS/audio-driver scheduling, and external vehicle action/reply playback.

## Measurement method

The benchmark instantiated the actual `InCarPipeline` and loaded the local cabin-trained Stage 1, Stage 2, Stage 3, and Stage 4 checkpoints. It used `torch.inference_mode()`, CPU wall-clock timing with `time.perf_counter()`, a fixed synthetic random input (seed 20261013), and 8 PyTorch CPU threads. The presence and KWS thresholds were set to zero with one-hit smoothing so the processing path would continue through KWS and arm the command state regardless of the random input. The command-completion timing used an already-armed state to guarantee Stage 4 inference. Threshold overrides make this a latency measurement only; they are not an accuracy or false-trigger test.

The input contains synthetic noise, not a recorded driver wake phrase or command. No quality, detection, or intent-accuracy conclusions should be drawn from this run.

## System specification

| Component | Specification observed in this run |
|---|---|
| System | Gigabyte B650 GAMING X AX V2 motherboard |
| CPU | AMD Ryzen 7 7700X, 8 cores / 16 logical processors |
| Memory | 47.1 GiB reported by Windows |
| GPU | NVIDIA GeForce RTX 5060 Ti, 16,311 MiB reported by NVIDIA driver; driver 610.88 |
| Operating system | Windows 11 Pro, 64-bit, build 26200 |
| Python | 3.14.6 |
| PyTorch | 2.14.1+cpu, 8 CPU threads |
| Accelerator used | CPU; this PyTorch build reports CUDA unavailable |

Although an RTX 5060 Ti is installed, this benchmark did **not** measure GPU latency. A CUDA-enabled PyTorch environment is required for a GPU comparison. The Python environment also lacks the optional `transformers` package. The local CRNN intent checkpoint does not use wav2vec; a minimal import shim was used for the unused wav2vec class import so this CRNN pipeline path could be measured. Model loading/startup time is not included.

## Models and checkpoints

- Stage 1: 16 kHz four-microphone cabin frontend + SpEx+-style target-speaker extraction — `stage1_spex_transfer_mc_cavemove_cabin/best.pt`
- Stage 2: scalar target-presence head — `stage2_presence_cavemove_cabin/best.pt`
- Stage 3: cabin-trained binary wake-word classifier — `stage3_kws_cavemove_binary/best.pt`
- Stage 4: CRNN intent classifier — `stage4_intent_cavemove_cabin/best.pt`

## CUDA remeasurement in `audioml_base`

The machine was remeasured in the project's `audioml_base` environment using the same pipeline, inputs, checkpoints, and gate overrides. This environment has Python 3.11.16, PyTorch 2.12.0+cu132, CUDA 13.2, and CUDA reports the RTX 5060 Ti. Each device had 3 warm-up calls and 20 timed repetitions. GPU timings synchronize CUDA before and after each call and include copying the three-second microphone/enrollment tensors from CPU memory to the GPU.

| Processing case | GPU median | GPU mean | GPU p95* | CPU median (same env) | Median speedup |
|---|---:|---:|---:|---:|---:|
| Normal call: Stage 1 + Stage 2 + Stage 3 | **24.2 ms** | 24.3 ms | 25.1 ms | 469.3 ms | **19.4×** |
| Armed command-completion call, including Stage 4 | **27.3 ms** | 31.0 ms | 28.6 ms | 474.4 ms | **17.4×** |

*Nearest-rank p95 over 20 samples; the command-completion set included one 99.0 ms maximum outlier. The median and p95 are less affected by that outlier. Peak allocated GPU memory during the CUDA run was 185.2 MiB. Median compute RTF was 0.0081 for a normal call and 0.0091 for command completion.

This CUDA result supersedes the earlier 0.44–0.45 s CPU-only result for the user's current environment. The CPU comparison is slower in `audioml_base` (about 0.47 s) than in the earlier Python 3.14 CPU-only environment (about 0.44 s); these are different software builds and should not be treated as a controlled CPU regression.

### Updated conclusion

With `audioml_base` and CUDA enabled, a complete 3-second processing call takes **about 24 ms on the RTX 5060 Ti** for the normal Stage 1–3 path, or **about 27 ms** when Stage 4 intent inference is included on an armed command call. GPU model-processing time is therefore far below the input context duration (median RTF below 0.01). This remains processing time after a complete context is available; it does not include the 3-second context accumulation, wake-to-command speaking time, audio-driver scheduling, or action/reply playback. The run uses synthetic input and forced gates for latency coverage, not accuracy validation.

## Earlier CPU-only run

In the initial Python 3.14 CPU-only runtime, a complete 3-second processing call took about **0.44–0.45 seconds**. That result is retained as historical context and is superseded by the CUDA-enabled `audioml_base` measurement above. The model still processes a non-causal 3-second context, so this compute timing does not include capture or buffering delay.
