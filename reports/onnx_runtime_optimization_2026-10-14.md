# ONNX runtime optimization note — 2026-10-14

## Scope

This note records a deployment-path optimization applied to the retained
`ckptfix4` ONNX export. It does not change trained weights, thresholds, the
PyTorch reference pipeline, or the production-readiness gates.

## Change

Before the change, `OnnxComponentRuntime.run_components()` evaluated every
exported graph on every 250 ms streaming hop, including the 2-second wav2vec
intent graph before a wake word had been accepted. The streaming state machine
cannot use an intent result until it is waiting for a post-wake command.

The runtime now supports `include_intent=False`, and
`OnnxInCarPipeline.forward()` sets that flag whenever it is not awaiting a
command. The parity harness keeps its default full-component mode, while the
live streaming path evaluates intent only in the command state. The behavior
is covered by a focused test that verifies the first wake-stage call omits
intent and the following command-stage call includes it.

ORT intra/inter-op thread counts are also explicit optional runtime settings.
The default remains provider-controlled; the host benchmark used 8 intra-op
threads and 1 inter-op thread on an AMD Ryzen 7 5800X3D (8 cores / 16 logical
processors).

## Evidence

The deterministic 10-second CAVEMOVE replay uses the actual
`StreamingRunner`, four microphone channels, and a sample-aligned playback
reference. It is a proxy deployment benchmark, not a real-driver accuracy
study.

| Configuration | Inference wall | Processed hop audio | Inference RTF | Capture-wall RTF | Dropped audio |
|---|---:|---:|---:|---:|---:|
| Full components every hop, 8 intra / 1 inter threads | 18.765 s | 7.25 s | 2.588x | 1.878x | 0 s |
| Lazy intent, 8 intra / 1 inter threads | 16.161 s | 7.25 s | 2.229x | 1.618x | 0 s |

This is approximately a 13.9% reduction in measured inference and capture
wall time on this host. The replay produced no wake or intent events because
the selected repeated proxy row is not an event-positive continuous session;
the 8-row composed parity check still exercises the PyTorch↔ONNX wake and
command state handoff.

The 6-second thread sweep showed the following capture-wall factors:

| Intra-op threads | 1 | 2 | 4 | 8 | 12 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| Capture-wall RTF | 4.09x | 3.22x | 1.88x | **1.39x** | 1.62x | 2.48x |

The short-run 8-thread result benefits from warm-up amortization; the longer
10-second value above is the stable comparison used for reporting. Eight
threads was retained as the host benchmark setting, not hard-coded as a
target-device setting.

As a cadence diagnostic, changing only the runner hop from 250 ms to 500 ms
processed 15 windows with zero dropped seconds, 1.093x inference RTF, and
0.824x capture-wall RTF. This makes the current host nearly keep up, but it
does not satisfy the intended latency budget: it increases decision cadence to
500 ms while retaining a 3-second non-causal context. It is a fallback
diagnostic, not a production fix.

## Remaining bottleneck

The isolated float ONNX benchmark measured approximately 375 ms p95 for the
3-second four-channel Stage-1 graph, 30.6 ms for speaker embedding, and
91.4 ms for the 2-second intent graph at 8/1 threads. Stage 1 therefore
dominates the current host streaming cost. The full path still exceeds a
250-ms hop and is not production-ready on this Python/CPU configuration.

The next valid performance work is target-specific: choose the automotive
SoC, export through its supported compiler/provider, use representative cabin
calibration for static/QAT quantization, and/or replace the non-causal
windowed Stage 1 with a causal/chunked front end. The rejected uncalibrated
dynamic-int8 diagnostic remains excluded.

## Reproduction

```powershell
python scripts/eval/benchmark_onnx_streaming_runner.py `
  --onnx-manifest output/onnx/retained_ckptfix4_2026-10-14/manifest.json `
  --seconds 10 --frame-ms 20 `
  --intra-op-threads 8 --inter-op-threads 1 `
  --output logs/onnx_streaming_runner_cpu_ckptfix4_10s_threads8_lazy_intent_2026-10-14.json
```

Artifacts:

- `scripts/eval/benchmark_onnx_streaming_runner.py`
- `scripts/eval/benchmark_onnx_components.py`
- `src/car_tse/deployment/onnx_runtime.py`
- `tests/test_onnx_runtime.py`
- `logs/onnx_streaming_runner_cpu_ckptfix4_10s_threads8_lazy_intent_2026-10-14.json`
- `logs/onnx_streaming_runner_cpu_ckptfix4_10s_threads8_lazy_intent_hop500_2026-10-14.json`
- `logs/onnx_components_cpu_ckptfix4_threads8_w3_r10_2026-10-14.json`
- `logs/release_manifest_current.json` (26 present hashed artifacts; sign-off false)
- `logs/production_readiness_audit_current_2026-10-14.json`
- `reports/production_readiness_snapshot_current_2026-10-14.md`

