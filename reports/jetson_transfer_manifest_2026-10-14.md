# Jetson transfer manifest — 2026-10-14

## Archives

| Archive | Size | SHA-256 | Contents |
|---|---:|---|---|
| `car_tse_jetson_runtime_ckptfix4_2026-10-14.zip` | 456,858,624 bytes / 435.69 MiB | `7503E366E6D4BF13271FB2FA3BC65493FC0F4509136CFB8BB21D7BAD85407CBC` | Retained ONNX runtime, optional compact export, configs, source, benchmark scripts, docs, checksums |
| `car_tse_all_checkpoints_2026-10-14.zip` | 2,757,225,261 bytes / 2.568 GiB | `34D23DFB02E6FBB0DD8824B92CEC6B5596A1887136720A0554D2704F19B9085B` | All 95 files under `checkpoints/`, plus `SHA256SUMS.txt` |

The uncompressed checkpoint payload is 3,293,508,696 bytes (3.07 GiB).
The archives were listed successfully with the Windows `tar` reader; the
checkpoint archive contains 95 `.pt` entries.

## Retained runtime graph

`onnx/retained_ckptfix4_2026-10-14/` contains the seven runtime components:

- `stage1.onnx` — 4-channel target-speaker extraction;
- `speaker_embedding.onnx` — enrollment/extracted speaker embeddings;
- `presence.onnx` — target-presence decision;
- `kws.onnx` — general wake detector;
- `target_kws.onnx` — enrollment-conditioned wake detector;
- `compact_verifier.onnx` — target-speaker match gate; and
- `intent.onnx` — 31-class FSC intent classifier.

`manifest.json` records opset 18, graph input contracts, checkpoint hashes,
and ONNX output hashes. The optional `compact_tse_ft1_2026-10-14/` directory
is included for follow-up model-size/quantization experiments and is not the
current primary benchmark identity.

## Host reference latency

Measured model-forward baselines on the development host:

- RTX 4080, FP16: wake stage 56.99 ms mean; command stage 76.21 ms mean.
- Host CPU, FP32: wake stage 561.15 ms mean; command stage 679.57 ms mean.
- ONNX streaming host diagnostic: lazy intent at a 500 ms hop, 8.195 s
  inference wall time over 7.50 s processed audio, RTF 1.093, with zero
  dropped windows.

These numbers are not Jetson results. The Jetson test must record Jetson
model, JetPack, runtime/provider, precision, power/clocks mode, warm-up,
mean/p50/p95 latency, RTF, and dropped audio.

## Verification on Jetson

```bash
sha256sum -c SHA256SUMS.txt
python scripts/eval/benchmark_onnx_components.py \
  --model-dir onnx/retained_ckptfix4_2026-10-14
```

The retained Stage 1 inputs are `[1,4,48000]` microphone audio, `[1,24000]`
enrollment audio, and `[1,48000]` playback reference. The intent input is
`[1,32000]`. Use a JetPack-compatible ONNX Runtime/TensorRT stack; the
Windows environment and absolute paths in the provenance manifest are not
portable.

This package is for latency and parity testing. The project remains
`research_engineering_preview` with `production_ready=false`; there are still
zero consented same-driver real-cabin sessions.
