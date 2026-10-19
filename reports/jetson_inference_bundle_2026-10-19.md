# Jetson inference bundle

Generated 2026-10-19 from the `multi-speaker-eval` branch. The transfer
artifacts are stored outside the Git worktree at `D:/Prawin_New/tse/jetson_bundles/`.

## Archives

| Archive | Size | SHA-256 | Use |
|---|---:|---|---|
| `car_tse_jetson_onnx_ft6_2026-10-19_v3.zip` | 349,987,240 bytes | `a2318f1c8d0eb87a4aba49ed4e786e03fcef33fbae7c1a58cabb99c791c8c0ef` | Recommended Jetson inference/latency transfer |
| `car_tse_jetson_inference_ft6_2026-10-19_v2.zip` | 816,542,628 bytes | `e308f7fb05d634f19d83059a2c21271cbead88899426e04567133e82195afe38` | Full bundle: ONNX plus PyTorch checkpoints and SpEx+ fallback |

Use the `v2`/`v3` archives. They include the vendored SpEx+ Python module
required by the ONNX bridge's import path. The earlier same-date archives are
superseded packaging attempts.

## Selected models

| Component | PyTorch checkpoint SHA-256 | Matching ONNX SHA-256 |
|---|---|---|
| Stage 1 four-channel TSE | `13ec7c58870feb975efb266e4064bf58575dbc6120f787a77a49c3bc4edda3ac` | `378aaa15781ceeb5fe6151c3cb0830f4e5736e1d6111e5fc6f545f8045a86185` |
| Speaker embedding | same Stage 1 checkpoint | `6ba5ed3335827ce3f33b1984ae4ee98cf1c746d705114f9b1ecf26842fd7ef88` |
| Presence gate | `d56b6c9d8f9efcfb211d50ba6c6ef9f6d0f628c0f417c51837582207e4331550` | `3cd06c0bac4bdcf157e4e5aedf83dc2f963656bc9b98b933b338afefa241f223` |
| General KWS | `c659372280c40b328ab724a8614a11c829dcc25a94ef2003fa480223287f1c3f` | `bf1dfde9a4a4a421b5bd2dfea759693fbab009c817f15ab0c22a4007c6a8f004` |
| Target-conditioned KWS | `7331991e12e6802ab1beb17827bd707d3cc407691c53ab4aa55cf12d4d125af1` | `0b23bd0921ddd7d2723bccaa5eea2e3b2bee0172713a1b8a7981b46afcf275ab` |
| Compact speaker verifier | `1767d672e5b1963827ca43b93976049b8e5fb828bb8f631a44d63fe50466ecea` | `dc667e279ae7abc6133634082edb5f739adad9ffdd42bc98ae97daac40fe899d` |
| FT6 intent head | `de6689d3a61664f16dab0f5e4ec6f9132877cda3d76707306760cd76d599813c` | `e544799007166ccafe904eb8e5219c868d191654a0142ec90b56e5b58f83f78f` |

The full bundle also carries the FT6 intent encoder and the SpEx+ pretraining
checkpoint for Python fallback. The ONNX graphs were exported at opset 18 and
each passed component-level PyTorch/ONNX parity.

## Qualification

This is an inference and device-benchmark bundle, not a production release.
The ONNX bridge uses the fixed-window command path; the current PyTorch FT6
research configuration uses the newer buffered/endpoint-aware command path.
Jetson latency, TensorRT provider compatibility, thermal behavior, audio I/O,
and real-driver safety metrics still need to be measured on the target device.

The large model files and archives are intentionally not committed to GitHub.
This report records their exact external paths and hashes for reproducibility.
