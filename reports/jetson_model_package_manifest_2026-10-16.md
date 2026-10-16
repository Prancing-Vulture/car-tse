# Jetson inference package manifest — 2026-10-16

This package is for inference/latency experiments on a Jetson. It is not a
production release and the multi-speaker numbers in the accompanying reports
are synthetic/cross-corpus proxy results, not real-driver evidence.

## Included raw checkpoints

The package includes the shared Stage 2 presence, Stage 3 regular KWS,
target-speaker KWS, compact command-time speaker verifier, and the Stage 4
wav2vec intent head plus fine-tuned encoder. It includes three Stage 1
separators so the Jetson comparison can reproduce the candidate study:

1. `stage1_spex_fsc_passenger_cochannel_large_ft1_2026-10-16` — larger
   train-only target/passenger co-channel adaptation.
2. `stage1_spex_fsc_passenger_cochannel_ft1_2026-10-16` — smaller adaptation.
3. `stage1_spex_dual_domain_rms010_ft1` — dual-domain baseline.

The primary balanced proxy candidate is the large Stage 1 model with the
post-wake compact speaker gate and intent confidence threshold 0.80. The
corresponding YAML is included. The no-gate YAMLs are included for ablations.

## Included deployment artifacts

The bundle also carries the previously exported ONNX runtime artifacts under
`onnx/`. Their filenames include their original export date and checkpoint
fix identifier; do not silently treat them as exports of the new large Stage
1 checkpoint. Use the raw PyTorch candidate for a new export if the Jetson
comparison is meant to test the 2026-10-16 candidate itself.

## Reproducibility

After copying the archive, extract it and verify `SHA256SUMS.txt`. The exact
pipeline YAML, checkpoint names, thresholds, and proxy evaluation commands are
in `configs/` and `reports/multispeaker_iteration_2026-10-16.md`.

Outer archive: `car_tse_jetson_multispeaker_models_2026-10-16_final.zip`

- Size: 885,336,960 bytes
- SHA-256: `B5AEA8EB32820DBC06EF49E8C40D8B9AB2EAEB46217E38C6168EB49B18F52753`

## Runtime caveats

- The wav2vec intent path still expects the Hugging Face
  `facebook/wav2vec2-base-960h` configuration/tokenizer assets unless the
  Jetson runner is pointed at a local equivalent.
- Measure Stage 1 + presence + KWS + compact verifier + intent together on
  the target Jetson; desktop GPU timings are not device timings.
- Keep all candidate directories separate. Do not overwrite the baseline
  checkpoint when testing the new separator.
