# External model provenance and safe loading

Evaluation date: 2026-10-19/20 (Asia/Kolkata)

## Trained SpEx+ baseline

- Source: [niobures/SpEx-Plus](https://huggingface.co/niobures/SpEx-Plus), pinned revision `2cab2f77d560734f83a9cfead45d7f128b804f0f`.
- File: `E:\car-tse-data\models\tier1_external\spexplus_wsj0_2mix_8k\last_best_checkpoint.pt`.
- SHA-256: `5ffcf87a45f46ece3fa43db5b4d7f9779a73392933fed0b563f0ead9bd9b492f` (matched the published file hash).
- Checkpoint metadata: epoch 96, best validation loss about -18.84; SpEx+ target extraction trained on WSJ0-2mix at 8 kHz. It is the no-CAVEMOVE-fine-tuning comparison.
- Loading: PyTorch restricted `weights_only=True`; deterministic module-name mapping only; strict state-dict load matched all keys; a finite forward pass returned 48,000-sample output for 3 seconds at 8 kHz. Evaluation resamples input from 16 kHz to 8 kHz and output back to 16 kHz.
- Safety: Hugging Face file status was reported safe and the published SHA-256 matched. No repository Python was executed for this checkpoint.

## SpeakerBeam baseline

- Source: [ESPnet LibriMix TD-SpeakerBeam](https://huggingface.co/espnet/Wangyou_Zhang_librimix_train_enh_tse_td_speakerbeam_raw), pinned revision `db03c34c0b8c26a87c0d6cac41229891c34b4a02`.
- File: `E:\car-tse-data\models\tier1_external\espnet_speakerbeam\99epoch.pth`.
- SHA-256: `7e1d6831d0df187ae8b23ed4d6804d07d0ec314336266e99f7da41a2f8ad055f` (matched published metadata).
- Loading: `weights_only=True`; strict load into the configuration-matched ESPnet architecture; inference dependencies installed under the isolated experiment runtime.
- License shown on the model card: CC-BY-4.0. The checkpoint is a target-extraction model trained on LibriMix.

## Rejected online candidate

- The same SpEx+ repository's `SpExPlus_LibriSpeech.zip` archive was pinned to revision `2cab2f77d560734f83a9cfead45d7f128b804f0f`; archive SHA-256 `9cdf093b9cb31e6a654ef2fffe0aa08931881b976a661629211bcd939ee879f5`.
- Its included `init checkpoint.pt.tar` was epoch 0 and behaved as an untrained initialization (pilot SI-SDR was about -39 dB). It was excluded from all main comparisons. Only its source layout was inspected; it was not executed.

## Selected local model

- Current selected Stage-1 checkpoint: `checkpoints/stage1_spex_fsc_passenger_cochannel_ft1_2026-10-16/best.pt`.
- SHA-256: `13ec7c58870feb975efb266e4064bf58575dbc6120f787a77a49c3bc4edda3ac`.
- Four-channel frontend and 16 kHz transfer-SpEx+; selected from the latest local runtime bundle as the current best.

## Selected FT6 pipeline graphs

- The continuous pipeline ablations use the already-present `car_tse_jetson_onnx_ft6_2026-10-19_v3/models/onnx/ft6/` bundle for presence, generic KWS, target-conditioned KWS, compact verification, and intent. The per-component output hashes and source-checkpoint hashes are in that directory's `manifest.json`; the exact hashes used by each run are repeated in its JSON `runtime_model_artifacts` field.
- Output SHA-256 values: presence `3cd06c0bac4bdcf157e4e5aedf83dc2f963656bc9b98b933b338afefa241f223`; generic KWS `bf1dfde9a4a4a421b5bd2dfea759693fbab009c817f15ab0c22a4007c6a8f004`; target KWS `0b23bd0921ddd7d2723bccaa5eea2e3b2bee0172713a1b8a7981b46afcf275ab`; compact verifier `dc667e279ae7abc6133634082edb5f739adad9ffdd42bc98ae97daac40fe899d`; intent `e544799007166ccafe904eb8e5219c868d191654a0142ec90b56e5b58f83f78f`.
- Stage 1 is the selected local PyTorch checkpoint above. Its hash matches the FT6 bundle manifest. The bundle ONNX graphs were used through ONNX Runtime CPU; an inference smoke pass checked fixed input/output contracts and finite outputs.
- The bundle's claimed large PyTorch intent fallback was absent from the checkout. The evaluation used its manifest-matched current intent graph rather than substituting an older local PyTorch checkpoint. ONNX Runtime 1.31.0 was installed into `audioml_base` without dependency changes; PyTorch remained `2.12.0+cu132` and CUDA was available for Stage 1.

## Matched local no-cabin-fine-tuning control

- The exact local transfer-SpEx+ initialization used to construct the selected model is `E:\car-tse-data\models\car_tse_spexplus_from_stage1.pt`, SHA-256 `62826ae27c1d83d077aa98bc050117f8b39f7dc22a3d71fd496d109e57cd6449`.
- It was loaded with `weights_only=True`; the report copy is the tensor-only `tse/base_spex_safe_state_dict.pt`. The selected checkpoint differs from this initialization only in 18 batch-normalization running-statistic/counter tensors. The full 3,000-row paired result and checks are in `tse_local_pretrained_control/run_manifest.json`.
