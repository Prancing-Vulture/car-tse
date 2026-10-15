# Dataset inventory for car-tse

This directory contains raw archives and extracted corpora selected for the
SpEx+-based in-car target-speaker extraction pipeline. The selection follows
the repository's four stages:

| Pipeline stage | Primary data | Role |
| --- | --- | --- |
| Target-speaker extraction | LibriSpeech + LibriMix generation | Clean enrollment/target speech and reproducible two-speaker mixtures |
| Cabin adaptation and presence | AISHELL-5 | Real four-channel in-car speech, near-field speaker references, TextGrid labels, and cabin-only noise |
| Acoustic augmentation | CAVEMOVE | Multichannel impulse responses and driving-condition noise, including speed, window, HVAC, and car-audio conditions |
| Wake-word detection | Speech Commands v2 | Public 16 kHz keyword corpus; application-specific wake recordings still belong in `custom_incar/wake/` |
| Intent classification | Fluent Speech Commands | 31 command intents with speaker-disjoint train/validation/test metadata |
| Auxiliary vehicle-noise robustness | Vehicle Interior Sound Dataset | 5,980 short vehicle-only clips; useful as noise-domain material, not as speech supervision |

## Source notes

- AISHELL-5: https://www.openslr.org/159/ (CC BY-SA 4.0). The training set
  is the primary real-cabin corpus; Dev/Eval1/Eval2 are held out evaluation
  material and `noise.tar.gz` is noise-only cabin audio. The full train,
  development, evaluation, and noise archives are present under
  `data/raw/aishell5`; extraction may take longer than the smaller corpora.
- CAVEMOVE: https://doi.org/10.5281/zenodo.13594243 (CC BY 4.0), with the API
  at https://github.com/SPL-FORTH-ICS/CAVEMOVE.
- LibriSpeech: https://www.openslr.org/12 (CC BY 4.0).
- LibriMix: https://github.com/JorisCos/LibriMix (MIT scripts). LibriMix is
  generated from LibriSpeech and WHAM; the complete default Libri2Mix build is
  hundreds of GB, so this workspace keeps the source corpora and generator
  instead of downloading the full expansion.
- Fluent Speech Commands mirror: https://doi.org/10.5281/zenodo.11106540
  (CC BY 4.0).
- Speech Commands v2: https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
  (CC BY; see the Speech Commands documentation for attribution).
- Vehicle Interior Sound Dataset: https://doi.org/10.5281/zenodo.5606504
  (CC BY 4.0). It contains no human speech and is retained only as auxiliary
  vehicle-noise data.
- ASR-CabNois: https://magichub.com/datasets/in-vehicle-noise-dataset/
  (CC BY-NC-ND 4.0; 5.08 hours, 44.1 kHz stereo, sign-in required for the
  2.47 GB download). Only the public sample is eligible for local inspection
  unless the user supplies an authenticated download.
- Nexdata 531-hour in-car noise: https://github.com/Nexdata-AI/531-Hours-In-Car-Noise-Data-by-Microphone-and-Mobile-Phone
  (commercial license). The public repository is sample/metadata only; the
  full corpus is not redistributed here.

## Layout

`raw/` holds downloaded archives and source repositories. Extracted datasets
keep their upstream top-level directory names where possible. `processed/` is
reserved for resampled audio, generated mixtures, and manifests; it is not
populated by the source downloads. `scripts/data/build_local_librimix.py`
creates a compact 8 kHz LibriSpeech-derived two-speaker bootstrap set in
`data/processed/Libri2Mix`; this is intentionally much smaller than the full
LibriMix expansion and is used to validate the training pipeline first.
`scripts/data/build_local_multichannel.py` derives paired four-channel
bootstrap mixtures from those mono mixtures while injecting a small amount of
AISHELL noise; the real AISHELL-5 microphone groups remain available for
separate cabin adaptation experiments.

The project config should point `data_root` at this directory after the
archives are extracted. Checksums and the exact files obtained are recorded in
`sources.json`.
