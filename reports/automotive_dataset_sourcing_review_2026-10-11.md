# Automotive dataset sourcing review

Date: 2026-10-11

## Decision summary

The most useful newly verified resource is the free ANIR V-class impulse-
response package. It has been downloaded to `D:\datasets\anir` and verified
with SHA-256. It is suitable for future spatial/cabin augmentation, but it is
not a speech, wake-word, passenger, or end-to-end evaluation corpus.

The most promising speech resource found is iCarB-Voice, but it is restricted:
the Zenodo record reports 35,460 samples from 197 drivers, two in-car
microphones, and controlled external/internal-noise conditions, while the
files are 121.7 GB and access requires an end-user license agreement for
non-commercial research. It cannot be downloaded or used without an eligible
institutional access request.

## Resources reviewed

| Resource | Evidence and access | Decision |
|---|---|---|
| ANIR Corpus | Official download offers free 48 kHz automotive background noise and impulse responses; the V-class IR archive is 15 MB. | Downloaded IRs only; augmentation candidate. |
| iCarB-Voice | 197 subjects, driver-seat recordings with roof/dashboard microphones and simulated/natural noise; restricted Zenodo files, 121.7 GB, non-commercial EULA. | Request access if the project has an eligible academic signatory; do not treat as available now. |
| ICMC-ASR / AISHELL-5 | Real multichannel in-car speech/noise, but Mandarin and no English Marvin same-driver protocol. | Useful acoustic/ASR diagnostics, not direct wake/intent evidence. |
| MagicHub / FutureBee / OORT commercial corpora | English automotive command data is advertised, but access and redistribution require a commercial request or are not openly licensed on the public page. | Do not download or train without a project license. |
| CI-AVSR | Cantonese in-car commands with simulated automotive noises; language and wake protocol do not match this project. | Not added to the English Marvin path. |
| ReMASC | English genuine/source/replayed voice-command corpus; 50 subjects, 49 vehicle commands, quiet/moving vehicle conditions, and several microphone arrays including a 7-channel 16-kHz array. The repository states the corpus is free for academic and commercial use; the complete set is about 92.3 GB and requires a free IEEE DataPort account. | A 10.9 MB public sample was downloaded to `D:\datasets\remasc`; retain as supplemental replay/array stress material, not as same-driver Marvin evidence. |

## ANIR artifact

- Archive: `D:\datasets\anir\v_class_impulse_responses.zip`
- SHA-256: `934F9B1F026DE9ADDD77FBB0EB5190169CF58AB7B18ADDDB6D83F6CF702148D1`
- Extracted: 682 `.fir` files
- Format: little-endian int32 sample count followed by float32 coefficients
- Sample rate: 48 kHz
- Provenance manifest: `data/manifests/external/anir_v_class_impulse_responses.json`

The archive is intentionally not wired into training yet. Channel geometry,
sample-rate conversion, and loudspeaker/microphone mapping must be specified
before it can be used. A future experiment should render the same clean target
speech through several ANIR microphone responses, add measured cabin noise,
and compare Stage 1 SI-SDRi and downstream false-action/hour against the
existing CAVEMOVE augmentation. It must remain a training/augmentation study,
not a substitute for real-driver evaluation.

The V-class background-noise archive is also complete:

- Archive: `D:\datasets\anir\v_class_noise_files.zip`
- SHA-256: `8CC71CA50F89909D475798761669ABEA788C6C2CA5F1E022480EB63F5022A441`
- Extracted: 1,302 WAV files across 42 conditions and 31 microphone IDs
- Format: 48 kHz WAV; filenames follow the official `con_*_mic_*.wav` scheme
- External manifest: `D:\datasets\anir\anir_v_class_noise_manifest.json`

The files remain outside the repository and are not yet part of training.

## Consequence for the next step

No online public dataset found in this review satisfies all of the required
conditions simultaneously: English “Marvin,” the same driver's enrollment and
command, passenger “Marvin,” no-command intervals, and a continuous matching
microphone capture. The highest-value missing artifact remains the consented
pilot set. ANIR can improve acoustic diversity after the rendering path is
validated; iCarB-Voice may improve speaker-conditioned training only after
access approval and a license review. ReMASC can add English moving-vehicle
command and replay/array stress tests, but its task is replay-attack defense
and its sample metadata does not provide the continuous enrollment → Marvin →
command → passenger/no-command protocol needed for production acceptance.

## ReMASC sample audit

The public sample archive was downloaded and SHA-256 verified:

- Archive: `D:\datasets\remasc\sample_v2.zip`
- Size: 10,867,475 bytes
- SHA-256: `CEA61DBFDD35874C64E2475BF8A9CC98363196509E32C43771691F09795643D8`
- Extracted files: 12 WAV files plus `meta.csv` and `readme.txt`
- Observed formats: 16/44.1 kHz and 1/2/4/6/7 channels
- Vehicle example: genuine 2-channel and replayed 6-channel recording for one
  vehicle-environment speaker; this is not the project’s four-channel contract

No ReMASC file was added to model training or counted as real-driver
end-to-end evidence. The provenance record is
`data/manifests/external/remasc_sample.json`.
