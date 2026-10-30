# In-car dataset audit — 2026-10-10

## Decision

No newly discovered public corpus was added to the English Marvin production
training path during this pass. The limiting requirement is not simply
vehicle noise; it is a permitted, speaker-consistent continuous sequence with
the same driver’s enrollment, Marvin wake word, command, passenger speech,
passenger Marvin, and no-command intervals. None of the audited public
sources provides that complete protocol in English.

## Existing project data

- CAVEMOVE supplies multichannel vehicle/cabin noise and is useful for
  synthetic acoustic mixtures and TSE regression.
- AISHELL-5 supplies real four-channel cabin speech and is useful for
  target-absent/noise-domain diagnostics. The current Eval2 exposure is
  negative-only for this project; it does not contain a driver enrollment,
  Marvin, and command sequence.
- Speech Commands provides Marvin positives and non-wake negatives, but its
  clips are isolated and are not continuous in-car driver sessions.
- Fluent Speech Commands supplies command/intent labels, but the speaker and
  wake-word recordings are not naturally linked to the same driver session.
- VISC and the downloaded vehicle-noise material are acoustic beds, not
  consented end-to-end driver/passenger sessions.

These sources remain appropriate for component training and cross-corpus
proxy tests, but their results are not real-driver reliability evidence.

## Public candidates reviewed

### ICSRC 2022 intelligent-cockpit corpus

Source: <https://iscslp2022-icsrc.org/>

The official challenge page describes 20 hours of real-world in-car Mandarin
speech recorded with a Hi-Fi microphone under different driving conditions,
including changing speed, open windows, music, and passenger-seat speakers.
It has 10-hour Eval and 10-hour Test partitions with 50 speakers in each, and
an “Others” category containing conversation/non-command material.

The data agreement on the same page limits use to non-commercial education,
research, and technology development; prohibits copying, redistribution, or
transmission outside the challenge; and explicitly prohibits using the Test
set for training or fine-tuning. The corpus is Mandarin and does not provide
the project’s English Marvin wake-word/enrollment/command continuity. It was
therefore not downloaded or used in a production-training run. If it is used
later for academic acoustic-domain analysis, the agreement must be accepted
and recorded separately, Eval must remain evaluation-only where required, and
it must not be presented as the project’s real-driver English benchmark.

### CI-AVSR

Source: <https://github.com/HLTCHKUST/CI-AVSR>

The repository describes a Cantonese in-car audio-visual command-recognition
dataset with clean, noisy, and out-of-domain noisy splits and points to a
processed Kaggle download. The repository is marked CC0, but that repository
license does not by itself establish the redistribution or commercial terms
for every processed audio/video asset. The language, audio-visual format, and
lack of an English Marvin/enrollment sequence make it unsuitable as a direct
production-data substitute. It remains a possible future acoustic/ASR
generalization study after the asset-level terms are verified.

### ASR-SCCabSC / in-vehicle scripted Mandarin speech

Source: <https://magichub.com/datasets/mandarin-chinese-scripted-speech-corpus-in-vehicle-scene/>

The listing describes a small Mandarin in-vehicle scripted-speech corpus with
dual-microphone recordings and a CC BY-NC-ND license, with account access
needed for download. Non-commercial/no-derivatives terms and the Mandarin
language mismatch make it inappropriate for the current production path
without explicit permission and a separate license review.

## Download/usage policy applied

1. Do not download a dataset merely because a public link exists.
2. Record language, channel geometry, speaker/session structure, license,
   redistribution terms, and whether training is permitted.
3. Keep challenge Test partitions out of training when the source rules say
   so.
4. Keep research-only or non-commercial data out of a production/commercial
   model unless permission is documented.
5. Do not call isolated clips, noise beds, or cross-corpus mixtures
   “real-driver evidence.”

## Next data action

Use the repository’s recorder and annotation templates to collect a small,
consented pilot rather than adding another language-mismatched corpus:

- three drivers if possible;
- at least two sessions per driver;
- one driver group for tuning and held-out driver(s) for test;
- 16 kHz, four-channel PCM matching the runtime input;
- enrollment, driver Marvin, driver command, passenger Marvin, passenger
  speech, no-command, noise-only, and playback/echo intervals;
- condition metadata for parked/idle, engine/HVAC, road motion, window/wind,
  music/playback, and passenger loudness;
- exact event timestamps and expected intent labels;
- written permission for the intended research and product use.

The validator currently reports zero session directories, so the pilot is the
next evidence gate rather than another unconstrained training sweep.
