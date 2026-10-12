# ANIR noise-only target-absent diagnostic

Date: 2026-10-11  
Status: diagnostic; no training promotion

## Setup

The retained reference pipeline was evaluated on all 42 ANIR V-class noise
conditions, using microphones 0-3 from each synchronized condition group. Each
group contributes 30 seconds after resampling from 48 kHz to 16 kHz, for 1,260
seconds (0.35 hours) total. The fixed LibriSpeech enrollment is absent from the
noise recordings.

Runtime settings match the retained reference safety configuration:

- Stage 1 dual-domain SpEx+ with the cabin frontend
- framewise presence threshold 0.15
- pre-wake speaker gate 0.24
- target-speaker Marvin KWS threshold 0.50
- two-window/two-hit smoothing with two-call refractory period

## Result

| Path | Events | Rate |
|---|---:|---:|
| Full gated path | 0 | 0.0/hour |
| KWS-only diagnostic | 3 | 8.57/hour |

The KWS-only positives occurred in conditions `04_00`, `09_00`, and `10_00`,
one event in each. Their maximum target-KWS scores were 0.573, 0.559, and
0.530; the full path rejected them because the presence/speaker gates did not
pass.

## Interpretation

This is useful evidence that the current gates suppress pure cabin-noise
triggers on this archive. It does not demonstrate passenger rejection, wake
recall, command accuracy, or production false-accept performance: the audio
contains no speech and the archive's microphone geometry differs from the
project's four-channel capture contract. The result supports using ANIR for
noise-only hard-negative testing and potentially for augmentation, but not
blindly adding it to training. Any augmentation study should compare the
existing CAVEMOVE baseline, freeze a held-out condition split, and report both
SI-SDRi and full-pipeline false-action/hour.

## Reproduction

```powershell
python scripts/data/build_anir_noise_manifest.py `
  --root D:\datasets\anir\v_class_noise_files `
  --output D:\datasets\anir\anir_v_class_noise_manifest.json

python scripts/eval/eval_anir_noise_target_absent.py `
  --noise-root D:\datasets\anir\v_class_noise_files `
  --max-streams 42 --max-seconds 1500 `
  --output logs/anir_noise_target_absent_all_conditions_2026-10-11.json
```
