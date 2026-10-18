# English spatial front-end ablation

Date: 2026-10-13  
Status: diagnostic proxy only; no production claim

## Purpose

This study checks whether an English, driver-seat spatial front end is a useful
next direction for the current wake-to-intent cascade. It compares:

1. `tse_first`: the current learned cabin front end followed by SpEx+;
2. `hybrid`: an offline CAVEMOVE-calibrated driver delay-and-sum beamformer
   followed by SpEx+ without the learned cabin front end;
3. `zone_only`: the same beamformer supplied directly to presence, target KWS,
   speaker verification and intent.

The scenes contain English Marvin wake clips, English Fluent Speech Commands,
passenger Marvin/non-wake clips, CAVEMOVE multichannel impulse responses, and
real CAVEMOVE vehicle-noise files. The driver microphone delays are fixed from
the selected CAVEMOVE impulse response and are not a production microphone
geometry calibration.

## Data contract limitation

This is not a same-driver evaluation. Speech Commands supplies the wake and
enrollment, while Fluent Speech Commands supplies the command from an unrelated
speaker. Consequently, the command-side speaker gate can reject a command even
when the wake event is correct. The report therefore records
`same_driver_wake_to_command=false`, and the results must not be described as
real-driver accuracy or as a valid release benchmark.

## Stage-1 adaptation

The English spatial manifest contains 384 speaker-disjoint training examples
and 96 validation examples (201/50 LibriSpeech speakers, zero overlap). Only
the cabin front end was trained; SpEx+ and the rest of the pipeline were frozen.
The best checkpoint was selected at epoch 1 with validation SI-SDRi **3.54 dB**.
Later epochs regressed to 2.46 and 3.10 dB, so the candidate was not promoted.

Checkpoint: `checkpoints/stage1_spex_english_spatial_cavemove_ft1/best.pt`  
Config: `configs/stage1_spex_english_spatial_cavemove_ft1.yaml`

## Full-pipeline result

The run used 8 scene identities × three cabin SNRs (`-5`, `0`, `+5 dB`),
passenger SNR `-5 dB`, 0.5-second hops, and the current strict thresholds. It
covered 24 scenes / 336 seconds (0.0933 hours) per variant.

| Variant | Wake recall | Intent accuracy | False wakes/h | False actions/h | Mean wake latency |
|---|---:|---:|---:|---:|---:|
| Learned `tse_first` | 12.5% | 0.0% | 0.0 | 0.0 | 2.0 s |
| `hybrid` beamformer + SpEx+ | 4.17% | 0.0% | 0.0 | 0.0 | 2.0 s |
| `zone_only` beamformer | 0.0% | 0.0% | 0.0 | 0.0 | n/a |

The matched wake/enrollment control used the exact wake clip as enrollment and
gave the same qualitative result: `tse_first` 12.5% wake recall,
`hybrid` 4.17%, `zone_only` 0%, and 0% intent for all variants. This control
removes wake/enrollment variation but **does not** remove the wake-to-command
speaker mismatch.

Artifacts:

- `scripts/eval/eval_english_spatial_zone_ablation.py`
- `scripts/data/build_english_spatial_stage1_manifest.py`
- `configs/eval_english_spatial_zone_ablation_stage1_spatial_ft1.yaml`
- `logs/english_spatial_zone_ablation_stage1_spatial_ft1_2026-10-13.json`
- `logs/english_spatial_zone_ablation_stage1_spatial_ft1_samewake_2026-10-13_retry.json`
- `reports/artifacts/english_spatial_zone_ablation_stage1_spatial_ft1_2026-10-13/`

## Decision

Do not promote the adapted Stage-1 checkpoint or change production thresholds
from this result. The experiment exposes two issues to fix before another model
comparison:

1. collect or record a genuine English same-driver sequence containing
   enrollment, Marvin, command, passenger speech and playback/no-command
   periods; and
2. keep the proxy's speaker-consistency fields explicit so cross-corpus results
   cannot be mistaken for end-to-end driver performance.

Once same-driver data exists, tune thresholds only on the tuning split, freeze
them, and compare the zone-first and TSE-first paths at matched false-accept
rates. Until then, more Stage-1 training would optimize around an invalid
speaker-identity contract.
