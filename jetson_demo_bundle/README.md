# Jetson one-file audio demo

Copy the contents of this folder into the root of the cloned `car-tse` repo,
merging the `scripts/` directory. This adds a small inference runner for the
already trained four-stage model; it does not require the CAVEMOVE or LibriMix
manifests.

## Inputs

- A speech mixture WAV. If it is mono, provide the multichannel cabin-noise WAV
  too; the script combines them at the requested SNR and selects the model's
  configured microphone channels. A pre-mixed four-channel WAV can be used
  without a separate noise file.
- A target speaker enrollment WAV. The script downmixes it to mono and
  resamples it to 16 kHz if needed.
- The SpEx+ base checkpoint at `checkpoints/car_tse_spexplus_from_stage1.pt`.
- The four stage checkpoints under `checkpoints/` with the names configured in
  `configs/eval_end_to_end_cavemove.yaml`.

The selected Stage 1 model uses a four-microphone cabin frontend. The provided
demo assets contain a mono speech mixture, a mono enrollment clip, and an
eight-channel CAVEMOVE noise recording; pass the noise file and the script
constructs a four-channel input. This is a proxy scene: its speech is from
Speech Commands, and the cabin recording contains noise rather than speech.

## Run

From the cloned repository root, with the Jetson virtual environment active:

```bash
export PYTHONPATH="$PWD/src:$PWD/third_party/speaker_extraction_SpEx${PYTHONPATH:+:$PYTHONPATH}"
python scripts/eval/process_audio_demo.py \
  --mixture tmp/cavemove_demo_bundle/data/cavemove_mixture.wav \
  --cabin-noise tmp/cavemove_demo_bundle/data/cavemove_cabin_noise.wav \
  --cabin-snr-db 0 \
  --enrollment tmp/cavemove_demo_bundle/data/enrollment.wav \
  --base-checkpoint /absolute/path/to/checkpoints/car_tse_spexplus_from_stage1.pt \
  --paths configs/paths.yaml \
  --device cuda
```

The script writes `logs/jetson_demo/extracted_speech.wav` and
`logs/jetson_demo/result.json`. The WAV contains the Stage 1 speaker-extracted
signal; the JSON reports the latest pipeline detection/intent result. A single
recording can end before the pipeline has observed both a wake word and a
following command, so `intent_ready` can be false even when audio extraction
worked.
