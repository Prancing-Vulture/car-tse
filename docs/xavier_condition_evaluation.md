# Jetson Xavier condition evaluation

This bundle supports repeatable condition-wise replay on the Xavier AGX. It
does not synthesize cabin conditions. Use real four-channel recordings made on
the target microphone array for the engine-noise, passenger-speech,
air-conditioning, and music-playback conditions.

The included `stream_rec/stream_trimmed_for_alignment.wav` is the requested
demo recording. It is stereo, so it is not a four-microphone condition trial
and must not be used to claim passenger rejection or per-condition accuracy.
Generated PNGs are local outputs and are not committed.

## 1. Get the branch and activate the Xavier environment

From the cloned repo on Xavier:

```bash
cd ~/Desktop/car-tse
git fetch origin
git checkout codex/jetson-condition-eval
git pull --ff-only
source .venv/bin/activate
```

Keep the existing Xavier Torch/CUDA build in this venv. The capture helper
needs `sounddevice`; the benchmark uses the repo's PyTorch, NumPy, and
SoundFile dependencies.

Create or update `configs/paths.yaml` with the Xavier checkpoint and data
locations. Confirm CUDA and checkpoint paths before starting:

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda)"
ls checkpoints/car_tse_spexplus_from_stage1.pt
```

The evaluation config also expects the Stage 1, 2, 3, and 4 checkpoints named
in `configs/eval_end_to_end_cavemove.yaml` under the configured checkpoint
root. The script records checkpoint hashes in its JSON output.

## Reproduce the stream recording and its figures

The branch contains the audio at `stream_rec/stream_trimmed_for_alignment.wav`.
Install Matplotlib in the existing venv if it is not already available:

```bash
python -m pip install matplotlib
```

Run the paced replay to regenerate both figures and save per-window metrics:

```bash
mkdir -p logs
python scripts/eval/reproduce_stream_recording.py \
  --audio stream_rec/stream_trimmed_for_alignment.wav \
  --enrollment data/enrollment.wav \
  --config configs/eval_end_to_end_cavemove.yaml \
  --paths configs/paths.yaml \
  --base-checkpoint checkpoints/car_tse_spexplus_from_stage1.pt \
  --device cuda \
  --command-delay-calls 5 \
  --command-timeout-calls 12 \
  --kws-threshold 0.4 \
  --mode all \
  --output-dir stream_rec/reproduced \
  --metrics-output logs/stream_recording_metrics.json \
  --windows-csv logs/stream_recording_windows.csv
```

This creates `stream_rec/reproduced/tse_speaker_and_removed_component.png`
and `stream_rec/reproduced/stagewise_processing_paper_style.png`. To rerun the
audio and write metrics without generating images, use the same command with
`--mode metrics`. The sample is stereo; the script downmixes it and repeats
the signal to four model inputs. It demonstrates the replay and plotting path,
not four-microphone spatial separation or condition-wise performance.

The recorder needs SoundDevice and PortAudio. Install them in the existing
Xavier environment without replacing its CUDA-enabled PyTorch:

```bash
sudo apt-get install -y libportaudio2
python -m pip install sounddevice
```

## 2. Record the condition trials

List the input devices and channel counts:

```bash
python scripts/eval/capture_xavier_condition.py --list-devices
```

For a device exposing eight raw channels where microphones are 0, 2, 4, and 6,
record a trial with:

```bash
python scripts/eval/capture_xavier_condition.py \
  --condition engine_noise \
  --duration 60 \
  --input-channels 8 \
  --channel-indices 0,2,4,6 \
  --device 0
```

Repeat with `passenger_speech`, `air_conditioning`, and `music_playback` after
setting up each real condition. For a four-channel device, use
`--input-channels 4 --channel-indices 0,1,2,3`. Record target-command trials
and target-absent trials for every condition; for the passenger condition,
include passenger-only speech and passenger wake-word negatives. Keep the same
target speaker enrollment file for matched trials. The WAV and metadata are
written under `data/raw/jetson_conditions/`, which is ignored by Git.

The recorder captures audio only. Record the condition settings and fill in
the event labels in the manifest after each session. For music, include a
`playback_ref_wav` entry when the aligned source/reference track is available.

## 3. Annotate the trials

Copy the template and replace its example filenames with the recorded WAV paths:

```bash
cp configs/xavier_conditions.example.jsonl configs/xavier_conditions.jsonl
nano configs/xavier_conditions.jsonl
```

Keep one row per trial. Set `expected_wake` to `true` for target wake trials and
`false` for target-absent trials. Set `expected_action` to the exact action label
expected from the pipeline, or `null` for a trial with no target command.
`expected_wake_end_s` and `expected_command_end_s` are seconds from the start of
the recording; fill them in to obtain wake and command decision latency from
the annotated utterance endings. `expected_action` must exactly match the
intent label map used by this checkout (for example, `activate|lights` if that
is the ground-truth command). Use `condition_details` to preserve the actual
engine speed, HVAC level, music track/volume, passenger-speaker setup, vehicle
state, and microphone channel selection. For passenger-only recordings, use
`expected_wake: false` even if the passenger says “Marvin.”

## 4. Run paced end-to-end condition replay

The paced mode feeds the saved four-channel signal at its original real-time
rate through `StreamingRunner` and measures the full gated pipeline call on the
Xavier GPU. It reports per-window p50/p95/p99 latency, hop deadline misses,
real-time factor, stream lag and dropped audio, wake/intent outputs, and
annotation-based outcomes when labels are present.

```bash
mkdir -p logs
(
  tegrastats --interval 1000 > logs/tegrastats_xavier.txt &
  TEGRASTATS_PID=$!
  trap 'kill "$TEGRASTATS_PID" 2>/dev/null || true' EXIT
  python scripts/eval/benchmark_xavier_conditions.py \
    --manifest configs/xavier_conditions.jsonl \
    --config configs/eval_end_to_end_cavemove.yaml \
    --paths configs/paths.yaml \
    --base-checkpoint checkpoints/car_tse_spexplus_from_stage1.pt \
    --device cuda \
    --replay-mode paced \
    --output logs/xavier_condition_latency.json \
    --windows-csv logs/xavier_condition_windows.csv
)
```

The JSON contains condition-wise latency percentiles, deadline miss rates,
false wakes per target-absent hour, target wake recall, exact intent accuracy,
latency from annotated phrase endings, environment details, and hashes of the
configurations, recordings, and checkpoints. The CSV retains one row per
inference window. Use `--replay-mode fast` only when measuring unpaced offline
throughput; do not label that result as a real-time replay.

For isolated Stage 1–4 model timings on one condition recording, run the
existing component benchmark once per four-channel recording:

```bash
python scripts/eval/benchmark_stagewise_latency.py \
  --config configs/eval_end_to_end_cavemove.yaml \
  --paths configs/paths.yaml \
  --mixture data/raw/jetson_conditions/engine_noise/REPLACE_WITH_TRIAL.wav \
  --enrollment data/enrollment.wav \
  --base-checkpoint checkpoints/car_tse_spexplus_from_stage1.pt \
  --device cuda --warmup 3 --repeats 15 \
  --output logs/stagewise_engine_noise.json
```

The isolated stage sum and the end-to-end per-hop measurement answer different
questions. Report both with their measurement scopes; do not call the isolated
stage sum a measured live pipeline call.

## Publication reporting notes

- Report model/config hashes, Xavier model and CUDA versions, sample rate,
  microphone channel selection, hop/window, power mode, and temperature/power
  telemetry alongside latency.
- Keep condition trials and negative exposure durations. False-wake/hour is
  only meaningful when target-absent exposure is recorded and annotated.
- Preserve each recording’s event annotations and raw per-window CSV. Avoid
  pooling condition results before reporting the per-condition values.
- Repeat each condition across independent trials and target speakers; one
  positive and one negative recording are only a pipeline check, not a stable
  publication estimate.
- For source-separation quality (such as SI-SDR improvement), also collect a
  time-aligned clean target reference. A noisy mixture by itself cannot verify
  the extracted target or prove that the residual is only noise.
- The committed stereo demo clip is for illustration and code provenance; it
  is not evidence for the four requested acoustic conditions.
