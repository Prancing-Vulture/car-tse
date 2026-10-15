# Five-step pipeline evaluation — 12 October 2026

## Short answer

The existing audio buffer now preserves the labeled command through Stage 2 window rejections in these replays. The complete pipeline still performs poorly: on the held-out cross-corpus proxy, it produced the expected final intent in 2 of 16 scenes (12.5%), while also producing 16 false wake events and 13 false actions in 224 seconds of simulated audio.

This is an engineering diagnostic, not a valid estimate of real same-speaker use. The wake clips are from Speech Commands and the commands are from Fluent Speech Commands; their speakers are not verified to match. The custom in-car wake, command, and playback folders are empty. Therefore the tests do not establish whether a real driver saying a wake phrase and then a command would achieve these rates.

## What was done against the five requested steps

### 1. Find and prepare realistic labeled continuous recordings

- Checked the configured custom in-car directories. `wake`, `commands`, and `playback` contain no recordings.
- No available manifest identifies a continuous same-speaker wake-then-command interaction with reviewed sample-level labels.
- There is also no dedicated long, target-absent in-car recording set with labeled passenger/media events, so a reliable false-actions-per-hour result cannot be measured.
- Used the repository's continuous-scene evaluator instead. It assembles labeled 14-second scenes with a target wake, a command, passenger wake/non-wake speech, and CAVEMOVE cabin noise. The wake and command are cross-corpus and may be different speakers.
- Scene labels include wake and command start/end times, expected intent, passenger events, SNR, and source recording IDs. These make the proxy useful for plumbing and debugging, but not production accuracy.

**Status:** real-world data requirement remains unmet. No synthetic speaker matching or generated speech was treated as real same-speaker evidence.

### 2. Check the exact audio delivered to Stage 4

- Replayed the actual Stage 1–4 checkpoint stack through the streaming runtime on CUDA, using `audioml_base`.
- Saved the exact two-second Stage 4 WAV input for every intent decision, plus scene audio and detailed per-window traces.
- Example from the held-out proxy: the labeled command spans 2.1–4.1 seconds and the saved Stage 4 input maps to 2.0–4.0 seconds. It contains 1.9 of the command's 2.0 seconds (95%). Stage 4 predicted `bring|socks`; the expected intent was `activate|lamp`.
- This supports that the fixed buffer is delivering command-time audio in this proxy. The 0.1-second tail is still outside the fixed two-second crop. That small crop mismatch is a possible improvement to test, but these results do not show it caused the wrong intent.
- The prior forced-rejection replay independently showed the same 2.0–4.0-second Stage 4 source span after an extra Stage 2 rejection, with sample-identical Stage 4 audio to its control.

Artifacts are under `reports/artifacts/continuous_pipeline_validation_proxy_8scene_20261012/` and `reports/artifacts/continuous_pipeline_heldout_proxy_8scene_gate035_20261012/`.

### 3. Measure stages and inspect where each scene fails

The integrated replay records each stage's decisions on the same scene. Existing held-out component reports give additional context, but they are not a substitute for matched same-speaker stage inputs:

- **Stage 1 (speaker extraction):** prior audits show transfer performance varies by domain: strong improvement on synthetic CAVEMOVE mixtures, but poor absolute SI-SDR on an AISHELL-5 real-cabin diagnostic. This points to a domain-transfer concern, not proof that extraction caused the current wrong intents. The current cross-corpus scene cannot resolve this because its wake and command speakers differ.
- **Stage 2 (speech/presence gate):** its held-out chunk F1 is 0.810 at threshold 0.40, with 0.907 recall but 0.333 false-accept rate. At 0.50, F1 is 0.786 and false-accept rate is 0.267. The integrated replay also showed that a 0.90 cutoff passed very few target wakes.
- **Stage 3 (wake detector):** a prior held-out TSE-candidate test reported F1 0.667 at threshold 0.29, with precision 0.525 and recall 0.914. That checkpoint/report is not the same as every model used in this replay, so it is context rather than a direct score for the current stack. In the current held-out replay, false wake events were frequent at the looser gate setting.
- **Stage 4 (intent model):** an existing clean FSC test reported 99.3% top-1 accuracy. That is on clean isolated commands, while this pipeline supplies two-second extracted/noisy crops. On the held-out proxy, only 2 of 14 scenes with a target wake had the correct final intent (14.3%). Several saved Stage 4 inputs contained most of the command but were classified incorrectly, so clean FSC accuracy does not transfer to the current pipeline conditions.

**Interpretation:** more than one stage is weak under the combined conditions. Stage 2's restrictive threshold suppresses target traffic; loosening it lets more target and non-target traffic through. Stage 3 contributes false wakes. Stage 4 often predicts the wrong intent after it receives a command-time crop. Stage 1 remains unisolated for the command speaker. The evidence does not justify naming a single confirmed root cause.

### 4. Run the complete cascade and report end-to-end outcomes

All five runs below used 16 scenes (8 target wake recordings × two cabin SNRs, −5 and 0 dB), 224 seconds total, and a 0.5-second hop. Validation thresholds were compared on validation manifests; the selected less-permissive tied point (command threshold 0.35) was then frozen for a single held-out proxy run.

| Split and settings | Wake recall | Correct final intent | False wakes | False actions |
|---|---:|---:|---:|---:|
| Validation: overall gate 0.50, command gate 0.20 | 8/16 (50.0%) | 2/16 (12.5%) | 7 | 5 |
| Validation: overall gate 0.50, command gate 0.35 | 8/16 (50.0%) | 2/16 (12.5%) | 7 | 5 |
| Validation: overall gate 0.50, command gate 0.50 | 8/16 (50.0%) | 0/16 (0%) | 6 | 4 |
| Validation: overall and command gates 0.90 | 1/16 (6.25%) | 0/16 (0%) | 0 | 0 |
| Held-out proxy: overall gate 0.50, command gate 0.35 | 14/16 (87.5%) | 2/16 (12.5%) | 16 | 13 |

The held-out result's high wake recall came with many false wake/action events, and only two correct intents. Thus, it does not meet the requested greater-than-90% intent-mapping target. False-event counts are shown directly; converting them to hourly rates would be misleading from only 224 seconds of simulated scenes. The passenger and background portions in these scenes do not replace long target-absent sessions.

### 5. Tune on validation, freeze, then evaluate held-out data

- Compared command-gate thresholds 0.20, 0.35, and 0.50 at fixed overall gate 0.50 and KWS threshold 0.29 on the validation scenes.
- Compared 0.90 overall/command gates as a restrictive operating point.
- Chose 0.35 only as the less-permissive of two tied validation settings; this is a diagnostic choice, **not a production recommendation**.
- Ran it once on the separate test manifests. The proxy's held-out E2E result was 12.5%, with 16 false wakes and 13 false actions.
- Each complete-pipeline run logged metrics, config, scene outcomes, and window traces to W&B. The report and audio artifacts were uploaded.

## W&B runs

- [Validation, command gate 0.20](https://wandb.ai/narainharreesh-vit/car-tse/runs/tyuhpcn7)
- [Validation, command gate 0.35](https://wandb.ai/narainharreesh-vit/car-tse/runs/cffnzbix)
- [Validation, command gate 0.50](https://wandb.ai/narainharreesh-vit/car-tse/runs/rofikggk)
- [Validation, overall/command gates 0.90](https://wandb.ai/narainharreesh-vit/car-tse/runs/wjvtcx14)
- [Held-out proxy, frozen command gate 0.35](https://wandb.ai/narainharreesh-vit/car-tse/runs/jr040fah)

## Conclusion and next action

The buffer is no longer the leading explanation for the wrong intents in these tested proxy scenes: Stage 4 received 95% of the labeled command in the inspected example and still selected the wrong intent. But this does not prove that Stage 4 alone is the root cause; the scenes use different speakers for wake and command, and other scenes fail earlier at the gates or detector.

Do not promote the 0.35 threshold or claim production readiness from these results. The next decisive evaluation requires reviewed, uninterrupted recordings from the same speaker: independent enrollment, wake phrase, and command in one stream, plus target-absent passenger/media/noise sessions. Then replay the fixed stack, listen to source-microphone and exact Stage 4 crops, and score by speaker/session. The current data cannot support that test, so collecting those recordings is the concrete blocker to a conclusive same-speaker pipeline diagnosis.

No model weights, SpEx code, or runtime code were changed for this evaluation. The new outputs are diagnostic reports, saved audio, and W&B runs.
