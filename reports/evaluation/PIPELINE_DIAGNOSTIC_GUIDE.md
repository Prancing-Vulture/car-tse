# What the pipeline tests show

## Simple summary

The wake detector catches more wake clips at a lower cutoff, but also mistakes more other audio for the wake word. The continuous check still produced no correct wake-to-command action, and its wake and command recordings were not speaker-matched. It is therefore a debugging proxy, not a valid estimate of same-speaker pipeline performance. The sampler-ratio check did not improve Stage 3. We need a speaker-matched continuous test before claiming a production root cause or promoting a model change.

## Stage 3 clip test

The held-out 384-clip test compared the earlier model and the SpEx-output-trained candidate at the same 0.29 cutoff:

| Model | Wake clips caught | Non-wake clips falsely accepted | F1 |
|---|---:|---:|---:|
| Earlier model | 58/128 | 43/256 | 0.507 |
| SpEx-output-trained candidate | 117/128 | 106/256 | 0.667 |

The candidate misses fewer real wake clips, but more than doubles false accepts. This is a real clip-level recall/F1 gain, not proof of better continuous in-car behavior. A separate candidate test at cutoff 0.50 gave F1 0.630 (97 true positives, 83 false positives, 31 misses); do not compare that number directly with the 0.29 result because the cutoff changed.

The follow-up sampler study fine-tuned the same candidate under natural, 2:1, and 4:1 negative-to-positive sampling, with three random seeds per setting. Their mean validation F1 values ranged only from 0.684 to 0.685, and no setting won consistently. **Keep the current v2 checkpoint; the sampling-ratio change did not provide reliable evidence of an improvement.** Details and all W&B runs are in [the sampler study](stage3_sampler_ratio_study.md).

## Continuous cutoff comparison

This controlled sweep used the same pipeline checkpoints, four target clips repeated at two cabin-noise levels (-5 and 0 dB), a 0.5-second hop, and a 0.50 Stage 2 gate. It has eight scene observations but only four underlying target clips. The only changed setting was the Stage 3 wake cutoff.

| Wake cutoff | Target wakes found | False wake events | False action events | Correct end-to-end passes |
|---:|---:|---:|---:|---:|
| 0.29 | 8/8 | 7 | 6 | 0/8 |
| 0.35 | 6/8 | 6 | 5 | 0/8 |
| 0.40 | 6/8 | 5 | 5 | 0/8 |
| 0.45 | 4/8 | 5 | 5 | 0/8 |
| 0.50 | 4/8 | 4 | 4 | 0/8 |

The lower cutoff finds more target wakes, but also admits more false events. The higher cutoff reduces some false events while missing more target wakes. No cutoff gives acceptable pipeline results. This is evidence of a trade-off on these simulated scenes, not a recommended production threshold or a false-alarms-per-hour estimate.

W&B runs: [0.29](https://wandb.ai/narainharreesh-vit/car-tse/runs/zfdg44l3), [0.35](https://wandb.ai/narainharreesh-vit/car-tse/runs/9x7z858e), [0.40](https://wandb.ai/narainharreesh-vit/car-tse/runs/i38377ki), [0.45](https://wandb.ai/narainharreesh-vit/car-tse/runs/755uiz8w), and [0.50](https://wandb.ai/narainharreesh-vit/car-tse/runs/5oveqqjz). Each run contains the scene and window traces.

## What the gate and timing checks show

On an earlier four-scene, one-noise-level comparison at wake cutoff 0.29 and 0.5-second hop:

- The 0.50 Stage 2 gate found 4/4 target wakes, with 1 false wake and 2 false actions.
- Lowering the gate to 0.40 still found 4/4, but raised false wakes to 2. It did not improve target wake recall in this test.
- Increasing the hop from 0.5 to 1.0 seconds reduced target wakes from 4/4 to 1/4. False wakes stayed at 1; false actions fell from 2 to 0. This trades away target detection without reducing false wakes in this small comparison.

These comparisons suggest that finer checks help catch short target wakes, while looser gating or confirmation alone does not solve false activations. They are small engineering probes, not estimates of real driving performance. See the corrected reports: [0.5-second hop / gate 0.50](diagnostic_v8_attribution_fixed_hop05_gate050.json), [0.5-second hop / gate 0.40](diagnostic_v8_attribution_fixed_hop05_gate040.json), and [1-second hop / gate 0.50](diagnostic_v8_attribution_fixed_hop1_gate050.json).

## What the exact command-buffer trace shows

A follow-up run now tracks the source-time segments actually appended to Stage 4's input. It repeats the same eight scenes and settings as the 0.29 cutoff run; wake detections, intent labels, and all headline counts are unchanged. The earlier v9 overlap percentages were computed from the latest stream time minus the buffer length. That calculation was wrong when presence checks skipped audio chunks. **Use the corrected v10 trace below; do not use v9's overlap percentages.**

The corrected trace splits the eight failures into two groups:

| Target-associated intent output | Stage 4 input from the source recording | Result |
|---|---|---|
| Four outputs at 7.5 s | Only 0.5 s, from 6.0–6.5 s; the labeled target command was 2.1–4.1 s and passenger wake audio began at 6.0 s | Late output, wrong intent, and no target-command audio in the recorded input segment |
| Three outputs at 6.5 s | 2.0 s, from 2.5–4.5 s; this overlaps 80% of the labeled target command | Wrong intent |
| One output at 4.5 s | 2.0 s, from 2.0–4.0 s; this overlaps 95% of the labeled target command | Wrong intent |

This confirms the exact stream-time segments selected in **these eight proxy scenes**. In four, the system kept waiting after the target wake and eventually passed only a short segment from the time the passenger wake recording began. In the other four, the Stage-4 input's time range overlapped most of the labeled command, but the predicted intent was still wrong. A source-time overlap does not prove SpEx preserved that speaker's words in its output, so these results do not isolate Stage 4's real-use accuracy.

There is a fundamental limit: the target wake is from Google Speech Commands, while the target command is a Fluent Speech Commands example whose speaker is not verified to match the enrolled wake speaker. SpEx and the presence gate are conditioned on the enrolled target speaker, so they may correctly suppress command speech from a different speaker. The 0/8 result therefore cannot tell us how well the system handles one person saying both the wake and the command. The clean isolated Stage 4 score also does not establish how it should classify separated, noisy command audio.

The next controlled test should use recordings where the same speaker says the wake and command. It should compare the exact Stage 4 input with the known command crop and log the presence decision for every skipped or appended chunk. Then evaluate Stage 4 on correctly aligned post-SpEx audio. This separates three questions: whether the wake decision arrives in time, whether presence gating keeps the command, and whether Stage 4 can classify what it receives. Keep Stage 4 and SpEx unchanged until that test identifies which change is justified.

Corrected run: [W&B v10 exact command-chunk trace](https://wandb.ai/narainharreesh-vit/car-tse/runs/itso0jkh), with local report [diagnostic_v10_command_capture_kws029_two_snr.json](diagnostic_v10_command_capture_kws029_two_snr.json). The capture instrumentation is evaluation-only and produced the same model outputs and headline metrics as v9.

## What to improve next

1. Collect longer continuous recordings with driver, passenger, radio/media, and road noise; label every wake and command time.
2. Find high-scoring non-wake examples in the training data that resemble the current false triggers. Train Stage 3 on such hard negatives using frozen SpEx outputs, while keeping validation and test sessions untouched.
3. Choose the score cutoff and confirmation rule on separate continuous validation recordings, measuring missed wakes, false wakes, wrong actions, and delay together.
4. Report the final result on held-out speakers and recording sessions. Do not claim a real false-alarm rate from these short synthetic scenes.

The attached research plan is directionally sound: published work evaluates streaming detection jointly with latency and accuracy, and a keyword-spotting study tests mining difficult negative training samples. Those papers motivate experiments; they do not guarantee the same gain on this project. [Google's streaming KWS study](https://research.google/pubs/streaming-keyword-spotting-on-mobile-devices/) and [Microsoft Research's hard-negative KWS study](https://www.microsoft.com/en-us/research/uploads/prod/2020/04/ICASSP2020_Max_pooling_KWS.pdf).
