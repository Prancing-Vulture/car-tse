# Car-TSE results report

**Updated:** 12 October 2026  
**Scope:** Car-TSE in-car voice pipeline, stages 0–4.  
**Evidence policy:** Metrics below are separated by dataset, split, and test type. They are not combined into one score when the recordings or thresholds differ.

> The supplied `audio-bss-tse/README.md` documents a different source-separation program (`gpt_conv` / AC-SIM). Its scores, run names, and goals do not describe this Car-TSE project and are not used here.

## Executive result

Car-TSE is a working research prototype, but its full wake-to-command pipeline is not yet reliable enough for production. The individual components can score well on their own datasets, while their combined real-time decisions still fail often.

The newest full-cascade pilot used 51 held-out synthetic scenes: 17 target wake + command, 17 target wake + no command, and 17 no-target-wake scenes. It correctly mapped **5/17 target commands (29.4%)**, detected a wake in **21/34 target-wake scenes (61.8%)**, and falsely detected a wake in **10/17 no-target-wake scenes (58.8%)**. A confidence cutoff of 0.80 limited false command actions to **1/17 wake + no-command scenes**, but did not prevent false wake detections. This small synthetic proxy is not a real-car accuracy estimate.

The W&B value near **0.83** is **5 correct intents among the 6 target-command cases where the pipeline emitted an action**. It is conditional accuracy after the pipeline chose to act. Across all 17 target-command scenes, the end-to-end command accuracy was **5/17 = 0.29**. Both values are correct; they answer different questions.

## Pipeline being measured

```text
Microphone stream
  → Stage 0: rolling timestamped buffer
  → Stage 1: SpEx target-speaker extraction
  → Stage 2: target-speech presence gate
  → Stage 3: “Marvin” wake-word detector
  → Stage 0: recover the buffered post-wake command audio
  → Stage 4: wav2vec encoder + command classifier
  → emit command or abstain
```

The recent command-head run trained only a small Stage 4 classifier (direct `768 → 9` linear layer, **6,921 parameters**). SpEx, Stage 2, Stage 3, and the wav2vec encoder were frozen. Stage 2/3 used their separate gates and wake scores; the optional learned Stage 2 + Stage 3 fusion head was **not active in this run**. SpEx module code and weights were not changed.

## Stream and continuous-scene results

These results use different recordings and answer different questions. In particular, target-absent streams measure false triggers but cannot measure wake recall.

| Test | Audio and exposure | Result | What it establishes |
|---|---|---|---|
| Cabin-noise continuous check | 1,191.8 seconds (19.9 min) of target-absent CAVEMOVE cabin noise | Full gated path: **0 smoothed wakes/hour** at presence threshold 0.90. With presence bypassed, KWS alone: 6 raw positive windows (**9.1/hour**) and 0 smoothed wakes/hour after temporal confirmation. | Stationary cabin noise alone was controlled in this recording. It says nothing about driver wake recall or passenger speech. The full path’s zero can also result from an overly strict gate. |
| Speech-rich target-absent check | 1,200 seconds (20 min) of AISHELL-5 real cabin speech/noise; no target wake positives | Original hybrid KWS at threshold 0.29: **87 smoothed events/hour**; at 0.81: **9/hour**. AISHELL-augmented KWS: **30/hour at 0.81**, **12/hour at 0.90**, **3/hour at 0.95**. Presence-gated full path: **0/hour** at presence 0.90; another tested profile (presence 0.75, KWS 0.95) also gave 0/hour. | Speech-rich negatives are much harder than cabin noise alone. Because this stream has no positive target wakes, 0 gated events/hour is not proof of a good operating point or high recall. |
| Earlier 16-scene continuous proxy | 16 synthetic 14-second scenes, 224 seconds total, 0.5-second hop; cross-corpus wake/command speakers | Held-out profile (overall gate 0.50, command gate 0.35): **14/16 wakes (87.5%)**, **2/16 correct intents (12.5%)**, **16 false wakes**, **13 false actions**. Validation profiles gave 0–2 correct intents out of 16. | Buffer and stage-flow debugging only. The brief exposure makes hourly false-event estimates misleading; the wake and command speaker identities were not matched. |
| New 51-scene command-head pipeline pilot | 17 target wake + command, 17 target wake + no command, 17 no-target wake; same-speaker synthetic GSC scenes with CAVEMOVE noise; test thresholds: presence 0.20, command gate 0.20, KWS 0.29, Stage 4 abstention 0.80 | Target wake recall **21/34 (61.8%)**; target command correct **5/17 (29.4%)**; no-command false actions **1/17 (5.9%)**; no-target false wakes **10/17 (58.8%)**; no-target false actions **0/17**. Among target-command scenes where an action was emitted: **5/6 correct (83.3%)**. | A stricter Stage 4 cutoff reduces false commands but cannot repair missed or false wake detections. This is a small synthetic proxy, not a natural in-car benchmark. |

The first two rows are the most informative available **continuous recordings**. The last two are short, assembled continuous-scene proxies. None contains the needed reviewed, uninterrupted, real-car driver recording with the same person enrolled, saying “Marvin,” then speaking a command, alongside passenger/media negatives.

## Buffer verification

The rolling-buffer fix was checked with the same two 14-second proxy scenes, checkpoints, and thresholds. In the control, Stage 2 rejected command-pending windows 3–6; in the forced-skip run, it also rejected window 7. In both cases Stage 4 still received the same **2.0–4.0 s** crop. The labeled command spans **2.1–4.1 s**, so **1.9/2.0 s (95%)** of it reached Stage 4. The saved inputs were sample-identical across the two runs.

This verifies buffer retention across those gate rejections. It does **not** prove an end-to-end accuracy gain: the predicted intent was wrong in both replays, and the cross-corpus command speaker was not verified to match the wake/enrolled speaker. A fixed two-second crop also omitted the final 0.1 s of the labeled command in this example.

## Component results and why they do not predict the full pipeline

| Component evaluation | Measured result | Limit |
|---|---:|---|
| Stage 1 target-speaker extraction, held-out synthetic CAVEMOVE | Mean SI-SDR improvement **+10.98 dB** over the mixture (400 clips). | Synthetic mixture score; it does not prove that commands from a same-speaker real driver remain intelligible after extraction. |
| Stage 1 real-cabin diagnostic, AISHELL-5 | Current output mean SI-SDR **−31.22 dB**, versus **−37.31 dB** for far-field input. A small frontend adaptation reached −30.57 dB but regressed CAVEMOVE validation. | Different real-cabin protocol; not a same-speaker wake-to-command test. |
| Stage 2 framewise presence, held-out test | At validation-locked threshold 0.40: **F1 0.810**, recall 0.907, false-accept rate 0.333 on 300 chunks. | Chunk-level classification is not the same as continuous target-presence decisions; operating point still has many false accepts. |
| Stage 3 `marvin` wake classifier, held-out test | At threshold 0.29: **F1 0.667**, precision 0.525, recall 0.914 on 384 isolated examples. | Isolated-window score; not continuous false wakes/hour and not identical to every checkpoint used by the full-cascade runs. |
| Stage 4 original 31-intent wav2vec, official clean FSC test | **99.31% top-1** on 3,793 clean utterances. | Clean isolated commands are unlike the extracted, noisy, cropped runtime input. |
| New isolated 9-class command head | Best validation macro-F1 **0.875**; held-out test macro-F1 **0.852** on 1,764 examples. Runtime-capture validation macro-F1 **0.767** on 95 actual Stage 4 captures. | These are classifier-only metrics; they do not include upstream missed wakes, wrong speakers, or false activations. |
| New head inside complete cascade | **5/17 target commands correct (29.4%)** in the 51-scene test. | This is the relevant pipeline result for the new head, but still a small synthetic proxy. |

## Shared-window Stage 2/3 comparison

To compare the **hard Stage 2 gate**, Stage 3 alone, and the learned Stage 2/3 fusion on identical examples, all three were rescored on the same 305 speaker-held-out validation windows (24 wake-positive, 281 negative). This is a window-level validation measure; it excludes temporal confirmation and is not an untouched final test.

| Approach | Operating point | TP / FP / FN / TN | F1 | Precision | Recall | Specificity | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| Configured hard gate | Stage 2 ≥0.90 AND Stage 3 ≥0.29 | 3 / 0 / 21 / 281 | **0.222** | 1.000 | 0.125 | 1.000 | 0.931 |
| Hard gate, best validation sweep | Stage 2 ≥0.70 AND Stage 3 ≥0.45 | 19 / 7 / 5 / 274 | **0.760** | 0.731 | 0.792 | 0.975 | 0.961 |
| Stage 3 score alone | Stage 3 ≥0.55 | 15 / 5 / 9 / 276 | **0.682** | 0.750 | 0.625 | 0.982 | 0.954 |
| Learned Stage 2/3 fusion | Fusion score ≥0.60 | 17 / 3 / 7 / 278 | **0.773** | 0.850 | 0.708 | 0.989 | 0.967 |

The configured `0.90/0.29` hard gate has a window-level F1 of **0.222** because it misses 21 of the 24 positive windows. Changing the hard-gate thresholds to `0.70/0.45` raises validation F1 to 0.760. The learned fusion reaches 0.773 on the same validation windows. Since the fusion checkpoint and cutoffs were selected using this validation split, those tuned figures are optimistic development results, not the final apples-to-apples test. Full threshold tables for F1, precision, recall, specificity, accuracy, false-positive rate, and false-negative rate are in [`STAGE23_APPLES_TO_APPLES_VALIDATION_2026-10-12.md`](evaluation/STAGE23_APPLES_TO_APPLES_VALIDATION_2026-10-12.md), with row-level values in the adjacent CSV and JSON. The evaluation is logged to [W&B](https://wandb.ai/narainharreesh-vit/car-tse/runs/kxhpbciw).

### Additional paired signal-path check

On 390 cross-corpus scenes (195 wake recordings × two SNRs, −5 and 0 dB), the actual runtime detected 289 wakes and got **49/390 (12.6%)** final intents correct. Offline Stage 4 got **390/390** raw-mixture command crops and **390/390** clean command references right, but only **69/390 (17.7%)** Stage 1 extracted command crops right. This strongly shows that the extractor changed the command audio in this proxy. However, the FSC command speaker differed from the enrolled wake speaker, so this result cannot prove that Stage 1 suppresses a real enrolled driver’s command. All **5,405** WAV paths referenced by its CSV were independently checked and exist.

## W&B runs

- [Final 51-scene run, Stage 4 confidence cutoff 0.80](https://wandb.ai/narainharreesh-vit/car-tse/runs/bwte7h8d) — W&B summary confirms 51 scenes, 0.80 cutoff, 0.294 end-to-end target command accuracy, 0.833 conditional accuracy among emitted target actions, and 0.588 no-target false-wake rate.
- [Earlier runtime-head run, cutoff 0.35](https://wandb.ai/narainharreesh-vit/car-tse/runs/7ec97d2h) — less restrictive command cutoff; different action trade-off.
- [Five-step held-out continuous proxy, cutoff 0.35](https://wandb.ai/narainharreesh-vit/car-tse/runs/jr040fah) — 16-scene cross-corpus evaluation, not directly comparable to the 51-scene run.
- [Buffer control](https://wandb.ai/narainharreesh-vit/car-tse/runs/a46u4ato) and [forced Stage 2 rejection](https://wandb.ai/narainharreesh-vit/car-tse/runs/fpfnyae6).
- [Stage 1 effect on wake and command audio](https://wandb.ai/narainharreesh-vit/car-tse/runs/vc4g0bz4) — 390-scene cross-corpus diagnostic.

## Verification notes

- Recounted the 51-scene rows directly from `gsc_command_full_cascade_report.json`: 17 rows in each of the three scene groups; 5 correct target commands; 1 false action in the 17 wake/no-command scenes; 10 false wakes and 0 false actions in the 17 no-target-wake scenes. The summary fields match those counts.
- Cross-checked the W&B local summary for run `bwte7h8d`. The `0.83` field is explicitly `full_pipeline/target_command_accuracy_when_action_emitted`; the all-target-scene field is `full_pipeline/target_command_accuracy = 0.29`.
- Recounted the 390 Stage 1 diagnostic rows from its CSV; wake count, end-to-end count, raw/clean/extracted command counts match the report. Verified all 5,405 referenced WAV files exist.
- Cross-checked the Stage 2 and Stage 3 reported metrics against their saved test JSON and the stream rates against the checked-in threshold CSV/report.
- These checks verify the recorded artifacts and arithmetic. They do not turn synthetic or cross-corpus tests into real-car validation.

## Conclusion and next evidence needed

The known buffer bug is fixed and its behavior is verified under the tested Stage 2 rejections. The strongest current blocker shown by the available end-to-end numbers is **reliable target-speaker handling and wake decisions under mixed speech**, followed by poor command accuracy on extracted runtime audio. The evidence does not yet isolate the real-driver failure to a single stage: most wake/command scenes pair recordings from different corpora and speakers.

The next decisive benchmark is a labeled continuous recording where the **same driver** provides enrollment, says the wake phrase, then speaks the command in the real microphone stream. Add target-absent sessions with passenger/media speech and cabin noise. Save the exact Stage 4 tensor and compare its audio to the reviewed command. Freeze checkpoints and thresholds before evaluation. Until then, report component scores and proxy stream results separately; do not claim >90% end-to-end intent mapping or production readiness.

## Evidence files

- `reports/evaluation/FIVE_STEP_PIPELINE_EVALUATION_2026-10-12.md`
- `reports/evaluation/BUFFER_FIX_REAL_MODEL_REPLAY_2026-10-12.md`
- `reports/evaluation/STAGE1_TSE_IMPACT_ALL_TEST_POINTS_2026-10-12.md`
- `reports/evaluation/gsc_command_full_cascade_report.json`
- `reports/evaluation/stage2_frame_presence_test.json`
- `reports/evaluation/stage3_gsc_marvin_tse_candidate_test.json`
- `reports/evaluation/stage4_intent_fsc_test_ft8.json`
- `reports/paper_figures_2026-10-05/stage3_aishell_threshold_stream.csv`
- Historical project context: `reports/project_benchmark_2026-10-03.md` and `reports/base_project_report_2026-10-04.md`.
