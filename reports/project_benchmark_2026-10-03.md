# Car-TSE project benchmark and status

Date: 2026-10-04  
Workspace: `D:/Prawin_New/tse/car-tse-main`  
Scope: benchmark, targeted enhancement, and audit. New experiments used uniquely named checkpoints; existing best checkpoints were not overwritten.

## Executive status

The project is a strong research prototype, not yet a production-ready in-car assistant.

The latest fine-tuned wav2vec intent model scores **99.31% top-1 on the official Fluent Speech Commands test split** and **96.60% on the official validation split**. This is close to the reported 99.7% clean FSC result, although the 99.7% paper uses a different architecture and distillation recipe: [Two-stage Textual Knowledge Distillation for End-to-End Spoken Language Understanding](https://arxiv.org/abs/2010.13105).

The current cabin-domain results are lower but more relevant to the target application:

| Pipeline block | Current measured result | Assessment |
|---|---:|---|
| Target-speaker extraction | +10.98 dB mean SI-SDR improvement on 400 held-out CAVEMOVE clips | Promising applied baseline; synthetic speech/noise protocol |
| Target-presence gate | 91.25% accuracy at default threshold, but 96.25% FAR; at threshold `0.90` it retained 0% smoothed wake recall in the target-positive proxy | Threshold is not production-calibrated; class imbalance and target-positive calibration dominate |
| Wake-word detector, AISHELL-5-augmented candidate | 80.74% F1; 78.46% recall; 83.15% precision; 25.6 false accepts/hour at validation-selected threshold `0.81` | Better on the merged held-out test, but real-stream calibration is worse at `0.81`; threshold/data mismatch remains |
| Continuous cabin-noise false-accept check, final runtime path | 0 full-pipeline wakes in 19.9 minutes; KWS-only raw rate 9.1/hour and smoothed rate 0/hour | Stationary cabin noise is controlled; speech-triggered FAR remains the main gap |
| Real speech-rich continuous hard-negative check, AISHELL-5 | Prior hybrid at `0.81`: 48 raw / 9 smoothed per hour; AISHELL-augmented KWS at `0.81`: 117 / 30, at `0.95`: 30 / 3; full gated path 0/hour | The new checkpoint needs real-stream threshold recalibration; no wake positives |
| Intent, latest TSE-domain fine-tune | 90.49% top-1 validation; 90.71% raw-cabin; 90.49% TSE-domain | Improved over the prior candidate; still not a real-cabin command benchmark |
| GPU inference, latest checkpoints | About 69.95 ms per 1-second benchmark input across Stage 1 + KWS + intent | Meets the rough GPU latency target in this non-streaming benchmark |
| CPU inference | About 275.3 ms per 1-second benchmark input | Too slow for a low-power CPU without optimization |

The project should currently be described as **“validated research prototype with strong clean intent classification and promising synthetic cabin TSE/KWS robustness.”** It should not yet be described as production-ready, SOTA, or validated on real driver/passenger conversations.

## Evaluation protocol and data coverage

### Stage 1: target-speaker extraction

- Training: 4,000 triplets.
- Validation: 400 triplets.
- Test: 400 triplets.
- Speech source: LibriMix/LibriSpeech-derived synthetic mixtures.
- Cabin condition: CAVEMOVE multichannel cabin noise, four selected channels `[0, 2, 4, 6]`.
- Test vehicle: held-out Volkswagen Golf CAVEMOVE condition.
- Enrollment: synthetic clean speech from the target speaker.
- Current model: transferred 16 kHz SpEx+-style extractor plus learned multichannel cabin frontend.

This is not yet a real driver-versus-passenger test. The current Stage 1 benchmark does not include real overlapping passenger speech, real driver commands, or a fully measured echo/playback path.

As a real-cabin diagnostic, 144 TextGrid-aligned AISHELL-5 target-present windows were evaluated against their near-field references. The current Stage 1 checkpoint produced **−31.22 dB mean SI-SDR** versus **−37.31 dB** for the far-field input baseline. A bounded two-epoch frontend adaptation (`stage1_spex_aishell5_adapt_ft1`) improved the real-cabin output only to **−30.57 dB** (+0.65 dB over the current Stage 1) while regressing the held-out CAVEMOVE validation from **11.00 dB to 9.90 dB**. It is therefore preserved as a rejected diagnostic checkpoint, not a replacement. The result shows that the synthetic CAVEMOVE gain does not transfer to AISHELL-5 real speech without a broader scale-aware/frontend adaptation protocol.

### Stage 2: target presence

- Source manifest: 4,400 examples: 4,000 target-present and 400 target-absent.
- Deterministic validation split: 880 examples: 800 present and 80 absent.
- The split is therefore highly imbalanced, so accuracy alone is misleading.

### Stage 3: wake-word detection

The wake word is currently `marvin`.

| Candidate | Test set | Positives / negatives | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Original cabin KWS | 1,147 | 195 / 952 | 84.22% | 52.29% | 82.05% | 63.87% |
| Hard-negative cabin KWS | 3,147 | 195 / 2,952 | 88.69% | 32.91% | 79.49% | 46.55% |
| TSE-domain KWS | 445 | 195 / 250 | 96.85% | 95.94% | 96.92% | 96.43% |
| Post-TSE hybrid KWS, corrected hard-negative test | 3,397 | 195 / 3,202 | 97.79% | 84.48% | 75.38% | 79.67% |
| AISHELL-5-augmented KWS, validation-selected `0.81` | 4,551 | 195 / 4,356 | 98.40% | 83.15% | 78.46% | 80.74% |

The original narrow TSE-domain candidate is still useful as a regression baseline because it is trained on the distribution produced by Stage 1. At the selected threshold `0.53`, it gives:

- Precision: 98.94%
- Recall: 95.90%
- F1: 97.40%
- False positives: 2/250 negatives
- Approximate false accepts/hour: 28.8, assuming independent 1-second windows

The independent-window false-accept rate is not, by itself, a continuous-stream automotive measurement. Temporal smoothing and refractory logic are now implemented and have been exercised on both long cabin-noise recordings and a sequential hard-negative proxy.

The corrected post-TSE hard-negative benchmark is the more realistic stress test for deployment. It passes 3,202 non-wake hard negatives through Stage 1 using a different enrollment speaker before KWS. The new hybrid KWS checkpoint (`stage3_kws_tse_hybrid_ft1`) is therefore the current full-pipeline candidate: it reaches 79.67% F1 at the threshold selected on validation (`0.29`), with 27 false positives / 3,202 negatives (30.4 false accepts/hour under the 1-second-window estimate). The narrow original 250-negative TSE-domain set remains useful for regression, but should not be used alone for a production claim.

The next KWS candidate (`stage3_kws_tse_aishell_ft1`) was initialized from that hybrid checkpoint and fine-tuned with real AISHELL-5 speech/noise hard negatives added to the existing hybrid cache. At the validation-selected threshold `0.81`, the merged held-out test gives 153 true positives, 31 false positives, and 42 false negatives: 80.74% F1, 78.46% recall, 83.15% precision, and 25.6 false accepts/hour under the same independent-window estimate. However, the real continuous stream exposes a distribution/calibration mismatch: the prior hybrid checkpoint at `0.81` produced 48 raw and 9 smoothed KWS-only events/hour, while the AISHELL-augmented checkpoint produced 117 raw and 30 smoothed events/hour. Raising the new checkpoint to `0.95` reduces this to 30 raw and 3 smoothed events/hour, but its target-positive proxy recall collapses. The new checkpoint is therefore an experimental candidate, not yet a clear deployment improvement.

A positive-expansion experiment (`stage3_kws_tse_morepos_aishell_ft1`) increased the TSE-domain wake examples from 1,000 to 1,710 while retaining hard negatives and AISHELL negatives. It reached 83.6% validation F1 and 81.9% test F1 at its offline operating points, but the validation-selected threshold produced 45 smoothed false events/hour on the real AISHELL stream. On the eight-speaker proxy, its smoothed recall was only 3.1%, 9.4%, 18.8%, and 9.4% at `-5`, `0`, `10`, and `20` dB SNR. This candidate is preserved for audit but rejected as the current model; more synthetic positives alone did not solve the distribution gap.

### Runtime conditioning correction

The downstream cache builders peak-condition Stage 1 outputs to a maximum amplitude of `0.95`, but the runtime path previously sent raw Stage 1 output to KWS. On eight real CAVEMOVE test clips, this produced raw KWS wake probabilities around `1e-12`; applying the same cache-matched conditioning moved ordinary non-wake scores into the expected range. The pipeline now keeps the raw Stage 1 waveform for SI-SDR and the presence head, while using cache-matched peak conditioning for KWS and intent. KWS also receives a configurable one-second crop by default, matching the one-second cached training examples instead of the full three-second Stage 1 context.

### Continuous-stream false-accept checks

The new evaluator `scripts/eval/eval_continuous_kws.py` slides the production Stage 1 context through 20 long CAVEMOVE target-absent recordings covering Alfa Romeo, Smart, and Volkswagen cabins, road/vehicle noise, and ventilation. The final runtime-matched run covers 1,191.8 seconds (19.9 minutes) and 1,200 one-second hops. At the provisional presence threshold `0.90`, no window passed presence and there were **0 full-pipeline wakes**. With presence gating disabled to isolate KWS, the raw detector produces 6 positive windows (**9.1/hour**), while 2-of-3 temporal confirmation plus refractory logic produces **0 wakes/hour**. This benchmark uses only pure cabin noise, so it does not measure passenger-speech false accepts.

This is encouraging evidence that stationary cabin noise is not the only or dominant false-accept driver. The speech-triggered portion was separately evaluated using `scripts/eval/eval_kws_hardneg_sequence.py` on 2,952 held-out post-TSE hard-negative examples. At threshold `0.29`, the independent-window proxy produces 31.7 positive windows/hour and 30.5 merged events/hour; 2-of-3 confirmation plus a two-call refractory period reduces this to **1.22 events/hour**. Because the sequence is formed by concatenating independent one-second clips, this is a temporal-calibration proxy rather than a real continuous acoustic result.

A second synthetic continuous proxy assembles six same-speaker wake clips and 30 passenger hard-negative clips into a shared CAVEMOVE noise bed at `-5`, `0`, `10`, and `20` dB SNR. With presence disabled to isolate KWS, raw wake recall is 16.7%, 50.0%, 50.0%, and 83.3% respectively; smoothed recall is 0%, 0%, 33.3%, and 33.3%. Smoothed false events remain 0, 150, 425, and 50/hour. At presence threshold `0.60`, positive-event coverage is retained but the presence pass rate varies from 36.8% to 95.1%. This proxy is useful for engineering diagnosis, not a real-cabin claim, and it shows that passenger-speech robustness and temporal recall still require improvement.

The same proxy was rerun with the AISHELL-augmented KWS checkpoint at threshold `0.81` to isolate the presence gate. With the original presence head at threshold `0.90`, presence pass rates were only 4.9%, 13.9%, 11.8%, and 5.6% at `-5`, `0`, `10`, and `20` dB SNR, and smoothed wake recall was **0% at every SNR**. Lowering presence to `0.60` restored smoothed recall to 0%, 16.7%, 33.3%, and 100%, but smoothed false accepts were 25, 400, 525, and 50/hour. The intermediate `0.75` sweep did not recover useful recall in this small proxy. This separates the immediate bottleneck: the original presence head/threshold suppresses target-positive speech, while lowering the threshold exposes unacceptable speech-triggered FAR.

The previously saved balanced-hard-negative presence candidate (`stage2_presence_tse_hardneg_ft1`) was also checked before starting another training run. At its validation-oriented threshold `0.95` it produced 0% smoothed wake recall across the same four scenes; at `0.60` it was effectively the same as the original presence head. Swapping the existing checkpoint therefore does not solve the target-positive gate failure; a purpose-built target-positive/absent calibration set is the next justified training input.

A cached presence-head fine-tune (`stage2_presence_gsc_aishell_cached_ft1`) was then trained from 604 same-speaker `marvin` positives with CAVEMOVE augmentation and 604 AISHELL-5 target-absent negatives. On the held-out cached test, it reaches 79.7% recall with 0% FAR at threshold `0.90` and 92.2% recall with 1.6% FAR at threshold `0.64`. On the continuous target-positive proxy at threshold `0.90`, smoothed recall becomes 0%, 0%, 0%, and 50% at `-5`, `0`, `10`, and `20` dB SNR, with smoothed false events of 0, 225, 225, and 50/hour. At threshold `0.75`, recall is 0%, 16.7%, 0%, and 83.3%, with false events of 25, 375, 325, and 50/hour. At presence `0.60` with KWS `0.81`, recall is 0%, 16.7%, 33.3%, and 100%, but false events rise to 25, 425, 525, and 50/hour. With the stricter KWS threshold `0.95` and presence `0.75`, recall falls to 0%, 0%, 0%, and 16.7%. This is a real improvement in high-SNR target retention, but it is not yet robust enough for deployment.

To reduce single-speaker variance, `scripts/eval/eval_synthetic_multispeaker_proxy.py` evaluated eight independent `marvin` speakers, four wake clips and 20 passenger negatives per scene, using the new presence checkpoint at `0.75` and KWS at `0.81`. Across 32 expected wakes per SNR, aggregate smoothed recall was **6.3%, 18.8%, 53.1%, and 25.0%** at `-5`, `0`, `10`, and `20` dB SNR; smoothed false-event rates were **70.3, 112.5, 140.6, and 42.2/hour**. This broader result shows that speaker/enrollment generalization is a systematic gap, not just an artifact of the original six-clip single-speaker proxy.

The same eight-speaker run using two short enrollment clips within the same 1.5-second enrollment budget improved low-SNR smoothed recall to **15.6% and 31.3%** at `-5` and `0` dB, compared with 6.3% and 18.8% for one-clip enrollment. Recall at `10` and `20` dB changed to 46.9% and 21.9%, with false-event rates of 112.5 and 51.6/hour. This is a promising enrollment-side enhancement, but its tradeoffs are not yet consistent enough to replace the single-clip baseline.

An expanded presence-calibration experiment (`stage2_presence_gsc_aishell_richer_ft1`) doubled the same-speaker positive sampling from two to four clips per speaker while keeping Stage 1 frozen and caching all embeddings. On its held-out cached test, it reaches 80.9% recall with 0.4% FAR at threshold `0.90`, and 91.4% recall with 1.6% FAR at `0.60`. On the eight-speaker proxy at `0.75`, smoothed recall is **6.3%, 18.8%, 50.0%, and 21.9%** with **70.3, 98.4, 93.8, and 42.2 false events/hour**; at `0.60`, recall is **6.3%, 18.8%, 53.1%, and 25.0%** with **70.3, 107.8, 135.9, and 42.2 false events/hour**. The cached metrics shift slightly, but proxy performance does not materially improve, so this checkpoint is preserved for audit and is not promoted as the default.

An AISHELL-5 target-present presence experiment then used TextGrid-aligned real far-field segments with near-field enrollment references (1,200/1,200 train rows and 144/144 validation/test rows). The fine-tuned head reached only 45.8% recall and 31.9% FAR at threshold `0.50` on the held-out real-cabin test; at `0.90` it rejected all positives (0% recall, 0% FAR). The existing cached presence head behaved similarly, with 0% recall at `0.90`. This is not a promoted model result: it exposes a larger acoustic/scale mismatch between the current Stage 1 frontend and real AISHELL target-present speech, even though AISHELL remains a valuable target-absent stream. The experiment is preserved under `stage2_presence_aishell_target_ft1` for diagnosing input conditioning and frontend adaptation.

### Real continuous AISHELL-5 hard-negative check

The evaluator `scripts/eval/eval_aishell5_continuous_kws.py` now runs the full Stage 1 → presence → conditioned KWS path over two target-absent AISHELL-5 `Eval2` sessions. The 1,200-second stream contains real four-channel far-field speech/noise with 85.5% labeled speech occupancy. It is a speech-rich false-accept benchmark, not a wake-word-positive or driver-command benchmark.

Using the original hybrid KWS checkpoint at threshold `0.29`, KWS-only detection produced 125 positive one-second windows (375/hour) and 29 smoothed events (87/hour). The prior hybrid at the stricter threshold `0.81` produced 16 windows (48/hour) and 3 smoothed events (9/hour). The AISHELL-augmented checkpoint produced 39 windows (117/hour) and 10 smoothed events (30/hour) at `0.81`; threshold `0.90` produced 19/57 and 4/12, while `0.95` produced 10/30 and 1/3. The presence-gated full path produced 0 smoothed events/hour in these target-absent runs at presence threshold `0.90`; the new presence gate at `0.75` plus KWS `0.95` also produced 0 full-path events/hour. This is a useful calibration result, not a KWS improvement claim: the new checkpoint must be retuned against real-stream negatives and target-positive recall together.

### Stage 4: intent classification

The old mel-CRNN and the newer wav2vec candidate were evaluated separately.

| Model / condition | Top-1 | Top-5 | Macro F1 | Examples |
|---|---:|---:|---:|---:|
| Old CRNN, bootstrap clean | 41.53% | 81.59% | 42.39% | 2,314 |
| Old CRNN, bootstrap CAVEMOVE cabin | 41.10% | 81.89% | 41.94% | 2,314 |
| Old CRNN, bootstrap TSE-domain | 48.62% | 84.57% | 50.64% | 2,314 |
| wav2vec, bootstrap clean | 92.09% | 98.49% | 92.03% | 2,314 |
| wav2vec, bootstrap raw cabin | 88.98% | 97.32% | 89.24% | 2,314 |
| wav2vec, bootstrap TSE-domain | 89.41% | 97.75% | 88.85% | 2,314 |
| wav2vec, latest fine-tune, bootstrap clean | **93.04%** | **98.62%** | **93.02%** | 2,314 |
| wav2vec, latest fine-tune, raw cabin | **90.71%** | **97.84%** | **91.05%** | 2,314 |
| wav2vec, latest fine-tune, TSE-domain | **90.49%** | **98.01%** | **90.13%** | 2,314 |
| wav2vec, latest fine-tune, official FSC validation clean | **96.60%** | **99.36%** | **96.40%** | 3,118 |
| wav2vec, latest fine-tune, official FSC test clean | **99.31%** | **99.89%** | **99.24%** | 3,793 |

The official FSC results are the correct comparison for the published 99.7% clean result. The earlier 2,314-example “clean” result was a local bootstrap slice of the official training CSV, not the official validation/test split.

The official manifests are speaker-disjoint:

- Train: 23,132 utterances, 77 speakers.
- Validation: 3,118 utterances, 10 speakers.
- Test: 3,793 utterances, 10 speakers.
- Train/validation/test speaker overlap: zero.

The current TSE-domain intent cache has not been generated for the official FSC validation/test splits, so robust intent numbers remain local-domain validation numbers rather than official noisy-test results.

## Stage-by-stage benchmark results

### Stage 1 TSE quality

Fresh 400-example CAVEMOVE test evaluation:

| Metric | Input mixture | Extracted output | Improvement |
|---|---:|---:|---:|
| Mean SI-SDR | -2.41 dB | 8.57 dB | **+10.98 dB** |
| Median | -2.37 dB | 10.49 dB | **+12.77 dB** |
| 10th percentile | -8.32 dB | 0.22 dB | **+4.06 dB** |
| 90th percentile | 3.30 dB | 15.07 dB | **+16.88 dB** |
| Negative improvement cases | 272 input cases | 40 output cases | 21 cases |

Comparison baselines on the same style of test:

- Clean transferred SpEx+ baseline: +10.73 dB SI-SDR improvement.
- Cabin-noise mean-pool baseline without the learned cabin frontend: +8.01 dB.
- Current multichannel cabin frontend: +10.98 dB.

The frontend therefore recovers roughly 3 dB over the simple noisy mean-pool baseline and slightly exceeds the clean transferred baseline on this synthetic CAVEMOVE protocol.

Important limitation: SI-SDR improvement does not guarantee command intelligibility, correct speaker identity, or downstream intent accuracy. The next TSE benchmark should include STOI/ESTOI, target confusion, ASR/WER or intent preservation, and real overlapping speakers.

### Stage 2 presence gate

Corrected deterministic validation result at the current pipeline threshold `0.50`:

- Examples: 880.
- True positives: 800.
- False negatives: 0.
- True negatives: 3.
- False positives: 77.
- Accuracy: 91.25%.
- FAR: **96.25%**.
- FRR: 0%.

This is not acceptable as a production operating point. Threshold sweep results show that the checkpoint contains useful separation, but the threshold is wrong for the imbalanced validation distribution:

- Best F1: threshold `0.72`, F1 97.72%, recall 99.0%, FAR 36.25%, FRR 1.0%.
- Best balanced accuracy: threshold `0.90`, balanced accuracy 90.34%, specificity 82.5%, recall 91.13%, FAR 17.5%, FRR 8.88%.

The next gate evaluation must use a larger, independently collected absent-target set and a threshold selected against a target false-accept rate, not F1 or raw accuracy.

A new presence-head candidate was fine-tuned from the original checkpoint using balanced post-TSE hard negatives. On the original deterministic split it reaches 88.75% specificity / 86.38% recall at threshold `0.95` (versus 82.5% / 91.13% for the original head at threshold `0.90`). On 2,952 held-out post-TSE absent clips it reaches 91.87% specificity at threshold `0.99`; that absent-only result has no recall component and is therefore a calibration signal, not a final gate score. The candidate is preserved as `stage2_presence_tse_hardneg_ft1`, but is not yet a production claim.

### Stage 3 KWS operating points

Threshold calibration on the held-out test sets:

| Candidate | Best-F1 threshold | F1 | Recall | Precision | Approx. FA/hour |
|---|---:|---:|---:|---:|---:|
| Original cabin | 0.59 | 66.04% | 72.31% | 60.78% | 344.1 |
| Hard-negative cabin | 0.69 | 54.13% | 60.51% | 48.96% | 150.0 |
| TSE-domain | 0.51 | 97.42% | 96.92% | 97.93% | 57.6 |
| TSE-domain, higher precision operating point | 0.53 | 97.40% | 95.90% | 98.94% | **28.8** |
| Corrected post-TSE hybrid, validation-selected threshold | 0.29 | **79.67%** | 75.38% | 84.48% | **30.4** |

The hard-negative model reduces false accepts but loses too much recall/precision on this test distribution. The TSE-domain model is the current preferred checkpoint.

### Stage 4 intent and the clean 99.7% comparison

The current wav2vec candidate uses `facebook/wav2vec2-base-960h`, with the last four transformer layers fine-tuned and a small intent head. It is not the same as the 99.7% vq-wav2vec/BERT plus two-stage textual distillation system.

Nevertheless, its official FSC test result is already 99.24% top-1. The remaining 0.46 percentage-point gap should not be treated as a cabin robustness gap. It is a clean-benchmark optimization gap. Text-teacher distillation is the most direct next experiment if the goal is to reproduce or exceed 99.7% under the same clean protocol.

## End-to-end pipeline check

Fresh 400-example end-to-end no-wake sanity runs use the latest Stage 1, Stage 2, AISHELL-5-augmented KWS, and wav2vec intent checkpoints, with cache-matched KWS conditioning and one-second input alignment:

```text
SI-SDR mean:       8.569 dB
accepted rate:     91.75%
presence rate:     91.75%
wake rate:         0% (0/400) with the current temporal decoder
examples:          400
device:            CUDA
KWS threshold:     0.81
```

This is not a wake-word accuracy result. These Stage 1 triplets contain ordinary LibriMix speech, not labeled `marvin` wake-word recordings, so the run is checking whether ordinary non-wake speech triggers the wake stage. The raw post-conditioning rate is 4/400; the configured temporal decoder suppresses all four. There is currently no fully aligned real-cabin wake-to-action test set.

Using the calibrated presence threshold `0.90` is a better provisional operating point than the default `0.50`, but it still needs confirmation on an independent absent-target set.

The latest normalized end-to-end configuration uses the latest intent fine-tune and AISHELL-5-augmented KWS checkpoint. With the original presence head at threshold 0.90 and KWS threshold 0.81, it reports 91.75% accepted/presence rate, 0% temporal-decoder wake rate, and the same 8.569 dB mean SI-SDR. These remain no-wake sanity checks, not wake-to-intent accuracy results.

The current experimental combination (`stage2_presence_gsc_aishell_cached_ft1`, presence `0.75`, AISHELL-augmented KWS `0.95`) was also run on the same 400-clip CAVEMOVE no-wake sanity set. It reports 100% presence acceptance, 0% temporal-decoder wake rate, and 8.569 dB mean SI-SDR. This confirms no immediate stationary-cabin regression, but it cannot establish target-positive wake recall.

The runtime temporal decoder is available in `configs/eval_end_to_end_cavemove_wav2vec_ft8_kws_aishell_ft1_presence90_smooth.yaml` and the current experimental configuration `configs/eval_end_to_end_cavemove_wav2vec_ft8_kws_aishell_ft1_presence_cached75_k95_smooth.yaml`: 2 hits in a 3-call KWS window plus a two-call refractory period. Both 400-clip no-wake sanity runs report 0% wake rate, and the long pure-noise KWS-only run also reports 0 wakes/hour. These are still not substitutes for a continuous driver/passenger recording.

The continuous CAVEMOVE and AISHELL-5 benchmarks now supply noise-only and speech-rich target-absent checks, but neither contains positive wake words paired with the enrolled target and therefore cannot measure wake recall or wake-to-intent accuracy.

```text
real cabin microphones → target driver speech → wake word → command intent → action/reply
```

Therefore, true end-to-end wake-to-intent accuracy and command/action completion are still unmeasured.

## Latency and footprint

Fresh single-thread inference benchmark for a 1-second input, 20 repeats on GPU and 10 repeats on CPU:

| Device | Stage 1 | KWS | Intent | Sum |
|---|---:|---:|---:|---:|
| GPU, prior checkpoints | 36.98 ms | 1.21 ms | 13.77 ms | **51.95 ms** |
| GPU, latest checkpoints | **51.60 ms** | **1.92 ms** | **16.43 ms** | **69.95 ms** |
| CPU | 168.06 ms | 2.09 ms | 105.19 ms | **275.34 ms** |

These measurements are model-forward timings, not a complete streaming benchmark. The current Stage 1 is non-causal with a 3-second context; the runtime KWS path now uses a 1-second crop and the intent model uses a 2-second crop. Streaming chunking, overlap, buffering, microphone I/O, resampling, and post-processing are not included.

Approximate current model footprint:

| Component | Parameters | Checkpoint size |
|---|---:|---:|
| Stage 1 TSE + cabin frontend | 10.73M | 43.1 MB |
| Stage 2 presence | 0.25M | 1.0 MB |
| Stage 3 KWS | 0.09M | 0.4 MB |
| wav2vec encoder | 94.37M | 377.6 MB |
| wav2vec intent head | 0.41M | 1.6 MB |
| **Total** | **105.85M** | **about 423.7 MB** |

The wav2vec encoder dominates the footprint. It is suitable for the current GPU prototype but is not yet a compact automotive-DSP deployment model.

## Dataset and storage inventory

Measured workspace footprint:

| Location | Size | Files |
|---|---:|---:|
| `data/raw` | 298.10 GiB | 187,002 |
| `data/processed` | 3.15 GiB | 24,006 |
| `data/cache` | 19.29 GiB | 556 |
| `checkpoints` | 0.75 GiB | 18 |
| `logs` | 2.0 MiB | 81 |

Largest raw corpora:

| Corpus | Local size |
|---|---:|
| AISHELL-5 | 247.41 GiB |
| LibriSpeech | 27.10 GiB |
| CAVEMOVE | 8.61 GiB |
| Speech Commands v2 | 5.37 GiB |
| Fluent Speech Commands | 5.00 GiB |
| Vehicle Interior Sound Dataset | 4.45 GiB |

The selected data roles are sensible:

- LibriSpeech/LibriMix: clean target/enrollment and synthetic mixtures.
- AISHELL-5: real four-channel in-car speech/noise/AEC material available for future real-cabin adaptation.
- CAVEMOVE: multichannel vehicle impulse responses and driving-condition noise currently used for cabin augmentation.
- Speech Commands v2: public keyword data and negatives.
- Fluent Speech Commands: 31-class intent data.
- Vehicle Interior Sound: vehicle-only noise augmentation, not speech supervision.

ASR-CabNois was not downloaded because the full MagicHub download requires authenticated access and has restrictive non-commercial/no-derivatives terms. The commercial Nexdata 531-hour corpus was also not redistributed. Those are data-availability limitations, not software failures.

## Verification and reproducibility findings

- Unit tests: **6 passed, 1 skipped**. The skipped test requires an unavailable vendor pretrained checkpoint; the added custom in-car manifest test verifies speaker-disjoint splits and enrollment attachment.
- Stage 1 speaker leakage check: no leakage detected across the generated Stage 1 manifests.
- Official FSC train/validation/test speakers: zero overlap.
- Existing Stage 2 evaluator had two issues: it accepted a missing manifest and omitted the transfer backend, producing misleading/incorrect results. The evaluator was corrected and the result above is from the corrected run.
- The original intent manifest builder created a 90/10 slice of `train_data.csv`, not official FSC validation/test manifests. A separate official-manifest builder was added; the existing training manifests and checkpoints were left unchanged.
- Latency evaluation now supports explicit `--device cpu|cuda` selection.
- A downstream input-scale/context mismatch was found between cached KWS examples and runtime inference; cache-matched peak conditioning and a one-second KWS crop were added and re-evaluated.
- The AISHELL continuous comparison was audited against the checkpoint paths recorded in each JSON artifact; the final report distinguishes the prior hybrid checkpoint from the genuinely AISHELL-augmented checkpoint and does not use the earlier mislabeled comparison as an improvement claim.
- Presence calibration embeddings are cached under `data/cache/stage2_presence_gsc_aishell_16k/` and `data/cache/stage2_presence_gsc_aishell_richer_16k/`, so future head-only experiments do not recompute Stage 1 every epoch.
- A speaker-disjoint custom in-car manifest builder is available at `scripts/data/build_custom_incar_manifest.py`; it accepts real wake/non-wake recordings plus optional enrollment clips without changing the frozen TSE frontend.
- AISHELL-5 TextGrid-aligned target-present cache/manifests and the rejected head-only experiment are preserved under `data/cache/stage2_presence_aishell_target_16k/`, `manifests/presence_aishell_target/`, and `checkpoints/stage2_presence_aishell_target_ft1/`.
- The Stage 1 dataset loader now honors fixed TextGrid crop offsets, seeks only labeled windows from long AISHELL recordings, and normalizes multichannel manifests to the configured microphone count. This enabled the bounded `stage1_spex_aishell5_adapt_ft1` audit without changing the default checkpoint.

## Current limitations and risk level

### High-priority gaps

1. Stage 2 is not calibrated for the joint target-recall/false-accept operating point: `0.90` suppresses all proxy wake events, while `0.60` restores some recall at a high speech-triggered FAR.
2. There is no real in-car driver/passenger command benchmark.
3. There is no long-duration real target-present driver wake benchmark. AISHELL-5 now provides a 20-minute speech-rich target-absent hard-negative stream, but it has no enrolled-target wake positives and is not a substitute for a real driver/passenger command set.
4. TSE is evaluated mainly with SI-SDR on synthetic speech/noise mixtures.
5. TSE-domain intent is measured on a local validation split, not official noisy FSC test data.
6. There is no ASR/WER or action-execution evaluation after extraction.
7. The wav2vec intent encoder is too large for the intended edge footprint.

### Claims supported today

- The pipeline can substantially improve synthetic cabin mixtures with the current multichannel TSE frontend.
- The TSE-domain KWS candidate is clearly better than the original cabin KWS candidate on the current held-out test.
- The AISHELL-augmented KWS checkpoint improves the merged held-out F1 to 80.74%, but its validation-selected threshold is not calibrated to the real stream: it gives 30 smoothed events/hour at `0.81`; `0.95` is needed to reach 3/hour, with target-positive recall still weak.
- The cached presence fine-tune improves high-SNR target retention in the continuous proxy, but 0–10 dB recall remains inadequate and speech-triggered false events remain high.
- Stationary CAVEMOVE cabin noise alone produces no smoothed false wakes in the current continuous check.
- The wav2vec intent model is very strong on clean FSC and remains above 88% top-1 on the current synthetic cabin/TSE-domain validation conditions.
- The architecture is a coherent staged system: target-speaker extraction → target presence → TSE-domain wake detection → intent/action.

### Claims not supported today

- Production readiness.
- SOTA TSE or SOTA full-pipeline performance.
- Real-world cabin robustness under passenger speech, echo, wind, and reverberation.
- A validated end-to-end wake-word-to-action accuracy.
- Sub-100 ms CPU or automotive-DSP latency.

## Recommended next milestone

The highest-value next milestone is a **calibrated, official, reproducible benchmark release**:

1. Run a scale-aware Stage 1 adaptation study across AISHELL-5 and CAVEMOVE with fixed target-aligned crops, input normalization ablations, and a dual-domain selection metric; the first small frontend-only attempt improved AISHELL by only 0.65 dB and regressed CAVEMOVE.
2. Use `scripts/data/build_custom_incar_manifest.py` to keep real wake, passenger/non-wake, and enrollment recordings speaker-disjoint, then fine-tune presence/KWS on that set. The richer same-proxy presence run did not materially improve the eight-speaker result, so more rows from `marvin` alone are not the next priority.
3. Recalibrate or fine-tune KWS against the real continuous negative stream; do not use the merged-test threshold `0.81` as the deployment threshold. Evaluate wake recall/FAR jointly because `0.95` currently reduces false accepts at the cost of target recall.
4. Add official FSC clean test/validation scoring to every intent experiment.
5. Build a small real-cabin evaluation set containing driver wake words, passenger speech, HVAC/road noise, playback echo, and command labels.
6. Measure continuous false accepts/hour, target miss rate, intent accuracy, action accuracy, and latency on that set.

The current best checkpoints and raw benchmark artifacts are preserved under `checkpoints/` and `logs/`; this report is a status snapshot, not a claim that the system has completed production validation.
