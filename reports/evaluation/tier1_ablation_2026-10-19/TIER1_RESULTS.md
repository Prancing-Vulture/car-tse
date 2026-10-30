# Tier-1 ablation results

This report records the 2026-10-19/20 run against the selected local Stage-1 checkpoint. The selected checkpoint SHA-256 is `13ec7c58870feb975efb266e4064bf58575dbc6120f787a77a49c3bc4edda3ac`.

## TSE ablations

The held-out set contains 3,000 LibriMix two-speaker examples, 500 per CAVEMOVE cabin-noise SNR from -5 to +20 dB. Target/passenger speech is duplicated identically across all four input channels; only the recorded cabin noise differs between channels. Passenger SIR is estimated from the aligned target and mixture waveforms. The per-example table contains the complete breakdown by SNR, target/passenger gender pairing, and estimated SIR bin.

SI-SDRi is relative to the four-channel delay-sum mean. Values below are mean ± standard deviation across test examples, not across training seeds. PESQ and STOI are wideband PESQ and standard STOI at 16 kHz.

| Variant | SI-SDRi (dB) | PESQ-WB | STOI |
|---|---:|---:|---:|
| Delay-sum only | 0.00 ± 0.00 | 1.085 ± 0.088 | 0.633 ± 0.099 |
| Oracle-noise MVDR only | 1.57 ± 2.00 | 1.094 ± 0.067 | 0.642 ± 0.099 |
| Public pretrained SpEx+ (WSJ0-2mix, no cabin fine-tuning) | 3.09 ± 7.60 | 1.176 ± 0.166 | 0.630 ± 0.183 |
| Public TD-SpeakerBeam (LibriMix) | 5.61 ± 6.24 | 1.234 ± 0.183 | 0.671 ± 0.165 |
| Local transfer-SpEx+ initialization, no cabin fine-tuning, delay-sum mean | 9.92 ± 3.96 | 1.310 ± 0.201 | 0.770 ± 0.120 |
| Selected SpEx+ on single channel 0 | 9.17 ± 3.97 | 1.284 ± 0.198 | 0.752 ± 0.129 |
| Selected SpEx+ on delay-sum mean | 9.90 ± 3.97 | 1.308 ± 0.201 | 0.770 ± 0.120 |
| Selected learned front end, 1-channel test-time mask | -0.44 ± 4.47 | 1.138 ± 0.135 | 0.595 ± 0.162 |
| Selected learned front end, 2-channel test-time mask | 8.80 ± 3.75 | 1.251 ± 0.178 | 0.749 ± 0.130 |
| Selected learned front end, all 4 channels | 10.20 ± 4.04 | 1.312 ± 0.202 | 0.773 ± 0.120 |
| Oracle-noise MVDR + selected SpEx+ | 10.83 ± 4.30 | 1.347 ± 0.209 | 0.791 ± 0.111 |

On this set, the selected four-channel learned front end adds about 0.30 dB SI-SDRi over the same selected SpEx+ model applied to delay-sum mono. It adds 1.40 dB over the two-channel test-time mask, while the one-channel mask fails. Oracle MVDR plus SpEx+ is about 0.63 dB above the learned front end; it uses the paired clean cabin-noise track for covariance, so it is an oracle comparator. MVDR alone adds only 1.57 dB over the delay-sum reference. The learned system's advantage is therefore mainly the extraction model; the added learned front-end gain over mean-mono is modest on this dataset.

The local initialization is the matched no-cabin-fine-tuning control. Its SpEx+ tensors were loaded with restricted tensor-only loading. Comparing its 456 tensors with the selected checkpoint shows only 18 changed values, all batch-normalization running means, variances, and counters. The selected SpEx+ on mean-mono scores 9.897 dB versus 9.920 dB for the local initialization: a paired difference of -0.023 ± 0.233 dB. The four-channel learned front end scores 10.201 dB, a paired gain of +0.280 ± 0.951 dB over the local initialization. This points to a modest front-end benefit; the public WSJ0 SpEx+ score is a useful external reference but is not a controlled adaptation comparison because its architecture and training recipe differ.

For estimated passenger SIR ≤ -3 dB (1,197 examples), mean SI-SDRi is 10.48 dB for the selected four-channel learned front end and 10.17 dB for selected SpEx+ on delay-sum mono. For >3 dB SIR (223 examples), they are 9.39 and 9.10 dB. By gender pair across all SNR/SIR cells, the selected front end averages 9.38 dB for same-gender pairs (1,486) and 11.01 dB for different-gender pairs (1,514). The local initialization averages 9.12 dB for same-gender and 10.70 dB for different-gender pairs. The complete per-SNR × gender × SIR table, including the matched local control, is in `tse/summary_by_snr_gender_sir_complete.csv`; per-example rows are in `tse/per_example_metrics_complete.jsonl`.

These results support noise robustness on the synthetic cabin-noise benchmark. They do not demonstrate spatial separation of passenger speech: target and passenger speech have identical waveforms in every microphone channel. The 1/2-channel frontend comparisons are test-time ablations that retain the first N channels and zero the rest; they are not independently trained N-channel models. The MVDR covariance uses the paired noise-only recording and is not deployable as evaluated.

## Seed stability

The three seed runs are four-epoch continuation fine-tunes initialized from the selected checkpoint; they retrain the learned front end, while SpEx+ remains frozen. These are seed-stability checks for front-end training, not three independent retrains from the original pretraining weights. Values below are mean ± sample standard deviation across the three seed-specific held-out SI-SDRi means. The detailed strata by cabin SNR, gender, and passenger-SIR bin are in `tse_seed_variance/summary_mean_std_over_seeds_by_snr_gender_sir.csv`.

| Active input channels | Seed mean ± std (dB) |
|---:|---:|
| 1 | 5.589 ± 0.147 |
| 2 | 9.088 ± 0.006 |
| 4 | 10.299 ± 0.003 |

The channel counts are test-time masks on four-channel-trained checkpoints, with the unused channels zeroed. The near-identical 2/4-channel behavior across seeds shows the ordering is stable under these continuation seeds; it does not turn the channel masks into separately trained microphone-count models.

## Continuous pipeline ablations under overlapping passenger speech

The pipeline matrix contains 90 synthetic 14-second scenes: 10 scenes at each combination of cabin SNR (-5, 0, +10 dB) and requested target-to-passenger SIR (-10, -5, 0 dB). Target and passenger wake/non-wake speech overlaps in the target command interval, with CAVEMOVE noise present. Actual co-active speech SIR is computed over samples where both speakers are active; it averaged -11.11, -6.11, and -1.11 dB for the three requested settings. The same selected Stage-1 checkpoint is used with the hash-manifested FT6 ONNX presence, generic KWS, target-conditioned KWS, compact verifier, and intent graphs. Hashes and scene details are recorded in each JSON run file.

| Requested target/passenger SIR | Actual overlap SIR mean (range) | Full TSE: wake / command capture / correct intent | Raw mono, no TSE: wake / command capture / correct intent | Direct command oracle accuracy |
|---:|---:|---:|---:|---:|
| -10 dB | -11.11 (-16.80 to -10.01) | 100% / 86.7% / 66.7% | 53.3% / 46.7% / 30.0% | 100% |
| -5 dB | -6.11 (-11.80 to -5.01) | 100% / 80.0% / 73.3% | 56.7% / 46.7% / 30.0% | 100% |
| 0 dB | -1.11 (-6.80 to -0.01) | 100% / 53.3% / 50.0% | 63.3% / 50.0% / 33.3% | 100% |

The percentages in each system column are scene-level wake recall, target-command capture rate, and joint correct-intent rate over 30 scenes. Conditional intent accuracy, cabin-SNR breakdowns, and every variant are in `pipeline_matrix/summary_by_requested_sir_cabin_snr.csv` and `pipeline_matrix/summary_by_requested_sir.csv`. In the hardest tested cell (requested -10 dB target/passenger SIR and -5 dB cabin SNR; 10 scenes), the full TSE path detected all 10 target wakes and completed 7 commands correctly; the raw/no-TSE path detected 2 wakes and completed 1 command correctly. This is the overlap condition where TSE changes the task outcome, unlike the near-equal aggregate intent scores on the less demanding data.

Removing the presence gate produced 78–86 false wake events and 60 false actions in each 30-scene SIR group, while the full gated path had no false actions in these scenes. The target-conditioned wake model without an extra verifier had 100% wake recall across these groups; generic KWS on TSE output had 0% recall, while generic KWS on the raw/no-TSE path had 53–63%. At threshold 0.15, adding the compact verifier to the target-conditioned path gave 60%, 60%, and 63% wake recall, with 30%, 13%, and 7% command capture at requested SIR -10, -5, and 0 dB. The separate speaker-similarity gate gave 23%, 37%, and 60% wake recall and 17%, 20%, and 23% command capture. Generic KWS plus the compact verifier had 0% wake recall. Neither verifier variant reduced false actions relative to the full path's already-zero count in these scenes. These threshold-specific results do not show that a verifier is generally harmful: wake clips are from Google Speech Commands while command/enrollment clips are from Fluent Speech Commands, so identity across wake and command is not established and a verifier can reject a cross-corpus positive.

The error-oracle results help localize the remaining gap. Direct inference on the labeled clean command crop reached 100% intent accuracy, while full streaming with oracle extraction reached 70% joint correct intent across the SIR groups. Oracle gating changed the result only modestly. This points to streaming command capture/timing and the cross-corpus setup as material error sources in addition to extraction. The actual intent classification result conditional on a captured command is reported separately from joint success in the summary CSV.

These are small cross-corpus diagnostic proxy runs, not real-driver evaluation or production reliability evidence. Each cabin-SNR × passenger-SIR cell has 10 scenes. Enrollment uses a separate utterance from the FSC command speaker, but the Marvin wake speaker is unverified; the three corpora do not provide continuous same-speaker wake-plus-command recordings. All raw per-scene outputs and exact model hashes are preserved under `pipeline_matrix/`.

## Reproducibility and model safety

The downloaded public SpEx+ and TD-SpeakerBeam weights, source revisions, licenses, SHA-256 values, strict loading checks, and the rejected epoch-0 archive are documented in `MODEL_SOURCES.md`. Main TSE results and the matched local-initialization control are combined in `tse/complete_run_manifest.json`; seed and pipeline run manifests preserve their source hashes and settings. The manuscript TeX source was not changed.
