# Research improvement plan for the car-TSE voice assistant

**Evidence snapshot:** 7 October 2026  
**Purpose:** Research-focused improvement plan and publication-readiness assessment  
**System:** Four-stage target-speaker-conditioned in-car voice-assistant prototype

## Abstract

This report assesses the `car-tse` repository and proposes a prioritized, evidence-led path toward a publishable in-car voice-assistant study. The system combines multichannel cabin processing, SpEx+-style target-speaker extraction (TSE), target-presence gating, wake-word detection, and intent classification. Existing repository measurements show a +10.98 dB mean SI-SDR improvement on held-out synthetic CAVEMOVE mixtures and 99.31% top-1 accuracy on the official clean Fluent Speech Commands test set. Those results do not establish driver-versus-passenger selectivity or reliable continuous wake-to-action performance. The repository itself documents a speech-rich AISHELL-5 target-absent stream with zero gated full-path false events in a selected configuration, alongside very high KWS-only false-event rates and weak/cross-corpus positive proxies. This report therefore prioritizes (1) a speaker-consistent real-cabin benchmark, (2) operating-point calibration using continuous false accepts per hour and target miss rate, (3) real multichannel TSE adaptation and ablation, (4) command/action robustness evaluation, and (5) streaming and edge optimization. Research literature supports the relevance and technical feasibility of these directions, but does not warrant predicting a numerical gain for this repository before controlled experiments. Estimated effort is expressed as person-weeks and assumes one researcher with repository and dataset access.

**Keywords:** target-speaker extraction; in-car speech; keyword spotting; continuous evaluation; speaker selectivity; reproducibility

## 1. System and evidence baseline

The intended path is:

```text
4-channel cabin audio (+ optional playback reference)
  → cabin frontend / target-speaker extraction
  → target-presence gate
  → temporal wake-word detector (“marvin”)
  → command capture and intent classification
  → action mapping
```

The checked-in project reports provide the following current evidence:

| Subsystem | Reported result | What it establishes—and does not establish |
|---|---|---|
| TSE | +10.98 dB mean SI-SDR improvement on 400 held-out synthetic CAVEMOVE mixtures | Useful synthetic-cabin enhancement baseline; not real driver/passenger selectivity, intelligibility, or action success |
| Real-cabin TSE diagnostic | AISHELL-5 target-present output around −31.22 dB mean SI-SDR versus −37.31 dB input; a short frontend adaptation gained only about 0.65 dB and regressed CAVEMOVE validation | A measurable domain gap, not evidence that AISHELL adaptation is solved |
| Presence gate | On an imbalanced validation split, threshold 0.50 gave 96.25% FAR; a high threshold around 0.90 reduced FAR but suppressed positives in proxy evaluation | Threshold and calibration need a representative joint positive/negative set |
| KWS | Strong narrow TSE-domain isolated-window scores; higher hard-negative and AISHELL continuous-stream false-event rates | Isolated F1 does not predict continuous stream operation; both target recall and false accepts/hour are needed |
| Intent | 99.31% top-1 on official clean FSC test; about 90% on local cabin/TSE-domain validation | Strong clean intent benchmark; not real post-wake, post-TSE command accuracy |
| End-to-end | Continuous examples are cross-corpus, with target wake and command not verified as the same speaker; one reported proxy achieved 25% wake recall and zero correct final intents | Diagnostic only; the positive path is not yet a valid same-driver evaluation |
| Runtime | About 69.95 ms GPU and 275.34 ms CPU model-forward sum for the benchmarked one-second input; current TSE uses a non-causal three-second context | Not end-to-end streaming latency. Wav2vec encoder is about 378 MB of a roughly 424 MB checkpoint total |

The underlying snapshots and caveats are recorded in [project_benchmark_2026-10-03.md](project_benchmark_2026-10-03.md), [continuous_pipeline_ablation_2026-10-04.md](continuous_pipeline_ablation_2026-10-04.md), and [continuous_full_pipeline_proxy_2026-10-04.md](continuous_full_pipeline_proxy_2026-10-04.md). Reported numbers are repository results and have not been independently rerun for this document.

## 2. Main assessment

The strongest research contribution available in the current project is not a claim of state-of-the-art component accuracy. It is a carefully evaluated, enrollment-conditioned, multichannel cabin speech pipeline and a reproducible analysis of how errors propagate from extraction through action selection. The critical missing experiment is a speaker-consistent continuous cabin benchmark: the same enrolled driver must provide the enrollment audio, wake word, and command, while passenger speech, other speakers, road/HVAC noise, and playback occur in annotated conditions.

This gap matters because target-speaker extraction is explicitly conditioned on identity. Cross-corpus pairing of a wake word and command from different speakers may cause a correct extractor to suppress the command. Separately, isolated KWS accuracy can coexist with excessive real-stream false alarms; work on real-world KWS false-alarm benchmarking has demonstrated this mismatch ([Sánchez Deutsch et al., 2023](https://doi.org/10.3233/FAIA230695)).

## 3. Recommended approaches

Effort estimates below are implementation/research estimates, not results reported by cited papers. They assume one experienced researcher, access to the existing local datasets/checkpoints, and no institutional review delay. New data collection effort is marked separately because recruiting, consent, vehicle access, and annotation may dominate elapsed time.

| Priority / approach | Proposed work and rationale | Expected improvement to this project | Estimated effort | Feasible? |
|---|---|---|---:|---|
| **P0 — Speaker-consistent continuous cabin benchmark** | Record or curate sessions where enrollment, “marvin,” and commands come from the same driver. Include passenger wake/non-wake speech, overlapping speech, multiple cabin states, playback, and noise. Split by speaker/session/vehicle; annotate events and action labels. Use AISHELL-5 for supplementary multichannel adaptation/diagnostics, not as a substitute for the target-language, same-driver wake benchmark. AISHELL-5 supplies four far-field channels, near-field references in training, and real vehicle scenarios ([paper](https://www.isca-archive.org/interspeech_2025/dai25c_interspeech.pdf), [official code](https://github.com/DaiYvhang/AISHELL-5)). | Converts proxy results into interpretable target recall, false activations/hour, wake-to-action accuracy, and subgroup/condition analysis. Highest value for scientific validity; no numerical gain is assumed. | 3–6 weeks engineering/annotation; **+4–12+ weeks elapsed** if new recording/recruitment is required | **Yes**, dependent on lawful data access and recording resources |
| **P0 — Define primary metrics and calibrate cascade jointly** | Choose operating points on development sessions only. Report target wake recall/miss rate, false accepts and false actions per hour, wake latency, command intent/action accuracy conditional and unconditional on wake success, plus confidence intervals. Freeze thresholds before held-out evaluation. Include per-stage ablations and error attribution. | Prevents optimizing accuracy/F1 on unrepresentative clips; directly addresses repository’s threshold-sensitive presence gate and KWS stream mismatch. Produces a defensible comparison protocol. | 1–2 weeks after a benchmark exists | **Yes** |
| **P1 — Real multichannel TSE and spatial ablations** | Compare current SpEx+ baseline with (a) mono/mean channel, (b) learned cabin frontend, and (c) spatially conditioned or seat-aware alternatives. Train/adapt on real multichannel target/reference pairs where permitted. Evaluate SI-SDRi plus intelligibility and speaker confusion; preserve vehicle/session-disjoint test data. SpEx+ and SpeakerBeam establish reference-conditioned extraction as a viable family ([SpEx+ code](https://github.com/xuchenglin28/speaker_extraction_SpEx), [SpeakerBeam paper/code](https://doi.org/10.1109/JSTSP.2019.2922820), [implementation](https://github.com/BUTSpeechFIT/speakerbeam)). | Tests whether the current synthetic gain transfers and whether spatial cues improve rejection of passenger speech. Likely to yield more meaningful evidence than another narrow KWS fine-tune; effect size is unknown. | 3–6 weeks | **Yes**, with aligned real multichannel supervision; **partial** if only existing corpus splits are available |
| **P1 — Train/select against hard negatives and open-set speakers** | Expand negative conditions with non-enrolled speakers, passenger speech, phonetic confusables, reverberant and far-field speech, and playback. Keep speaker-disjoint train/dev/test and calibrate on continuous sessions. Train presence and KWS with the same extraction/normalization/runtime distribution. | Targets the dominant observed failure mechanism (speech leakage and distribution/calibration mismatch); may reduce false actions while preserving target recall, but must be measured at a fixed operating point. | 2–4 weeks after labeled data/cache preparation | **Yes** |
| **P1 — Improve objective and downstream utility evaluation** | Retain SI-SDRi for signal fidelity, add STOI/ESTOI or ASR/command preservation, target-speaker confusion, and action correctness. Report outcomes for the full cascade, not only enhanced audio. Add no-TSE, no-presence, and oracle-gate comparisons. | Separates “cleaner waveform” from “the intended driver’s command was preserved and acted on correctly”; reveals stage interactions. | 1–3 weeks | **Yes** |
| **P2 — Streaming TSE redesign / causal inference** | Measure algorithmic look-ahead, real-time factor, p50/p95 end-to-end latency, memory, and sustained load. Prototype a causal/lightweight TSE baseline against the current non-causal system. SpeakerBeam-SS reports a real-time TSE architecture and a 78% real-time-factor reduction versus its conventional causal Conv-TasNet baseline while matching its performance on its evaluated setup; this is a feasibility precedent, not a guaranteed result here ([paper](https://arxiv.org/abs/2407.01857)). | Could reduce delay and compute, but may trade off extraction quality. Establishes whether the 3-second context can be removed for the application. | 4–8 weeks | **Yes as a research prototype;** production suitability needs target hardware |
| **P2 — Reduce intent model footprint** | Distill the wav2vec intent model into a compact acoustic encoder or small-footprint KWS/SLU model; evaluate structured pruning/quantization only after defining the hardware target. Compare accuracy, memory, power, and latency. | Addresses the dominant model size (about 378 MB) and CPU latency. Any accuracy/size trade-off is empirical; no claimed compression ratio is assumed. | 3–6 weeks | **Yes**, target device required for final claim |
| **P2 — Playback echo and vehicle condition evaluation** | Use the playback reference where available; create labeled playback-on/off and HVAC/road/wind/window conditions. Measure echo-triggered false accepts and target recall separately. AISHELL-5 includes stereo reference signals for AEC in sessions where the car stereo is on ([official repository](https://github.com/DaiYvhang/AISHELL-5)). | Establishes robustness to a major unmeasured cabin disturbance and determines whether the current reference branch helps. | 2–4 weeks once data exists | **Partial**; reference-aligned playback data is required |
| **P3 — Threshold-free or calibrated confidence analysis** | Report DET/ROC curves and calibration (e.g., reliability plots/Brier score) for presence and KWS, but select the actual operating point using continuous-stream utility and false-accept constraints. | Makes threshold behavior transparent and supports reproducible configuration selection; does not replace continuous validation. | 1 week | **Yes** |

## 4. Minimum publishable experimental design

For a credible journal submission, the next study should be designed around a pre-registered evaluation protocol rather than another sequence of isolated candidate fine-tunes:

1. **Define the target claim narrowly.** For example: *“An enrollment-conditioned multichannel cascade for driver-directed wake-word and intent recognition under passenger speech and cabin noise.”* Avoid “production-ready” or SOTA without matched baselines and target-hardware evidence.
2. **Create independent partitions.** Keep speakers and, where possible, vehicles and recording sessions disjoint. Never tune thresholds on the final test streams. Publish manifest-generation rules and data licenses.
3. **Use matched baselines.** Compare raw single microphone, fixed channel/mean, current learned frontend, current TSE, and relevant reference-conditioned extraction baselines. Ensure each baseline sees identical audio and enrollment references.
4. **Report cascade and component results.** Primary outcomes should include target wake recall, false wakes/hour, false actions/hour, wake latency, command intent/action accuracy, and confidence intervals. Secondary outcomes may include SI-SDRi, STOI/ESTOI, KWS precision/recall, and model footprint. Explain event-merging/refractory rules used for hourly rates.
5. **Use repeated or stratified analyses.** Stratify by driver/passenger, overlap, cabin noise/SNR, vehicle, microphone channel, playback, and speaker familiarity. Estimate uncertainty at the session or speaker level rather than treating overlapping windows as independent trials.
6. **Release sufficient artifacts.** Provide versioned configuration, checkpoint identifiers/hashes, preprocessing/cache lineage, evaluation scripts, threshold-selection protocol, and non-sensitive metadata/audio where consent and license allow.

## 5. Recommended sequence and decision gates

**Gate A — Valid positives.** Do not claim end-to-end success until an independent set contains same-driver enrollment, wake, and command instances. Existing cross-corpus proxies should remain explicitly labeled diagnostic.

**Gate B — Useful operating point.** Select presence and KWS thresholds on development streams at a stated false-action budget, then report target recall and action success on untouched test streams. A setting that achieves zero false wakes by rejecting all target speech is not an improvement.

**Gate C — Generalization.** Demonstrate performance across speakers and at least multiple cabin/noise conditions. Compare the learned spatial frontend with simple channel baselines to show the multichannel module contributes measurable value.

**Gate D — Deployment claims.** Only claim real-time/edge readiness after causal streaming and target-device p50/p95 latency, memory, and power measurements. The current model-forward figures exclude important latency components and are not enough for this claim.

## 6. Conclusion

The repository has a coherent research prototype, useful synthetic TSE results, strong clean intent classification, and enough evaluation infrastructure to support a stronger paper. Its limiting issue is validity of the positive end-to-end evidence: target-speaker enrollment, wake, and command are not yet consistently represented in the same real continuous cabin recordings. The highest-return next step is therefore a speaker-consistent benchmark with locked evaluation and continuous false-action metrics, followed by multichannel TSE and hard-negative experiments on that benchmark. Streaming redesign and intent-model compression are worthwhile second-phase studies, but should follow the accuracy and data-validity milestone. Literature and released code establish that these approaches are technically grounded; only controlled experiments on this project can establish their gains.

## References

1. Ge, M. et al. “SpEx+: A Complete Time Domain Speaker Extraction Network.” *Interspeech* (2020). [Paper](https://www.isca-archive.org/interspeech_2020/ge20_interspeech.html) · [Official implementation](https://github.com/xuchenglin28/speaker_extraction_SpEx).
2. Žmolíková, K. et al. “SpeakerBeam: Speaker Aware Neural Network for Target Speaker Extraction in Speech Mixtures.” *IEEE JSTSP* 13(4), 800–814 (2019). [DOI](https://doi.org/10.1109/JSTSP.2019.2922820) · [Implementation](https://github.com/BUTSpeechFIT/speakerbeam).
3. Sato, H. et al. “SpeakerBeam-SS: Real-time Target Speaker Extraction with Lightweight Conv-TasNet and State Space Modeling.” (2024). [Paper](https://arxiv.org/abs/2407.01857).
4. Dai, Y. et al. “AISHELL-5: The First Open-Source In-Car Multi-Channel Multi-Speaker Speech Dataset for Automatic Speech Diarization and Recognition.” *Interspeech* (2025). [Paper](https://www.isca-archive.org/interspeech_2025/dai25c_interspeech.pdf) · [Dataset/baseline repository](https://github.com/DaiYvhang/AISHELL-5).
5. Sánchez Deutsch, S., Huerta Casado, I., and Escrig Escrig, J. “A Real-World Dataset for Benchmarking False Alarm Rate in Keyword Spotting.” (2023). [DOI](https://doi.org/10.3233/FAIA230695).

