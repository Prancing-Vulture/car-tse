# Pipeline failure audit and production improvement plan

**Snapshot:** 12 October 2026  
**Purpose:** Record the errors and limitations found while investigating poor end-to-end wake-to-intent results, and define the evidence-based improvement plan.  
**Scope:** Repository reports, local evaluation artifacts, evaluator/runtime code, and completed code audits. This is an evidence audit, not a claim that every historical run was independently reproduced.

## Short conclusion

Several earlier measurements and comparisons were invalid or misleading, and some have since been corrected. The most important unresolved test flaw is that the continuous proxy does not verify that the enrolled speaker, wake speaker, and command speaker are the same person. The corrected v10 trace now shows which source-time audio chunks were appended to Stage 4, but it does not prove which words survived target-speaker extraction.

The current evidence identifies weaknesses in Stage 1 transfer, Stage 2 operating-point calibration, Stage 3 false accepts, and the end-to-end proxy. **It does not yet prove one runtime code defect is the root cause of same-speaker pipeline failure.**

The existing tests are enough to stop broad, exploratory ablations and select the next engineering work. They are not enough to promise production performance, because the main continuous positive test does not verify a same-speaker wake-and-command interaction. The next experiments should be limited to (1) an instrumented same-speaker replay that locates audio loss or misclassification, and (2) a locked end-to-end acceptance test after changes.

## A. Confirmed implementation, data, and measurement errors

| # | Error found | Why it mattered | Status and evidence |
|---|---|---|---|
| 1 | **Stage 2 evaluator accepted a missing manifest and omitted the transfer backend.** | It could produce a result without evaluating the intended data/model path, making that result misleading. | Corrected. The benchmark report explicitly records both issues and says the reported result came from the corrected evaluator. See [benchmark audit](../project_benchmark_2026-10-03.md#verification-and-reproducibility-findings). |
| 2 | **An intent manifest presented as an official evaluation split was actually a 90/10 slice of the training CSV.** | It was not the official Fluent Speech Commands validation/test set, so it could not support an official-split claim. | An official-manifest builder was added. Existing training manifests and checkpoints were not changed. See the [benchmark audit](../project_benchmark_2026-10-03.md#verification-and-reproducibility-findings). |
| 3 | **Some Stage 3 cached examples did not match runtime audio scale and crop length.** | The detector was trained/evaluated on inputs different from what the runtime supplied; offline-to-runtime comparisons were therefore unreliable. | Cache-matched peak conditioning and a one-second KWS crop were added and re-evaluated. See [cache/runtime correction](../project_benchmark_2026-10-03.md#verification-and-reproducibility-findings) and [`build_gsc_marvin_tse_kws_cache.py`](../../scripts/data/build_gsc_marvin_tse_kws_cache.py). |
| 4 | **The Stage 1 AISHELL-5 loader mishandled some labeled audio.** | It did not honor fixed TextGrid crop offsets, seek the labeled spans in long recordings as intended, or consistently match the configured microphone count. This could feed the wrong audio region or channel shape to the model. | Corrected for the AISHELL audit. See the [benchmark audit](../project_benchmark_2026-10-03.md#verification-and-reproducibility-findings) and [`datasets.py`](../../src/car_tse/data/datasets.py). |
| 5 | **One earlier AISHELL KWS comparison mislabeled the checkpoint identity.** | It made the comparison appear to be between model variants that the saved result artifacts did not actually use. | Checkpoint paths in the result JSONs were audited; the later report distinguishes the hybrid and AISHELL-augmented checkpoints and drops the earlier comparison as an improvement claim. See the [benchmark audit](../project_benchmark_2026-10-03.md#verification-and-reproducibility-findings). |
| 6 | **An earlier pipeline evaluator attributed wake events using the decision-window center.** | A short wake can overlap a positive KWS crop even when that crop's center falls outside the labeled wake interval. The old rule could miss or misattribute a confirmed wake, especially at coarse hops. | The later evaluator attributes events using positive supporting crops and their overlap with the labeled wake. See [`eval_continuous_full_pipeline_proxy.py`](../../scripts/eval/eval_continuous_full_pipeline_proxy.py#L84) and the corrected [v8 diagnostic reports](./PIPELINE_DIAGNOSTIC_GUIDE.md#what-the-gate-and-timing-checks-show). |
| 7 | **The v9 Stage 4 capture-time estimate assumed the audio buffer was contiguous.** | It inferred the source interval from the latest stream time minus buffer length. Presence gating can skip chunks, so that calculation can assign the wrong source times and wrong command-overlap percentages. | Superseded by v10, which records the exact appended source segments. Do not use v9 overlap percentages or the earlier exact interval claim of 4.1–6.1 seconds. See the [corrected guide](./PIPELINE_DIAGNOSTIC_GUIDE.md#what-the-exact-command-buffer-trace-shows) and [v10 result](./diagnostic_v10_command_capture_kws029_two_snr.json). |
| 8 | **The generated continuous scenes pair wake and command clips without verifying the same speaker.** | Stage 1 extraction and the presence gate are speaker-conditioned; they may correctly suppress another person's command. Thus a failed command in this proxy is not a valid measurement of same-driver operation. | Still a limitation of the current proxy. The generator independently selects an intent row and labels scenes `speaker_consistent=false`. See [`eval_continuous_full_pipeline_proxy.py`](../../scripts/eval/eval_continuous_full_pipeline_proxy.py#L899) and [v10 limitations](./diagnostic_v10_command_capture_kws029_two_snr.json). |
| 9 | **The proxy has few independent target examples, and early short-run hourly rates were over-interpretable.** | Eight observations at two noise levels reuse four target clips. A handful of false events over a few seconds cannot establish a real false-alarms-per-hour rate. | Later reports explicitly call these engineering probes and do not extrapolate a defensible hourly rate. See the [continuous cutoff results](./PIPELINE_DIAGNOSTIC_GUIDE.md#continuous-cutoff-comparison). |
| 10 | **W&B epoch logging was opt-in, and some historical runs lacked useful epoch-by-epoch metrics.** | Being logged in to W&B does not itself send training metrics. The run command must enable logging; missing historical epochs cannot be reconstructed afterward. | Stage 2 and Stage 3 training scripts now include epoch metric logging. The source confirms the logging path, but not every old remote run history was independently checked. See [`train_stage2_frame_presence.py`](../../scripts/train/train_stage2_frame_presence.py#L564) and [`train_stage3_kws.py`](../../scripts/train/train_stage3_kws.py#L339). |
| 11 | **The proposed hard-negative cache path is not directly compatible with the available artifacts.** | The audit found an expected manifest missing and a two-second Stage 4 cache where the Stage 3 KWS experiment needs one-second examples. Reusing the wrong cache would compare or train on mismatched inputs. | This is a preparation blocker; the compatible manifest/cache must be generated before that experiment. It is not evidence of the current runtime failure. Relevant builders include [`build_kws_hybrid_cache.py`](../../scripts/data/build_kws_hybrid_cache.py) and [`build_stage4_fsc_cavemove_manifests.py`](../../scripts/data/build_stage4_fsc_cavemove_manifests.py). |

## B. What the corrected v10 command trace actually establishes

The v10 run repeated the eight proxy scenes at the same settings as the v9 trace. Its headline counts and model predictions were unchanged; the change was the capture-time instrumentation. The exact appended source intervals split the outputs into two groups:

- **Four late outputs at 7.5 seconds:** Stage 4 had only 0.5 seconds from source interval 6.0–6.5 seconds. That interval coincides with passenger-wake onset, not the labeled target command. The outputs were late and wrong.
- **Four outputs within the evaluator's deadline:** Three Stage 4 inputs covered source interval 2.5–4.5 seconds (80% overlap with the labeled command); one covered 2.0–4.0 seconds (95% overlap). All four predicted the wrong intent.

This establishes which source-time chunks were appended in these scenes. It **does not** establish that the target command was intelligible in the separated SpEx output, because source-time overlap alone cannot prove that the command words survived extraction. The speaker mismatch also prevents interpreting this as a valid same-driver trial. The corrected run is in the [v10 JSON](./diagnostic_v10_command_capture_kws029_two_snr.json) and [W&B](https://wandb.ai/narainharreesh-vit/car-tse/runs/itso0jkh).

## C. Measured model and pipeline weaknesses (not software bugs)

These are observed weaknesses or trade-offs. They should not be described as confirmed code defects.

### Stage 1: target-speaker extraction

The project reports a useful **+10.98 dB mean SI-SDR improvement** on held-out synthetic CAVEMOVE mixtures. In an AISHELL-5 real-cabin diagnostic, the current output was about −31.22 dB mean SI-SDR versus −37.31 dB for the input baseline. A short adaptation improved AISHELL by only about 0.65 dB and regressed held-out CAVEMOVE validation from 11.00 dB to 9.90 dB. This shows a domain-transfer problem; it does not prove the SpEx implementation is broken. See the [benchmark](../project_benchmark_2026-10-03.md#stage-1-target-speaker-extraction).

### Stage 2: presence gate

The pilot's test-set **chunk-level** F1 was 0.810 at threshold 0.40, with a 0.333 false-accept rate. At threshold 0.50 its chunk F1 was 0.786 and false-accept rate 0.267. The earlier model scored 0.786 chunk F1 at 0.40. These are not pipeline F1 scores. Chunk and frame metrics use different units and, in these reports, may use different operating thresholds; they must not be compared as if they were the same measurement. See [pilot results](./stage2_frame_presence_test.json) and [initial model results](./stage2_frame_presence_initial_test.json).

The continuous cutoff sweep shows the same recall/false-trigger trade-off: at wake cutoff 0.29 the proxy found 8/8 target wakes but also had 7 false wakes and 6 false actions; at 0.50 it found 4/8, with 4 false wakes and 4 false actions. No setting was acceptable on that small proxy. See the [cutoff results](./PIPELINE_DIAGNOSTIC_GUIDE.md#continuous-cutoff-comparison).

### Stage 3: wake-word detector

At the same 0.29 cutoff, the SpEx-output-trained candidate improved held-out clip F1 from 0.507 to 0.667 and caught 117/128 wake clips rather than 58/128. Its false accepts also increased from 43 to 106. In a separate three-seed sampler study, natural, 2:1, and 4:1 negative-to-positive sampling had almost identical mean validation F1 (about 0.684–0.685); the sampling change was not a reliable improvement. See the [diagnostic guide](./PIPELINE_DIAGNOSTIC_GUIDE.md#stage-3-clip-test) and [sampler study](./stage3_sampler_ratio_study.md).

### Stage 4: intent recognition

The reported 99.31% top-1 result is on clean official Fluent Speech Commands test audio. It does not establish accuracy on noisy post-SpEx command audio. In v10, all four in-deadline outputs whose source intervals overlapped most of the labeled command still predicted the wrong intent; because the voices were not matched and the separated words were not independently verified, the test cannot identify whether Stage 4 itself, extraction, or the pairing caused those errors.

## D. Leads that remain unproven

- **No confirmed runtime capture bug:** the completed code audit did not reproduce a defect in the wake-to-Stage-4 capture path. The runtime uses presence decisions and appends only accepted new audio; this behavior still needs evaluation on valid same-speaker continuous recordings.
- **Two unverified runtime edge cases:** switching between timestamped and untimestamped input calls, and silence-triggered inference occurring on the same frame as timeout. Neither has been established as the cause of an observed failure.
- **Timing is not a proven sole root cause:** the v10 trace shows four late outputs without target-command source-time audio and four wrong intents despite command-time overlap. It does not explain whether the actual command content survived extraction.
- **No SpEx cause established by the latest diagnosis:** v9-to-v10 work corrected evaluation instrumentation; it did not change SpEx weights. A prior bounded AISHELL adaptation checkpoint was evaluated and rejected for promotion because its CAVEMOVE result regressed.

## Recommended next test to resolve the open cause

Use continuous recordings in which the same enrolled person says the wake word and command. For each trial, retain the source audio, extracted Stage 1 output, every Stage 2 accept/reject decision, exact chunks appended to Stage 4, and the resulting intent. Then compare the known command crop, the separated command audio, and Stage 4's actual fixed-length input. This will distinguish:

1. a wake decision that arrives too late;
2. a presence gate that drops command speech;
3. extraction that removes or distorts the command; and
4. an intent model that misclassifies audio it actually received.

Until that test is run, keep the current 0/8 proxy result labeled as a cross-corpus engineering diagnostic, not as a valid same-speaker pipeline score.

## E. Recommended changes by stage

These are **proposed changes, not changes already implemented**. The aim is to improve the full wake-to-action path, not to maximize isolated stage scores. No numerical gain is promised before measurement.

| Stage | Evidence-based problem | Recommended change | What it should improve |
|---|---|---|---|
| **0 — Audio input and timing** | The v9 report could not map buffered audio correctly when a gate skipped chunks. The v10 instrumentation fixed the report, but actual same-speaker command preservation has not been validated. | Keep one timestamped rolling buffer with the source-time span for every chunk. On wake confirmation, preserve the relevant post-wake audio and continue collecting until a command endpoint is detected. Record the exact waveform and source spans delivered downstream. The runtime already has timestamp overlap handling and conditional pre-roll; verify and strengthen this behavior rather than assuming it is absent. | Fewer commands lost because wake confirmation or a gate arrives late; reliable proof of what audio each downstream stage received. |
| **1 — Target-speaker extraction (SpEx)** | Stronger synthetic CAVEMOVE results do not transfer cleanly to AISHELL-5. The short AISHELL adaptation made only a small AISHELL gain and regressed CAVEMOVE validation. | For the first remediation pass, **keep the current SpEx code and default checkpoint frozen**. Build same-speaker driver enrollment/wake/command recordings with passenger and cabin-noise examples. Compare extracted driver speech with the known command, including intelligibility and downstream command preservation. Consider adaptation only if this evidence identifies extraction as a limiting stage. | Avoid spending effort changing SpEx without evidence; if extraction is the problem, train against the acoustic conditions and speaker interactions the vehicle pipeline actually faces. |
| **2 — Target-presence gate** | The Stage 2 pilot had chunk F1 about 0.81 at threshold 0.40, but a 0.333 false-accept rate. Higher thresholds can suppress target speech; lower thresholds admit more speech from non-target sources. | Train/adjust the gate on frozen Stage 1 outputs from matched driver-positive and passenger/radio/no-target examples. Calibrate its cutoff on continuous development recordings against a declared false-action budget and target miss rate. Preserve per-chunk accept/reject logs. | Keep more genuine driver command audio while limiting passenger/noise audio admitted to later stages. |
| **3 — Wake-word detection** | The candidate's clip F1 improved from 0.507 to 0.667 at the same cutoff, but false accepts rose from 43 to 106. The sampler-ratio study did not produce a repeatable gain. Continuous cutoff sweeps also showed a recall/false-trigger trade-off. | Train using hard non-wake examples from the same post-SpEx/runtime distribution: passenger speech, phonetic confusables, radio, and cabin noise. Tune score cutoff, temporal confirmation, and refractory behavior together on continuous development data. Do not lower the cutoff alone. | Better driver-wake recall without a corresponding increase in false wake/action events. |
| **4 — Intent recognition and action** | Clean FSC top-1 accuracy does not show how the model handles noisy separated cabin commands. In v10, four timely outputs overlapped the command's source time but were all classified incorrectly; the speakers were mismatched and the separated word content was not verified. | After Stage 0–1 audio is validated, create correctly aligned post-SpEx command inputs from same-speaker continuous recordings and fine-tune/evaluate Stage 4 on those inputs. Add a confidence-based abstain response; require confirmation for consequential actions. Evaluate compression or distillation only after accuracy is stable. | More correct intents on real pipeline audio, fewer confident wrong actions, and eventually a smaller model for deployment. |

### Cross-stage behavior

Stages 2 and 3 must be calibrated together. Stage 2 decides which audio Stage 3 can see; Stage 3 decides whether that audio is a wake word. Optimizing either one alone can improve its local score while worsening full-pipeline false actions or missed commands. Select their joint operating point using complete continuous sessions, then freeze it before final testing.

The complete pipeline must also be evaluated directly. If five serial steps each had 90% success and failures were independent, the full-path success would be about 59% (`0.90^5`). Even 98% at each step would be about 90% end-to-end (`0.98^5`). Actual stage failures are not necessarily independent, so these are illustrations; full-pipeline results, not this multiplication, decide acceptance.

## F. Minimum remaining test plan

No further random threshold sweep, sampler-ratio sweep, or small cross-corpus scene set is recommended. The following two tests answer specific unresolved questions.

### Test 1 — Instrumented same-speaker replay

Use continuous recordings where the enrolled person says the wake word and then a labeled command. Include a few representative cabin conditions and delays between wake and command. Save, with timestamps:

1. source microphone audio;
2. Stage 1 extracted audio;
3. every Stage 2 score and accept/reject decision;
4. the exact samples and source-time spans appended to Stage 4;
5. the exact fixed-length Stage 4 waveform and predicted intent.

This test determines whether the main failure is late wake confirmation, Stage 2 dropping the command, Stage 1 losing/distorting the driver's words, or Stage 4 misclassifying words it received. It is a diagnostic test, not the final production score.

### Test 2 — Locked end-to-end acceptance test

After changes are selected using development recordings, evaluate once on held-out speakers and sessions (and held-out vehicles where available). Include target-present driver commands and long target-absent stretches with passengers, media/playback, and representative vehicle noise. Do not tune thresholds on this final set.

Report at minimum:

- complete wake-to-correct-action success and the number of trials;
- target wake misses and false wakes;
- false actions per hour on target-absent recordings;
- command intent accuracy after a confirmed wake;
- median and 95th-percentile wake-to-action latency;
- results by speaker, vehicle/noise condition, and command type;
- uncertainty intervals, with speakers/sessions—not overlapping audio windows—as the independent units where applicable.

The test must be long enough to support the chosen false-action limit. A few short scenes cannot establish a low hourly false-action rate. The product owner must define the allowable false-action rate based on the consequences of each action; there is no universal production threshold.

## G. Production acceptance and execution rules

- Treat **greater than 90% full-pipeline success** as a minimum project goal only if that matches the intended use. It is not, by itself, a safety or production criterion. Set a separate false-action budget, latency limit, and abstention behavior.
- Report component metrics for diagnosis, but accept or reject the system using held-out full-pipeline outcomes.
- Keep data partitions speaker/session/vehicle-disjoint as appropriate. Use development data for model and threshold selection and a locked test set only for final reporting.
- For future coding/training/evaluation runs, use the requested `audioml_base` environment and enable W&B explicitly. Log configuration, split/manifest identifiers, checkpoint hashes, train/validation metrics by epoch, selected thresholds, and final test metrics. Do not use final test results for epoch-wise model selection.
- Keep SpEx source and default weights unchanged during the first remediation pass. Revisit them only if the instrumented same-speaker test shows that the driver's command is lost or materially damaged by extraction.
- No model retraining or runtime change is claimed by this document. It records the proposed work and the evidence needed to justify each change.
