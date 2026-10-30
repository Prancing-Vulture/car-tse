# Wake-to-intent remediation plan

**Snapshot:** 12 October 2026  
**Purpose:** Turn the pipeline audit, primary research, diagnostic-readiness check, and repository tests into a staged plan that locates failures before changing models. This is an engineering plan; it is not a production-readiness claim or a forecast of score gains.

## Current decision

The next step is to validate the diagnostic recorder and its manifest with unit and parity tests, then run it on a continuous recording where the enrolled person says both the wake phrase and the labeled command. No eligible same-person recording is present in the checked custom in-car folders, so that replay is currently blocked on data. Until a matched recording shows where words are lost, keep runtime behavior, model weights, and SpEx unchanged. Do not launch another threshold or sampler sweep.

The audit's 0/8 result is a cross-corpus debugging result: the wake and command clips were not verified to be from the same person. v10 corrects the old v9 buffer-time calculation and records source intervals actually appended in those scenes. Four inputs had only a late 0.5-second interval; four overlapped most of the labeled command but received the wrong intent. Source-time overlap does not establish that the separated waveform contained the command words. These observations do not identify a same-driver root cause.

## Evidence and current limits

The repository audit records confirmed historical measurement/data errors and their corrections. It also records that the present runtime review did not reproduce a capture defect. Stage 1 has better results on synthetic CAVEMOVE audio than on the AISHELL real-cabin diagnostic; the short AISHELL adaptation did not justify promotion because its held-out CAVEMOVE validation regressed. Stage 2 and Stage 3 show operating-point trade-offs, and the three-seed sampler-ratio study did not identify a repeatable winner. Stage 4's clean Fluent Speech Commands score does not establish accuracy on noisy post-SpEx commands. These component results help describe risks; only held-out full-pipeline results can support acceptance.

The repository validation snapshot ran in `audioml_base`: the focused pipeline/evaluator suite passed 15 tests, and the then-current full suite passed 22 with one pretrained-checkpoint test skipped. Existing coverage exercises one overlapping-window deduplication case, one framewise early pre-roll case, and variable-size timeout accumulation. It does not cover rejected-gap provenance, timestamped/untimestamped transitions, simultaneous endpoint and timeout, or long-command retention. These are test gaps, not known causes. See the [validation report](REPO_VALIDATION_2026-10-12.md) for commands, environment, and exact caveats.

The matched replay is not ready: the configured custom wake, command, and playback folders are empty. Speech Commands and Fluent Speech Commands have separate speaker identities, so their speaker tokens cannot be joined to fabricate a same-person example. The [readiness report](MATCHED_DIAGNOSTIC_READINESS_2026-10-12.md) records this data check and the proposed trace contract.

Research supports the order of work, not a numeric gain prediction. SpEx+ is conditioned on a target reference, so identity must be matched before judging separation. Streaming KWS research measures accuracy together with latency; endpointing research studies early cutoff against delay; intent calibration and selective prediction motivate a measured repeat/defer path once the input is trustworthy. Hard-negative KWS methods motivate diverse, difficult training examples, while false-alarm testing guidance shows why seconds of negative audio cannot establish an hourly event rate. The verified primary sources and dataset compatibility checks are in the [primary research report](PIPELINE_PRIMARY_RESEARCH_2026-10-12.md), including [SpEx+](https://www.isca-archive.org/interspeech_2020/ge20_interspeech.html), [streaming KWS](https://doi.org/10.21437/Interspeech.2020-1003), [hard-negative KWS](https://doi.org/10.1109/ICASSP40776.2020.9053009), [endpoint detection](https://arxiv.org/abs/2401.08916), and [intent calibration](https://doi.org/10.1609/aaai.v36i10.21314).

## Staged work

| Stage | Work and decision | Why it helps | Status |
|---|---|---|---|
| **0. Trust the evidence** | Validate the diagnostic manifest and trace schema. Add parity tests showing that captured Stage 4 samples equal the tensor passed at the model boundary, and that recorded appended segments reproduce the runtime command buffer. Keep one event definition for target wake, target command, wrong target intent, and unsolicited action. | Prevents another incorrect timing or attribution result from driving a model change. | **In progress:** the matched-diagnostic helper and tests are being built in a separate task; review and tests remain pending. Historical v10 exact-segment instrumentation exists, but local provenance still needs a retained run ID/command record. |
| **1. Locate the first lost information** | When same-speaker continuous audio is available, replay it once with locked current checkpoints and full trace: source microphone audio, raw and normalized SpEx output, each Stage 2 score/decision including rejected calls, exact appended sample spans, exact fixed-length Stage 4 tensor, logits, predicted intent, and timestamps. Add a separately recorded clean target command only if one exists and can be aligned. Have a human verify the command words in each relevant crop. Use ASR only as an aid, not as proof of intelligibility. | Shows whether the first failure is labeling, extraction, gating, capture/endpointing, or intent classification. | **Blocked on data:** no suitable same-person recording is available in the checked custom folders. No replay has been run. |
| **2. Make only the diagnosis-supported change** | Follow the conditional table below. Use development recordings to compare any change with the frozen baseline. | Limits changes to the stage that evidence implicates and avoids weakening another stage's false-trigger protection. | **Proposed.** No threshold, code change, retraining, or SpEx change is justified yet. |
| **3. Lock full-pipeline acceptance** | Freeze code, checkpoints, configuration, event policy, and test manifest. Evaluate once on independent speakers/sessions and vehicles where available, with positive trials and long target-absent recordings. | Measures the user-visible result under conditions excluded from fitting and threshold selection. | **Not run.** Depends on Stage 1 findings, selected changes, a held-out set, and product owners defining risk/latency budgets. |

## Conditional changes

| What the matched evidence shows | Response | Why this should help | Guardrail |
|---|---|---|---|
| The reference/source labels are wrong, or speaker identity cannot be established. | Reject the trial or manifest row as matched evidence; repair labels and collect/verify a valid same-person recording. | Correct provenance is necessary before measuring a speaker-conditioned system. | Do not infer identity by joining corpus speaker tokens, row order, demographics, or unverified voice similarity. |
| The source microphone command is audible, but its words are materially lost or damaged in the aligned frozen SpEx output. | Only then investigate SpEx configuration, conditioning/reference quality, channel alignment, and cabin-domain behavior; compare candidate extraction changes on development recordings. | Listening to paired source-mic and extracted crops can identify extraction as a plausible bottleneck. Use SI-SDR or another clean-reference metric only when a separate clean target recording exists and its timing/alignment has been verified. | SpEx source and default weights stay fixed in the first diagnostic pass. A change to SpEx requires this evidence and held-out regression checks. |
| The command is audible in extraction but Stage 2 rejects one or more command spans. | Test a development-only command hold/hysteresis policy and timestamped retention of gate-rejected spans, while still requiring target-presence evidence for wake/action decisions. | Short gate drops or pauses may otherwise remove command pieces from the downstream buffer. | Do not automatically bypass the gate. Compare missed target speech and unsolicited actions together. Preserve per-chunk scores, decisions, and source spans. |
| Stage 2 accepts the command, but Stage 4's captured input begins late, ends early, or cuts off labeled words. | Adjust pre-roll, endpointing, timeout, or crop policy only against labeled command retention. Include pauses and commands longer than the current 2-second input. Measure audio lookahead and compute time separately. | A policy tied to labels can preserve complete commands while bounding delay. | Current 3-second context/1-second KWS configuration incurs audio lookahead; evaluator timing does not include measured compute. Do not describe timing as the proven cause until the replay demonstrates it. |
| The exact Stage 4 tensor contains the intended words and is correctly aligned, but the model predicts the wrong intent. | Build a labeled post-SpEx training set from training speakers/sessions and use it for fine-tuning only if needed. Keep separate development speakers/sessions for model and threshold selection, confidence calibration, and the repeat/defer policy. Keep the final test speakers/sessions locked. | Aligns Stage 4 training with the input distribution it receives and can avoid acting on uncertain commands. | Do not treat maximum softmax as calibrated confidence. Do not train on development or locked-test recordings. Do not start Stage 4 training before input correctness is verified. |
| Stage 3 is implicated by same-domain wake or false-wake evidence. | Inspect development false triggers for broad categories (passenger speech, phonetic confusables, media, cabin noise). Search for and mine related examples from the **training partition only**, keeping source speakers/sessions disjoint. Compare against the frozen model on the same development sessions. | Diverse hard negatives can target the observed confusers better than arbitrary class-ratio changes. | Never copy exact validation/test audio into training. Keep the locked test untouched. Tune cutoff, confirmation, and refractory policy jointly on development full sessions; promise no threshold in advance. |

If the first failure is unclear, improve measurement or collect another labeled trial; do not select a stage based on source-time overlap alone.

## Diagnostic trial procedure

For every trial, record an independent enrollment clip and one uninterrupted microphone stream from the same pseudonymous speaker and session. Label wake and command onset/offset in integer source samples, the transcript, and expected intent. Include representative pauses and cabin conditions. Add passenger or playback speech as separate labeled negative conditions. Compare source-microphone and aligned extracted crops, normalized downstream audio, accumulated accepted audio, and the exact Stage 4 tensor by listening. The microphone mixture is not a clean target reference. If a separate clean target recording exists, pair and align it to the trial before using it as an oracle or calculating clean-reference metrics; otherwise do not invent or splice a “clean command” signal.

The trace should retain the sample rate, channel order, source checksum, manifest/config/checkpoint identity, per-call window and fresh sample spans, overlap/deduplication, Stage 1 waveform and conditioning, all Stage 2 logits/probabilities and accept/reject reasons, Stage 3 score and temporal support when evaluated, buffer sizes and appended/rejected source spans, endpoint/timeout state, exact Stage 4 input tensor, output logits, predicted intent, and decision timestamps. Save audio locally even if remote logging fails. Distinguish `not_evaluated` after a gate rejection from a low KWS score.

The v10 evaluator's current decision timestamp is the end of its full three-second context window, although the KWS classifier sees a one-second crop. This captures audio lookahead but not model compute time. The Stage 4 model receives a fixed two-second crop, and longer samples are clipped to that window. The replay must report the actual source samples represented in the two-second tensor and whether labels show truncation. The vehicle action module is currently a stub mapping an intent ID to a label or `noop_*`; current “correct action” metrics measure intent dispatch labels, not vehicle actuator execution.

## Locked acceptance protocol

Use a new test manifest separated from training and threshold-selection recordings by speaker and session; hold out vehicles where available. Freeze the full model stack, configuration, event grouping, and confidence policy before the run. Include target-present wake-and-command trials and long target-absent exposure with passengers, media, and representative road/cabin conditions. Report the number of speakers, sessions, vehicles, trials, and target-absent hours so readers can judge the independent evidence.

Report each quantity separately:

- full wake-to-correct-intent success and its trial count;
- wake misses and false wakes under a declared event-grouping rule;
- intent accuracy conditional on a confirmed target wake;
- wrong intents/actions on valid target commands;
- unsolicited actions during target-absent periods, with negative exposure hours and rate;
- abstentions/repeats and a risk-versus-coverage curve;
- median and 95th-percentile wake-to-action latency, including context lookahead, endpoint delay, and measured compute; state the timestamps used;
- results by speaker, session, vehicle/noise condition, and command type.

Use uncertainty intervals clustered by speaker/session (and vehicle where the sample supports it), not overlapping windows as independent observations. For rare false actions, report the exposure and a count-based confidence bound. As an illustration only, zero events in 30 target-absent hours gives a one-sided 95% Poisson upper bound near `2.996 / 30 = 0.10` events per hour. That is not a product criterion, and the Poisson assumption should be stated; variation and clustering across sessions may require a more conservative analysis. Product owners must define allowable false-action and latency budgets based on consequence. This plan does not invent those budgets or claim that any particular number is safe.

Because the action layer is a stub, software acceptance can verify intent dispatch only. If the product is expected to control vehicle functions, actuator integration, confirmation behavior, and hardware-in-the-loop or equivalent action verification require a separate acceptance test before any claim about real vehicle action correctness.

## Run provenance and W&B

Use `audioml_base` for model replays and future evaluation/training. Model evaluators expose W&B as an opt-in option, so pass `--wandb` explicitly and select `--wandb-mode online` or `offline`. An offline run is saved locally and is **not synced** until explicitly synced; a local offline directory is not evidence of a hosted run. Retain the run URL/ID and local report/audio paths.

For each replay and locked evaluation, record the git revision, dirty status and patch hash, manifest/schema version and hash, config hash, checkpoint paths and SHA-256 hashes, software/environment versions, selected thresholds, split identities, and W&B run identity/mode. Preserve pre-existing working-tree changes; do not reset or overwrite them. Unit and parity tests need no artificial W&B run.

## Implemented, proposed, and blocked

| State | Items |
|---|---|
| **Implemented evidence/corrections** | The audit documents corrected Stage 2 evaluator prerequisites, official intent manifests, Stage 1 AISHELL loader handling, runtime-matched Stage 3 scaling/crop, wake-event attribution, and v10 exact appended-segment timing. Current source includes a W&B evaluation helper and optional evaluator flags. These are working-tree artifacts and the tree was already dirty at validation; check revision and patch provenance before treating them as a release baseline. |
| **Implemented tests, with narrow coverage** | Existing tests cover timestamp overlap deduplication, one framewise pre-roll case, and timeout accumulation, among other component checks. The 12 October validation snapshot passed 22 tests and skipped the missing pretrained-checkpoint test. See the validation report for exact commands. |
| **In progress** | A standalone matched-diagnostic helper and manifest validator with parity tests are being built. They still require review and successful tests before the recorder can be called ready. |
| **Proposed** | Same-speaker replay; conditional stage-specific fixes; development calibration; hard-negative mining from training data only if Stage 3 is implicated; locked held-out acceptance; actuator verification if relevant. |
| **Blocked or not established** | Matched replay is blocked by absent recordings. No runtime root cause is confirmed. No new checkpoint or threshold is promoted. Production readiness and physical vehicle action correctness are unestablished. |

## Related evidence

- [Pipeline failure audit and production improvement plan](PIPELINE_ERRORS_AUDIT.md)
- [Pipeline diagnostic guide and corrected v10 interpretation](PIPELINE_DIAGNOSTIC_GUIDE.md)
- [Primary research and dataset compatibility report](PIPELINE_PRIMARY_RESEARCH_2026-10-12.md)
- [Matched diagnostic readiness and trace contract](MATCHED_DIAGNOSTIC_READINESS_2026-10-12.md)
- [Repository validation snapshot](REPO_VALIDATION_2026-10-12.md)
- [Corrected v10 exact command-capture report](diagnostic_v10_command_capture_kws029_two_snr.json)
