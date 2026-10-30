# Stage fault isolation test — 12 October 2026

## Decision

Use a **matched recording + stage-input checks + one-stage-at-a-time bypasses**. The test names a model fault only when that model received usable input and still made the wrong decision. A skipped stage is recorded as `not evaluated`, never scored as a model miss.

**This test cannot be run on the current speech clips as a same-speaker pipeline test.** The configured custom recording folders are empty, and Speech Commands wake speakers are not linked to Fluent Speech Commands command speakers. Running it on those clips would repeat the known confound. The matched replay utility already enforces speaker/session evidence and saves the per-stage inputs; the missing requirement is new labeled audio.

## Required recording set

For each positive trial, record simultaneously:

1. The configured far-field microphone array stream used by the pipeline.
2. A close-talk reference microphone worn by the same enrolled speaker. This is the reference used to check whether the command was present/intelligible before and after extraction; it is not passed to the production pipeline.
3. A separate enrollment recording from that speaker.

In the array stream, the enrolled person says the wake phrase, pauses naturally, then says one Fluent Speech Commands intent. Include at least 20 positive trials across at least 5 speakers for the first diagnostic, with quiet and cabin-noise conditions. Record at least 20 target-absent trials containing cabin noise and passenger/media speech. These numbers are a **fault-finding pilot**, not proof of production reliability; the locked acceptance set must be larger and include hours of target-absent audio.

Annotate wake and command start/end sample numbers, exact transcript, expected intent, speaker/session, vehicle/noise condition, and the source/enrollment file hashes. A reviewer listens to each stream and verifies same-speaker provenance. Use the existing required manifest schema in `scripts/eval/replay_matched_continuous_diagnostic.py`; its `speaker_match_verified` field is a reviewed assertion, not an automatic voice-identity detector.

## Run sequence

Freeze the checkpoints and thresholds before the diagnostic. For every trial:

1. Run the unchanged Stage 0–4 cascade using `scripts/eval/replay_matched_continuous_diagnostic.py` in `audioml_base`.
2. Save the far-field source, close-talk reference, every Stage 1 output, Stage 2 probabilities and accept/reject reason, Stage 3 scores and confirmation state, command-buffer source segments, exact Stage 4 input tensor, intent scores, and timestamps. Keep each artifact tied to its source interval and checksum.
3. Blindly compare the close-talk reference, raw array crop, normalized Stage 1 crop, and exact Stage 4 crop against the reviewed transcript. Two reviewers independently mark whether the wake/command words are intelligible; disagreements are adjudicated before assigning a Stage 1 fault.
4. Repeat the same frozen trials with only one diagnostic intervention changed at a time. Keep original model scores in every run.
5. Log settings, per-trial outcomes, and artifacts to W&B. Do not tune on the held-out speakers/sessions.

## Interventions that isolate the stages

| Replay | Change exactly this | What it answers |
|---|---|---|
| Baseline | Nothing | Where did each real trial stop or become wrong? |
| Stage 1 reference | Replace only the extracted speech with the aligned close-talk reference at the same downstream interface | Does clean target speech restore the downstream path? If yes, suspect Stage 1/extraction; compare its output against the reference to confirm. This is an upper-bound diagnosis, not a proposed product design. |
| Stage 2 oracle | Force the presence gate to accept only time spans labeled as target speech; leave KWS and intent models unchanged | Does a known-correct gate recover wakes/commands that normal Stage 2 rejected? If yes, Stage 2 is blocking useful input. Also count false passes on target-absent/passenger spans. |
| Stage 3 oracle | Force a wake event only at the annotated target-wake boundary; leave command capture and Stage 4 unchanged | Does the rest of the pipeline complete when the real wake is supplied? If yes, Stage 3 missed a usable wake. This is paired with an offline KWS score on every labeled wake crop, including crops Stage 2 skipped. |
| Stage 4 crop control | Feed Stage 4 the manually verified command interval from the *actual Stage 1 output*, bypassing only command buffering/cropping | If this gets the intent right while the normal crop does not, the buffer/crop path is at fault. If both are wrong, the error is later than capture. |
| Stage 4 clean-reference control | Feed Stage 4 the aligned close-talk command crop | If both this and the Stage 1 crop are wrong, Stage 4 classification/training is faulty for that intent. If the clean reference is right but Stage 1 output is wrong, investigate extraction or Stage 4 robustness to extracted audio using the listening/recognizability check. |

The oracle runs are diagnostic controls, not headline performance results. They reveal whether downstream stages can succeed when a suspected upstream decision is made correctly.

## Fault assignment rules

Assign each trial to its **earliest verified failure** and separately list later failures when they can be tested offline:

| Stage | Call it faulty only when… |
|---|---|
| **Stage 0: timing/buffer** | The reviewed command is present in source/Stage 1 audio, but its labeled samples or recognizable words are missing from the exact Stage 4 tensor. A source-time estimate alone is not enough; inspect the saved tensor and segment map. |
| **Stage 1: extraction** | The close-talk reference and array source preserve the reviewed words, but the normalized Stage 1 output repeatedly loses or damages them. The Stage 1 reference replay should recover downstream performance. |
| **Stage 2: presence gate** | Reviewed target speech is present and recognizable in Stage 1 output, but Stage 2 rejects the corresponding fresh speech interval; the Stage 2 oracle recovers downstream trials. Track false accepts on labeled negatives at the same time. |
| **Stage 3: wake detector** | Stage 2 passed an intelligible labeled wake crop, but KWS did not score/confirm it at the frozen operating point. Also attribute false confirmations on labeled negatives to Stage 3 when Stage 2 passed them. |
| **Stage 4: intent model** | The exact Stage 4 tensor contains the reviewed command words, but the model returns the wrong intent. The clean-reference control distinguishes a general classifier failure from poor robustness to extracted/noisy speech. |

If two stages fail independently, report both. Do not force a single winner. The **primary pipeline limiter** is the earliest stage whose oracle substitution increases correct end-to-end trials most, with false-action cost reported beside it.

## Required output and decision

Publish one row per trial with: baseline final result; first failing stage; whether each later model was evaluated; its score/prediction; reference-audio review; and each oracle-control result. Summaries must show:

- correct wake-to-intent trials / positive trials;
- wake misses and false wakes;
- correct intent / confirmed target wake;
- false actions and the target-absent exposure duration;
- exact Stage 4 command coverage and reviewed word intelligibility;
- per-stage standalone results on the exact same trial inputs;
- baseline versus each oracle intervention, grouped by speaker/session and noise condition.

The model to fix first is the one supported by both (a) repeated component failures on usable labeled input and (b) a clear end-to-end recovery when only that stage is corrected. If neither condition holds, the result is “not isolated”; collect more trials or inspect the audio rather than changing weights.

## Current readiness and what is already verified

- The capture utility records Stage 1 raw/normalized tensors, Stage 2 logits and gate reason, Stage 3 score/confirmation state, command appends, and the exact Stage 4 tensor. I reran its observer-parity and manifest-validation suite in `audioml_base` with `PYTHONPATH=.`: **5 passed**. This verifies the capture/validation plumbing, not model accuracy.
- The continuous cross-corpus proxy has already shown command-time audio can reach Stage 4 despite a rejected Stage 2 window. That establishes a buffer behavior check only; it is not this fault-isolation test.
- No oracle-intervention implementation or same-speaker recordings are present yet. The five proxy evaluations logged to W&B cannot identify a definitive real-speaker model fault.
- No weights or SpEx code should be changed until the first matched diagnostic identifies the earliest failure and its oracle control confirms recovery.

## Command once the recordings exist

```powershell
conda run -n audioml_base python scripts/eval/replay_matched_continuous_diagnostic.py `
  --manifest E:/car-tse-data/custom_incar/manifests/matched_continuous.jsonl `
  --config configs/eval_continuous_pipeline_p0_e.yaml `
  --paths configs/paths.e_research.yaml `
  --output-dir reports/evaluation/artifacts/matched_continuous `
  --wandb --wandb-mode online
```

The command is the baseline capture. The five intervention replays above must be added around this capture before the protocol can automatically produce oracle comparisons. Do not claim they have run until their per-trial reports exist.
