# Multi-speaker bottleneck diagnosis

Date: 2026-10-16  
Branch: `multi-speaker-eval`  
Status: diagnostic; no new training was started for this diagnosis

## Bottom line

The current end-to-end ceiling is not caused by the basic wake detector
missing the enrolled driver's wake word. The dominant failure is target
attribution after (and, in some cases, before) wake detection:

1. The pre-wake path has no explicit speaker gate. The frame presence model
   detects speech activity, not whether the speech belongs to the enrolled
   driver.
2. The enrollment-conditioned target-KWS model is used as a wake score, but
   it is not sufficiently robust to passenger/co-channel speech in the current
   proxy. Its per-window score is not currently logged separately, which makes
   this subcomponent harder to tune than it should be.
3. Once a wake is accepted, the command path still relies on short streaming
   windows and a confidence buffer. In overlap, the first post-wake windows
   contain both the command and passenger material, and there is no reliable
   target-conditioned end-of-command decision.
4. The overlap training data has identical values in all four synthetic
   channels. It therefore contains no seat or direction cue. A model trained
   on it can learn co-channel denoising, but cannot learn the driver-seat
   separation needed by the product architecture.

The correct next model work is therefore target-conditioned wake/attribution
training and overlap-aware extraction/endpointing. Raising the existing
presence or speaker thresholds alone is not a solution: it suppresses
passenger actions by suppressing valid driver commands as well.

## Evidence from the current 1,000-scene runs

All numbers below are synthetic matched-speaker proxy results, not real-driver
or production claims.

| Condition and current stack | Target wake recall | Target intent | Fully correct | No committed command | Wrong executed actions | Strict false wakes/h | Strict false commands/h |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sequential/passenger-only, clean Stage-1 + train-split intent + command gate + post-action-16 | 99.9% | 92.9% | 92.9% | 52 scenes | 19 | 38.57 | 1.03 |
| Simultaneous overlap, small passenger-mixture Stage-1 + train-split intent + command gate + post-action-16 | 100.0% | 66.2% | 66.2% | 313 scenes | 25 | 73.03 | 0.26 |

Source logs:

- `logs/multispeaker_passenger_cleanidentity_commandgate015_conf080_postaction16_1000_2026-10-16.json`
- `logs/multispeaker_overlap_small_commandgate015_conf090_postaction16_1000_2026-10-16.json`

The overlap run has 684 ambiguous passenger-overlap command events. The
target wake is detected in every scene, but only 687 scenes produce a command
event and only 662 scenes are fully correct. This separates the problem from
wake recall: the system is hearing the scene, but it is not consistently
extracting/accepting the driver's command.

The sequential run shows the complementary pattern: target wake recall is
99.9% and target intent is 92.9%, while the remaining failures include 52
scenes with no committed command and 19 wrong executed actions. This is mostly
a command-window/attribution and safety-policy problem, not a basic acoustic
wake-recognition problem.

## Component-by-component diagnosis

### 1. Target-speaker extraction: dominant in simultaneous overlap

The current overlap stress test sums target and passenger audio into the same
waveform on all four synthetic channels. The matched-prefix audit found a
7.03-point fully-correct change when Stage-1 extraction was bypassed, so TSE is
material. However, the existing extractor is not the only bottleneck: the
co-channel benchmark provides no spatial information, and the passenger-mixture
adaptations were trained with the same limitation.

The large passenger-mixture Stage-1 adaptation improved overlap only modestly
while reducing sequential fully-correct performance from 84.9% to 67.2% in the
earlier matched comparison. The small adaptation is currently the best overlap
candidate, but it remains only about 66% fully correct. This is evidence for a
conditional extractor/data mismatch, not evidence that a larger amount of the
same identical-channel mixture will solve the problem.

### 2. Wake/source attribution: dominant source of false wakes and passenger leaks

The canonical multi-speaker configuration has:

- `speaker_gate_threshold: null` before wake;
- `command_speaker_gate_threshold: 0.15` only after wake;
- a frame presence head with no enrollment input;
- `speaker_fusion_mode: none`.

Therefore the pre-wake path cannot use the compact speaker verifier as a
hard target check. The target-KWS checkpoint is loaded and used, but its single
score is not enough to reject all passenger/co-channel cases.

The source labels in the current logs show the problem directly:

- Sequential: 965 target-sourced wake windows, 73 passenger-wake windows,
  45 passenger-nonwake windows, and 10 windows overlapping both passenger
  sources. The strict false-wake rate is still 38.57/hour after the
  post-action guard.
- Overlap: 760 windows overlap target plus passenger sources, while 239 are
  passenger-only wake/nonwake windows. The strict false-wake rate is 73.03/hour.

The post-action refractory state reduces repeated actions and wake churn, but
it cannot identify the first passenger-triggered wake. It is a safety guard,
not a target-speaker solution.

### 3. Presence gate: not the main identity solution

The configured Stage-2 head is a frame/activity presence head. It is not an
enrollment-conditioned target-speaker detector. In the logged command windows,
presence probabilities remain high on mixed/passenger material, so lowering the
threshold does not separate the driver from the passenger.

The earlier matched gate study confirms the threshold trade-off:

- Sequential compact gate at 0.30: passenger-only false actions fell from 91
  to 22 and wrong actions from 566 to 79, but target intent fell from 94.0%
  to 62.2%.
- Overlap compact gate at 0.30: executed non-target actions fell from 94 to 1,
  but target intent fell from 61.5% to 37.1%.

This is useful as a safety fallback, but it is not a production operating
point. The model needs target-conditioned activity/identity evidence, not a
higher generic speech threshold.

### 4. Command capture and endpointing: major sequential and overlap loss

The runtime still evaluates short post-wake windows with a two-second intent
input and a three-call confidence buffer. The current overlap scene places the
passenger wake at about 2.4--3.4 seconds and passenger non-wake speech at about
3.2--4.7 seconds, while the target command starts at about 2.1 seconds. The
early command windows are therefore acoustically ambiguous by construction.

Per-scene examples show the runtime suppressing or timing out commands when the
speaker similarity is below the 0.15 command threshold, and committing wrong
intent actions when a mixed window is confidently classified. This explains
why lowering the intent confidence threshold from 0.90 to 0.60 increased
overlap fully-correct scenes to 73.5% on one run but also increased wrong
executed actions from 25 to 76 and strict false commands from 0.26/hour to
4.63/hour. The apparent accuracy gain is unsafe confidence calibration, not a
real solution.

The next runtime/model should detect the end of the target command using a
target-conditioned VAD/endpoint signal, accumulate the complete command, and
then classify it once. It should not rely on the highest-confidence short crop
when that crop may contain the wake tail or passenger speech.

### 5. Intent classifier: secondary in sequential, coupled to TSE/windowing in overlap

The intent model is strong on clean target-only data and remains about 92.9%
target-intent accurate in the best current sequential proxy. Its overlap
validation score is higher than the end-to-end score (the train-split mixed
candidate reached about 75.5% validation accuracy / 75.2% macro-F1, but the
full pipeline reached only about 66%). The gap indicates a distribution mismatch
between cached training inputs and the actual streaming, gated, speaker-mixed
windows. Retraining the intent head alone is therefore unlikely to fix the
current overlap bottleneck until extraction and endpointing are corrected.

### 6. Evaluation/data limitations are themselves a limiting factor

The current multi-speaker results cannot establish real-cabin production
performance because they use synthetic clip assembly, no consented continuous
driver sessions, no spatial microphone differences, no playback-reference
channels, and no measured target Jetson runtime. The overlap construction is a
useful co-channel stress test, but it is simultaneously harsher than a real
array in its lack of spatial cues and less representative in its lack of real
room acoustics.

## Ranked bottlenecks

1. **Target attribution before and after wake** — passenger material can arm
   the command state; the pre-wake path has no hard speaker gate.
2. **TSE under simultaneous speech** — the current extractor/data do not
   preserve a reliable target stream in co-channel overlap.
3. **Command endpointing/window mismatch** — short fixed windows are often
   partial or contaminated; this creates both no-action and wrong-intent
   failures.
4. **Intent calibration on extracted streaming audio** — useful after the
   first three are improved, but not the first intervention.
5. **Safety state policy** — post-action refractory is helpful and should stay,
   but it only limits repeated actions; it does not solve source attribution.

## Training decision

The next training iteration should be split into two controlled candidates,
with unique checkpoints and plots:

1. **Enrollment-conditioned target-KWS adaptation**: train/fine-tune the
   target-KWS model with FSC-domain same-speaker positives, passenger wake and
   non-wake hard negatives, and Stage-1 outputs produced by the same candidate
   extractor. Add per-window target-KWS logging and tune the pre-wake threshold
   on train/validation speakers only.
2. **Overlap-aware target extractor**: train a new Stage-1 candidate with the
   speaker encoder/backbone allowed to adapt, plus channel-diverse spatial
   mixtures where available. The identical-channel mixture run is retained as
   a smoke test, not as the production architecture. Preserve a clean and
   sequential regression gate so overlap gains cannot silently destroy the
   92% sequential operating point.

Only after those candidates pass the regressions should the intent head be
retrained on the new extractor's full, endpointed command outputs. The
existing post-action refractory policy remains enabled as a safety control.

## Reproduction references

- `reports/multispeaker_iteration_2026-10-16.md`
- `reports/mac_review_reconciliation_2026-10-15.md`
- `src/car_tse/pipeline/in_car_pipeline.py`
- `scripts/eval/eval_continuous_full_pipeline_proxy.py`
