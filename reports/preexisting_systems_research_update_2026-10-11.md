# Pre-existing systems research update

Date: 2026-10-11  
Scope: architecture and data references for a target-speaker, in-car voice assistant.

## Bottom line

The strongest reusable design pattern is a two-factor streaming decision:

1. a compact phonetic/keyphrase detector answers “did the audio contain
   Marvin?”;
2. an independent target-speaker activity or verification branch answers “was
   Marvin spoken by the enrolled driver?”;
3. a temporal state machine decides whether the following audio is a command;
4. intent/action execution is allowed only after the speaker and timing gates
   agree.

This is materially different from increasing the capacity of the intent head.
It matches the current diagnosis: the repository’s TSE and clean intent
components are useful, but the remaining risk is target/passenger calibration
and long-negative operating behavior. The existing repository already contains
research implementations of the personal-activity head, target KWS, speaker
gates, margin veto, command delay, and long-negative replay. None is promoted
without same-driver tuning and held-out-driver testing.

## Reference patterns and what they imply here

### 1. Personalized keyphrase detection is a multi-signal problem

Google’s personalized keyphrase system combines streaming ASR/keyphrase
evidence with text-independent speaker verification. It also uses speaker
separation and adaptive noise cancellation; the reported design direction is
that speaker verification reduces false triggers, while separation and ANC
reduce false rejects.

**Repository implication:** keep the target-KWS probability and speaker
identity probability as separate observables. Do not collapse them into one
uncalibrated score. The action gate should require both to pass, with a
threshold selected on same-driver tuning data.

Reference: <https://research.google/pubs/personalized-keyphrase-detection-using-speaker-and-environment-information/>

### 2. Personal VAD/TS-VAD uses frame-aligned target activity

Personal VAD and target-speaker VAD predict target activity over time while
conditioning on a speaker representation. The important distinction from a
generic speech-presence classifier is that non-speech and non-target speech
are separate outcomes. This is a closer match to “passenger said Marvin” than
one scalar speech-present score.

**Repository implication:** the three-way `PersonalActivityHead` is the right
research direction, but its probabilities are not yet calibrated for the
cross-corpus cabin proxy. Keep the framewise implementation, train it with
same-driver/target/passenger labels, and evaluate it with false actions/hour,
not only chunk F1.

References:

- <https://arxiv.org/abs/1908.04284>
- <https://arxiv.org/abs/2204.03793>
- <https://arxiv.org/abs/2005.07272>

### 3. Streaming target-speaker extraction must protect clean speech

VoiceFilter-Lite is a useful deployment reference because it is speaker
conditioned, streaming, small, and designed to avoid over-suppressing clean
target speech. Its feature-domain mask, asymmetric loss, adaptive suppression,
and separate/bypassable model are useful ideas even though this repository
uses a waveform SpEx+ family extractor.

**Repository implication:** do not replace SpEx+ solely because another model
is newer. First compare clean-speech preservation, passenger suppression,
real-time factor, memory, and end-to-end command accuracy on identical chunks.
A future causal/lightweight TSE candidate can borrow the adaptive suppression
and clean-preservation evaluation, but only after the data contract is fixed.

Reference: <https://research.google/pubs/voicefilter-lite-streaming-targeted-voice-separation-for-on-device-speech-recognition/>

### 4. Multichannel front ends should exploit spatial information and noise
coherence

Supervised multichannel keyword spotting and modern in-car systems use the
microphone array as more than duplicated mono input. Spatial filtering,
cross-channel noise coherence, and a learned feedback path can help the KWS
backend and front end together.

**Repository implication:** the current mean-pool/learned-residual cabin
front end is only a baseline until a matching four-channel recording setup is
available. The next spatial study should compare mean-pool, fixed
beamforming/GCC features, and a learned causal spatial block while reporting
channel geometry, clean-speech preservation, false actions/hour, and latency.

Reference: <https://research.google/pubs/supervised-noise-reduction-for-multichannel-keyword-spotting/>

### 5. Production KWS toolkits treat state as part of the model

WeKWS and Google’s streaming KWS work emphasize small-footprint streaming
operation, temporal state, and low latency. This supports evaluating hop size,
confirmation hits, refractory state, and post-wake command delay together.

**Repository implication:** the 250-ms candidate is a latency experiment, not
the new default. Its AISHELL-5 target-absent follow-up increased full-path
events from 3.0/hour to 6.0/hour on the comparable 1,200-second slice, so it
remains rejected until real-driver tuning data supports it.

References:

- <https://github.com/wenet-e2e/wekws>
- <https://github.com/google-research/google-research/tree/master/kws_streaming>

### 6. Enrollment can evolve, but only behind a reliability filter

The recent EvoTSE work addresses a limitation that is relevant to long
drives: a static enrollment can become stale as the acoustic environment and
the speaker’s voice change. Its central idea is to update the enrollment from
historical extracted segments only when they pass a reliability filter.

**Repository implication:** an adaptive enrollment bank is a later-stage
feature, not a shortcut around current speaker mismatch. A future implementation
must update the driver profile only after independent target-activity,
speaker-verification, and command-consistency checks. Passenger speech,
assistant playback, and low-confidence extractions must be ineligible for
profile updates. The current multi-reference bank experiment was rejected
because an uncalibrated mean embedding reduced target recall.

Reference: <https://arxiv.org/abs/2604.06810>

## Dataset and protocol findings

The public corpora fall into complementary categories; none found in this
pass provides the complete English protocol needed for acceptance.

| Resource | What it is useful for | Why it cannot be the final Marvin proof |
|---|---|---|
| ICMC-ASR / AISHELL-5 | Large real multichannel in-car speech and far-field noise; the challenge page describes 1,000 hours of in-car multi-speaker speech plus more than 400 hours of far-field noise. | Mandarin, challenge/data-agreement constraints, and no same-driver English enrollment → Marvin → command → passenger-Marvin sequence. |
| ICSRC 2022 | 20 hours of real in-car Mandarin speech across speed, window, music, and passenger-seat conditions, with explicit non-command material. | Mandarin and a challenge split/protocol rather than the required English personalized wake-word sequence. |
| SPID | About 25 hours of spontaneous in-car German dialogue with front/back speakers and controlled simulated driving noise; useful for Lombard and passenger-overlap ideas. | German, research-only access, no Marvin wake/action labels, and not the project’s microphone/input contract. |
| CENSREC-3 | Driver isolated words under 16 combinations of speed and cabin conditions; useful for a historical noisy in-car speech-interval protocol. | Isolated words and research/development distribution; no continuous personalized wake-to-command interaction. |
| CI-AVSR | Cantonese in-car commands, clean/noisy/OOD-noisy splits, and audio-visual data; useful for command/noise generalization experiments after asset terms are verified. | Cantonese, audio-visual format, and no same-driver English Marvin protocol. |
| ReMASC | English vehicle voice commands, real microphone arrays, genuine/source/replayed recordings, and replay/vehicle stress. | Replay-attack task; no continuous enrollment-to-wake-to-command session with passenger Marvin and labeled no-command intervals. |
| iCarB-Voice / commercial automotive corpora | Potentially closer driver-seat English/noise coverage. | Restricted or commercial access; cannot be used or redistributed without an approved license. |

Official source pages:

- <https://icmcasr.org/>
- <https://iscslp2022-icsrc.org/>
- <https://dss-kiel.de/index.php/media-center/data-bases/spid-corpus>
- <https://research.nii.ac.jp/src/en/CENSREC-3.html>
- <https://github.com/HLTCHKUST/CI-AVSR>
- <https://github.com/YuanGongND/ReMASC>

## What should change in our project

### Highest-value next change: data and calibration

Collect a small permitted pilot using the existing recorder/annotation
contract. Each held-out test session must contain, in one continuous capture:

- the same driver’s short enrollment, driver “Marvin,” and a labeled command;
- passenger speech and passenger “Marvin” near the driver events;
- no-command and noise-only intervals;
- condition labels for parked/idle, HVAC, road, window/wind, music/playback,
  and passenger loudness;
- the actual four-channel microphone layout expected by the pipeline.

Use driver/session-disjoint tuning and test splits. Freeze the wake, activity,
speaker, and action thresholds on tuning only. If no safe tuning point meets
the false-action constraint, report that explicitly and do not silently select
the highest-recall unsafe point.

### Next architecture experiment after pilot data exists

Train a compact, causal target-activity/verification branch with three labels:
non-speech, enrolled driver, and non-target/passenger. Keep a separate compact
phonetic KWS output for “Marvin.” Evaluate these combinations:

1. KWS only;
2. KWS + current scalar presence;
3. KWS + three-way target activity;
4. KWS + independent speaker verification;
5. KWS + target activity + speaker verification + temporal veto.

The primary safety metric should be false actions/hour over labeled negative
exposure, with driver wake recall, command accuracy, and response latency as
companion metrics. This is the closest practical implementation of the
pre-existing personalized-keyphrase and personal-VAD patterns.

### Later optimization, not the immediate fix

Once the joint metric is stable, compare a causal/lightweight TSE or
VoiceFilter-Lite-style feature mask, spatial beamforming/ANC, and a compact
BC-ResNet/MatchboxNet/WeKWS-style KWS backbone. Quantize or distill the large
intent stage only after measuring whether it preserves command accuracy under
the real cabin conditions. The present approximately 424 MB footprint and
development GPU timings are optimization targets, not evidence that the
accuracy bottleneck is model size.

## Current decision

No new dataset is being added to model training from this research pass. The
current downloaded ReMASC sample remains a supplemental English vehicle/replay
stress asset; AISHELL-5 remains a real-cabin target-absent diagnostic; ANIR
remains a noise/impulse-response augmentation asset. The next production-
relevant evidence still requires a consented, same-driver pilot with matching
hardware. The current reference configuration stays unchanged, and the
250-ms fast-stream candidate stays opt-in and rejected for safety promotion.
