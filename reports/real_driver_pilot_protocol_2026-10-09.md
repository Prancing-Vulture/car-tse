# Real-driver pilot recording and labeling protocol

This protocol is for a small, repeatable pilot of the in-car assistant. It is
not a substitute for a larger real-world study.

## Split design

Use a speaker-disjoint split:

- `tuning`: drivers D01 and D02; used to choose thresholds and any operating
  settings;
- `test`: driver D03; never used to tune thresholds or inspect predictions
  beyond the pre-declared scoring procedure.

If more speakers are available, use at least two held-out test drivers. Keep
whole sessions together; do not put clips from one recording in both splits.

## Session directory

Each session is a directory with this shape:

```text
sessions/<session_id>/
  audio.wav
  session_metadata.json
  annotations.jsonl
```

`audio.wav` should be PCM, 16 kHz, and use the production microphone channel
layout. The metadata must include consent, driver ID, session split, channel
map, microphone geometry, and acoustic condition. The templates in this report
are machine-readable starting points.

## Event labels

Use half-open intervals `[start_s, end_s)` in seconds from the beginning of
`audio.wav`. Required event types are:

- `enrollment`: target driver's reference speech;
- `driver_wake`: driver says `Marvin`;
- `driver_command`: command following a driver wake;
- `passenger_wake`: passenger says `Marvin`;
- `passenger_speech`: passenger speech without a wake word;
- `no_command`: driver/passenger speech where no action is intended;
- `noise_only`: no speech, but vehicle/background noise is present;
- `playback`: optional vehicle playback/reference signal.

For command events, record the transcript and expected intent/slots. For wake
events, record the exact phrase and speaker role. Overlapping events are
allowed; for example, a passenger event can overlap a driver command.

## Suggested pilot script

For each condition, record several repetitions with natural timing:

1. 3-second enrollment clip from the driver;
2. driver: `Marvin`, then a command from the supported intent set;
3. driver speech without `Marvin` and explicitly no intended action;
4. passenger: `Marvin`, then a command-like phrase, with no intended action;
5. passenger speech without the wake word;
6. at least 20 seconds of noise-only/silence;
7. repeat after changing HVAC/road/passenger condition.

The operator should announce condition changes outside labeled event intervals
or record them in metadata. Keep a session log that maps the session ID to the
consent record and any equipment changes.

## Permission rules

Do not use a session unless `consent.permission_to_use` is true and the owner,
license, and permitted purpose are recorded. Remove or replace recordings if
consent is withdrawn. Synthetic mixtures and public corpora may be used for
development, but must not be labeled as real-driver positive evidence.

## Scoring freeze

Before opening held-out test predictions, record the selected presence,
command-presence, KWS, speaker-gate, and intent-confidence thresholds plus the
code commit in the evaluation output. The tuning split may be used for a
threshold grid and error
analysis. The test split is then scored without changing thresholds, smoothing,
speaker gating, or component selection.

The evaluator records explicit release gates rather than leaving
`production_valid` ambiguous. Its default held-out targets are at least 90%
driver-wake recall, at least 90% end-to-end command accuracy, at most 1.0
conservative unmatched-wake false accept per hour of labeled negative
exposure, and at most 1.0 false action per hour of labeled negative exposure,
together with at least three drivers and three complete sessions. Negative
exposure is the union of passenger-wake, passenger-speech, no-command,
noise-only, and playback intervals; overlapping labels are counted once. These
values are command-line configurable, but any change must be written into the
report before the held-out test is opened. The conservative false-accept rate
counts every unmatched wake, including one outside a labeled interval, over
that negative-exposure denominator; a label-covered interval-specific rate and
the historical total-recording rate are also reported for diagnostics.
