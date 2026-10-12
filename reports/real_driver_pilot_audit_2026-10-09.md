# Real-driver pilot audit and evaluation plan

Date: 2026-10-09  
Status: **real-driver evidence is not yet available**

Update 2026-10-10: the workspace was rechecked and still contains zero
session directories. A consent-guarded capture utility is now available at
`scripts/data/record_real_driver_session.py`; it writes the required metadata
layout but does not invent event annotations.

## Executive finding

The repository does not currently contain a permitted, speaker-consistent,
continuous recording with all of the following in one session:

1. a driver's short enrollment utterance;
2. that same driver's wake word, `Marvin`;
3. that same driver's command;
4. passenger speech, including a passenger saying `Marvin`;
5. no-command and noise-only periods; and
6. timestamps and vehicle-condition metadata.

`data/raw/custom_incar` contains only its README and no WAV recordings. The
existing audio is useful for component development, but it cannot be reported
as a real-driver end-to-end test set:

| Local corpus | What it can support | Why it is not the requested test set |
|---|---|---|
| AISHELL-5 | Permitted real in-car, multi-channel negative/diagnostic tests | No aligned driver enrollment + `Marvin` + command sequence |
| CAVEMOVE | Cabin impulse/noise augmentation and TSE diagnostics | Cabin/noise material, not a labeled driver assistant session |
| LibriSpeech | Clean speech/enrollment and source separation training | Not a vehicle session or wake-word command recording |
| Fluent Speech Commands | Intent training/evaluation | No `Marvin` wake event or vehicle session |
| Speech Commands v2 | Wake-word training/evaluation | No same-driver command/session context |
| Vehicle Interior Sound | Vehicle-noise augmentation | No speech, enrollment, or command events |
| Nexdata sample | Inspection only unless commercial use is authorized | License/permission is not established for this project |

The AISHELL-5 license is CC BY-SA 4.0 according to its official release page
([OpenSLR 159](https://us.openslr.org/159/)). It may be used for the existing
real-cabin target-absent diagnostic, but not relabeled as a positive driver
benchmark. Commercial or sign-in datasets are not used as evidence here unless
the project obtains and records permission.

## What can be claimed now

The current repository can support a reproducible **cross-corpus proxy**
benchmark and a real-cabin **target-absent** benchmark. Those tests are useful
for debugging false accepts and component interactions, but they do not prove
that a real driver is tracked from enrollment through wake word and command.
The proxy deliberately uses separate public clips for wake and command, so its
speaker consistency must remain marked `false`.

The current thresholds will be treated as frozen engineering settings for the
proxy runs. A proper threshold-tuning/test split cannot be performed until
consented recordings exist; tuning on the eventual test sessions would be
invalid.

## Small pilot to collect

Minimum process pilot:

- 3 consenting drivers, with 2 drivers reserved for tuning and 1 driver held
  out for the first test. This is enough to validate the process, not enough
  for a broad reliability claim.
- 2 sessions per driver, approximately 10--15 minutes each. Keep the second
  session of the held-out driver untouched until thresholds and settings are
  frozen.
- Record the same microphone geometry used by the system, preferably 4-channel
  PCM WAV at 16 kHz. Record a channel map and microphone positions.
- Store consent/permission metadata before using any audio. Do not upload or
  publish a recording without the speaker's explicit permission.
- Use safe parked/controlled conditions for the pilot; do not create a
  distracted-driving situation.

Each session should contain several repetitions of:

1. enrollment: a short natural speech clip (target is <=3 seconds);
2. driver says `Marvin` followed by a command;
3. driver speech without an intended command;
4. passenger says `Marvin` followed by speech or a command-like phrase;
5. passenger speech without `Marvin`;
6. silence/no-command intervals; and
7. noise-only intervals.

Repeat across at least parked-idle, HVAC/road-noise, and passenger-present
conditions. Record condition changes as metadata, not just in a filename.

The recommended on-disk format and event fields are in:

- `reports/real_driver_pilot_protocol_2026-10-09.md`
- `reports/real_driver_pilot_session_template.json`
- `reports/real_driver_pilot_annotations_template.jsonl`

## Evaluation that will be run after collection

Thresholds are selected only on the tuning-driver sessions and then frozen.
The held-out sessions are evaluated once with those settings. The report will
include, by condition:

- driver wake recall: detected driver wake events / labeled driver wake events;
- post-wake command accuracy: correct intent / driver wake events with a
  labeled command;
- false actions per hour: actions during passenger, no-command, and noise-only
  intervals divided by eligible hours;
- response latency: command/wake event to action, with median and p95; and
- counts and exposure hours, so rates are not presented without denominators.

The same frozen examples will be used for the ablation matrix: full pipeline;
no TSE; no presence gate; no speaker matching; no wake-word smoothing; and
command recognition without wake-word gating. Every row will save its config,
commit, input session IDs, and raw event outputs.

## Current limitation

No recording access, consented speakers, or permitted continuous sessions were
provided in the workspace. Therefore this task can prepare the collection and
evaluation machinery and run proxy/negative diagnostics, but it cannot create
real-driver positive results without inventing evidence. The first report will
clearly state the number of drivers and recording hours as zero until the pilot
is supplied.
