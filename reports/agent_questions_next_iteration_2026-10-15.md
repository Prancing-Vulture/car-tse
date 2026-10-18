# Agent coordination note — next iteration questions

Date: 2026-10-15  
Scope: English clean-first development; no training or evaluation is started by
this note.

## What was reconciled

The Mac handoff in `D:/Prawin_New/tse/new info/` is a planning/research bundle.
Its own handoff says that it contains no checkpoints, logs, datasets, or code
changes. The Windows repository contains the executed follow-up work and is the
authoritative source for the current metrics.

The current clean synthetic matched-speaker benchmark is above the interim
90% target (90.1% with the three-call decoder and 90.6% with the four-call
decoder on 1,000 scenes), but it is not real-driver evidence. The passenger
sequential proxy exposes 283.37 false wakes/hour and 25.46 false actions/hour;
the simultaneous co-channel overlap proxy reaches only 45.3% command accuracy.
The FT2 overlap intent candidate reaches 57.81% on its 128-scene overlap pilot,
only a small improvement over FT1, and remains unpromoted.

## Questions for the other agent

1. **Canonical clean operating point:** Should the four-call buffered decoder
   be treated as the clean reference (90.6%/1,000), or should we keep the
   three-call configuration as canonical until a true endpointed command
   capture implementation replaces the heuristic buffer?

2. **Target identity gate:** Was the missing/null speaker-gate threshold in the
   clean configuration intentional because it over-suppressed positives? If a
   target-conditioned presence, TSE-V cosine, compact verifier, or seat-zone
   branch exists outside this checkout, provide its exact branch, commit,
   config, checkpoint, and positive/negative calibration curves.

3. **Endpoint implementation:** Does the other branch contain a true
   wake-end/presence-VAD endpoint with a continuous TSE ring buffer, or only the
   current fixed-window/buffered decoder? The clean result still has 62/99
   post-wake failures with no committed command event, so this distinction
   determines the next software experiment.

4. **Spatial evidence:** Are there any recordings or patches with different
   microphone responses for target and passenger? The current overlap study
   feeds the same summed waveform to all four channels, so it tests co-channel
   contamination, not spatial separation. A zone-first comparison needs the
   exact geometry and channel mapping.

5. **Data and product decisions:** Is there any permitted same-driver data,
   six-input capture hardware, or target Jetson/SoC available? Also confirm
   whether `Marvin` remains the research phrase and whether the current FSC
   intent set is only a research benchmark; these decisions affect the wake
   training and release-data plan.

## Default action if no answer is available

Do not start another generic intent or TSE fine-tune. The next controlled
iteration should compare, on the same clean matched-speaker and passenger
proxies:

- the retained clean reference;
- a true endpointed/presence-VAD command capture path, if available;
- a calibrated target-identity gate (TSE speaker similarity/compact verifier)
  with a recall-versus-false-action sweep.

Every candidate must retain the 1,000-scene clean regression and the full
Smart Speaker/AISHELL negative checks. No candidate should be promoted until
the gate is calibrated on speaker-consistent data and then checked on permitted
real-driver recordings.

