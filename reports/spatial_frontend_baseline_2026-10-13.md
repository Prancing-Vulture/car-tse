# Classical spatial-front-end baseline

Date: 2026-10-13  
Status: implementation complete; evaluation and production promotion pending
measured microphone geometry.

## What was added

`src/car_tse/frontend/spatial.py` now provides two deterministic baselines:

- integer-sample delay-and-sum for a low-cost streaming reference;
- offline frequency-domain MVDR with diagonal loading for architecture
  comparison studies;
- GCC-PHAT delay estimation as a calibration diagnostic.

The implementation is channel-count checked, uses zero padding instead of
circular chunk wrapping, and accepts explicit steering delays and weights.
Four focused unit tests cover alignment, time-first input, delay calibration,
and finite-length MVDR output.

## What this does not claim

The repository does not yet contain the production microphone positions or a
measured driver-seat steering vector. Therefore no arbitrary delay vector is
promoted as a seat-zone configuration, and no spatial accuracy or passenger-
rejection number is reported from this implementation. AISHELL-5 provides
useful four-channel diagnostic audio, but its channel geometry is not the
project's future vehicle geometry.

## Next comparison

Once the pilot hardware geometry is recorded, evaluate three paths at matched
false-accept exposure:

1. current TSE-first reference;
2. zone-first delay-and-sum/MVDR followed by the existing wake path;
3. hybrid zone front end with candidate-only TSE.

Report wake recall, passenger acceptance, false actions/hour, command accuracy,
CPU cost, and p95 latency by acoustic condition. This module is an enabling
baseline, not evidence that the architecture is production-ready.

