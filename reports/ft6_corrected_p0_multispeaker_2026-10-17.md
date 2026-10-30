# FT6 corrected P0 multi-speaker evaluation

**Date:** 2026-10-17
**Branch:** multi-speaker-eval
**Model:** FT6 runtime-jitter intent checkpoint with the frozen Stage-1 TSE,
presence head, general KWS, target KWS, and compact verifier
**Evaluation size:** 1,000 scenes per condition, 3.889 synthetic hours per
condition
**Training:** none in this evaluation
**Effective configuration SHA-256:** f6200849979397f55c496c678c0a3be6fd17692dd6c44edadb3a58d73a8fad2c

## Protocol

This is the corrected P0 proxy protocol. It uses:

- matched-speaker target wake/command pairing;
- dynamic passenger selection from 136 passenger wake and 136 passenger
  non-wake speakers;
- speaker-disjoint held-out evaluation manifests;
- full-length command audio;
- zero cabin/road noise, so this isolates passenger speech and co-channel
  interference;
- four synthetic microphone channels;
- the canonical three-call command buffer;
- 0.5-second hop;
- the FT6 soft command speaker gate: threshold 0.15, rescue confidence 0.85,
  rescue floor 0.0;
- no checkpoint changes.

These are synthetic multi-speaker proxies, not real-driver cabin evidence. The
overlap condition sums target and passenger speech into a co-channel mixture
and contains no spatial seat cue.

## Results first

| Metric | Sequential passenger | Simultaneous overlap |
|---|---:|---:|
| Scenes | 1,000 | 1,000 |
| Duration | 3.889 h | 3.889 h |
| Target wake recall | **99.70%** | **100.00%** |
| Target intent accuracy overall | **96.20%** | **91.70%** |
| Intent accuracy conditioned on target wake | 96.39% | 91.70% |
| Fully-correct scenes | **962/1,000 = 96.20%** | **917/1,000 = 91.70%** |
| Strict fully-correct action scenes | 914/1,000 = 91.40% | 890/1,000 = 89.00% |
| Scenes with wrong intent action | 28 = 2.80% | 60 = 6.00% |
| Wrong-intent action events | 28 | 60 |
| Executed non-target actions | 3 | 0 |
| Executed non-target actions/hour | 0.771/h | 0/h |
| Passenger-only false actions | 1 | 0* |
| Passenger-only false actions/hour | 0.257/h | 0/h* |
| False-action events/hour | 0.771/h | 0/h |
| Strict false wakes/hour | 49.371/h | 61.714/h |
| Strict false commands/hour | 14.143/h | 7.714/h |
| Mean target-wake latency | 2.125 s | 2.147 s |
| P95 target-wake latency | 3.500 s | 3.500 s |

The FT6 sequential result is a substantial improvement over the older
corrected P0 baseline in exact-scene accuracy and non-target action count. It
does not solve the safety problem: 0.771 executed non-target actions/hour and
49.371 strict false wakes/hour remain far above a production target.

The overlap result is lower than sequential by 4.50 percentage points. Its
main loss is command/intent correctness, not target wake recall. The zero
passenger-only action value must not be read as successful passenger rejection:
999 of 1,000 scenes contain overlapping passenger-wake intervals, so source
attribution is ambiguous by construction. The wrong-intent, wrong-executed
action, strict-false-wake, and overlap-command metrics are the meaningful
overlap diagnostics.

## Sequential condition

Artifact: logs/multispeaker_passenger_ft6_p0_dynamic_full_1000_2026-10-17.json

Detailed metrics:

- target wake recall: 997/1,000;
- target intent accuracy: 962/1,000;
- conditioned intent accuracy: 96.389%;
- false wake events: 8, or 2.057/h under the legacy event counter;
- strict false wakes: 192, or 49.371/h;
- false-action events: 3, or 0.771/h;
- executed actions: 990;
- executed non-target actions: 3, or 0.771/h;
- wrong-intent action events: 28;
- wrong executed action events: 28;
- scenes with wrong intent/action: 28;
- strict target command events: 935;
- strict false command events: 55, or 14.143/h;
- passenger wake overlap events: 43;
- passenger unambiguous wake events: 43, or 11.057/h;
- passenger-only false actions: 1, or 0.257/h;
- target-wake latency: 2.125 s mean, 3.500 s p95.

The recorded wake-source attribution counts are 997 target-tolerance-support
events, 6 passenger-nonwake events, and 2 passenger-wake events. The broad
target-tolerance category is retained for historical compatibility; strict
source-attributed and action-level metrics are the safer primary measures.

## Simultaneous-overlap condition

Artifact: logs/multispeaker_overlap_ft6_p0_dynamic_full_1000_2026-10-17.json

Detailed metrics:

- target wake recall: 1,000/1,000;
- target intent accuracy: 917/1,000;
- conditioned intent accuracy: 91.700%;
- false wake events: 0 under the legacy counter;
- strict false wakes: 240, or 61.714/h;
- false-action events: 0 under the legacy action counter;
- executed actions: 977;
- executed non-target actions: 0;
- wrong-intent action events: 60;
- wrong executed action events: 60, or 15.429/h;
- scenes with wrong intent/action: 60;
- strict target command events: 947;
- strict false command events: 30, or 7.714/h;
- passenger-wake overlap events: 999;
- passenger unambiguous wake events: 239, or 61.457/h;
- passenger-overlap command events: 976;
- ambiguous overlap command events: 976, or 250.971/h;
- target-wake latency: 2.147 s mean, 3.500 s p95.

All events were attributed to target-tolerance-support by the current
association rule. That is expected for co-channel overlap and demonstrates why
the overlap condition cannot be judged using passenger-only false actions
alone.

## Comparison with earlier FT6 128-scene checks

The 128-scene checks were useful for rapid iteration, but the corrected
1,000-scene result is the primary current proxy evidence:

| Condition | Earlier 128-scene FT6 | Corrected 1,000-scene FT6 |
|---|---:|---:|
| Sequential fully correct | 96.09% | **96.20%** |
| Sequential target wake recall | 100% | **99.70%** |
| Overlap fully correct | 89.84% | **91.70%** |
| Overlap target wake recall | 100% | **100%** |

The difference is expected from the larger, dynamically sampled scene set and
the full-length command protocol. The 1,000-scene values should replace the
128-scene values for the current FT6 proxy headline, while both remain
synthetic and non-production evidence.

## Decision

**FT6 is retained as the strongest current downstream candidate, but it is not
promoted to production.** It improves command accuracy and reduces
non-target actions relative to the earlier P0 baseline, but the strict
false-wake rate and the remaining wrong-intent actions are still too high.

The next safe step is threshold/policy tuning on a separate tuning split:

1. sweep the compact speaker-gate operating point and report the full
   target-recall versus non-target-action curve;
2. evaluate one-action-per-wake refractory behavior while preserving the
   three-call buffer;
3. compare endpointing only after the canonical buffer baseline is frozen;
4. re-run clean, sequential, and overlap regressions with the selected
   thresholds;
5. keep the 1,000-scene test artifacts untouched.

No further checkpoint promotion should occur until a real-driver pilot and a
target-device latency measurement are available.
