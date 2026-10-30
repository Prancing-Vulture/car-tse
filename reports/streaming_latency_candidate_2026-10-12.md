# Streaming latency and command-window study

Date: 2026-10-12  
Status: diagnostic operating-point study; no production promotion

## Motivation

The fixed pipeline evaluates a three-second Stage 1 window on a one-second
hop. Its two-hit wake smoothing produced roughly 1.9 seconds of proxy wake
latency. Pre-existing streaming KWS systems treat hop size, temporal state,
refractory policy, and post-wake command timing as one evaluated system. The
short-hop experiment also exposed that the first command window can overlap
the wake word.

An opt-in `command_delay_calls` state-machine parameter was added. It delays
intent inference after a wake by a configured number of accepted streaming
calls; the historical default is zero. A longer refractory period was tested
for denser streams to merge duplicate wake events.

## ReMASC proxy results

All rows use the same 32 synthetic scenes / 448 seconds, ReMASC vehicle stress
bed, group-split three-way activity head, independent-top margin `-0.67`, and
speaker gates. These are not real-driver measurements.

| Hop | Delay calls | Refractory calls | Wake | Intent | False wakes/h | False actions/h | Mean latency | P95 latency |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00 s | 0 | 2 | 90.6% | 78.1% | 64.3 | 0.0 | 1.879 s | 2.5 s |
| 0.50 s | 0 | 2 | 93.8% | 56.3% | 64.3 | 0.0 | 1.350 s | 2.0 s |
| 0.25 s | 0 | 2 | 96.9% | 56.3% | 192.9 | 64.3 | 0.968 s | 1.625 s |
| 0.25 s | 2 | 2 | 96.9% | 59.4% | 128.6 | 0.0 | 0.968 s | 1.625 s |
| 0.25 s | 4 | 2 | 96.9% | 84.4% | 128.6 | 0.0 | 0.968 s | 1.625 s |
| 0.25 s | 4 | 8 | **96.9%** | **81.3%** | **64.3** | **0.0** | **0.968 s** | **1.625 s** |

The four-call delay restores a clean command window at the fast hop. The
longer refractory period removes duplicate wake events without changing the
observed target recall or latency. The resulting candidate is faster than the
one-second reference and preserves the proxy safety result, but its negative
exposure is still far too small and synthetic for a release claim.

## AISHELL-5 target-absent follow-up

The fast hop was replayed over 1,200 seconds of real four-channel AISHELL-5
Eval2 audio with the fixed enrollment speaker absent. With a two-window/two-hit
KWS policy and the same `-0.67` activity margin, both eight-call and sixteen-
call refractory settings produced **6.0 full-path events/hour**. The one-
second reference produced 3.0/hour on the comparable 1,200-second diagnostic.
The longer refractory reduced KWS-only event rate from 192.0/hour to 147.0/hour
but did not reduce the full-path rate. This rejects the fast setting as a
current safety improvement; it must not replace the reference configuration.

## Decision and next evidence

Keep the fast-stream configuration as an opt-in research candidate:
`configs/eval_end_to_end_cavemove_personal_activity_group_p025_margin_m067_fast_stream.yaml`.
Do not change the production/reference configuration until the same settings
are calibrated on permitted driver tuning sessions and evaluated on held-out
drivers. The real deployment latency must also be measured on the intended
automotive CPU/DSP; RTX 4080 timing is only a development reference.

## Artifacts

- `src/car_tse/pipeline/in_car_pipeline.py`
- `scripts/eval/eval_continuous_full_pipeline_proxy.py`
- `tests/test_pipeline_drop.py`
- `configs/eval_end_to_end_cavemove_personal_activity_group_p025_margin_m067_fast_stream.yaml`
- `logs/continuous_full_pipeline_proxy_remasc_personal_activity_margin_m067_hop050_8scenes_2026-10-12.json`
- `logs/continuous_full_pipeline_proxy_remasc_personal_activity_margin_m067_hop025_8scenes_2026-10-12.json`
- `logs/continuous_full_pipeline_proxy_remasc_personal_activity_margin_m067_hop025_delay2_8scenes_2026-10-12.json`
- `logs/continuous_full_pipeline_proxy_remasc_personal_activity_margin_m067_hop025_delay4_8scenes_2026-10-12.json`
- `logs/continuous_full_pipeline_proxy_remasc_personal_activity_margin_m067_fast_stream_hop025_8scenes_2026-10-12.json`
- `scripts/eval/eval_aishell5_continuous_kws.py`
- `logs/aishell5_target_absent_personal_activity_hop025_ref8_1200s_2026-10-12.json`
- `logs/aishell5_target_absent_personal_activity_hop025_ref16_1200s_2026-10-12.json`
