# Dual-domain Stage 1 + Stage 2 handoff study

Date: 2026-10-10  
Status: bounded proxy study; not a real-driver evaluation

## Purpose

This study checked whether the dual-domain Stage 1 checkpoint could be carried
through the presence, wake-word, and command handoff without inheriting stale
thresholds. It also tested whether the extracted signal is the right input for
the post-wake intent model.

The positive scenes are cross-corpus proxies: the wake-word clips and command
clips are not verified to be spoken by the same person. AISHELL-5 is used only
as a target-absent real-cabin diagnostic. No real-driver reliability claim is
made.

## Checkpoints and operating point

- Stage 1: `checkpoints/stage1_spex_dual_domain_rms010_ft1/best.pt`
- Stage 2: `checkpoints/stage2_frame_presence_dual_domain_nonwake_ft1/best.pt`
- KWS: `checkpoints/stage3_kws_gsc_marvin_tse_ft2/best.pt`
- Target speaker KWS: `checkpoints/stage3_target_kws_gsc_marvin_ft1/best.pt`
- Intent: `checkpoints/stage4_intent_wav2vec_ft8_resume_tse_domain/best.pt`
- Operating config: `configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_operating_candidate.yaml`

The new framewise presence head has the following held-out validation result:

| Metric | Value |
|---|---:|
| Frame F1 at threshold 0.50 | 0.904 |
| Frame FAR at threshold 0.50 | 0.009 |
| Frame FRR at threshold 0.50 | 0.055 |
| Best frame F1 (threshold 0.65) | 0.913 |
| Chunk F1 at threshold 0.50 | 0.996 |
| Chunk FAR at threshold 0.50 | 0.003 |
| Chunk FRR at threshold 0.50 | 0.000 |

The previous 0.70 presence threshold was not reusable: the new head outputs
were calibrated at a much lower probability scale. A joint proxy sweep selected
presence threshold 0.07, command-presence threshold 0.10, KWS threshold 0.50,
and two-hit smoothing. These thresholds are provisional until tuned on real
driver sessions.

## Full continuous proxy

Eight scenes were replayed at cabin SNRs -5, 0, 10, and 20 dB with passenger
speech and a passenger wake-word clip. The total exposure was 112 seconds.

| Variant | Wake recall | Intent accuracy | False actions/hour | False wake events |
|---|---:|---:|---:|---:|
| Full pipeline | 100.0% | 12.5% | 0.0 | 5 |
| No TSE | 100.0% | 75.0% | 96.4 | 9 |
| No presence | 100.0% | 12.5% | 450.0 | 20 |
| No speaker matching | 87.5% | 12.5% | 0.0 | 3 |
| No wake smoothing | 100.0% | 0.0% | 32.1 | 6 |

The full path is conservative and suppresses false actions, but command
accuracy remains the main bottleneck in this cross-corpus setup. The no-TSE
accuracy must not be interpreted as a better system: it also creates many
false actions, which is unacceptable for vehicle control.

## Post-wake intent handoff ablation

The same eight scenes were replayed with the full safety path. Only the signal
fed to the intent model after an accepted wake was changed.

| Handoff | Wake recall | Intent accuracy | False actions/hour |
|---|---:|---:|---:|
| Extracted signal (current full path) | 100.0% | 12.5% | 0.0 |
| Raw multichannel mixture after wake | 100.0% | **37.5%** | 0.0 |
| Mean logits: extracted + raw | 100.0% | 25.0% | 0.0 |

This is the most useful current engineering lead: TSE/presence/speaker gating
appears valuable for deciding whether to listen, while the current intent
checkpoint can perform better when given the raw post-wake mixture. The raw
handoff is only a candidate; it must be tested on same-driver recordings and
under passenger interference before promotion.

At passenger SNR +10 dB, the full path remained at 100% wake recall and 12.5%
intent accuracy with 0 false actions/hour. At passenger SNR -10 dB, it had
100% wake recall, 12.5% intent accuracy, and 32.1 false actions/hour. The raw
handoff also reached 37.5% intent accuracy at -10 dB, but retained the same
32.1 false actions/hour; it is therefore not yet a safety improvement in the
harsher passenger condition. This small proxy set is too small to establish a
monotonic noise trend.

## Speaker-consistency diagnostic

The default proxy uses a Marvin wake clip for enrollment and an unrelated FSC
speaker for the command. To isolate that mismatch, two diagnostic replays were
run at passenger SNR +10 dB:

| Enrollment source | Interpretation | Wake recall | Intent accuracy | False actions/hour |
|---|---|---:|---:|---:|
| Wake clip | Realistic cross-corpus proxy; wake and command speakers differ | 100.0% | 12.5% | 0.0 |
| Same command clip | Leakage-prone oracle; enrollment reuses command audio | 100.0% | 100.0% | 32.1 |
| Different FSC utterance, same command speaker | Partial speaker-consistency diagnostic; wake speaker still differs | 100.0% | 100.0% | 64.3 |

This sharply narrows the diagnosis: command intent is much better when the
enrollment speaker matches the command speaker, but the safety gate still
produces false actions. The same-command result is not evidence of model
accuracy because the command audio is reused for enrollment. The different-
utterance result removes that lexical reuse, but it is still not a real-driver
test because the wake-word speaker is not verified to match and the scene is
synthetically assembled.

### Speaker-gate follow-up

An explicit cosine speaker-similarity gate was swept on the same-speaker-other
diagnostic. Thresholds 0.15, 0.18, 0.20, and 0.22 preserved 100% wake recall
and 100% intent accuracy while reducing false actions from 64.3/hour to
0/hour; the 0.22 setting also reduced false wake events to one over 112
seconds, at a mean wake latency of 2.125 seconds. This is a promising safety
mechanism for real-driver tuning, but it cannot be selected from the current
proxy: on the wake/command cross-corpus proxy, even a 0.10 gate reduced wake
recall to 12.5%, and thresholds at or above 0.12 reduced it to 0%. The gate
therefore exposes the proxy's speaker mismatch rather than establishing a
production threshold.

Under a louder passenger condition (passenger SNR -10 dB), the same-speaker
diagnostic showed the expected trade-off: 0.22 retained 100% wake/intent
accuracy but still produced 32.1 false actions/hour; 0.27 reduced false
actions to 0/hour but reduced wake recall to 75% and intent accuracy to
37.5%. This makes the gate a tunable safety/recall control, not a solved
threshold.

Artifacts:

- `logs/continuous_ablation_matrix_stage2_same_speaker_gate_sweep_2026-10-10.json`
- `logs/continuous_ablation_matrix_stage2_cross_corpus_gate_sweep_2026-10-10.json`
- `logs/continuous_ablation_matrix_stage2_same_speaker_gate_passenger_m10_2026-10-10.json`
- `logs/continuous_ablation_matrix_stage2_same_speaker_gate_passenger_m10_high_2026-10-10.json`

### Hard-negative target-KWS replay

The repository already contained a trained hard-negative target-KWS checkpoint,
so it was evaluated without starting a new training run. The dual-domain Stage
1/Stage 2 checkpoints, thresholds, scenes, and intent model were held fixed.

| Target-KWS checkpoint | Passenger SNR | Wake recall | Intent accuracy | False actions/hour | False wake events |
|---|---:|---:|---:|---:|---:|
| Current target KWS | +10 dB | 100.0% | 12.5% | 0.0 | 5 |
| Hard-negative target KWS | +10 dB | 100.0% | 12.5% | 0.0 | 3 |
| Current target KWS | -10 dB | 100.0% | 12.5% | 32.1 | 6 |
| Hard-negative target KWS | -10 dB | 100.0% | 12.5% | 32.1 | 5 |

The hard-negative checkpoint reduces proxy false wake events modestly without
changing target recall or command accuracy. It does not solve the harsh-
passenger false-action failure, and its benefit is based on only eight scenes
per condition. It is retained as the next wake-stage candidate for a real-
driver tuning split.

Artifacts:

- `configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_hardneg_candidate.yaml`
- `logs/continuous_ablation_matrix_dual_domain_stage2_hardneg_2026-10-10.json`
- `logs/continuous_ablation_matrix_dual_domain_stage2_hardneg_passenger_m10_2026-10-10.json`

### Joint safety-threshold follow-up

The evaluator now supports an optional post-wake intent-confidence gate. On
the same partial speaker-consistency proxy at passenger SNR -10 dB, the
speaker gate at 0.22 reduced false actions from 192.9/hour to 32.1/hour while
retaining 100% wake and intent accuracy. Adding an intent-confidence threshold
of 0.53 removed the remaining proxy false action while retaining 100% wake and
intent accuracy. Thresholds 0.55 and 0.60 also stayed safe but suppressed one
target command, reducing intent accuracy to 87.5%. A command-presence threshold
of 0.15 produced the same safety result but also reduced intent accuracy to
87.5%.

The 0.22/0.53 pair is retained as a diagnostic safety candidate only. It must
be recalibrated on permitted tuning drivers and then frozen before any held-out
test. It is not a production threshold.

Artifacts:

- `configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_safety_diagnostic_candidate.yaml`
- `logs/continuous_ablation_matrix_stage2_intent_confidence_gate_passenger_m10_2026-10-10.json`
- `logs/continuous_ablation_matrix_stage2_intent_confidence_gate_passenger_p10_2026-10-10.json`
- `logs/continuous_ablation_matrix_stage2_joint_gate_command_presence_passenger_m10_2026-10-10.json`

### Additional ft2 Stage 1/head comparison

An additional low-rate ft2 Stage 1 frontend and separately aligned intent head
were replayed on the same eight partial-speaker-consistency scenes. Without a
speaker gate, ft1 and ft2 both reached 100% wake and intent accuracy with 64.3
false actions/hour. With a 0.18 speaker gate, ft2 retained 100% wake/intent
with 0 false actions/hour; at 0.20 it fell to 87.5% intent, and at 0.22 it
fell to 87.5% wake and intent. The ft2 frontend remains useful for AISHELL-5
SI-SDRi, but its aligned downstream head is not an end-to-end improvement on
this proxy.

- `configs/stage1_spex_dual_domain_rms010_ft2.yaml`
- `configs/stage4_intent_tse_dual_domain_ft2.yaml`
- `logs/continuous_ablation_matrix_dual_domain_ft2_intent_v1_2026-10-10.json`
- `logs/continuous_ablation_matrix_stage2_ft2_head_same_speaker_gate_sweep_2026-10-10.json`

## Real-cabin target-absent check

One 120-second AISHELL-5 Eval2 session was run with the dual-domain Stage 1,
dual-domain frame presence head, and the calibrated operating point. The fixed
LibriSpeech enrollment speaker is absent from this recording, so this measures
false triggering only.

| Metric | Result |
|---|---:|
| Duration | 0.0333 hours |
| Speech occupancy | 80.6% |
| Full-path smoothed trigger rate | 0.0/hour |
| KWS-only raw positive-window rate | 30.0/hour |
| KWS-only smoothed event rate | 0.0/hour |

This is encouraging for the conservative gated path, but it is only one short
target-absent session and does not test a true driver wake word.

### AISHELL target-speaker KWS diagnostic

The same 120-second target-absent session was replayed through the optional
target-speaker KWS head, using the same dual-domain Stage 1/Stage 2 path. This
is a diagnostic of cross-corpus calibration and rejection only: the session
contains no labeled positive wake word, and its fixed LibriSpeech enrollment
does not belong to the AISHELL speakers.

| KWS mode | Threshold | Full-path smoothed events/hour | KWS-only smoothed events/hour | Max KWS score |
|---|---:|---:|---:|---:|
| Generic KWS baseline | 0.50 | 0.0 | 0.0 | 0.554 |
| Current target-speaker KWS | 0.50 | 630.0 | 810.0 | 0.601 |
| Hard-negative target-speaker KWS | 0.50 | 150.0 | 180.0 | 0.680 |
| Hard-negative target-speaker KWS | 0.70 | 0.0 | 0.0 | 0.680 |

The hard-negative target-speaker head improves substantially over the current
target-speaker head at 0.50, but both are much worse than the generic head at
that inherited threshold. A 0.70 diagnostic threshold suppresses all events
in this 120-second target-absent slice, but it has not been tuned against
positive same-driver wake recordings and therefore is not a promoted setting.
The result reinforces that target-speaker KWS needs calibration on a permitted
same-driver tuning split before it can replace the generic KWS path.

The one-pass threshold replay makes the trade-off explicit (full-path smoothed
events/hour):

| Target-speaker head | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 |
|---|---:|---:|---:|---:|---:|
| Current | 630.0 | 30.0 | 0.0 | 0.0 | 0.0 |
| Hard-negative | 150.0 | 60.0 | 30.0 | 0.0 | 0.0 |

This replay reuses the same extracted windows; it is an evaluation-tool
optimization, not a new training result. The threshold must eventually be
selected on a separate permitted tuning split containing both driver positives
and passenger/no-command negatives.

A later AISHELL-5 hard-negative adaptation is documented separately in
`reports/aishell5_target_kws_adaptation_2026-10-11.md`. Its selected split-gate
candidate reduced the full-path rate to 0/hour over a 1,200-second Eval2
target-absent exposure, while KWS-only remained 75/hour. That longer negative
study is a follow-up candidate, not a replacement for the real-driver pilot.

## Decision and next step

The dual-domain Stage 1 candidate passes the full 144-row-per-domain
SI-SDRi no-regression check and is retained for integration experiments. The
new presence head is usable only with its recalibrated operating point. The
raw post-wake intent handoff is the next candidate to validate, not a promoted
default.

The next decisive step is to obtain permitted continuous same-driver cabin
sessions with enrollment, driver “Marvin,” command, passenger speech,
passenger “Marvin,” and no-command intervals. Tuning sessions must be kept
separate from held-out test drivers. Until then, all end-to-end numbers here
remain diagnostic proxy results.

## Reproduction

```powershell
python scripts/eval/eval_threshold_ablation_sweep.py `
  --config configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_operating_candidate.yaml `
  --paths configs/paths.yaml

python scripts/eval/eval_continuous_ablation_matrix.py `
  --config configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_operating_candidate.yaml `
  --paths configs/paths.yaml --scenes 2 `
  --cabin-snr-db -5 0 10 20 --passenger-snr-db 10 `
  --output logs/continuous_ablation_matrix_dual_domain_stage2_intent_handoff_2026-10-10.json

# Repeat with a much louder passenger to test the harsher handoff condition.
python scripts/eval/eval_continuous_ablation_matrix.py `
  --config configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_operating_candidate.yaml `
  --paths configs/paths.yaml --scenes 2 `
  --cabin-snr-db -5 0 10 20 --passenger-snr-db -10 `
  --output logs/continuous_ablation_matrix_dual_domain_stage2_intent_handoff_passenger_m10_2026-10-10.json

python scripts/eval/eval_continuous_ablation_matrix.py `
  --config configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_operating_candidate.yaml `
  --paths configs/paths.yaml --scenes 2 --cabin-snr-db -5 0 10 20 `
  --passenger-snr-db 10 --enrollment-source same_speaker_other `
  --output logs/continuous_ablation_matrix_dual_domain_stage2_same_speaker_other_2026-10-10.json

python scripts/eval/eval_continuous_ablation_matrix.py `
  --config configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_hardneg_candidate.yaml `
  --paths configs/paths.yaml --scenes 2 --cabin-snr-db -5 0 10 20 `
  --passenger-snr-db 10 `
  --output logs/continuous_ablation_matrix_dual_domain_stage2_hardneg_2026-10-10.json

python scripts/eval/eval_aishell5_continuous_kws.py `
  --paths configs/paths.yaml --split Eval2 --session-ids 001 `
  --max-seconds 120 `
  --stage1-config configs/stage1_spex_dual_domain_rms010_ft1.yaml `
  --stage1-checkpoint checkpoints/stage1_spex_dual_domain_rms010_ft1/best.pt `
  --presence-checkpoint checkpoints/stage2_frame_presence_dual_domain_nonwake_ft1/best.pt `
  --presence-type frame --presence-threshold 0.07 --kws-threshold 0.50 `
  --smooth-window 2 --required-hits 2 --refractory-calls 0

# Optional target-speaker KWS diagnostic; use a permitted tuning split before
# selecting a threshold for any positive/negative driver evaluation.
python scripts/eval/eval_aishell5_continuous_kws.py `
  --paths configs/paths.yaml --split Eval2 --session-ids 001 `
  --max-seconds 120 `
  --stage1-config configs/stage1_spex_dual_domain_rms010_ft1.yaml `
  --stage1-checkpoint checkpoints/stage1_spex_dual_domain_rms010_ft1/best.pt `
  --presence-checkpoint checkpoints/stage2_frame_presence_dual_domain_nonwake_ft1/best.pt `
  --presence-type frame --presence-threshold 0.07 --kws-threshold 0.70 `
  --smooth-window 2 --required-hits 2 --refractory-calls 0 `
  --target-kws-config configs/stage3_target_kws_gsc_marvin.yaml `
  --target-kws-checkpoint checkpoints/stage3_target_kws_gsc_marvin_hardneg_ft1/best.pt
```
