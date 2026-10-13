# Agent handoff context — session 2 (in-car target-speaker voice assistant)

Generated: 2026-10-13  
Repository: `D:/Prawin_New/tse/car-tse-main`  
Branch: `full-pipeline` (all work pushed; worktree clean as of this document)  
Companion document: `reports/agent_handoff_context_2026-10-12.md` (session-1
context: mission, architecture, operational rules — still valid; this document
supersedes its status/evidence sections).

## 1. Current honest status

`research_engineering_preview`, `production_ready=false`. The system is a
four-stage research prototype (SpEx+ dual-domain TSE → frame presence →
target-speaker Marvin KWS → wav2vec2 intent) with a new real-time streaming
runner and a compact speaker-verifier identity gate. Since session 1 it gained
substantially better safety evidence, a capture/inference ring buffer, and a
validated CPU operating point.

Still blocked (all data/hardware):
same-driver continuity, passenger rejection, study size, capture hardware
match, frozen split, threshold freeze, reproducible release state.
This host exposes only a **single-channel Fifine microphone** (verified
2026-10-12); no 4-channel interface exists, so the pilot cannot be recorded
here. No real-driver session exists anywhere.

Evidence-present gates: no-command safety diagnostics (real cabin + clean
speech negatives), latency, resource budget, regression artifacts.

Do not claim production readiness, SOTA, or real-driver reliability.

## 2. Environment and how to run

- Python env: `D:\Prawin_New\tse\car-tse-main\.venv` (Python 3.12.14,
  torch 2.14.1+cu132, transformers 5.19, scipy, matplotlib, sounddevice).
- Run everything with the venv interpreter, from the repo root:
  `& "D:\Prawin_New\tse\car-tse-main\.venv\Scripts\python.exe" ...`
- Tests: `.venv\Scripts\python.exe -m pytest tests/` → **43 passed, 1 skipped**.
- GPU: RTX 4080 16 GB (idle between runs). CPU: Ryzen 7 5800X3D (8C/16T; use
  8 threads; 16 is slower). A concurrent heartbeat agent edits docs and runs
  jobs in this repo — preserve its changes, don't reset the worktree.

## 3. Recommended runtime configuration

`configs/eval_end_to_end_cavemove_target_kws_smart_speaker_compact_cabin_smart_thr030_2026-10-12.yaml`
(compact verifier backend, gate 0.30; presence 0.15, target-KWS 0.50, 2-of-2
smoothing, command gate 0.30, intent confidence 0.53).

Checkpoints (all preserved; none overwritten):

| Role | Checkpoint | Params |
|---|---|---:|
| Stage 1 TSE | `stage1_spex_dual_domain_rms010_ft1/best.pt` | 10.73 M |
| Presence | `stage2_frame_presence_dual_domain_nonwake_ft1/best.pt` | 33.7 k |
| General KWS | `stage3_kws_gsc_marvin_tse_ft2/best.pt` | 93.4 k |
| Target KWS | `stage3_target_kws_gsc_marvin_hardneg_aishell5_ft1/best.pt` | 389 k |
| **Compact verifier (recommended)** | `compact_speaker_verifier_cabin_smart_ft1/best.pt` | 176.6 k |
| Intent | `stage4_intent_wav2vec_ft8_resume_tse_domain/{best,encoder_best}.pt` | 94.78 M |

Total 106.2 M params / 425.24 MB. Alternative: `compact_speaker_verifier_cabin_aware_ft1`
(same safety, lower positive retention) and `compact_speaker_verifier_librispeech_ft1`
(v3, rejected for the system — see section 7). The SpEx splitgate config
(`..._neg_safety_splitgate_candidate.yaml`) remains the historical reference.

## 4. Accuracy — current best numbers

Per-stage (protocols in `reports/system_accuracy_latency_report_2026-10-12.md`):

- **Stage 1**: AISHELL-5 matched SI-SDRi 6.24 dB (baseline 2.39); CAVEMOVE 10.59
  (10.48); overall 8.42 (6.43), 288-example regression.
- **Stage 2 presence**: val frame@0.5 acc 98.84 / F1 90.4 / FAR 0.90%; chunk
  acc 99.79 / F1 99.59.
- **Stage 3 KWS standalone**: general val F1 66.9; target val F1 43.9
  (negative-heavy; operative behavior is the cascade).
- **Stage 3b verifier (v2)**: Dev AUC 0.938, Eval1 AUC 0.9706, TPR 79.7% @
  FPR 0.53% (validation-selected); cache pass @0.30: GSC targets 93.0%,
  AISHELL 2.0%, Smart 0%, non-wake 5.7%.
- **Stage 4 intent**: clean FSC 99.31% top-1; TSE-domain val 90.49% top-1,
  98.01% top-5, ECE 0.023.

Full-system:

- **AISHELL-5 Eval2, 18 sessions, 3.889 h, target absent (real 4-ch cabin)**:
  SpEx reference 18 events / 4.63 per hour; compact v2 @0.30 → **0 events**
  (95% upper bound 0.95/h); also **0 events at 0.25 s hop with 8-of-8
  confirmation** (2026-10-13, KWS-only 71.74/h).
- **Smart Speaker all-file (3.193 h, 2,110 files, clean Alexa speech)**:
  reference 86 / 26.94 per hour; compact v2 @0.30 → **11 / 3.45 per hour**;
  0 false actions.
- **15-minute continuous streams (one stateful pass, 0.25 s hop, noise +
  silence + sparse events)**: speaker-consistent wake stream 12/12 wakes,
  1.49 s mean latency, 2 false wakes (one passenger Marvin leak at 2.9 dB),
  0 false actions; cross-corpus-wake stream 0/12 (extractor suppresses the
  different speaker — documented artifact, not a bug).
- **Positive proxies (cross-corpus)**: expanded 32-scene 78.1% wake / 68.8%
  intent (88% conditioned); ReMASC stress 96.9% wake / 81.3% intent, 0 false
  actions, 0.97 s latency.
- Clean intent unchanged (~99.3%). All positive evidence is synthetic or
  cross-corpus; the wake/intent speaker identity is not verified.

## 5. Latency and streaming

Per-component (RTX 4080 FP16 mean / this CPU FP32 mean):

| Component | GPU | CPU |
|---|---:|---:|
| Stage 1 extraction 3 s × 4 ch | 66.5 ms | 423 ms |
| Presence 3 s | 1.6 ms | 1.1 ms |
| Target KWS 1 s | 2.2 ms | 2.1 ms |
| General KWS 1 s | 1.5 ms | 2.0 ms |
| Compact verifier 2 × 1 s | 5.0 ms | 3.0 ms |
| Intent wav2vec 2 s | 17.4 ms | 95 ms |
| Composite wake stage | 57.0 ms | 561 ms |
| Composite command stage | 76.2 ms | 680 ms |

Deployment operating points (streaming runner with capture thread):
GPU 0.25 s hop RTF 0.28 / 0 skips; CPU 0.25 s hop RTF 1.73 → 9 s skipped of
30 s; **CPU 1.0 s hop RTF 0.47 / 0 skips / 0.9 s peak lag** (reference timing).
Dynamic int8 only affects intent (Stage 1 convs dominate). Strictly bounded
latency needs chunked inference; 0.25 s CPU needs GPU, quantization, or a
lighter Stage 1.

New streaming infrastructure: `src/car_tse/pipeline/streaming_runner.py`
(ring buffer; push never blocks inference; drop_oldest/catch_up with explicit
`skipped_seconds`; stats) + `scripts/eval/streaming_runner_simulation.py`;
7 unit tests in `tests/test_streaming_runner.py`.

## 6. Session-2 commits (all pushed to origin/full-pipeline)

- `d671f12` production-readiness tooling and reports (session-1 dirty tree)
- `f2e9382` cabin-aware compact verifier + long-negative evaluation
- `aa7d8d6` Smart Speaker cross-domain check
- `6250cae` Smart-Speaker-augmented verifier + readiness audit refresh
- `5038206` long-form 15-minute stream evaluator
- `373403c` streaming ring buffer and overload-aware runner
- `bba9dd2` consolidated accuracy/latency report + per-component benchmark
- `a278708` zero-event interval label fix
- `2c23d13` verifier data scaling + CPU operating point (+ this session-2
  handoff and fast-hop safety entry in the follow-up commit)

## 7. Negative results — do not repeat without new data

- Verifier v3 (LibriSpeech-scaled, supervised) rejects generic mismatches far
  better (Libri negatives 18.9% vs 45.9%) but keeps fewer same-speaker
  positives (68.0% vs 93.0%); mean/min/product ensembles do not dominate v2 on
  validated benchmarks. v2 stays recommended.
- 16-thread CPU is slower than 8; int8 dynamic quantization barely moves the
  wake stage (Stage 1 convs dominate).
- 0.25 s hop with 2-of-2 confirmation remains unsafe (0.5 s evidence);
  equivalent 8-of-8 is safe (0 events). 4-of-4 (1 s evidence) was interrupted
  and has no log — rerun before quoting.
- Prior rejections (see session-1 handoff): fixed RMS conditioning, Stage 1
  hard-negative FT, margin variants, fast stream with the old gating, mined
  KWS v2/v3/v4, compact verifier with old threshold transfer.

## 8. Next steps (ranked)

1. **Real-driver pilot** (decisive): 4-channel interface + consented sessions
   per `reports/real_driver_pilot_protocol_2026-10-09.md`; then validate,
   freeze thresholds on tuning drivers, evaluate held-out. Until this exists,
   every recall/passenger number stays synthetic.
2. **Rerun the interrupted 4-of-4 fast hop** on AISHELL 18 sessions (one
   command; ~25 min) and, if safe, measure stream-A latency at that setting.
3. **Tighten the zero-event bound** with more held-out negative exposure
   (target 15–20 h total; Eval2 is exhausted, so this needs new sessions or
   clearly-labelled synthetic beds).
4. **Causal/chunked Stage 1** to reach drop-free 0.25 s hop on CPU and reduce
   the 425 MB footprint (intent is 89% of parameters).
5. Passenger-Marvin hard negatives in target-KWS/verifier training to close
   the one observed stream leak.

## 9. Operational rules (carry-over, still binding)

- Preserve the dirty worktree / concurrent heartbeat edits; no broad resets.
- Do not overwrite established checkpoints; unique names for every output.
- Run focused tests, then the full suite, after code changes.
- Dual-domain Stage 1 regression before promoting TSE; long-negative + positive
  checks before threshold changes.
- Never call a proxy real-driver/production/SOTA; zero counts are bounds, not
  guarantees.
- Do not commit or push unless the user explicitly requests it in the current
  interaction.
- Keep the 15-minute heartbeat running. For PDFs read the PDF skill first; the
  latest PDF is `output/pdf/in_car_voice_assistant_ablation_report_2026-10-11.pdf`.

## 10. Key report files

- `reports/system_accuracy_latency_report_2026-10-12.md` — consolidated
  accuracy/latency (start here).
- `reports/compact_speaker_verifier_cabin_safety_2026-10-12.md` — verifier
  safety study, streams, audit refresh.
- `reports/research_journey.md` — full experiment log (latest entries
  2026-10-12/13).
- `reports/production_readiness_snapshot_compact_cabin_smart_2026-10-12.md` —
  machine-readable readiness snapshot.
