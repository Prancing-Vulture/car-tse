# Progress comparison: initial project to current production-readiness effort

Date: 2026-10-12  
Workspace: `D:/Prawin_New/tse/car-tse-main`

## Executive summary

The project has improved substantially as an engineering and evaluation system, but it has not crossed the production-readiness line.

The most meaningful measured changes are:

- The promoted dual-domain Stage 1 candidate improves the AISHELL-5 diagnostic by **+3.86 dB SI-SDR improvement** over the original Stage 1 checkpoint on a matched 288-example regression, while preserving CAVEMOVE at essentially the same level (**+0.11 dB** change).
- The closest expanded same-speaker proxy improved from **25% target-wake recall / 0% final intent / 64.3 false actions per hour** to **78.1% / 68.8% / 0 observed false actions per hour**. This is a strong diagnostic improvement, but it is still cross-corpus and not a real-driver result.
- A smaller, selected 8-scene safety proxy reaches **100% wake and 100% intent with 0 observed false actions/hour**, but its exposure is only 112 seconds and the wake recording is not verified to be from the enrollment/command speaker. It must not be treated as a production metric.
- Long target-absent evidence is now much stronger: from an initial 120-second/0-event check to **3.889 hours across 18 AISHELL-5 sessions**, where the current full path records **4.63 smoothed events/hour**. The rate is not yet good enough, but the newer result is far more informative and exposes the real calibration problem.
- Clean intent accuracy is essentially maintained at about **99.3% top-1**; there is no material improvement to claim there.
- The complete checkpoint remains about **424 MB**; the edge-footprint problem is not solved.
- Test coverage increased from **7 passed, 1 skipped** to **36 passed, 1 skipped**.

The honest status change is therefore: **much better evidence, stronger domain adaptation, and a safer/ more measurable cascade; still not production-ready because real same-driver continuous recordings and a frozen held-out test are missing.**

## How the comparison was defined

The initial point is the first full base-project benchmark in `reports/base_project_report_2026-10-04.md`, supplemented by the first continuous proxy in `reports/continuous_full_pipeline_proxy_2026-10-04.md`.

The current point uses the latest validated artifacts available on 2026-10-12. Where the test protocol changed, the comparison is labelled as diagnostic rather than presented as a like-for-like score. In particular:

- The original continuous proxy had four 14-second cross-corpus scenes and a target wake/command speaker mismatch.
- The later 32-scene proxy has 448 seconds of audio, uses a same-speaker enrollment/command construction, but the wake clip is still not independently verified to be from that same speaker.
- Neither proxy is a real-driver evaluation.

## Metric comparison

| Area | Initial baseline | Current result | Change | Interpretation |
|---|---:|---:|---:|---|
| Stage 1, original CAVEMOVE held-out benchmark | +10.98 dB SI-SDRi | Not directly re-run on the identical 400-example protocol | — | Synthetic cabin extraction was already strong; this was not the main bottleneck. |
| Stage 1, matched AISHELL-5 regression | 2.39 dB SI-SDRi with original checkpoint | 6.24 dB SI-SDRi with promoted `ft1` candidate | **+3.86 dB** | The real-cabin diagnostic improved without materially regressing CAVEMOVE. |
| Stage 1, matched CAVEMOVE regression | 10.48 dB SI-SDRi with original checkpoint | 10.59 dB SI-SDRi with promoted `ft1` candidate | **+0.11 dB** | Synthetic cabin performance was maintained. |
| Stage 1, matched two-domain overall | 6.43 dB SI-SDRi | 8.42 dB SI-SDRi | **+1.99 dB** | Better balance across the two tested domains. |
| Continuous target wake recall, first 4-scene proxy | 25% (1/4) | 78.1% (25/32) on expanded proxy | **+53.1 percentage points** | Large diagnostic gain, but not a real-driver claim. |
| Continuous final intent accuracy, first 4-scene proxy | 0% (0/4) | 68.8% (22/32) on expanded proxy | **+68.8 percentage points** | The guarded cascade now passes many more target interactions. |
| False actions on the same expanded proxy | 64.3/hour equivalent | 0 observed/hour | **0 observed in current run** | Encouraging, but the proxy is short and not production evidence. |
| Selected 8-scene safety proxy | Not available in the initial study | 100% wake, 100% intent, 0 observed false actions/hour | — | Useful smoke/safety result only: 112 seconds and an unverified wake-speaker identity. |
| Harder ReMASC vehicle-stress proxy | Not available | 96.9% wake, 81.3% intent, 0 observed false actions/hour | — | More realistic stress evidence; still synthetic/cross-corpus. |
| Clean official FSC intent test | 99.31% top-1 in the base report | 99.26% top-1 in the current dual-domain head audit | **-0.05 percentage points** | Statistically/practically unchanged; no improvement should be claimed. |
| Narrow selected KWS F1 | 97.40% selected operating point | Broader current tests are lower and harder; full-path negative testing is now available | Not comparable | The important improvement is deployment-style testing, not a higher isolated F1. |
| Long target-absent full-path exposure | 120 seconds, 0 events/hour reported | 14,002 seconds / 3.889 hours, 18 sessions, 4.63 events/hour | **~117× more exposure** | The current number is worse-looking but much more credible; it reveals residual false accepts. |
| GPU forward timing | About 69.95 ms in the base measurement | 48.55 ms FP16 wake stage; 62.34 ms FP16 post-wake command stage | About **30.6% lower** wake-stage time in that measurement | Precision/context definitions differ; complete streaming latency is still unproven. |
| CPU forward timing | About 275 ms in the base measurement | 522 ms wake / 642 ms post-wake in the current full 3-second-context measurement | Not a demonstrated improvement | The current CPU number includes a fuller production-shaped path and is not directly comparable. |
| Complete model footprint | About 423.7 MB | 424.5 MB | **+0.8 MB / +0.2%** | Essentially unchanged; wav2vec remains the edge-footprint bottleneck. |
| Automated test suite | 7 passed, 1 skipped | 36 passed, 1 skipped | **+29 passing tests** | Reproducibility and safety/evaluator coverage improved materially. |

## What changed technically

### 1. Stage 1 became domain-aware

The initial system was primarily CAVEMOVE-shaped. The later work added dual-domain regression and a Stage 1 fine-tuning path using AISHELL-5 diagnostics while retaining a CAVEMOVE non-regression check. The promoted `ft1` checkpoint is the safer downstream choice; the later `ft2` result was kept diagnostic because it was not the selected downstream handoff.

This is the clearest model-quality gain: AISHELL-5 SI-SDRi rose from 2.39 to 6.24 dB on the matched regression, while CAVEMOVE rose slightly from 10.48 to 10.59 dB.

### 2. The wake-to-command handoff became guarded

The current path includes target-speaker frame activity, a target-speaker KWS branch, speaker gates, separate wake/command thresholds, temporal smoothing, refractory behavior, an intent-confidence veto, and an explicit wake-to-command state transition. The wake frame is not reused as the command window.

These changes explain why the current short safety proxies can reach zero observed false actions even when the unguarded/early proxy produced false actions. They do not remove the need for a real-cabin test.

### 3. Hard-negative and long-negative testing was added

The initial report mostly had isolated KWS scores and a very short continuous negative check. The current work adds AISHELL-5 speech-rich target-absent streams, hard-negative target-KWS checkpoints, threshold sweeps, smoothing/refractory experiments, and a 18-session aggregate.

The result is more actionable rather than simply better: at the current operating point, KWS-only smoothed events are 96.67/hour over the 18-session AISHELL-5 negative run, while the complete target-speaker-gated path reduces this to 4.63/hour. That is still above a typical production target, but it identifies exactly where further work is needed.

### 4. The evaluation is more honest and reproducible

Since the base report, the project now has:

- a real-driver session manifest/validator and pilot protocol;
- production-readiness gates;
- frozen-config/checkpoint identity reporting;
- condition-specific proxy outputs;
- negative-exposure rate definitions;
- 95% confidence-interval support for event rates and recalls;
- a rate-denominator audit so false accepts are reported per hour of labelled negative exposure when real sessions exist;
- additional ablation and research-journey documentation;
- 36 passing automated tests.

This is a major evidence-quality improvement even where the measured model score did not improve.

## What did not improve enough

1. **Real-driver evidence is still absent.** The readiness snapshot still reports zero valid sessions. There is no same-driver continuous recording containing enrollment, `Marvin`, command, passenger speech, passenger `Marvin`, and no-command periods.

2. **False acceptance is not at a production level.** The best long target-absent current reference is 4.63 full-path events/hour over 3.889 hours. This is much more credible than the initial 0/hour over 120 seconds, but it is not yet a satisfactory release result.

3. **The full proxy does not yet reach the requested 80% consistently.** The harder expanded proxy is 78.1% wake and 68.8% final intent. The ReMASC stress configuration reaches 96.9% wake and 81.3% intent with zero observed false actions, but it is a separate diagnostic setup and still does not establish generalization.

4. **The clean intent model was already good.** The current result is within 0.1 percentage point of the base result. Work should not focus on increasing clean FSC accuracy until the noisy, target-speaker-conditioned command path is measured on real sessions.

5. **The deployment footprint remains too large.** The full checkpoint is still roughly 424 MB, and the current Stage 1 context is non-causal with a 3-second window. Automotive DSP/edge readiness has not been demonstrated.

## Production-readiness status: then versus now

| Readiness question | Initial status | Current status |
|---|---|---|
| Does the cascade run end to end? | Yes, as a research prototype | Yes, with more gates and reproducible diagnostics |
| Does it work on isolated clean components? | Mostly yes | Still yes; intent remains about 99.3% clean top-1 |
| Does it reject speech-rich target-absent cabin audio? | Only a short, weakly informative check | Partially demonstrated over 3.889 hours, but 4.63 full-path events/hour remains |
| Does it reliably handle the same driver across enrollment, wake, and command? | No evidence | Still no real-driver evidence; proxy is improved but insufficient |
| Is passenger rejection proven in real cabin recordings? | No | No; only synthetic/cross-corpus proxies exist |
| Is the model production-ready? | No | **No**; critical gates remain blocked |

The current readiness snapshot is `research_engineering_preview`, not a release candidate. The blocked gates include same-driver continuity, passenger rejection, study size, matching capture hardware, frozen split, threshold freeze, and reproducible release state.

## Bottom line

Since the initial stage, the project has moved from a promising but poorly connected cascade to a substantially better controlled research system:

- **Stage 1 real-cabin diagnostic:** materially better (+3.86 dB SI-SDRi on the matched AISHELL regression).
- **Expanded continuous proxy:** materially better (+53.1 points wake recall; +68.8 points final intent versus the first proxy; 0 observed false actions in the current 32-scene run).
- **Clean intent:** essentially unchanged and already strong.
- **Latency/size:** partial GPU-speed improvement in the optimized measurement, but no edge-ready footprint or complete streaming proof.
- **Evidence quality:** substantially better, with ~117× more target-absent exposure, CI support, denominator correction, readiness gates, and 29 additional passing tests.
- **Production status:** unchanged at the decision level—**not ready until a consented, speaker-consistent, held-out real-cabin pilot is recorded and passed.**

The next improvement effort should therefore target real-driver data collection and joint calibration of Stage 1, target activity, speaker verification, KWS, and action vetoes—not another clean-intent fine-tune.

## Source artifacts

- Initial benchmark: `reports/base_project_report_2026-10-04.md`
- Initial continuous proxy: `reports/continuous_full_pipeline_proxy_2026-10-04.md`
- Current Stage 1 regression: `logs/stage1_dual_domain_regression_full_2026-10-10.json`
- Current expanded same-speaker proxy: `logs/continuous_full_pipeline_proxy_ft1_positive_fullmatrix_same_speaker_m10_32scenes_2026-10-11.json`
- Current selected safety proxy: `logs/continuous_full_pipeline_proxy_hardneg_aishell5_neg_splitgate_p015_wake024_cmd022_same_speaker_m10_2026-10-11.json`
- Current ReMASC stress proxy: `logs/continuous_full_pipeline_proxy_remasc_personal_activity_margin_m067_fast_stream_hop025_8scenes_2026-10-12.json`
- Current long target-absent run: `logs/aishell5_continuous_targetkws_hardneg_aishell5_ft1_p015_sg024_kws050_smooth22_eval2_all18_dedup_2026-10-11.json`
- Current readiness snapshot: `reports/production_readiness_snapshot_ratefix_ci.md`
