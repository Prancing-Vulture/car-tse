# Agent handoff context — in-car target-speaker voice assistant

Generated: 2026-10-12  
Repository: `D:/Prawin_New/tse/car-tse-main`  
Workspace root: `D:/Prawin_New/tse`  
Current branch at last audit: `full-pipeline`  
Current HEAD at last audit: `a2e62de Add configurable Stage 1 RMS conditioning experiment`

## 1. Mission

Build and evaluate a reliable in-car voice assistant that:

1. hears the enrolled driver in a noisy cabin;
2. detects the wake word `Marvin` only when spoken by the target driver;
3. understands the command that follows;
4. executes an action only when the target driver intended it;
5. rejects passengers, passenger wake words, background speech, road/HVAC/wind noise, and playback/echo;
6. is eventually suitable for low-latency, resource-constrained automotive deployment.

The current project is a research prototype, not a production release. The user requested continued improvement, efficient training, reproducible documentation, and periodic progress through the active 15-minute heartbeat. Do not stop or delete that heartbeat unless the user explicitly asks.

## 2. Current honest status

The project has improved materially as an engineering/evaluation system, but it is **not production-ready**.

The decisive limitation is still data/evidence: there are no valid real-driver continuous sessions containing the same driver's enrollment, `Marvin`, command, passenger speech, passenger `Marvin`, and no-command periods.

Current status should be described as:

> A staged GPU-capable research prototype with strong clean intent accuracy, improved dual-domain target-speaker extraction, guarded wake-to-command logic, long target-absent diagnostics, and incomplete real-cabin end-to-end validation.

Do not claim SOTA, production readiness, real-driver reliability, or broad 80%+ end-to-end accuracy from the synthetic/cross-corpus proxies.

## 3. Runtime architecture

```text
4-channel cabin audio + optional playback reference
        ↓
learned cabin frontend: echo-reference branch + spatial channel collapse
        ↓
16-kHz SpEx+-style target-speaker extractor
        ↓
framewise target-presence/activity gate
        ↓
target-speaker Marvin KWS
        ↓
streaming state machine: wake arms command capture; wake frame is not decoded as intent
        ↓
fixed extracted command window
        ↓
wav2vec 2.0 31-class Fluent Speech Commands intent model
        ↓
vehicle action + optional spoken reply
```

Important implementation behavior:

- enrollment is short and target-conditioned;
- the wake frame is not reused as the command frame;
- there is a wake-to-command timeout;
- target presence/activity, speaker gates, KWS threshold, temporal smoothing/refractory, and intent confidence can veto an action;
- current Stage 1 inference uses a non-causal 3-second context, so the measured forward time is not a complete streaming latency claim.

Main implementation files:

- Stage 1/TSE: `src/car_tse/models/stage1_system.py`
- Cabin frontend: `src/car_tse/frontend/cabin_frontend.py`
- Presence/activity: `src/car_tse/models/presence_head.py`
- KWS: `src/car_tse/models/kws_model.py`
- Pipeline/state machine: `src/car_tse/pipeline/in_car_pipeline.py`
- Intent: `src/car_tse/models/wav2vec_intent.py`
- Continuous proxy evaluator: `scripts/eval/eval_continuous_full_pipeline_proxy.py`
- Real-driver evaluator: `scripts/eval/eval_real_driver_sessions.py`
- Readiness audit: search `scripts` for `production_readiness` and inspect `reports/production_readiness_snapshot_ratefix_ci.md`.

## 4. Current recommended checkpoints/configuration

The safest current downstream baseline is the split-gate candidate configuration:

`configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_hardneg_aishell5_neg_safety_splitgate_candidate.yaml`

Recommended checkpoint roles:

- Stage 1: `checkpoints/stage1_spex_dual_domain_rms010_ft1/best.pt`
- Presence: `checkpoints/stage2_frame_presence_dual_domain_nonwake_ft1/best.pt`
- General/TSE KWS: `checkpoints/stage3_kws_gsc_marvin_tse_ft2/best.pt`
- Target KWS: `checkpoints/stage3_target_kws_gsc_marvin_hardneg_aishell5_ft1/best.pt`
- Intent head: `checkpoints/stage4_intent_wav2vec_ft8_resume_tse_domain/best.pt`
- Intent encoder: matching `encoder_best.pt` in that directory.

The original Stage 1 baseline remains useful for regression:

`checkpoints/stage1_spex_transfer_mc_cavemove_cabin/best.pt`

Experimental checkpoints are preserved. Do not promote a new one merely because it improves a short proxy; require dual-domain extraction checks plus long negative and positive tests.

## 5. Initial baseline versus current metrics

Initial baseline sources:

- `reports/base_project_report_2026-10-04.md`
- `reports/continuous_full_pipeline_proxy_2026-10-04.md`

Current comparison source:

- `reports/progress_comparison_initial_to_current_2026-10-12.md`

Important numbers:

| Metric | Initial | Current | Meaning |
|---|---:|---:|---|
| Original CAVEMOVE Stage 1 held-out SI-SDRi | +10.98 dB | Not identical protocol | Synthetic cabin extraction was already strong |
| Matched AISHELL Stage 1 regression | 2.39 dB | 6.24 dB with promoted `ft1` | **+3.86 dB**, clearest model-quality gain |
| Matched CAVEMOVE Stage 1 regression | 10.48 dB | 10.59 dB | Preserved synthetic performance |
| First continuous proxy wake recall | 25% (1/4) | 78.1% (25/32) expanded proxy | +53.1 percentage points, still diagnostic |
| First continuous proxy final intent | 0% | 68.8% (22/32) expanded proxy | +68.8 points, still cross-corpus |
| False actions in those proxy runs | 64.3/hour equivalent | 0 observed/hour | Encouraging but not production evidence |
| Long target-absent exposure | 120 s | 3.889 h / 18 AISHELL-5 sessions | Current evidence is ~117× longer |
| Long target-absent full-path rate | 0/hour on tiny run | 4.63/hour current reference | More credible, still too high for release |
| Clean FSC intent top-1 | 99.31% | 99.26% current dual-domain audit | Essentially unchanged |
| Complete model footprint | ~423.7 MB | 424.5 MB | Edge footprint remains unsolved |
| Automated tests | 7 passed, 1 skipped | 36 passed, 1 skipped | +29 passing tests |

The comparison has an important caveat: the proxy protocols changed. The later positive proxy is more speaker-consistent in enrollment/command construction, but its `Marvin` wake clip is not verified to be from the same speaker. It is not real-driver evidence.

## 6. Current strongest diagnostic results

### Expanded same-speaker proxy

Artifact:

`logs/continuous_full_pipeline_proxy_ft1_positive_fullmatrix_same_speaker_m10_32scenes_2026-10-11.json`

Results:

- 32 scenes, 448 seconds;
- target wake recall: 78.125%;
- final intent accuracy: 68.75%;
- intent accuracy conditioned on wake: 88%;
- false wake events: 0 observed/hour;
- false action events: 0 observed/hour;
- mean target wake latency: 2.1 s;
- p95 target wake latency: 2.5 s;
- `production_valid=false` by design because this is still a diagnostic proxy.

### Selected 8-scene safety proxy

Artifact:

`logs/continuous_full_pipeline_proxy_hardneg_aishell5_neg_splitgate_p015_wake024_cmd022_same_speaker_m10_2026-10-11.json`

Results:

- 8 scenes, 112 seconds;
- 100% target wake recall;
- 100% final intent accuracy;
- 0 observed false wake events/hour;
- 0 observed false action events/hour;
- mean/p95 wake latency: 2.5 s.

This is a short selected diagnostic and must not be reported as production performance.

### ReMASC vehicle-stress proxy

Artifact:

`logs/continuous_full_pipeline_proxy_remasc_personal_activity_margin_m067_fast_stream_hop025_8scenes_2026-10-12.json`

Results:

- 32 scenes, 448 seconds;
- 96.875% target wake recall;
- 81.25% final intent accuracy;
- 64.3 diagnostic false wakes/hour;
- 0 observed false actions/hour;
- mean latency: 0.968 s;
- p95 latency: 1.625 s.

The fast stream was rejected as a production default because the comparable AISHELL target-absent run rose to about 6 false events/hour versus about 3/hour for the 1-second reference. Keep it as a latency diagnostic, not the promoted default.

### Long target-absent AISHELL-5

Artifact:

`logs/aishell5_continuous_targetkws_hardneg_aishell5_ft1_p015_sg024_kws050_smooth22_eval2_all18_dedup_2026-10-11.json`

Results:

- AISHELL-5 Eval2;
- 18 sessions;
- 14,001.9 seconds / 3.889 hours;
- all recordings target-absent relative to the fixed enrollment;
- full-path smoothed events: 18;
- full-path event rate: 4.628/hour;
- KWS-only smoothed event rate: 96.673/hour.

This is a negative-only safety result. It says nothing about target-driver wake recall.

## 7. Current latency/resource evidence

Current full forward benchmark:

- FP16 CUDA wake stage: mean 48.55 ms, p50 48.21 ms, p95 51.56 ms;
- FP16 CUDA post-wake command stage: mean 62.34 ms, p50 62.57 ms, p95 64.44 ms;
- FP32 CUDA full-path benchmark: wake mean 100.81 ms, command mean 115.22 ms;
- CPU full-path benchmark: wake mean 522.22 ms, command mean 641.67 ms;
- model parameters: 106,027,183;
- checkpoint bytes: 424,488,203 (~424.5 MB);
- current 3-second context is non-causal.

Artifacts:

- `logs/full_pipeline_latency_gpu_2026-10-11.json`
- `logs/full_pipeline_latency_2026-10-11.json`
- `logs/full_pipeline_latency_cpu_2026-10-11.json`

Do not claim sub-100-ms end-to-end streaming latency. The numbers are model-forward measurements and omit microphone I/O, buffering, resampling, overlap/look-ahead, scheduling, and vehicle action execution.

## 8. Dataset/data situation

Locally available/used:

- LibriSpeech / Libri2Mix for clean speaker mixtures and enrollment;
- CAVEMOVE multichannel cabin mixtures/noise;
- AISHELL-5 real four-channel far-field/near-field speech/noise;
- Google Speech Commands v2 for `Marvin`/non-wake KWS;
- Fluent Speech Commands for 31-class intent;
- Vehicle Interior Sound material for vehicle/cabin noise;
- ANIR V-class impulse-response/noise resources;
- ReMASC sample stress resource only.

Approximate local raw storage from the base audit:

- `data/raw`: 298.10 GiB;
- AISHELL-5: 247.41 GiB;
- LibriSpeech: 27.10 GiB;
- CAVEMOVE: 8.61 GiB;
- Speech Commands v2: 5.37 GiB;
- Fluent Speech Commands: 5.00 GiB;
- Vehicle Interior Sound: 4.45 GiB.

Additional external resources reviewed but not all downloaded:

- ReMASC full corpus is roughly 92.3 GB and requires a free IEEE DataPort account;
- MagicHub/commercial in-vehicle datasets have licensing/access constraints;
- no permitted public English corpus was found containing the full same-driver continuous sequence needed for final evaluation.

Relevant documentation:

- `reports/automotive_dataset_sourcing_review_2026-10-11.md`
- `reports/preexisting_systems_research_update_2026-10-11.md`
- `data/sources.json`

## 9. Real-driver readiness audit

Current audit:

- `reports/production_readiness_snapshot_ratefix_ci.md`
- machine-readable: `logs/production_readiness_snapshot_ratefix_ci.json`

Current status: `research_engineering_preview`, `production_ready=false`.

Blocked gates:

- same-driver continuity;
- passenger rejection;
- minimum study size;
- capture hardware match;
- frozen split;
- threshold freeze;
- reproducible release state.

Measured/evidence-present gates:

- no-command safety diagnostics;
- latency;
- resource budget;
- regression artifacts.

The real-driver session directory is currently empty:

`data/raw/custom_incar/sessions`

Hardware preflight previously found only a 1/2-channel host input, not the required matched 4-channel capture setup. Do not fabricate or relabel synthetic data as real-driver evidence.

## 10. Research and design conclusions

The pre-existing-systems review found that comparable robust systems generally combine:

- compact streaming keyword detection;
- target-speaker activity/verification;
- temporal state/action vetoes;
- multichannel/spatial cues where available;
- careful false-accept-per-hour evaluation.

The current project already has partial equivalents, but not a real-driver benchmark. The system-level cascade may be a useful engineering contribution, but current evidence does not establish component-level SOTA or scientific novelty.

Research references and links are documented in:

- `reports/preexisting_systems_research_update_2026-10-11.md`
- `reports/research_journey.md`

## 11. Known rejected experiments

Keep these for reproducibility; do not promote without a new benchmark:

- fixed RMS conditioning as runtime default: improves AISHELL SI-SDRi but increases negative events;
- Stage 1 hard-negative fine-tune: first validation regressed SI-SDRi;
- fast 250-ms stream: improves latency but worsens long-negative false events;
- framewise/target-top activity-margin variants: recall/safety tradeoffs were not better overall;
- same-domain/enrollment-conditioned/hard-negative KWS variants: isolated gains did not translate reliably to the continuous proxy;
- 2-of-2 smoothing variant: increased false actions in one proxy;
- compact CRNN/distilled intent candidates: materially lower intent accuracy and not promoted;
- compact speaker-verifier candidates: latency/safety tradeoff not yet sufficient for default promotion.

## 12. Important resources outside the active repository

The active Git repository is nested at `D:/Prawin_New/tse/car-tse-main`, but several relevant resources are stored elsewhere. Another agent should know about these before declaring that the workspace has been fully audited.

### `D:/datasets` — canonical external data/model store

- `D:/datasets/anir/`
  - ANIR V-class impulse responses and vehicle-noise material;
  - extracted multichannel noise plus `anir_v_class_noise_manifest.json`;
  - source archives include approximately 6.8 GB of V-class noise and a 14.95 MB impulse-response archive.
- `D:/datasets/remasc/`
  - `sample_v2.zip` is only a small ReMASC sample (~10.9 MB);
  - `stress_bed/ReMASC/noise/remasc_vehicle_replayed_6ch_tiled.wav` is the locally generated 6-channel, tiled vehicle-noise stress bed (~23 MB);
  - the full ReMASC corpus is not local.
- `D:/datasets/smart_speaker_command_26417548/`
  - University of Rochester Smart Speaker Command Dataset, with approximately 2,110 WAV files across seven speakers;
  - each speaker has roughly 300 recordings of `Hey Alexa` followed by a command, including a solo/mixture recording structure;
  - source audio is 48 kHz, 16-bit, mono;
  - the command set contains smart-speaker tasks such as lights, timers, music, reminders, and calls;
  - documentation PDFs are under `SmartSpeakerCommandDataset_RutowskiEtAl_2023/`;
  - it is already integrated into the repository’s diagnostic path via `scripts/eval/eval_smart_speaker_command_negative.py` and `scripts/data/build_target_kws_smart_speaker_negative_cache.py`;
  - on the evaluated 250-file Speaker1 subset, the current candidate reduced full-path false wakes from 728.09/hour to 17.90/hour and the 14-file action sample had zero false actions; this is clean single-channel negative evidence only, not a production result;
  - a full all-speaker evaluation was started as `logs/smart_speaker_command_negative_all_2026-10-12.json` and must be completed before using an all-speaker result;
  - this may help study same-speaker wake-to-command structure and speaker consistency, but its wake phrase is `Hey Alexa`, not `Marvin`; verify registration/licensing/terms before any new training or redistribution.
- `D:/datasets/precise_community_data/`
  - a Git checkout of Mycroft Precise Community Data;
  - includes a pre-trained `Marvin` Precise model at `marvin/models/marvin-en-0.3.0-20190817-jarbas.tar.gz`;
  - it does **not** contain a local raw Marvin WAV corpus under that directory; the model README says its wake-word data came from Google Speech Commands and its non-wake data included a private dataset;
  - includes English not-wake clips, noise archives, and license files;
  - useful as an external wake-word baseline/reference, not a drop-in replacement for the current PyTorch TSE-domain KWS, and its reported accuracy is not directly comparable to this project’s false-accept/hour evaluation.
- `D:/datasets/pretrained_models/`
  - Hugging Face cache for `microsoft/wavlm-base-plus-sv`;
  - the local `pytorch_model.bin` is about 404.5 MB;
  - this was used for compact speaker-verifier experiments and is outside the Git repository.
- `D:/datasets/precise_community_data/` and the Smart Speaker dataset are not automatically part of the repository’s reproducible checkout. Any future use should record provenance, license/terms, hashes, and the exact manifest in `data/sources.json`.

### `E:/prawin/snn` — separate separator/SNN research tree

This is a separate research project, not wired into `car-tse-main`:

- `E:/prawin/snn/baseline/` contains clean Libri2Mix Conv-TasNet controls, an enhanced separator, training/evaluation scripts, and `SPEC_V1.md`/`SPEC_V2.md`;
- `E:/prawin/snn/data/` contains Libri2Mix-derived data;
- `E:/prawin/snn/runs/` contains trained and probe runs, including Conv-TasNet enhanced variants and SpEx+/JEPA probes;
- `E:/prawin/snn/replications/` contains speaker-extraction replication work;
- `E:/prawin/snn/third_party/LibriMix/` contains related metadata/code;
- it may contain useful TSE/separator ideas and checkpoints, but no checkpoint should be substituted into the car pipeline without matching input/output conventions and a dual-domain regression.

### Other adjacent directories

- `D:/Prawin_New/snn_replication_noise/` contains WHAM noise archives and a small test bundle; it is an optional noise source, not part of the current car-pipeline data manifests.
- `D:/Prawin_New/jepa-models/` and `D:/Prawin_New/ssd-jepa/` are separate JEPA/vision experiments and are not current in-car TSE assets.
- `D:/Prawin_New/prawin/` contains unrelated personal projects and archives; do not scan, modify, or treat it as part of this repository.
- `D:/Prawin_New/tse/` itself also contains handoff/report artifacts outside the Git repo: `output/pdf/`, `latex_guides/`, `tmp/`, `build_guides.py`, `car-tse-models-full-2026-10-04.zip`, and `car-tse-good-samples-2026-10-05.zip`.

### Live-state clarification

The earlier readiness snapshot recorded 179 changed worktree entries. A fresh read-only check after adding the handoff material reports **182 changed entries**; the repository remains dirty by design. The count of **78 checkpoint files** (`*.pt`/`*.pth`) is confirmed. Do not interpret the changed-file count as a clean release state.

## 13. Latest evaluator-quality improvements

Recent code/documentation changes include:

- real-driver safety rates now use the union of labelled negative intervals when available;
- historical total-recording rates remain for compatibility/diagnostics;
- Wilson confidence intervals for wake recall/command accuracy;
- Poisson/Garwood intervals for false-accept and false-action rates;
- readiness audit includes rate-denominator and CI evidence;
- focused tests cover the negative-exposure denominator behavior;
- current full suite: **36 passed, 1 skipped**.

## 14. Safest next step

Do not start another blind clean-intent or Stage 1 training run first.

Recommended order:

1. Prepare/validate the pilot recording workflow with the required 4-channel microphone setup.
2. Record or obtain consented sessions with the same driver providing enrollment → `Marvin` → command.
3. Add passenger `Marvin`, passenger non-wake speech, silence/no-command periods, and vehicle-noise conditions.
4. Keep tuning and held-out test drivers/sessions separate.
5. Freeze thresholds on tuning data only.
6. Run the complete pipeline and component ablations on the held-out sessions.
7. Report driver wake recall, command accuracy conditioned on correct wake, false wakes/hour, false actions/hour, p50/p95 response latency, and condition breakdowns.
8. Only then decide whether joint retraining of Stage 1/presence/target-KWS is justified.

If real recording access is unavailable, continue with dataset audit, synthetic stress expansion, evaluator/reproducibility work, and architecture/resource experiments, but label every result as synthetic or cross-corpus.

## 15. Operational rules for the next agent

- Preserve the dirty worktree and existing user changes; do not reset, clean, or delete broadly.
- Use `apply_patch` for edits.
- Do not overwrite the established checkpoints.
- Give every new training/evaluation output a unique checkpoint/log/report name.
- Run the focused test and then the full test suite after code changes.
- Require a dual-domain Stage 1 regression before promoting a TSE checkpoint.
- Require long target-absent and positive proxy checks before changing thresholds.
- Do not call a proxy “real-driver,” “production,” or “SOTA.”
- Do not treat a zero event count from a very short recording as a reliable FAR estimate.
- Keep the active 15-minute heartbeat running.
- Do not commit or push unless the user explicitly requests it in the current interaction.
- If asked to generate/update a PDF, first read and follow the PDF skill instructions; the latest existing PDF is `output/pdf/in_car_voice_assistant_ablation_report_2026-10-11.pdf`.

## 16. One-paragraph handoff

The repository contains a four-stage in-car assistant: multichannel SpEx+-style target-speaker extraction, framewise target activity, target-speaker `Marvin` KWS, and wav2vec FSC intent classification. Since the 2026-10-04 base benchmark, dual-domain Stage 1 fine-tuning improved matched AISHELL SI-SDRi from 2.39 to 6.24 dB while preserving CAVEMOVE, and the expanded cross-corpus continuous proxy improved from 25% wake/0% final intent/64.3 false actions per hour to 78.1% wake/68.8% intent/0 observed false actions per hour. Long target-absent testing is now 3.889 hours over 18 AISHELL-5 sessions, with 4.63 full-path events/hour and 96.67 KWS-only events/hour. Clean intent remains about 99.3%, the model remains about 424 MB, and tests are 36 passed/1 skipped. These are meaningful research-engineering gains, but there are still zero real-driver sessions, no proven passenger rejection, no frozen held-out real-cabin split, and no production sign-off. The next highest-value step is a small consented 4-channel real-driver pilot and frozen threshold/ablation evaluation; do not present synthetic proxies as real-world reliability.
