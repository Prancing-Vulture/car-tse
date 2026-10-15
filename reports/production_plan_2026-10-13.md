# Production plan: in-car target-speaker voice assistant

Date: 2026-10-13
Status of the system today: `research_engineering_preview`, `production_ready=false`.
Current product scope: **English only for the next development cycle**. Mandarin
AISHELL-5 remains an auxiliary acoustic/negative diagnostic, not a product
training or headline-accuracy domain. This document started as a plan, but the
workspace now also contains the first English spatial-proxy training and
evaluation results described in §2.6 and the research journey. It builds on
`agent_handoff_context_2026-10-12.md`, the session-2 handoff,
`production_readiness_gates.md` and `real_driver_pilot_protocol_2026-10-09.md`,
and adds external research (sources at the end).

---

## 0. Bottom line

The current system is a good research cascade. Production-grade means more than
better numbers. It needs changes on five fronts:

1. **Product definition.** "Marvin" (from GSC) and the 31 FSC intents are research
   proxies. Production needs a real wake phrase and an intent/slot ontology.
   The immediate prototype scope is English only: AISHELL-5 is retained for
   target-absent cabin/noise diagnostics, but is excluded from English product
   accuracy claims and English wake/intent training.
2. **Architecture for a car.** In a car the driver's seat position is fixed and
   known. A spatial (seat-zone) cue is cheaper and more robust than a voiceprint,
   and it works without enrollment. The current design is TSE-first: it runs
   SpEx+ on a 3 s window every hop, which measured about 423 ms for the Stage-1
   model-forward component on this CPU benchmark (not complete end-to-end
   product latency) and suppresses wakes from a different speaker. This plan replaces it with a cheap always-on
   detector followed by zone, speaker and TSE stages that run only on candidates.
   It also adds echo cancellation (AEC) for infotainment playback, which is
   untested today.
3. **Evidence at production scale.** Today there are 0 real drivers. The release
   gates require ≥3 drivers. That threshold is a pilot minimum, not a product
   claim. A credible release needs dozens of speakers, several vehicles and a
   condition matrix sized so the confidence bounds clear the gates (section 4).
4. **Embedded engineering.** The target is a C++/ONNX runtime with int8 models on
   a named automotive SoC, not Python plus PyTorch on an RTX 4080. It also needs
   deterministic builds, a model registry and monitoring after release.
5. **Safety, security and privacy process.** This means SOTIF-style hazard
   analysis of unintended actions, risk tiers per action, ASPICE MLE
   traceability, the ISO/SAE 21434 cybersecurity standard for adversarial and
   replay audio, and GDPR Art. 9 handling of voiceprints.

At a realistic pace, with a small team (2–3 ML, 1–2 embedded/audio, 1
data-operations person, part-time safety and legal), this is roughly **6–9
months** from now to a limited fleet release. The critical path is data
collection, not modelling.

---

## 1. Phase 0: Requirements and reproducibility (weeks 0–3)

Nothing downstream is meaningful until these are frozen.

### 1.1 Product requirements (write down and get sign-off)

| Item | Decision needed | Recommendation |
|---|---|---|
| Wake phrase | Brand phrase, 3+ syllables, phonetically distinct | Replace "Marvin". Two-word phrases with 4–6 syllables have far fewer confusers. |
| Languages and locales | Which language ships first | **English first for the current prototype.** Keep AISHELL-5 only as an auxiliary acoustic/negative diagnostic; do not mix its Mandarin speech into English headline metrics. |
| Intent ontology | Intent list, slots, out-of-scope handling | 30–150 intents with slots (temperature value, seat, contact name…), plus an explicit "out of scope" class. |
| Action risk tiers | Which actions can run without confirmation | Tier A (low risk: media, HVAC ±): execute. Tier B (calls, messages, navigation changes): confirm. Tier C (windows while driving, anything safety-related): confirm or disallow. |
| Who may command | Driver only, or any occupant scoped to their own zone | Driver-only for vehicle controls. Optionally zone-scoped for comfort controls (a passenger adjusts their own seat or HVAC). Zone awareness covers both. |
| Enrollment | Required or optional | Optional. The system must work without enrollment using the zone, and enrollment should only improve it. This is the Personal VAD 2.0 requirement of working with and without enrollment [6]. |
| Release gates | Numbers frozen before any held-out data is opened | See §1.2. |

### 1.2 Proposed release gates (tighten the current ones)

The current gates (≥90% wake recall, ≥90% command accuracy, ≤1.0 false accepts
per hour, ≤1.0 false actions per hour) are reasonable for a pilot and too loose
for a product. Consumer wake-word products are commonly quoted at under about
0.1 false accepts per hour with a low single-digit false-reject rate [7]. That
figure comes from a vendor and should be checked against your OEM's
requirements. The standard way to report this is FRR at a fixed FA/h, per noise
condition [7][8].

| Gate | Pilot (exists) | Production proposal |
|---|---|---|
| Driver wake recall | ≥90% | ≥95% overall, ≥90% in the worst condition stratum, judged on the 95% Wilson lower bound |
| Command accuracy after wake (intent + slots) | ≥90% | ≥93% overall, ≥88% worst stratum |
| False accepts, all non-driver audio | ≤1.0/h | ≤0.1/h on the 95% upper bound, measured with music, nav and TTS playback present |
| Passenger wake acceptance (driver-only actions) | not a gate | ≤2% (95% upper bound) |
| False actions | ≤1.0/h | ≤0.05/h for Tier A. 0 observed for Tier B/C plus confirmation. |
| Wake latency (end of phrase → wake event) | partial | p95 ≤ 300 ms on target hardware |
| Wake-to-action latency | partial | p95 ≤ 1.0 s after end of command |
| Resources | partial | Always-on path within an agreed DSP/CPU core budget and memory budget on the target SoC (set after hardware choice) |
| Robustness | none | One-microphone failure, clipping, 0–130 km/h, windows open, rain, HVAC max, music at 85 dBA |

### 1.3 Reproducibility (cheap, start immediately)

The local copies in `~/Downloads/Prawin` are not git repositories and contain no
checkpoints or venv. The canonical state lives on the Windows host.

- Put code under git with one canonical remote. Tag every evaluated state.
- Version the data: DVC or a manifest of content hashes for every corpus, split
  and synthetic bed. `data/sources.json` is a start.
- Keep a model registry with checkpoint hash, config hash, git SHA, training-data
  manifest hash and evaluation-report link. This closes the existing
  "reproducible release state" gate.
- CI runs unit tests, a small golden-audio regression (bit-exact or tolerance)
  and an export check (ONNX parity) on every merge.
- Pin the environment (lockfile or container). `torch 2.14.1+cu132` and
  `transformers 5.19` must be recoverable.

**Exit criteria:** requirements doc signed, gates frozen, any released number
reproducible from a tag.

### 1.3.1 Evaluated-state manifest (2026-10-14)

The first standalone release manifest is now generated by
`scripts/reporting/build_release_manifest.py`. For the retained runtime it
hashes the runtime and Stage-1 configs, intent label map, test manifest, six
runtime checkpoints/encoders, and records Git/environment identity. All 12
listed artifacts are present and hashed. The manifest deliberately reports
`worktree_clean=false`; it is reproducibility evidence, not a release sign-off.

The retained YAML was also corrected so its `test` split and
`meta/intent_label_map.json` reference resolve to real files. This fixes stale
configuration references without changing model weights or evaluation
thresholds.

---

## 2. Phase 1: Data programme (weeks 1–16, runs in parallel; the critical path)

### 2.1 Capture hardware

- Buy a multichannel interface with **≥6 inputs**. Four go to cabin
  microphones at the production positions and geometry. Two carry the
  infotainment loudspeaker reference signal (stereo loopback or line-out tap),
  which AEC needs. The current protocol lists playback as optional. It has to be
  mandatory.
- Wherever possible, use the production microphone part, mounting and
  preamp/ADC chain. Microphone frequency response and positioning shift the
  acoustics more than most model changes do.
- Add close-talk lavalier microphones on driver and passenger. They are not
  model inputs. They give ground-truth speaker activity, so labelling becomes
  semi-automatic. AISHELL-5 recorded near-field headsets alongside its door
  microphones for the same reason [3].
- Record vehicle CAN or OBD speed, plus a window/HVAC log if available. These
  give free condition labels.

### 2.2 Study design (scale beyond the 3-driver pilot)

Run it in two waves:

- **Wave 1 (pilot, 2–4 weeks):** 8–10 speakers, 1 vehicle, the current
  protocol plus playback. Purpose: validate the tooling and labelling and find
  failure modes. Not a release dataset.
- **Wave 2 (release, 8–12 weeks):**
  - **≥80 speakers**, split by speaker and session into roughly 40 for training
    and tuning and roughly 40 held out. Balance gender, age band and accent. The
    voices of people over 60 and of children (as passengers) matter.
  - **≥3 vehicle types**, for example a compact car, an SUV and an EV. Cabin
    acoustics and noise spectra differ a lot.
  - Condition matrix: speed (parked, about 50 km/h, about 110 km/h) × HVAC
    (off, mid, max) × windows (closed, one open) × playback (none, music, nav
    prompt, phone call) × occupancy (driver alone, plus front passenger, plus a
    rear passenger).
  - Scripted and **unscripted** segments, the latter being free conversation,
    phone calls and singing along to music. Unscripted audio is where false
    accepts come from.

### 2.3 Sample-size arithmetic (why these numbers)

- **Zero-event rate bound (rule of three):** with 0 events in T hours, the 95%
  upper bound is about 3/T per hour. Showing ≤0.1/h therefore needs **≥30 h of
  held-out negative exposure with zero events**, and in practice 50–100 h so a
  few events can be tolerated. The current 3.889 h of AISHELL Eval2 gives a
  rule-of-three 95% upper bound of approximately **0.77/h** when zero events
  are observed; older 0.95/h references are stale or conservative and should
  not be mixed with this denominator.
- **Proportions:** to show passenger acceptance ≤2% with 0 failures, you need
  ≥150 passenger wakes, because 3/150 = 2%. For recall, with 95% observed on
  about 200 trials per stratum, the Wilson lower bound is about 92%. So budget
  about 200 driver wakes per major condition stratum on the held-out side.
- **Speaker clustering:** errors cluster by speaker. Report bootstrap confidence
  intervals resampled by speaker, not by utterance. Three drivers cannot support
  any generalisation claim.

### 2.4 Labelling and governance

- Keep the existing half-open-interval JSONL schema. Pre-label speaker activity
  from the close-talk microphones and run an ASR pre-pass for transcripts, then
  have a human verify. Double-annotate 10% and report agreement.
- Consent: voiceprints used to identify someone count as special-category
  biometric data under GDPR Art. 9, which needs explicit consent [10]. Ordinary
  recordings are personal data. Get written consent covering training use,
  retention and withdrawal, keep the existing `consent.permission_to_use`
  contract, and do a DPIA. Get legal review. The sources found here are
  secondary, not primary legal text.
- Freeze the held-out set behind access control. Only the evaluation pipeline
  reads it.

### 2.5 Supplementary data (useful but never release evidence)

- **AISHELL-5:** 100+ h of in-car 4-channel audio plus 40 h of real cabin noise
  [3]. Keep it for simulation and pretraining. Eval2 is exhausted as a safety
  set.
- **TTS-synthesised wake phrases with cabin impulse responses and noise:** good
  for bootstrapping a new wake phrase. Practitioners consistently report that
  synthetic data alone leaves residual false accepts, and that real recorded
  negatives (quiet rooms, TV, music) remove most of them. Use roughly a 1:10
  positive:negative ratio with phonetic near-miss confusers [9].
- Cabin impulse responses measured in each target vehicle with a sweep, so
  simulation matches the real geometry.

**Exit criteria:** Wave 2 collected, labelled, validated by
`validate_real_driver_sessions.py`, and frozen. The held-out split is never
touched during development.

### 2.6 English-only progress and current interpretation (2026-10-13)

The first English spatial experiment is now in the repository. It uses English
Speech Commands and Fluent Speech Commands rendered through CAVEMOVE's
driver/passenger impulse responses and real vehicle-noise files. A
speaker-disjoint LibriSpeech/CAVEMOVE Stage-1 adaptation was trained on 384
examples and selected at epoch 1 from a 96-example validation split (best
validation SI-SDRi 3.54 dB). It is **not promoted**: the downstream proxy did
not improve over the current Stage-1 checkpoint.

The spatial ablation evaluated 24 scenes (8 scenes × three cabin SNRs; 5.6
minutes total) with the full stateful pipeline. With the current strict gate,
the Stage-1/TSE-first path recalled 12.5% of wake events and produced 0% intent
accuracy and 0 false actions/hour in this tiny proxy. A classical hybrid path
recalled 4.2%, and beamforming-only recalled 0%. These numbers are not real-car
metrics. More importantly, the proxy's wake/enrollment comes from Speech
Commands while the command comes from an unrelated Fluent Speech Commands
speaker, so the command-side speaker gate is expected to reject it. The
benchmark now records this explicitly as `same_driver_wake_to_command=false`.

The result is therefore a data-contract finding, not evidence that the proposed
zone architecture is intrinsically worse: a genuine same-driver English
recording is still required. Do not tune release thresholds or promote the
English Stage-1 checkpoint from this cross-corpus proxy.

### 2.7 Clean-English preservation transfer (2026-10-13)

The clean English path was then isolated before further noisy work. The
corrected evaluator explicitly loaded the fine-tuned Wav2Vec intent head; the
earlier clean near-zero intent result was caused by a CRNN/Wav2Vec harness
mismatch. A speaker-disjoint FSC identity manifest was used to adapt Stage 1
without replacing the retained noisy checkpoint. On a 32-speaker, 64-scene /
896-second clean proxy, the clean candidate reached 96.9% wake recall and
93.8% overall intent at KWS threshold 0.25; at threshold 0.40 it reached
93.8% wake recall and 90.6% overall intent, with 96.7% intent conditional on
wake. Proxy false wakes were 16.1/h and 8.0/h respectively, but the exposure
was under 0.25 hours and had no same-driver wake-to-command continuity.

The clean Stage-1 candidate fixed the prior TSE-to-intent handoff failure, but
its bounded CAVEMOVE SI-SDR regression was −2.80 dB relative to the retained
dual-domain checkpoint. A small joint clean-plus-CAVEMOVE fine-tune preserved
the clean result but did not improve the noisy proxy. The clean checkpoint is
therefore a measured clean-path branch, not a universal production promotion.
Details and exact artifacts are in
`reports/clean_english_stage1_transfer_2026-10-13.md`.

The clean component boundary was also isolated: direct TSE-to-intent was
100% on an 8-scene silent proxy, while the presence-gated path was lower. A
clean-English presence-head fine-tune recovered the clean proxy but produced
257.1 false wakes/hour on the CAVEMOVE regression, so it was rejected. This
reinforces the plan's rule that clean-domain improvements must survive a
noisy-domain regression before promotion.

### 2.8 Capture-hardware preflight (2026-10-14)

The recording host was checked before collecting any real-driver data. The
read-only preflight found six visible input devices, but none supported the
required four-channel, 16 kHz contract; the available devices exposed at most
two channels. No audio was recorded and no session was created. This confirms
that the current machine cannot satisfy the pilot capture requirement. The
preflight artifact is `logs/capture_hardware_preflight_2026-10-14.json` and
the reusable checker is `scripts/data/preflight_capture_hardware.py`.

This does not change the production decision. A four-channel cabin array is
the minimum needed for the current evaluator, while the proposed production
interface remains six inputs (four cabin microphones plus two playback
reference channels for AEC). Hardware procurement or attachment is therefore
the next critical-path action before a real-driver pilot can begin.

The recorder now makes this distinction explicit: `model4_pilot` preserves the
current four-channel research contract, while `production6` refuses to record
unless at least six channels and a declared playback reference are supplied.
The production6 preflight was also run explicitly: all six visible devices
failed the six-channel/16 kHz check (maximum two input channels), so no audio
or session directory was created. Artifact:
`logs/capture_hardware_preflight_production6_2026-10-14.json`.

The production capture metadata now has to declare the channel roles explicitly
(`microphone_channel_indices` and `playback_reference_channel_indices`). The
real-session evaluator and live streaming runner preserve that mapping: four
mic channels feed the model, while the playback-reference channels are averaged
and passed sample-aligned to the learned echo-cancellation front-end. A wide
capture without this mapping is rejected. This closes the software-side AEC
plumbing gap; it does not replace the requirement for a real six-input device
or real-driver sessions.

### 2.9 Endpoint buffering experiment (2026-10-14)

The clean endpoint diagnosis exposed a state-machine failure mode: committing
the first eligible post-wake frame can select a partial command from an
overlapping window. An opt-in confidence-buffered decoder now waits for one
additional candidate when the first intent confidence is below 0.90. A
dual-crop diagnostic also compares fixed and KWS-trimmed windows.

On the 32-scene clean cross-corpus proxy, intent improved from 87.5% with the
latest-frame endpoint to 96.875% (31/32), with 100% wake recall and no observed
false events. The improvement did not transfer to the small noisy regression:
the clean Stage-1 branch measured 75% wake / 62.5% intent / 96.4 false
wakes/hour, while the retained dual-domain branch measured 50% wake / 0%
intent / 0 false wakes on 16 scenes. Therefore the buffered paths remain
diagnostic and are not part of the retained production candidate. The plan's
rule remains: endpoint policy must be evaluated on the frozen same-driver
tuning/test split and at matched false-accept exposure before promotion.

---

## 3. Phase 2: Architecture rework (weeks 2–12, starts on synthetic data and moves to Wave 1)

### 3.1 Target architecture

```
4 mics + playback ref
   │
   ▼
[A] Front end (always on, causal, ≤20 ms algorithmic latency)
    linear AEC per mic → neural residual-echo suppressor
    → zone separation/beamforming (driver, front passenger, rear)
   │  driver-zone stream (+ other zones for zone-scoped commands)
   ▼
[B] Always-on speaker-independent wake detector (≤250 k params, int8, streaming)
   │  candidate (low threshold, high recall)
   ▼
[C] Verification cascade, runs only on candidates (~1–2 s of audio):
    zone/DOA confidence  +  second-pass wake model (larger)
    + speaker verifier vs enrollment (optional)  → calibrated fusion score
   │  accepted wake
   ▼
[D] Command capture: endpointing (driver-zone personal VAD)
   ▼
[E] Streaming ASR + NLU (or distilled E2E SLU)  → intent + slots + confidence
   ▼
[F] Action policy: risk tier, confirmation, cooldown, safety interlocks
```

How this differs from today, and why:

| Today | Proposed | Reason |
|---|---|---|
| SpEx+ TSE on every 0.25–1 s hop (3 s window) | Causal zone front end always on. TSE runs only on candidates, or is dropped if zone separation suffices | 423 ms CPU per hop is the bottleneck. A causal, chunked TSE/separator is feasible: published causal TSE reaches about 32 ms algorithmic latency [1] and multichannel directional extraction about 2 ms [2]. Zoneformer combines zone separation, enhancement and AEC on-device for cars [5]. |
| Identity = enrollment voiceprint only | Zone cue (geometry) + voiceprint fusion | The passenger Marvin leak at 2.9 dB is exactly the case where a spatial cue is strongest. Zone works with no enrollment. |
| No AEC | Linear AEC plus a neural residual suppressor using the reference channel | Music, nav prompts and the assistant's own TTS are the most common real-car false-trigger and barge-in sources. Cross-zone echo coupling is a known in-car problem [5][11]. |
| wav2vec2-base, 31 FSC intents, fixed 2 s window | Streaming ASR + NLU, or a distilled E2E SLU | wav2vec2 is 89% of the 425 MB. Production needs slots, variable-length commands and an extensible ontology. Distillation plus quantization can reach about 99% FSC accuracy at 83–700× smaller size [12]. Streaming E2E SLU is also established [13]. |
| 2-of-2 / 8-of-8 frame confirmation heuristic | Calibrated second-pass verifier on the candidate segment | It replaces hop-dependent smoothing rules, which have been fragile (2-of-2 unsafe at 0.25 s hop, 8-of-8 safe), with a single calibrated score. |

### 3.2 Decision experiment before committing (weeks 2–5)

Use AISHELL-5 4-channel audio plus the existing synthetic beds, then Wave 1,
to compare three wake-path variants at matched FA/h:

1. **Current:** TSE-first (reference).
2. **Zone-first:** beamformer/zone separator → SI wake → verifier. No TSE.
3. **Hybrid:** zone-first plus TSE on candidates only.

Report FRR at fixed FA/h (0.1, 0.5, 1.0), passenger acceptance, CPU cost and
cross-speaker behaviour. Choose using data, not by assumption. My expectation is
that (2) or (3) wins on cost and passenger rejection, but this must be measured.

### 3.3 Component plans

- **Front end:** start with a classical baseline (MVDR or delay-and-sum per
  zone, plus WebRTC-style linear AEC). It is cheap, explainable and a strong
  baseline. Then train a causal neural zone separator or residual suppressor in
  the Zoneformer style [5], with quantization-aware training. Measure stability
  when one microphone fails.
- **Wake detector:** retrain for the product phrase on TTS bootstrap data plus
  Wave 1 real data. Mine hard negatives continuously from unscripted audio,
  music with lyrics and talk radio. Include passenger wakes as negatives for the
  driver-only head; this closes next step 5 in the handoff.
- **Speaker verifier:** keep the 176 k compact verifier as the fast path.
  Consider the asymmetric design, a large model at enrollment and a small one at
  runtime, which recovers a meaningful EER gap at no runtime cost [14]. Use
  multi-utterance in-car enrollment that is passive and gradual, updated from
  confident accepted commands with drift guards. Calibrate scores per condition
  so the fused threshold transfers.
- **Personal VAD / endpointing:** a speaker- and zone-conditioned VAD (130 k
  parameters in the original paper [6]) gives robust command endpointing and
  removes the fixed 2 s intent window.
- **Intent:** if the ontology is small and closed, use a distilled E2E SLU
  student (QUADS-style joint quantization and distillation [12]). If it is
  large, open or slot-heavy, use a small streaming ASR plus a grammar or small
  NLU. That is more maintainable and localisable. Always include an
  out-of-scope class and confidence calibration (ECE is already measured, which
  is good).
- **Action policy (new, needed for safety):** per-tier confirmation, a cooldown
  after a false-wake cancel, a vehicle-state interlock (no window or sunroof
  commands above X km/h unless confirmed), and an undo for Tier A actions. This
  turns residual model errors into recoverable UX events rather than hazards.

**Exit criteria:** architecture chosen by experiment, and every model meets its
standalone gate on Wave 1 tuning data.

---

## 4. Phase 3: Training, tuning and freeze on real data (weeks 12–20)

1. Train or fine-tune every component on Wave 2 training speakers plus
   simulation. Keep established checkpoints, use unique names, and keep the
   dual-domain regression before any promotion. These carry over from the
   operational rules.
2. Tune thresholds jointly on tuning speakers only, under both constraints (FA/h
   and false actions/h). The evaluator already supports this, using the
   negative-exposure denominator.
3. **Freeze** thresholds, configs and checkpoints and record the hashes in the
   registry.
4. Run the held-out evaluation **once**. Report every gate with confidence
   intervals, per condition stratum, per vehicle and per demographic group,
   with speaker-bootstrap intervals and raw event logs.
5. If a gate fails, do error analysis on **tuning** data only, iterate, and pay
   for a fresh held-out tranche, or at minimum report how many held-out reuses
   occurred. Never quietly retune on the test set.

**Exit criteria:** every §1.2 gate passes on held-out data, plus a demographic
fairness check (no group's FRR worse than 1.5× overall).

---

## 5. Phase 4: Embedded port (weeks 8–24, overlaps)

1. **Choose the target SoC early** (week 2–3). Every latency and budget number
   depends on it. Typical cockpit SoCs have a DSP or NPU plus ARM cores.
2. Export path: PyTorch → ONNX → vendor toolchain, for example QNN/QAIRT with
   AIMET quantization on Qualcomm, or the equivalent for other vendors [15][16].
   Known pitfalls:
   - Operators the NPU cannot run fall back to CPU, so audit operator coverage
     per model.
   - Float input and output on a quantized NPU model can cost more than the
     compute itself. One benchmark measured 30.6 ms with float I/O against
     12.1 ms with quantized I/O [16]. Keep audio-feature I/O quantized end to end.
   - NPU first-call compile and load is slow, so precompile context binaries
     and warm up at boot [16].
   The workstation now has a repeatable ONNX preflight at
   `scripts/export/export_onnx_components.py`. All six retained components
   export and pass ONNX Runtime CPU parity at fixed prototype shapes. The
   shared log-mel frontend required an export-only real DFT convolution because
   complex `torch.stft` is rejected by the legacy exporter. This is not yet a
   target-SoC or quantized result: Stage 1 uses traced fixed lengths, and the
   Wav2Vec intent graph is about 379 MB, so distillation/quantization remains
   necessary for the stated edge footprint.
   A local ONNX Runtime CPU benchmark records Stage 1 at 371/376 ms p50/p95
   for a 3-second window and intent at 91/94 ms p50/p95 for 2 seconds; KWS,
   presence, target-KWS and compact verification are each below 2 ms p95 on
   this desktop. These numbers are a reproducible workstation baseline only,
   not evidence for the selected automotive SoC or the 24-hour soak criterion.
   A dynamic-int8 feasibility pass was also run on the exported graphs. It
   reduced file sizes to roughly 27--29% of float, but made every small graph
   slower on this CPU (for example KWS 0.73→10.31 ms p95 and presence
   4.04→11.45 ms) and produced 6.54% Stage-1 and 12.22% target-KWS relative-
   L2 drift. Stage 1 became 5.66 s p95, and intent/compact-verifier conversion
   failed. This path is rejected; it is not a production quantization result.
   The next conversion must use representative cabin calibration, static or
   QAT quantization, and the selected vendor execution provider.
   A static MinMax QDQ diagnostic was then run on the compact intent graph with
   128 calibration and 128 disjoint cabin-domain evaluation rows. It completed
   but collapsed held-out accuracy from 55.47% float to 4.69% INT8, with 1.128
   relative-logit error and only a 2.15% file-size reduction; p95 latency rose
   from 7.13 to 8.10 ms. It is rejected and remains isolated. This is evidence
   that workstation post-training quantization is not yet a safe deployment
   path, not evidence that the compact architecture itself is unsuitable.
   A separate clean-only endpoint experiment fixed buffered-candidate state
   selection by choosing the longest consecutive intent run. It reached 64/64
   clean proxy actions with zero false actions, but required three 0.5-second
   post-wake hops (2.22 s mean / 2.50 s p95 command completion after wake).
   The same policy fell to 50% wake and 0% intent on a 16-scene noisy proxy,
   so it remains an isolated clean diagnostic and is not the retained runtime.
   A 250 ms-hop rescue was also rejected: three hops gave 75% clean intent,
   while four hops gave 87.5% intent and 128.6 false wakes/hour on the short
   proxy. The latency target therefore needs a streaming intent/endpoint model,
   not simply a smaller evaluation hop.
   A clean-first 12.7M-parameter CRNN intent candidate is now available. Its
   FP32 official clean-test score is 83.7% (macro-F1 85.6%), the clean
   end-to-end proxy is 87.5%, and clean→TSE fine-tuning raises that proxy to
   93.75%. It is much smaller and faster than Wav2Vec, but the same transferred
   candidate is only 6.25% intent on the short noisy proxy. It remains a
   candidate for a larger, properly mixed training study rather than a release
   replacement; the front-end/wake bottleneck must be solved first.
   Full-pipeline resource measurements for the compact candidate are 60.8 ms
   GPU p95 wake compute / 104.3 ms command compute and 505/527 ms CPU p95;
   the candidate checkpoint set is 98.4 MB. These are workstation numbers,
   not target-SoC evidence, and CPU remains above the real-time budget.
   Two clean-side latency shortcuts were rejected. An unpaired default-hop
   replay with early thresholds 0.90/0.98 reached 87.5% wake and 65.6%/59.4%
   intent; the matched 0.5-second-hop 0.90 study reached 100% wake / 96.875%
   intent with 0.69/1.0-second mean/p95 command completion, versus 100% /
   100% and 1.5/1.5 seconds for the 1.0-threshold baseline. The faster point
   is not a release choice because it loses actions and lacks held-out,
   speaker-consistent evidence. A new 1.5-second 12.7M-parameter clean CRNN
   reached 88.4% official FSC accuracy / 88.9% macro-F1 but only 53.1% intent
   on the raw clean proxy; a TSE-cache fine-tune reached 40.625% intent. Both
   are isolated. The correct next optimization is a streaming intent/endpoint
   model trained with matched command timing, not a shorter fixed crop.
   An opt-in causal CRNN was also evaluated. It reduced the intent head to
   5.79M parameters / 24.3 MB ONNX and 4.40 ms CPU p95, while reaching 89.0%
   clean FSC accuracy (89.7% macro-F1). TSE-domain fine-tuning raised that
   standalone score to 89.7% / 90.4% but left the clean end-to-end proxy at
   53.125% intent. It is isolated rather than promoted: deployment efficiency
   is promising, but the current cross-corpus command contract does not prove
   streaming correctness.
3. **Quantization-aware training** for the front end and wake models. The
   finding that dynamic int8 barely helps Stage 1 is expected: convolutions need
   static or QAT int8.
4. **Runtime in C++**, not Python. Keep the capture ring buffer design from
   `streaming_runner.py`: push never blocks, explicit skipped-seconds
   accounting. Add a watchdog, a fixed memory arena, real-time thread priorities
   and a bit-exact golden test that compares against the Python reference.
5. **Profile on the device** at p50, p95 and p99 under concurrent load
   (navigation, media decoding) and thermal throttling, not on a desktop GPU.
6. **Graceful degradation:**
   - Microphone failure → fall back to fewer channels.
   - Overload → the wake path keeps priority and the command path queues.
   - Model load failure → push-to-talk only.

**Exit criteria:** the full pipeline on target hardware meets the latency and
resource gates with 0 audio drops over a 24 h soak test.

---

## 6. Phase 5: Verification and validation (weeks 20–26)

- **Held-out functional gates** (Phase 3, rerun on the device build, which must
  match the Python reference within tolerance).
- **Long-duration negative soak:** ≥50–100 h of real driving with no intended
  commands (radio, podcasts, conversation, phone calls), on the device build.
  This is the only honest route to a sub-0.1/h claim.
- **Stress and robustness:** speeds, weather, windows, HVAC, a microphone
  disconnected mid-session, clipping, sample-rate or clock drift, and the
  assistant's own TTS (self-trigger).
- **Security (ISO/SAE 21434, UN R155 scope):** replayed driver recordings,
  the driver's voice playing over car speakers or a phone, TTS voice clones of
  the enrolled driver, inaudible or ultrasonic injection, and adversarial
  perturbations hidden in music. SOTIF explicitly excludes deliberate attacks,
  so they belong under cybersecurity [17]. Mitigations: the zone/DOA cue
  (playback comes from speaker positions, not the driver's mouth), AEC
  reference cross-check, liveness or anti-spoofing scoring, and Tier B/C
  confirmation.
- **SOTIF-style analysis (ISO 21448):** enumerate triggering conditions
  (accents, noise, out-of-distribution phrasing, foreseeable misuse such as a
  child in the rear seat) and show each is covered by data or by the action
  policy. SOTIF covers ML functions that fail to "comprehend the situation",
  including out-of-distribution inputs [17].
- **Process evidence (Automotive SPICE 4.0 MLE.1–4 plus SUP.11 for ML data
  management):** traceability from ML requirements → data → model → test [18].
  Check the base practices against the official VDA PAM v4.0. Secondary sources
  disagree on process names.
- **Usability:** a small in-car user study covering task completion, perceived
  false triggers, and how often confirmations become annoying.

**Exit criteria:** a safety case, a security test report and a V&V report, all
signed off.

---

## 7. Phase 6: Staged release and operations (week 26 onward)

1. **Shadow mode** on an internal fleet of 10–50 cars for 4–8 weeks. The system
   runs fully but takes no actions. It logs only on-device counters (wakes,
   cancels, confidence histograms), plus audio only from opted-in drivers.
2. **Limited release** with Tier A actions only. Tier B and C are enabled after
   the field FA/h and cancel-rate proxies hold.
3. **Monitoring KPIs:**
   - Wake events per hour.
   - "Cancel / no response within N s" as a false-accept proxy.
   - Repeat-command rate as a false-reject proxy.
   - Latency percentiles.
   - Per-locale and per-vehicle drift alarms.
4. **Over-the-air model updates** with staged rollout, an A/B holdback and
   automatic rollback if a KPI regresses. Every update re-passes the frozen
   regression suite and a fresh held-out tranche.
5. **Privacy in operation:**
   - The voiceprint never leaves the vehicle.
   - Users can delete their voiceprint, and the system re-enrolls after a
     driver profile change.
   - Audio retention is limited to opt-in data.
   - Check AI Act transparency obligations. Secondary sources date Art. 50
     duties to Aug 2026, subject to the Omnibus changes [10]. Get legal review.

---

## 8. What to do in the next two weeks (no new data needed)

1. Git, the registry and hashed release manifests (closes the reproducibility
   gate).
2. Draft the requirements doc from §1.1–1.2 and get decisions on the wake
   phrase, language, ontology and target SoC.
3. Order capture hardware: a ≥6-input interface, production-equivalent
   microphones, a loudspeaker reference tap and close-talk lavaliers.
4. **Completed 2026-10-13:** reran the 4-of-4 fast-hop AISHELL check across all
   18 Eval2 sessions (3.889 h). The full path produced 0 smoothed events/hour
   at the selected proxy configuration; the machine-readable result is
   `logs/aishell5_target_absent_compact_cabin_smart_thr030_hop025_hits4_all18_2026-10-13.json`.
   This closes the interrupted-proxy bookkeeping item, not the real-driver
   validation gate.
5. **Started 2026-10-13:** prototype the classical zone baseline and the
   English CAVEMOVE spatial ablation. The matched-FAR comparison on real
   passenger audio is still open; the current 24-scene proxy is explicitly
   cross-corpus and must not be used to select a production checkpoint.
6. Start TTS bootstrap data for the chosen wake phrase.
7. Extend the pilot protocol to make playback mandatory and record the
   reference channel. Add unscripted segments and a passenger-wake target count
   of ≥150 per wave.
8. **Completed 2026-10-14:** corrected the ONNX exporter so wrapped Stage-1
   and compact-verifier checkpoints load into their underlying models before
   export. Regenerated the candidate in
   `output/onnx/retained_ckptfix2_2026-10-14/`, verified all six exported
   handoffs against the trained PyTorch path on eight deterministic CAVEMOVE
   cabin-noise rows, and added the result to the hashed release manifest.
   This closes an export correctness defect, not the target-SoC or real-driver
   release gates. Host-CPU measurements remain a reference only.

## 9. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Real data collection slips (consent, vehicles, people) | Blocks every gate | Start procurement and ethics review now. Run Wave 1 with internal volunteers. |
| Zone-first approach fails in some cabins (e.g. rear/front crosstalk) | Passenger leakage | Hybrid with candidate-only TSE (§3.2 variant 3). Per-vehicle impulse-response calibration. |
| Target SoC cannot run the front end within budget | Latency and drops | Choose hardware early. Use a classical front end as fallback. Use the NPU for the wake path only. |
| Overfitting thresholds to a small held-out set | Optimistic release numbers | One-shot held-out evaluation, fresh tranches, speaker bootstrap. |
| Wake phrase has common confusers | High FA/h | Confuser-mined negative set before committing the phrase. Consider a two-word phrase. |
| Regulatory or privacy surprise (voiceprint) | Launch blocker | Enrollment optional by design. On-device only. DPIA early. |

---

## Sources

External research done on 2026-10-13. Vendor and blog sources are marked; treat
their numbers as indicative.

1. WeSep, a modular, cue-composable TSE framework with causal BSRNN (about 32 ms latency, ONNX streaming): https://arxiv.org/pdf/2607.27436
2. Low-latency multichannel directional speech extraction (about 2 ms algorithmic latency): https://arxiv.org/html/2407.04879v1
3. AISHELL-5, an in-car multichannel multi-speaker dataset (Interspeech 2025): https://arxiv.org/abs/2505.23036
4. StarTSE, streaming chunk-wise TSE: https://arxiv.org/pdf/2604.19635
5. Zoneformer, an on-device neural beamformer for in-car multi-zone separation, enhancement and AEC (Interspeech 2023): https://www.isca-archive.org/interspeech_2023/xu23b_interspeech.html ; in-car multi-zone challenges: https://www.arxiv.org/pdf/2510.10687
6. Personal VAD 2.0: https://arxiv.org/abs/2204.03793 ; Personal VAD: https://arxiv.org/pdf/1908.04284
7. Picovoice wake-word guidance and benchmarks (vendor): https://picovoice.ai/products/voice/wake-word/ , https://picovoice.ai/blog/wake-word-benchmarks/ ; Apple, Personalized Hey Siri: https://machinelearning.apple.com/research/personalized-hey-siri
8. 2020 Personalized Voice Trigger Challenge (FRR at 1 FA/h and speaker-verification metrics): https://arxiv.org/pdf/2101.01935
9. Synthetic and real negatives for wake words (practitioner and vendor reports): https://huggingface.co/gdiamos/hey-claude , https://livekit.com/blog/livekit-wakeword , https://arxiv.org/pdf/2011.01460
10. Voice data under GDPR and the EU AI Act (secondary sources, needs legal review): https://www.softcery.com/lab/eu-voice-ai-regulations-founders-guide , https://waywithwords.net/blog/how-does-gdpr-apply-to-speech-datasets
11. In-car multi-zone echo and feedback (patents, Cerence): https://patents.google.com/patent/US9641934 , https://www.arxiv.org/pdf/2210.03363
12. QUADS, quantized distillation for SLU (Interspeech 2025): https://arxiv.org/html/2505.14723v1
13. Streaming end-to-end SLU: https://arxiv.org/pdf/2105.10042
14. Asymmetric enroll-verify speaker verification (ECAPA-TDNNLite): https://arxiv.org/pdf/2110.04438 ; LightCAM: https://arxiv.org/pdf/2402.06073
15. ONNX Runtime QNN Execution Provider: https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
16. Qualcomm NPU benchmark (I/O quantization overhead): https://github.com/usefulsensors/qc_npu_benchmark ; Qualcomm Aura translator (Whisper on NPU): https://www.qualcomm.com/developer/project/aura-translator
17. SOTIF (ISO 21448) overview: https://visuresolutions.com/automotive/iso-21448/ ; Machine learning in certified systems: https://arxiv.org/pdf/2103.10529
18. Automotive SPICE 4.0 MLE processes: https://ul.com/sis/resources/automotive-spice-machine-learning-engineering-mle , https://blogs.sw.siemens.com/polarion/automotive-spice-4-0-whats-new/

## 2026-10-14 implementation update

The current evaluated deployment candidate is
`output/onnx/retained_ckptfix3_2026-10-14/manifest.json`, not the earlier
`retained_ckptfix2` directory. It contains seven independently checked graphs,
including the SpEx+ speaker-embedding path. Eight deterministic CAVEMOVE
cabin-noise rows passed PyTorch↔ONNX handoff parity, and a variable-length
check passed 1.0→2.0 s, 1.5→3.0 s, and 2.0→4.0 s enrollment/extracted windows.
The release manifest has all artifacts and hashes, but sign-off remains false:
there are still zero consented real-driver sessions, no matching six-input
capture device, no frozen speaker-disjoint split, and a dirty worktree. These
deployment checks strengthen reproducibility; they do not replace the real-car
study or target-SoC validation required by this plan.

The first current-export quantization feasibility pass is explicitly rejected:
uncalibrated dynamic-int8 reduced file size for some graphs but introduced
material parity drift and was slower on the local CPU; the speaker, verifier,
and wav2vec graphs also failed conversion. Do not use the diagnostic graphs
under `output/onnx/retained_ckptfix3_2026-10-14/dynamic_int8/`. The plan's
calibration/QAT and target-vendor-runtime work remains necessary.

## 2026-10-14 runtime optimization update

The retained ONNX streaming path now avoids evaluating the wav2vec intent graph
until the wake-to-command state is active. The parity harness remains in
full-component mode, so this is a deployment scheduling optimization rather
than a change to model outputs or thresholds. On the host CPU, a 10-second
four-microphone/playback-reference replay improved from 2.588x to 2.229x
inference RTF and from 1.878x to 1.618x capture-wall RTF, with 29 windows and
zero dropped seconds in both runs. The result is still above the intended
real-time budget and must not be generalized to the target SoC.

The isolated profile identifies Stage 1 as the dominant cost (about 375 ms
p95 for a 3-second four-channel window); the next performance milestone is
therefore a causal/chunked or target-compiled Stage 1, not another clean-intent
training run. The host benchmark used 8 intra-op and 1 inter-op ORT threads,
but the thread choice is not a production-device recommendation.

A cadence sensitivity run at a 500 ms hop reached 1.093x inference RTF and
0.824x capture-wall RTF on the same host, with no dropped seconds. It is useful
as a diagnostic fallback, but its decision cadence is already too slow for the
planned interaction budget and it still uses a 3-second non-causal window.

Detailed record: `reports/onnx_runtime_optimization_2026-10-14.md`.

## 2026-10-14 implementation update: composed ONNX runtime

The retained export now has a reusable composition layer in
`src/car_tse/deployment/onnx_runtime.py`. It validates four microphone inputs,
requires the sample-aligned playback reference, connects the exported
SpEx+ speaker embedding to target-KWS, and implements the retained fixed-window
wake-to-command state machine. Endpoint-trimmed strategies remain explicitly
unsupported in this runtime until separately validated.

`logs/onnx_composed_pipeline_parity_ckptfix3_cpu_8samples_2026-10-14.json`
shows all seven ONNX handoffs and all compared state-machine outputs matching
the PyTorch reference on eight deterministic CAVEMOVE proxy rows. This closes
an integration/reproducibility gap; it does not close the real-driver,
passenger-rejection, target-SoC, quantization, or production sign-off gates.
The report is included in `logs/release_manifest_current.json`, which now
contains 24 present, hashed artifacts and still correctly reports
`release_signoff_ready=false`.

The composed check was then expanded to all 32 rows in the retained CAVEMOVE
proxy manifest. All state-machine decisions matched the PyTorch reference.
Five component-row comparisons crossed the isolated graph harness's tighter
numeric rule, so the artifact reports both that strict count and an explicit
integration tolerance (relative L2 ≤0.2% or absolute error ≤0.01). Every
component passed under the bounded integration tolerance. This is still CPU
proxy parity and does not change the requirement for real-driver, target-SoC,
quantization, and safety evidence.

## 2026-10-14 implementation update: live-ring and export contract

The current ONNX candidate is ckptfix4. Enrollment and extracted speaker
audio now have distinct symbolic time dimensions, eliminating the earlier ORT
shape-reuse warnings during repeated calls. A 10-second deterministic stream
through StreamingRunner processed 29 windows, dropped 0 seconds, and kept
playback-reference routing sample-aligned. Host CPU inference was 2.58×
real-time on processed hop audio (about 1.87× capture wall time), so the
current float/Python path fails the intended real-time budget and is retained
as a diagnostic deployment baseline only.

The corrected export, 32-row composed parity, variable-length check, component
latency benchmark, and live-ring benchmark are all included in the 25-artifact
release manifest. This improves reproducibility and exposes the performance
gap; it does not close the target-SoC, quantization, real-driver, passenger
rejection, or safety gates.

The current manifest additionally hashes the lazy-intent 250 ms benchmark and
the 500 ms cadence diagnostic, for 26 present artifacts total. Its sign-off
flag remains false, as expected until target-hardware and real-driver evidence
exists.

## 2026-10-14 clean-path and causal follow-up

The first opt-in causal 16 kHz Stage 1 implementation passed structural and
prefix-causality checks, but swapping it into the existing checkpoint reduced
clean validation SI-SDR from 41.67 dB to 4.16 dB. A separate mixed-precision
fine-tune produced NaN train/validation SI-SDR in its first epoch and was
stopped. No causal checkpoint was promoted. This confirms that the latency
plan requires a model trained under the streaming contract plus numerical
hardening, not an inference-time architectural swap.

For clean proxy operation, the retained models with 500 ms hop and endpoint
trim reached 100.0% wake recall and 84.375% intent accuracy on 128 synthetic
noise-free rows with zero observed false events. This clears the provisional
clean proxy target but does not alter the production gates: the wake and
command speakers are not verified to be the same driver, the scenes are not
real cabin recordings, and negative exposure is too short for a release
claim. Artifact: `logs/clean_endpoint08_64scenes_clean_hop05_2026-10-14.json`.

The corresponding open-set check makes the release implication explicit:
the clean-only KWS/no-gate candidate produced 1,209.38 false wakes/hour and
419.51 false actions/hour on 3.193 hours of non-Marvin Smart Speaker speech.
The hard-negative/TSE-domain KWS with SpEx gating instead rejected nearly all
of the cross-corpus positive proxy (4.6875% wake, 0% intent), because its wake
enrollment and command speaker are not the same person. This is not a model
choice to freeze; it is a data-validity result. The next production milestone
is a speaker-consistent English cabin pilot, after which threshold calibration
and gate selection can be performed on tuning drivers and evaluated on held-
out drivers.

The full hard-negative/TSE-domain KWS sweep is now complete. On all 2,110
non-Marvin clips (3.193 hours), KWS-only smoothing generated 234 events
(73.30/hour), while the full path with the frame-presence gate generated 155
(48.55/hour). Presence gating therefore removed 33.8% of these open-set false
events, but the rate remains far above 0.1/hour. The zero false actions in the
14-file sequential sample cover only 68.65 seconds and are not a release
claim. This confirms that the next production work should focus on the
speaker-consistent cabin pilot, seat-zone/voiceprint calibration, and
target-device measurement rather than more clean positive-only tuning.
