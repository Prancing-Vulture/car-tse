# Pre-existing systems review for the in-car assistant

Date: 2026-10-11  
Scope: target-speaker extraction (TSE), target-speaker activity/presence, personalized wake-word detection, command understanding, and edge deployment.

## Executive conclusion

The current cascade is directionally consistent with established research, but it is not yet a production-proven implementation of those ideas. The most relevant references do not treat speaker extraction, speaker activity, and keyphrase detection as independent generic models. They condition the streaming front end and the decision layers on the same enrolled speaker representation, and they evaluate false triggers over long negative audio under the same operating conditions as the positive examples.

The best next architecture upgrade is therefore a speaker-conditioned streaming front end—personal VAD/TS-VAD or a target-speaker activity head—fed by the same enrollment embedding used by TSE and the wake detector. In parallel, the four-microphone path should add spatial information or beamforming once matching cabin recordings exist. Replacing the current wake network with BC-ResNet or MatchboxNet is a reasonable edge experiment, but it is not the first fix: our current negative tests show that calibration, speaker consistency, and continuous-cabin evidence dominate the risk. Newer work also supports separating keyword evidence from speaker identity and combining them with a late-fusion veto, rather than trusting one wake score.

No reference below is being presented as a drop-in pretrained solution for “Marvin” in this repository. The papers provide design patterns and baselines; they do not remove the requirement for permitted, speaker-consistent, real-cabin recordings.

## How the references map to our pipeline

| Area | Established reference pattern | Implication for this project |
|---|---|---|
| TSE | SpEx/SpEx+ use a target-speaker reference to guide time-domain extraction; SpEx++ makes short-reference use more practical through multi-stage refinement and frame-level reference information. | Our SpEx+-style Stage 1 is a defensible baseline. The highest-value improvements are short-enrollment robustness, causal/streaming behavior, and training mixtures that match the cabin—not another isolated clean SI-SDR run. |
| Multichannel TSE | SpeakerBeam and later multichannel systems use speaker conditioning together with spatial cues, and neural/beamforming hybrids are designed for low-latency multichannel enhancement. | Our duplicated or single-channel proxy inputs cannot test the intended four-microphone advantage. A real 4-channel cabin set is required before selecting a spatial front end. |
| Personal activity/presence | Personal VAD predicts target speech at frame level, separating non-speech, target speech, and non-target speech. Personal VAD 2.0 focuses on streaming, enrollment/enrollment-less robustness, and small runtime cost. TS-VAD similarly predicts target activity per time frame from acoustic features and speaker representations. | The current scalar presence gate should eventually become a speaker-conditioned, time-aligned activity decision. This directly addresses passenger rejection and “Marvin” spoken by the wrong person. |
| Personalized wake word | Google’s personalized keyphrase work combines streaming ASR, speaker verification, target-speaker separation, and adaptive noise cancellation. The results emphasize that speaker verification reduces false triggers while separation/ANC reduces false rejects. | Keep the wake detector target-conditioned and evaluate it after TSE/activity gating. Add a calibrated verification score and long negative exposure; do not optimize only clip-level F1. |
| Small-footprint KWS | BC-ResNet and MatchboxNet are compact streaming-friendly command/KWS backbones; BC-ResNet reports high Google Speech Commands accuracy with low compute, while MatchboxNet emphasizes 1-D time/channel separable convolutions and noise augmentation. | These are good candidates for a distilled or replacement always-on KWS head. They are not automatically better for target-conditioned “Marvin”; the training set must contain speaker-consistent positives and hard passenger/no-command negatives. |
| Edge runtime | VoiceFilter-Lite explicitly targets tiny, streaming, low-latency on-device separation and reports 8-bit quantization. Personal VAD 2.0 also treats model size, CPU, memory, and latency as design constraints. | Our current Wav2Vec intent stage dominates the approximately 424 MB footprint. After accuracy is stable, distill/quantize the intent stage and benchmark on the actual target hardware. The rejected compact CRNN experiments show that compression before data alignment can cost accuracy. |

## Primary references

### Target-speaker extraction

- [SpEx+: A Complete Time Domain Speaker Extraction Network](https://arxiv.org/abs/2005.04686) ties the reference and mixture encoders and reports gains over the earlier SpEx baseline. This supports keeping the enrollment path architecturally aligned with the extraction path.
- [Multi-stage Speaker Extraction with Utterance and Frame-Level Reference Signals (SpEx++)](https://arxiv.org/abs/2011.09624) uses multiple stages and frame-level reference information to make short enrollment more useful. It is the clearest reference for a future short-enrollment upgrade, but it would need a matching latency and memory study in this repo.
- [Improving speaker discrimination of target speech extraction with time-domain SpeakerBeam](https://arxiv.org/abs/2001.08378) studies spatial features and auxiliary speaker-identification loss. This is relevant to our passenger-rejection problem because extraction quality alone does not guarantee speaker identity is preserved.
- [Multi-channel target speaker extraction with refinement](https://arxiv.org/abs/2302.07928) combines multichannel causal enhancement, speaker conditioning, and a low-latency design. It is a useful architectural reference for the eventual four-microphone path, not a claim that our current single-channel SpEx+ path is equivalent.

### Speaker-conditioned activity and wake detection

- [Personal VAD: Speaker-Conditioned Voice Activity Detection](https://arxiv.org/abs/1908.04284) formulates target activity as a frame-level, speaker-conditioned problem and reports a very small model. This is closer to the failure we currently see than a generic binary speech-presence classifier.
- [Personal VAD 2.0](https://arxiv.org/abs/2204.03793) addresses streaming operation, enrollment variability, and edge constraints. It suggests using the enrolled voice representation throughout the streaming decision path rather than only at TSE time.
- [Target-Speaker Voice Activity Detection](https://arxiv.org/abs/2005.07272) predicts each speaker’s activity per frame and also investigates a multichannel attention extension. Its time-aligned output is a better conceptual match for gating commands than a single window-level presence score.
- [Personalized Keyphrase Detection using Speaker and Environment Information](https://arxiv.org/abs/2104.13970) combines streaming keyphrase recognition with speaker verification, separation, and adaptive noise cancellation. The combination is particularly relevant to our target-driver-versus-passenger requirement.
- [VoiceFilter-Lite: Streaming Targeted Voice Separation for On-Device Speech Recognition](https://research.google/pubs/voicefilter-lite-streaming-targeted-voice-separation-for-on-device-speech-recognition/) is a direct edge reference for a small, streaming, speaker-conditioned separator. It highlights the “must not hurt clean speech” constraint and reports an 8-bit runtime path.
- [SpeakerBeam-SS: Real-time Target Speaker Extraction with Lightweight Conv-TasNet and State Space Modeling](https://arxiv.org/abs/2407.01857) is a newer streaming reference: it uses state-space modeling and a lighter causal frontend to reduce real-time cost while preserving extraction quality. It is a candidate pattern for a later Stage 1 latency study, not a reason to replace the current checkpoint before the data contract is fixed.

### Compact wake/command backbones

- [Broadcasted Residual Learning for Efficient Keyword Spotting (BC-ResNet)](https://arxiv.org/abs/2106.04140) and the [official Qualcomm implementation](https://github.com/Qualcomm-AI-research/bcresnet) provide a compact baseline for an always-on KWS head.
- [MatchboxNet](https://arxiv.org/abs/2004.08531) uses 1-D time/channel-separable residual blocks and noise augmentation. It is a useful alternate backbone for an edge KWS or small command head.
- [Personalized Keyword Spotting for User-Defined Keywords Leveraging Text-Independent Speaker Verification](https://arxiv.org/abs/2606.20106) is a recent preprint that combines a phonetic keyword branch with a compact speaker-verification branch and allows late fusion to act as an independent speaker veto. Its reported 1.55M-parameter budget is attractive for edge deployment, but its benchmark is not our cabin protocol and should be treated as a design reference rather than a transferred result.

## What we learned from applying these ideas to our repository

The following experiments are repository evidence, not results quoted from the papers.

### Hard-negative data audit

The [University of Rochester Smart Speaker Command Dataset](https://rochester.figshare.com/articles/dataset/Smart_Speaker_Command_Dataset/26417548) was downloaded under its CC BY 4.0 license. It contains 2,110 clean 48 kHz mono clips from seven speakers, scripted as “Hey Alexa” plus a command. Because the wake phrase is not “Marvin,” it is useful as a non-Marvin continuous hard-negative diagnostic, but it is not cabin audio and it does not provide positive Marvin examples.

The Mycroft [Precise Community Data](https://github.com/MycroftAI/Precise-Community-Data) repository was inspected. It contains contributor-specific licensing files and many short clips, but no raw `marvin` clip collection suitable for treating as a verified positive training set; the `marvin` directory contains a model artifact. It was not added to the training set.

### Negative evaluation

The evaluator duplicates each mono Smart Speaker clip across the configured four channels solely to exercise the existing interface. This is deliberately labeled a diagnostic and must not be reported as real-cabin or multichannel evidence.

| Run | Duration/files | Full-path false wakes | KWS-only events | Action sample |
|---|---:|---:|---:|---:|
| Retained reference, 250 clips | 0.335 h / 250 | 2,085.8 per hour | 12,031.3 per hour | 105.9 false actions/hour over 14 files |
| Retained reference, speaker gate 0.40 | 0.335 h / 250 | 0.0 per hour | 12,031.3 per hour | not run |
| Full Smart hard-negative candidate | 0.335 h / 250 | 26.9 per hour | 107.4 per hour | 0.0 over 14 files |
| Thin Smart candidate | 0.335 h / 250 | 2,461.8 per hour | 13,263.7 per hour | 105.9 false actions/hour over 14 files |
| Thin candidate, KWS threshold 0.70 | 0.335 h / 250 | 0.0 observed | 0.0 observed | 0.0 observed |

These numbers are not contradictory: the current evaluator records conservative window/event diagnostics and the exact action sample is intentionally small. More importantly, the threshold-0.70 thin run also produced 0% positive wake recall on its positive proxy. A threshold that suppresses every negative is not a useful operating point.

The evaluator was subsequently corrected to merge near-contiguous positive
windows into operating events. On the same 250-file subset, the corrected
replay measured 728.1 merged full-path events/hour and 1,817.2 merged
KWS-only events/hour for the retained reference, versus 17.9 and 50.7/hour
for the full Smart hard-negative candidate. The exact 14-file action sample
remained 105.9 false actions/hour for the reference and 0.0 for the
candidate. The earlier table is preserved as the original window-count
diagnostic; the corrected artifacts are:

- `logs/smart_speaker_command_negative_250_dedup_baseline_2026-10-11.json`
- `logs/smart_speaker_command_negative_250_dedup_candidate_2026-10-11.json`

### Candidate training and positive proxy

The new Smart-negative candidates were trained only as experiments:

| Candidate | Positive proxy | Decision |
|---|---:|---|
| Full merged Smart hard-negative candidate | 14.84% wake recall, 2.34% overall intent, 15.79% conditioned intent | Rejected; too much positive loss |
| Thin merged candidate | 81.25% wake recall, 75.00% overall intent, 92.31% conditioned intent on 16 scenes | Promising diagnostic, but rejected because the negative evaluation did not improve and the sample is too small |
| Thin candidate at KWS 0.70 | 0% wake/intent | Rejected; unusable recall |
| Distilled candidate | 31.25% wake/intent, 100% conditioned intent on 16 scenes | Rejected; insufficient wake recall |

The retained proxy reference remains approximately 68.75% wake recall and 59.375% overall intent on 32 synthetic scenes, with 86.36% intent accuracy conditional on a wake. The Marvin clip is not verified to be spoken by the same speaker as the command/enrollment in that proxy, so it is not real-driver evidence.

### Main interpretation

The experiments support the literature’s emphasis on speaker-conditioned decisions and long negative exposure. A hard-negative KWS model can reduce some observed negative events, but if its positive wake recall collapses it is not an improvement. Conversely, a small positive proxy can look strong while hiding false-trigger behavior and speaker mismatch. We therefore retain the earlier reference configuration and do not promote any of the Smart-negative candidates.

## Recommended upgrade order

1. **Collect the permitted pilot first.** Use the existing validation contract: the same driver must provide enrollment, “Marvin,” and the command in a continuous session, with passenger “Marvin,” passenger speech, no-command periods, and annotated cabin conditions. Keep tuning drivers/sessions disjoint from held-out test drivers/sessions. The current capture preflight found no matching four-channel device, so the hardware gap must be resolved before claiming a four-channel result.
2. **Train a personal activity head.** Replace or augment the current presence gate with a three-way target/non-target/non-speech frame classifier conditioned on the enrollment embedding. Start with a small causal model inspired by Personal VAD 2.0; compare it against the current presence + speaker-gate cascade using false-action/hour as the primary safety metric.
3. **Split wake evidence into two veto-capable branches.** Keep a compact phonetic KWS branch for “Marvin” and add a calibrated text-independent speaker-verification score from the enrollment. Combine them with an explicit late-fusion rule or learned calibrator so passenger speech can fail the speaker branch even when the words are correct. Train this only after collecting same-driver positives and passenger-“Marvin” negatives.
4. **Make wake detection speaker-consistent.** Train positive “Marvin” recordings from the same driver/session family as enrollment and command, and include passenger Marvin, near-wake phonetic confusions, radio/road/HVAC, and long no-command audio. Use BC-ResNet, MatchboxNet, or the newer dual-branch design only as compact experiments after the data contract is in place.
5. **Add spatial information.** Once a real 4-channel set exists, compare the current mono/SpEx+ path with a beamforming or multichannel speaker-conditioned front end. Keep the latency budget explicit; the reference multichannel work demonstrates that causal low-latency designs are possible, but it does not guarantee the same result for our cabin geometry.
6. **Study streaming TSE efficiency.** After accuracy is stable, benchmark a causal/state-space or lighter Conv-TasNet-style extractor against SpEx+ on identical chunks, measuring SI-SDRi, clean-speech preservation, real-time factor, and memory.
7. **Optimize the intent stage last.** The current Wav2Vec stage is the largest footprint. Distill or quantize it only after the joint wake-to-command score is stable, and measure end-to-end CPU/DSP latency, memory, and degradation on noisy cabin commands.
8. **Freeze thresholds on tuning data and report long negatives.** Select thresholds using both false accept and false action constraints. Report driver wake recall, command accuracy, false accepts/hour, false actions/hour, and response latency by condition. Do not substitute clip-level F1 or a clean studio benchmark for this report.

## First personal-activity experiment

The personal-activity recommendation was implemented as an isolated binary
target-activity candidate. `PersonalFramePresenceHead` uses the SpEx+
enrollment embedding to modulate framewise log-mel features. Its internal
row-split validation reached chunk F1 1.000 at threshold 0.40, but the
end-to-end proxy exposed the calibration problem: threshold 0.15 produced
81.25% wake recall, 62.50% overall intent, and 96.4 false actions/hour;
threshold 0.40 produced 50.00% wake recall, 37.50% overall intent, and zero
observed false actions on the small proxy. The checkpoint was therefore
rejected. This is consistent with the references: speaker conditioning is
promising, but operating points must be selected on continuous
speaker-consistent data, not on an internal activity F1 score.

Detailed experiment record: `reports/personal_presence_candidate_2026-10-11.md`.

## Enrollment-bank experiment

The review also suggested stabilizing the target representation with several
short references. An optional runtime bank was tested on identical 16-scene
cross-corpus cabin proxies. One reference produced 81.25% wake recall and
68.75% overall intent with zero observed false actions; an uncalibrated mean of
three references produced 6.25% wake recall and 6.25% overall intent. The bank
was rejected because it rejected nearly all target wakes. Future multi-
utterance enrollment must be quality-weighted or learned from same-driver
data and selected on tuning data; it is not a drop-in improvement.

Detailed experiment record: `reports/enrollment_bank_candidate_2026-10-11.md`.

## Late speaker-verification fusion experiment

The literature recommendation was implemented as an opt-in late-fusion path:
the target-KWS probability is combined with a sigmoid-mapped SpEx cosine
similarity before temporal smoothing. The historical hard gate remains the
default. On the eight-scene cabin proxy, `min` fusion at center 0.24/scale
0.04 retained 100.0% wake and intent with zero false actions, but matched the
retained Smart-negative diagnostic at 728.1 full-path events/hour. Center
0.26 reduced that negative rate only to 695.3/hour while dropping positive
wake and intent to 87.5%; product fusion at center 0.24/scale 0.04 dropped
positive wake and intent to 25.0%. The candidate was rejected because the raw
SpEx similarity is not calibrated for this domain. A dedicated speaker-
verification branch still requires same-driver target/passenger tuning data.

Detailed experiment record: `reports/late_speaker_fusion_candidate_2026-10-11.md`.

## Runtime-domain speaker-verifier calibration audit

To separate threshold problems from representation problems, the frozen SpEx
embedding was calibrated on the cached Stage-3 cabin-mixture preparation data.
This is closer to runtime than clean LibriSpeech pairs because the extracted
embedding comes from a target-KWS mixture, but it remains synthetic
CAVEMOVE/GSC-Marvin data. The held-out test AUC was 0.515; at a validation
operating point near 1% FPR, test TPR was only 1.56%. Applying the candidate in
the end-to-end proxy reduced target wake recall to 56.25% and overall intent to
43.75%, while observing zero false events on the small proxy. It was rejected:
the current extracted embedding is not a usable calibrated speaker-verifier
signal, so changing the sigmoid center or threshold cannot solve the problem.

This result strengthens the literature-based recommendation to train a
separate speaker-conditioned personal-activity or verifier head using
speaker-consistent driver/passenger sessions. The detailed record is in
`reports/speaker_verifier_calibration_2026-10-11.md`.

## Pretrained speaker-encoder follow-up

The next reference-driven diagnostic used Microsoft's pretrained
`microsoft/wavlm-base-plus-sv` as an independent speaker encoder. On the full
synthetic cabin cache, raw WavLM cosine reached 0.839 held-out test AUC,
compared with 0.515 for the current SpEx extracted embedding. A small frozen-
WavLM relation head reached 0.896 AUC. However, at a validation-selected
approximately 1% FPR operating point, the relation head still produced only
37.5% test TPR and 2.34% test FPR. It is therefore a better representation
candidate, not a safe production verifier.

The resource check is also decisive: one 1-second WavLM forward measured 25.7
ms p50 / 31.6 ms p95 on the RTX 4080, with 101.1M parameters and about 444 MB
peak CUDA allocation. It must be distilled or replaced with a much smaller
speaker branch before edge promotion. The complete candidate record is in
`reports/wavlm_speaker_verifier_candidate_2026-10-11.md`.

## Additional automotive data sourcing

The official ANIR corpus was reviewed and its 15 MB V-class impulse-response
archive was downloaded to `D:\datasets\anir` (682 48-kHz `.fir` files). It is a
good future spatial/cabin augmentation source, but contains no wake/intent
labels and does not satisfy the same-driver evaluation contract. iCarB-Voice is
more directly relevant to speaker-conditioned in-car modeling, but its 121.7-
GB files are restricted to approved non-commercial EULA users, so it was not
silently added. The sourcing decision and hashes are recorded in
`reports/automotive_dataset_sourcing_review_2026-10-11.md`.

The complete ANIR V-class noise archive was then indexed and evaluated
separately. Across all 42 conditions (0.35 hours, microphones 0-3), the
retained gated path produced 0 observed noise-only events/hour; the KWS-only
stream produced 8.57/hour. This is a useful hard-negative diagnostic, not
evidence of driver or passenger performance. The detailed result is in
`reports/anir_noise_target_absent_2026-10-11.md`.

## Reproducibility artifacts

- `scripts/data/build_target_kws_smart_speaker_negative_cache.py`
- `scripts/eval/eval_smart_speaker_command_negative.py`
- `scripts/train/train_stage3_target_kws.py`
- `data/cache/stage3_target_kws_smart_speaker_negative_16k/`
- `data/cache/stage3_target_kws_gsc_marvin_hardneg_aishell5_smart_speaker_16k/`
- `logs/smart_speaker_command_negative_250_2026-10-11.json`
- `logs/smart_speaker_command_negative_250_candidate_2026-10-11.json`
- `logs/smart_speaker_command_negative_250_thin_candidate_2026-10-11.json`
- `logs/smart_speaker_command_negative_250_thin_candidate_kws070_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_smart_speaker_hardneg_same_speaker_m10_32scenes_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_smart_speaker_thin_same_speaker_m10_8scenes_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_smart_speaker_thin_kws070_same_speaker_m10_8scenes_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_smart_speaker_distill_same_speaker_m10_8scenes_2026-10-11.json`
- `scripts/eval/calibrate_cached_spex_speaker_verifier.py`
- `logs/speaker_verifier_calibration_gsc_cache_stage1_ft2_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_late_fusion_gsc_cache_calibrated_m10_2026-10-11.json`
- `scripts/eval/eval_pretrained_wavlm_speaker_verifier.py`
- `scripts/train/train_wavlm_speaker_verifier_head.py`
- `scripts/eval/benchmark_pretrained_wavlm_speaker_verifier.py`
- `logs/wavlm_speaker_verifier_cabin_diagnostic_2026-10-11.json`
- `logs/speaker_verifier_wavlm_cabin_ft1_2026-10-11.json`
- `logs/wavlm_speaker_verifier_latency_2026-10-11.json`
- `configs/stage2_personal_presence_dual_domain_nonwake_ft1.yaml`
- `checkpoints/stage2_personal_presence_dual_domain_nonwake_ft1/best.pt`
- `logs/continuous_full_pipeline_proxy_personal_presence_same_speaker_m10_8scenes_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_personal_presence_p040_same_speaker_m10_8scenes_2026-10-11.json`

## Compact runtime follow-up

The pretrained-speaker recommendation was implemented as a small, opt-in late
speaker-veto backend. The distilled log-mel verifier has only 176,579
parameters and measured 5.14 ms p50 / 5.24 ms p95 for enrollment plus
extracted one-second inputs on the RTX 4080. However, its low-FPR validation
point gave only 16.4% held-out TPR at 2.34% FPR, and the full synthetic cascade
detected 0/2 wakes at that threshold. A deliberately relaxed 0.80 point
detected 1/16 wakes and 0/16 commands. It was rejected for promotion.

The practical lesson is important: the existing problem is not solved by
dropping in a smaller pretrained verifier or by changing thresholds on the
cross-corpus proxy. The verifier must be trained with the same driver's
enrollment/wake/command and passenger/no-command negatives, and it should be
calibrated jointly with the causal activity and wake-smoothing decisions. Keep
the compact branch as a reproducible implementation candidate, not as a claim
of production readiness. See
`reports/compact_speaker_verifier_runtime_candidate_2026-10-11.md`.

## ANIR augmentation follow-up

The ANIR V-class noise archive was converted into 42 matched four-channel
16-kHz noise beds and used to train an opt-in personal-presence candidate with
604 additional target-positive copies. On the same 16-scene cross-corpus proxy,
the candidate exactly matched the prior personal-presence result: 81.25% wake
recall, 62.50% overall intent, 208.93 false wakes/hour, and 96.43 false
actions/hour. It was rejected for promotion. ANIR remains useful as a noise
diversity asset, but it cannot compensate for the missing same-driver,
passenger, and no-command recordings.

Detailed experiment record:
`reports/anir_presence_augmentation_candidate_2026-10-11.md`.

## Three-way personal-activity follow-up

The next architecture-backed experiment made the binary presence weakness
explicit. `PersonalActivityHead` predicts non-speech, enrolled-driver speech,
and non-target/passenger speech per frame, then exposes only the driver-speech
probability to the existing cascade. A 2,408-row diagnostic manifest retained
the 1,208 GSC/AISHELL rows and added 1,200 non-Marvin Speech Commands clips
with different-speaker Marvin enrollment and CAVEMOVE noise.

The four-epoch candidate reached validation chunk F1 0.697 at threshold 0.40.
On the fixed 16-scene cross-corpus proxy, threshold 0.25 produced 75.0% wake
recall, 56.25% overall intent, 208.93 false wakes/hour, and 48.21 false
actions/hour, versus 81.25%, 62.50%, 208.93/hour, and 96.43/hour for the
existing binary personal head. Threshold 0.30 reduced wake recall to 25.0%
and threshold 0.40 rejected every target scene. The candidate is rejected for
promotion; it is retained as the implementation to train on the real pilot's
same-driver and passenger labels. Details are in
`reports/personal_activity_3way_candidate_2026-10-12.md`.

The 0.25 point was also replayed on the 250-file Smart Speaker non-Marvin
diagnostic. Merged full-path events fell modestly from 728.1/hour to 674.4/hour,
but KWS-only events remained 1,817.2/hour and the 14-file sequential action
sample remained 105.9 false actions/hour. This does not meet the safety bar.

The candidate was then retrained with enrollment-speaker-disjoint groups:
1,925 training rows across 912 groups and 483 validation rows across 225 groups,
with zero overlap. Validation chunk F1 rose to 0.755, but the fixed proxy at
0.25 returned 75.0% wake, 56.25% intent, and 96.43 false actions/hour; at
0.35 it rejected all target scenes. This confirms that internal activity F1
and cross-corpus full-pipeline behavior remain poorly calibrated. The detailed
record is in `reports/personal_activity_group_split_candidate_2026-10-12.md`.

## AISHELL-5 continuous target-absent follow-up

The continuous AISHELL-5 evaluator was extended to instantiate the new
three-way activity head directly, and the group-split candidate was replayed
on all 18 real four-channel `Eval2` sessions (3.889 hours total). The fixed
LibriSpeech enrollment speaker is absent from these recordings, so this is a
negative rejection diagnostic only.

At presence 0.25, speaker gate 0.24, KWS 0.50, and two-hit smoothing, the full
path produced 4.89 merged events/hour; the KWS-only stream produced 96.69/hour.
The aggregate KWS sweep was 11.83, 5.66, 4.89, and 3.09 full-path events/hour
at thresholds 0.30, 0.40, 0.50, and 0.60. This indicates useful rejection from
the speaker-conditioned gates on real multichannel cabin speech, but it does
not establish driver wake recall, command accuracy, or a production false
accept rate. Thresholds were not selected on real-driver tuning data.

Holding KWS at 0.50 and sweeping only presence gave 9.51, 8.74, 6.94, 4.89,
1.80, and 0.77 full-path events/hour at presence thresholds 0.10, 0.15, 0.20,
0.25, 0.30, and 0.35. The KWS-only rate stayed at 96.69/hour, isolating the
negative-rejection contribution of the activity/speaker gate without implying
that the stricter thresholds preserve target recall.

The evaluator also now omits zero-second rows when a duration budget ends and
stores aggregate threshold-sweep values in the root result for reproducible
reporting. Details: `reports/aishell5_personal_activity_target_absent_2026-10-12.md`.

## Research-informed compact intent candidate

The literature review was followed by a controlled downstream-model study.
VoiceFilter-Lite and personalized-keyphrase systems emphasize small
speaker-conditioned feature branches, clean-speech preservation, and an
independent speaker-verification veto. SpeakerBeam-SS suggests that future
Stage 1 efficiency work should use causal/state-space modeling rather than
simply increasing recurrent capacity. These references support a compact
candidate study, but they do not turn cross-corpus proxy results into
real-driver evidence.

The hidden-384 two-layer distilled CRNN was trained with the same cached
TSE/cabin data, teacher logits, normalization, temperature 3.0, and KD weight
0.5 as the retained hidden-256 student. It scored 50.4% on held-out cabin/TSE
validation and 73.3% on clean FSC test, versus 56.3% and 80.5% for the
existing student. On a 40-scene same-speaker proxy at cabin SNRs -5/0/5/10 dB,
it achieved 75.0% wake recall, 50.0% overall intent, 66.7% intent given wake,
and zero observed false actions. It was rejected because both intent metrics
were lower and the footprint grew to 23.94M parameters / 96.1 MB. The default
pipeline remains unchanged. Full details:
`reports/research_informed_intent_student_candidate_2026-10-12.md`.

The pipeline loader now reads CRNN dimensions from the evaluation configuration
when a non-default candidate is selected. This prevents a checkpoint/config
mismatch and makes future architecture comparisons reproducible.

## ReMASC vehicle-speech stress replay and gate calibration

The public ReMASC sample was converted into a 60-second, six-channel, 16-kHz
vehicle stress bed by resampling and tiling one replayed cabin recording. The
bed was mixed with the existing synthetic target/enrollment/wake/command
scenes at cabin SNRs -5, 0, 5, and 10 dB. This is a bounded cross-corpus
diagnostic: the vehicle bed is real cabin audio, but the driver, wake word,
command, and passenger clips are not one continuous real-driver recording.
It must not be reported as a real-driver false-accept rate.

| Configuration | Wake recall | Intent accuracy | Diagnostic false wakes/h | Diagnostic false actions/h |
|---|---:|---:|---:|---:|
| Retained binary/frame gate | 100.0% | 90.6% | 257.1 | 64.3 |
| Group-split 3-way activity, target threshold 0.25 | 100.0% | 96.9% | 257.1 | 128.6 |
| 3-way activity plus target-minus-passenger margin -0.67 | 90.6% | 78.1% | 64.3 | 0.0 |
| Group-split 3-way activity, target threshold 0.35 | 0.0% | 0.0% | 0.0 | 0.0 |
| 3-way activity 0.25, command gate 0.35 | 100.0% | 0.0% | 192.9 | 192.9 |

Each row contains 32 scenes / 448 seconds. The 3-way head improves target
command classification at its permissive operating point, but its target
probability is not yet separated from passenger speech: it allows passenger
“Marvin” events to produce actions. Raising the same threshold to 0.35 rejects
the passenger but also rejects every target wake. An asymmetric command gate
suppresses the target command in this proxy and still does not produce a safe
operating point. This is evidence for a data/calibration gap, not evidence
that the 3-way architecture is intrinsically worse.

The result matches the established system pattern: Personal VAD 2.0 and
target-speaker VAD use speaker-conditioned frame decisions; personalized
keyphrase systems combine keyword evidence with a separate text-independent
speaker-verification decision; multichannel hotword systems exploit spatial
and adaptive-noise-cancellation cues; and production KWS toolkits treat
streaming state, long negative exposure, and threshold/refractory policy as
part of the evaluated system. For this repository, the next high-value
implementation is a same-driver, target/passenger/non-speech activity and
speaker-verification training/calibration set, followed by an explicit
two-factor action veto. A new KWS backbone alone is not justified by the
current evidence.

As a further runtime-only test, the new target-minus-passenger margin was
replayed at -0.70, -0.68, -0.67, and -0.66. On the ReMASC stress proxy,
-0.67 gave 90.6% wake recall, 78.1% intent accuracy, 64.3 diagnostic false
wakes/hour, and zero observed false actions/hour. The neighboring -0.68 point
reintroduced 128.6 false actions/hour, while -0.66 reduced wake recall to
62.5%. This is a useful calibration shape, but the cutoff was selected after
observing the proxy and cannot be frozen for deployment.

The same -0.67 diagnostic margin was then evaluated over all 18 AISHELL-5
Eval2 sessions (3.889 hours), where the fixed enrollment speaker is absent.
Full-path smoothed target-absent events fell from 4.89/hour to 3.60/hour;
KWS-only events stayed at 96.69/hour. This is a modest real-cabin negative
rejection improvement, not a driver-recall or production-FAR result.

Artifacts:

- `scripts/data/build_remasc_cabin_stress_bed.py`
- `D:\datasets\remasc\stress_bed\ReMASC\noise\remasc_vehicle_replayed_6ch_tiled.wav`
- `logs/continuous_full_pipeline_proxy_remasc_vehicle_stress_8scenes_2026-10-12.json`
- `logs/continuous_full_pipeline_proxy_remasc_personal_activity_group_p025_8scenes_2026-10-12.json`
- `logs/continuous_full_pipeline_proxy_remasc_personal_activity_group_p035_8scenes_2026-10-12.json`
- `logs/continuous_full_pipeline_proxy_remasc_personal_activity_group_p025_cmdp035_8scenes_2026-10-12.json`
- `configs/eval_end_to_end_cavemove_personal_activity_group_p025_cmdp035.yaml`
- `configs/eval_end_to_end_cavemove_personal_activity_group_p025_margin_m067.yaml`
- `logs/continuous_full_pipeline_proxy_remasc_personal_activity_group_p025_margin_m0p66_8scenes_2026-10-12.json`
- `logs/continuous_full_pipeline_proxy_remasc_personal_activity_group_p025_margin_m0p67_8scenes_2026-10-12.json`
- `logs/continuous_full_pipeline_proxy_remasc_personal_activity_group_p025_margin_m0p68_8scenes_2026-10-12.json`
- `logs/continuous_full_pipeline_proxy_remasc_personal_activity_group_p025_margin_m0p70_8scenes_2026-10-12.json`
- `logs/aishell5_target_absent_personal_activity_group_all18_margin_m067_2026-10-12.json`
- [ReMASC official repository](https://github.com/YuanGongND/ReMASC)
- [Personal VAD 2.0](https://arxiv.org/abs/2204.03793)
- [Personalized Keyphrase Detection using Speaker and Environment Information](https://research.google/pubs/personalized-keyphrase-detection-using-speaker-and-environment-information/)
- [Supervised Noise Reduction for Multichannel Keyword Spotting](https://research.google/pubs/supervised-noise-reduction-for-multichannel-keyword-spotting/)
- [WeKWS production-oriented KWS toolkit](https://github.com/wenet-e2e/wekws)

## Frame-aligned activity-margin follow-up

The runtime was audited against the framewise target/non-target decisions used
by personal-VAD systems. The historical margin selected target and non-target
frames independently, so two frame-aligned alternatives were added as opt-in
research modes: `target_top` and `framewise_top`. On the same 32-scene / 448-
second ReMASC stress proxy, a target-top cutoff of `-0.30` produced 50.0% wake
recall, 28.1% intent accuracy, 64.3 diagnostic false wakes/hour, and zero
observed false actions/hour; framewise-top produced 68.8%, 40.6%, 64.3/hour,
and zero. The target-top mode reduced the all-session AISHELL-5 Eval2
target-absent rate only from 3.60/hour to 3.34/hour, with no change in the
96.69/hour KWS-only rate. Both modes are rejected for promotion because their
proxy recall loss outweighs the small negative-domain gain. Full details are
in `reports/personal_activity_margin_aggregation_2026-10-12.md`.

## Streaming-state and latency follow-up

Production-oriented KWS systems treat hop size, temporal confirmation,
refractory state, and post-wake command timing as part of the evaluated
system. The repository was therefore tested at 500 ms and 250 ms hops. The
250 ms stream initially mixed the wake word into the first intent window; an
opt-in four-call command delay fixed that overlap. With an eight-call
refractory period, the ReMASC stress proxy reached 96.9% wake, 81.3% intent,
64.3 diagnostic false wakes/hour, zero observed false actions/hour, and 0.968
seconds mean wake latency. The one-second reference was 90.6% wake, 78.1%
intent, 64.3 false wakes/hour, zero false actions/hour, and 1.879 seconds.
This is a diagnostic latency/state-machine candidate, not a real-driver or
automotive-hardware result. A 1,200-second AISHELL-5 target-absent follow-up
produced 6.0 full-path events/hour at the 250 ms hop with both 2-second and
4-second refractory settings, versus 3.0/hour for the comparable one-second
reference, so the fast setting is rejected for safety promotion. Full details
are in `reports/streaming_latency_candidate_2026-10-12.md`.

## Limitations

This review does not establish state of the art, production readiness, or superiority over the cited systems. Most references report task-specific benchmarks with different datasets, microphone layouts, and latency definitions. The Smart Speaker data is clean studio speech; ICMC-ASR and other in-car corpora may help with cabin acoustics but do not supply the exact English same-driver enrollment → Marvin → command → passenger/no-command protocol required here. Dataset licenses and any institutional agreements must be checked again before redistribution or publication.
