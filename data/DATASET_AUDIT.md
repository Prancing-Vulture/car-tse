# Dataset inventory and provenance audit

**Repository:** `car-tse`  
**Audit snapshot:** 2026-10-05  
**Purpose:** source and derived dataset documentation for reproducibility, audit, and journal-paper methods writing.

## Scope and evidence standard

This inventory combines the checked-in source registry (`data/sources.json`), data preparation code, model configs, evaluation code, and checked-in benchmark reports. Public corpus descriptions below are cross-checked against the linked dataset landing pages. A dataset being mentioned in a path template or helper is not evidence that it was downloaded or used in a reported experiment. “Acquisition registry” means the repository's own record, not a fresh filesystem verification.

**Important snapshot limitation:** this checkout contains no `data/raw`, `data/processed`, `data/cache`, or `manifests` directories. Consequently, the archives and generated rows reported in `sources.json` and benchmark reports could not be independently enumerated or checksum-verified in this checkout. The recorded paths, archive sizes, and statuses below reflect committed metadata; they do not establish that the corresponding local files are present now. The reports do record prior measured data footprints, but those measurements are not a substitute for the missing raw-data audit trail.

## Executive inventory

| Dataset / asset | Type and role in this project | Evidence of use | Acquisition / status in repo metadata |
|---|---|---|---|
| LibriSpeech | Read English speech; clean target and enrollment utterances, source material for mixtures | Stage 1 triplets, enrollment references, synthetic continuous speech and intent-independent controls | Three archives registered complete; see details below |
| LibriMix / local LibriSpeech-derived mixtures | Synthetic two-speaker mixtures and source-separated target references | Stage 1 pretraining/bootstrap and synthetic diagnostics | Full standard LibriMix expansion is not registered as downloaded; local bootstrap recipe exists |
| WHAM! | Ambient-noise source used by the official LibriMix recipe | Mentioned in LibriMix provenance; no direct WHAM path is used by the local bootstrap generator | Config template lists WHAM, but `sources.json` has no WHAM acquisition entry |
| AISHELL-5 | Real Chinese in-car multichannel speech, near-field references, TextGrid annotations, noise-only cabin audio | Cabin/domain diagnostics, target-absent negatives, target-aligned experiments and continuous negative-stream evaluation | Five full archives registered complete; local presence cannot be checked in this checkout |
| CAVEMOVE | Three-car impulse responses and driving/ventilation noise | Cabin-noise augmentation, synthetic cabin mixtures and evaluation | Full archive registered complete_verified with MD5 |
| Google Speech Commands v2 (GSC) | Short English keyword recordings; includes `marvin` positives and non-wake-word examples | KWS, presence calibration, synthetic continuous and multi-speaker proxy evaluation | Archive registered complete |
| Fluent Speech Commands (FSC) | English spoken commands with 31 slot-combination intents | Stage 4 intent training/evaluation; command hard negatives for KWS | Archive registered complete_verified with MD5 |
| Vehicle Interior Sound Dataset (VISC) | Vehicle-only sound clips; potential noise-domain material | Listed as auxiliary in data documentation and project report; no checked-in recipe proves it enters the principal experiments | Archive registered complete_verified with MD5 |
| MSWC English | Multilingual Spoken Words Corpus English subset; possible KWS positives/negatives depending on label mapping | Legacy KWS manifest builder can take up to 5,000 files as positive label 1; no repository acquisition record or current reported usage evidence | Config path only; acquisition unverified |
| Hey Snips | KWS evaluation negatives in legacy manifest builder | `build_kws_manifest.py` scans it into eval as label 0 | Config path only; acquisition unverified; no split/manifest present here |
| Custom in-car recordings | User-collected wake/non-wake/enrollment recordings | Builder supports speaker-disjoint manifests, but reports state the real corpus is not yet present | Expected input only; not in source registry; no files in this checkout |
| ASR-CabNois | Restricted-access in-car noise | Not used; cited as a possible source only | Explicitly not downloaded; access and license constraints recorded |
| Nexdata 531-hour in-car noise | Commercial in-car noise | Not used; public sample/metadata only | Explicitly not redistributed; commercial license recorded |
| SLURP | Spoken language understanding corpus | Path appears in example path config; no loaders/configs/reports establish use | Mention-only; not registered/downloaded |
| SpEx+ pretrained checkpoint | Model initialization asset, **not a dataset** | Transfer model initialization | README requires local checkpoint; checkpoint provenance/checksum is not captured in dataset registry |

## Dataset records

### 1. LibriSpeech

- **Nature:** approximately 1,000 hours of 16 kHz read English audiobooks, segmented and aligned; source audio derives from LibriVox. The corpus is licensed CC BY 4.0. [OpenSLR SLR12](https://www.openslr.org/12/)
- **Project role:** clean target speech and speaker enrollment for target-speaker extraction. The example data layout also describes it as clean speech for enrollment. The mixture recipe uses separate utterances from two different speakers.
- **Registered local root:** `data/raw/librispeech`.
- **Registered archives:** `train-clean-100_fixed.tar.gz` (6,387,309,499 bytes; MD5 `2a93770f6d5c6c964bc36631d331a522`, complete_verified), `dev-clean.tar.gz` (337,926,286 bytes; MD5 `42e2234ba48799c1f50f24a7926300a1`, complete), and `test-clean.tar.gz` (346,663,984 bytes; MD5 `32fa31d27d2e1cad72775fee3f4849a9`, complete). These sizes/checksums are transcription of `sources.json`, not revalidated here.
- **Subsets selected by local bootstrap code:** `train-clean-100`→train, `dev-clean`→validation, and `test-clean`→test. At most 4,000/400/400 four-second examples are generated by default, respectively. Each source is converted to mono 8 kHz; a random four-second crop is selected or zero-padded. The RNG seeds are fixed (42 plus split offsets 0/101/202).
- **Split caveat:** the bootstrap code picks target/distractor speaker IDs by deterministic arithmetic over sorted speaker IDs, but the repository does not show a persisted source-utterance manifest, seed/config snapshot, or leakage report for the *exact generated bootstrap instance*. The full LibriSpeech split-builder can create speaker splits, but that does not by itself prove its output was used by the bootstrap generator. Avoid describing the bootstrap as speaker-disjoint unless the generated manifests and leakage check are supplied.
- **Reported use:** Stage 1 LibriSpeech/LibriMix source mixtures; a benchmark report documents local bootstrap and validation/evaluation derived from those examples. The report says the 2,314-example intent slice came from FSC rather than LibriSpeech.

### 2. LibriMix and local derived mixtures

- **Official dataset recipe:** LibriMix is an open-source speech-separation corpus generated from LibriSpeech and, for noisy conditions, WHAM! noise. Its standard recipe varies number of speakers, sample rate, min/max duration mode, and clean/noisy mixture type; standard outputs include `mix_clean`, `mix_both`, and `mix_single`. The default complete Libri2Mix expansion is reported at about 430 GB by its maintainers. [LibriMix repository and citation](https://github.com/JorisCos/LibriMix)
- **Project distinction:** the local helper `scripts/data/build_local_librimix.py` is explicitly a **workspace bootstrap generator, not the official LibriMix recipe**. It generates two-speaker, fixed-length 4 s mono mixtures at 8 kHz from locally available LibriSpeech clean speech; no WHAM noise is added there. It writes separate `mix_both`, `s1` target and `s2` distractor WAVs under `train-100`, `dev`, and `test` trees (the local path convention is not proof of official LibriMix equivalence). Default row limits: 4,000 train / 400 validation / 400 test.
- **Mixing procedure:** target speaker and distractor speaker differ according to deterministic selection; each has a random utterance and crop. Source RMS is attenuated by a factor based on `max(1, RMS*4)`. Distractor gain is sampled uniformly from 0.7–1.3. Mixture and sources are peak-scaled together if mixture peak exceeds 0.98. Preserve the exact script revision, CLI arguments, and generated metadata for any paper claim.
- **Multichannel bootstrap derivative:** `build_local_multichannel.py` maps each mono mixture to four channels, with fixed gains `(1.00, 0.94, 1.06, 0.89)` and delays `(0, 3, -4, 7)` samples unless `--randomize` is set. In randomized mode, gains are 0.85–1.15, delays are −16…16 samples, and SNR is sampled from −5, 0, 5, 10, 15, 20, or 30 dB. If AISHELL noise files are discoverable under the configured `noise` root, aligned recordings are injected; otherwise the recipe proceeds without them. Very low Gaussian noise is also added. **This is simulated four-channel data, not real microphone-array recording.**
- **Manifest mapping:** `build_librimix_triplets.py` pairs `mix_both` with `s1` and an enrollment utterance for the target speaker; if no enrollment pool is provided, it reuses the target source itself. That fallback creates same-utterance enrollment/target leakage and must be checked/disabled for a formal benchmark.
- **WHAM status:** standard LibriMix provenance refers to WHAM, but the local bootstrap has no WHAM input. The example path config contains a WHAM entry, but `sources.json` does not document a local WHAM download. Describe WHAM as an official recipe dependency only when the actual generated manifests confirm the official noisy recipe was used.

### 3. AISHELL-5

- **Nature:** Chinese multi-speaker speech recorded inside a hybrid electric car, with four far-field microphones, near-field microphones worn by speakers in the training portion, TextGrid annotations, and a separate noise-only set. The official description reports 165 participants, 94 h train, 3.3 h Dev, 3.3 h Eval1, 3.58 h Eval2, and about 40 h noise. License: CC BY-SA 4.0. [OpenSLR SLR159](https://www.openslr.org/159/)
- **Project role:** real-cabin domain evaluation/adaptation, speech-rich target-absent negatives, cabin noise, and bounded target-present experiments. It has no project wake phrase labels and is not a driver-wake positive set.
- **Registered root:** `data/raw/aishell5`.
- **Registered archives:** `train_full.tar.gz` (54,598,541,447 bytes), `Dev_full.tar.gz` (1,916,430,826), `Eval1_full.tar.gz` (1,727,013,906), `Eval2_full.tar.gz` (1,885,457,834), and `noise_full.tar.gz` (13,361,773,095), all marked complete. A `train.tar.gz.partial` record is marked superseded_partial_resume_only. The committed registry does not give checksums for these archives.
- **Project processing:** the scripts refer to far-field channels `DX01C01`–`DX04C01`; train-side near-field and TextGrid data are used for target-present crop experiments. The evaluation report documents two 10-minute Eval2 sessions as a target-absent continuous speech/noise stream (1,200 s, 85.5% speech occupancy), and reports a distinct 1,200/144/144 train/validation/test target-present extraction experiment. These are project-selected subsets, not official dataset split sizes.
- **Evaluation interpretation:** AISHELL-5 far-field speakers do not match the fixed LibriSpeech enrollment speaker in the continuous negative-stream evaluator. This is a useful speech-rich false-accept stream, not an enrolled-target recall benchmark. The language/domain difference (Chinese in real cabin vs predominantly English source speech) also confounds acoustic-domain adaptation claims.
- **Data transformations:** multichannel mixture path attachment, fixed TextGrid-aligned crops, resampling/cache creation and channel selection are implemented in scripts. Report sample rate, channels, crop-selection rules, and whether near-field audio is enrollment, supervision, or both for each experiment.

### 4. CAVEMOVE

- **Nature:** multichannel audio from three cars, including impulse responses and noise across static/moving driving conditions, with compact-array and distributed-microphone configurations. The maintainers provide a Python API to combine user-provided dry speech with noise and optional ventilation or car-audio interference. [Zenodo record, DOI 10.5281/zenodo.13594243](https://doi.org/10.5281/zenodo.13594243) · [API repository](https://github.com/SPL-FORTH-ICS/CAVEMOVE)
- **License:** repository records CC BY 4.0. Confirm the exact license statement and attribution in the retrieved record/documentation before redistribution; preserve the Zenodo version in a paper.
- **Registered root/archive:** `data/raw/cavemove`; `cavemove_dataset_full.zip` (2,917,523,835 bytes; MD5 `f23b49d284878337fba4fe5169460d67`, complete_verified) and `Documentation.pdf` (7,213,365 bytes; MD5 `ef87365182bac1efe25f6c4d7befa321`, complete).
- **Project role:** add CAVEMOVE cabin noise to clean LibriSpeech-derived targets/mixtures; create multichannel cabin examples; evaluate synthetic cabin TSE, KWS, presence and intent-domain behavior. The cabin manifest builder admits only WAVs in folders named `noise` or `ventilation`; it excludes impulse responses, radio IRs, AISHELL noise, VISC and synthetic multichannel augmentation in that specific recipe.
- **Noise split method:** builder reserves one named car (default controlled by CLI) for test and shuffles remaining files into train/validation; validation ratio and seed are configurable. It selects noise files and SNRs from −5, 0, 5, 10, 15, 20 dB. This is file-level split except for held-out vehicle; document the selected vehicle and any fallback behavior.
- **Metrics in checked-in reports:** a 400-example CAVEMOVE test is reported for Stage 1 SI-SDR. The continuous synthetic speech evaluator uses Speech Commands `marvin` target clips, non-wake passenger hard negatives, and long CAVEMOVE noise beds. These are generated proxies and must be labeled synthetic, not real in-car conversations.

### 5. Google Speech Commands v2

- **Nature:** short English, single-word recordings. Version 2 has 105,829 WAV files and 35 word-folder classes in the upstream release; the standard benchmark convention separates a core vocabulary from unknown and silence handling. [TensorFlow Speech Commands documentation](https://www.tensorflow.org/datasets/catalog/speech_commands) · [upstream archive](https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz)
- **Project role:** KWS training/evaluation and positive `marvin` wake-word examples; non-wake words become target-absent/presence/KWS negatives or passenger hard negatives. It is not cabin-recorded speech.
- **Registered root/archive:** `data/raw/speech_commands_v2`; `speech_commands_v0.02.tar.gz`, 2,428,923,189 bytes, complete. `sources.json` gives no checksum or explicit license field. The data README says “CC BY”; verify the upstream archive's bundled LICENSE/README, attribution requirements, and any third-party material before publication or redistribution.
- **Project labels are task-specific:** the legacy KWS manifest builder maps `_silence_`, `_unknown_`, and ten explicit classes (`yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`) to a 12-class map. Other class folders are skipped by this builder. Separate target-KWS scripts select `marvin`; positive/negative labels therefore differ between recipes. Record the exact manifest builder and labels used for each reported model.
- **Transformations:** caching/resampling to 16 kHz occurs in several newer recipes; older configs use 8 kHz. Stage 3 baseline config is 8 kHz, 1 s segments, 36 output classes, while newer binary-KWS configurations have different mappings. Do not conflate them.

### 6. Fluent Speech Commands (FSC)

- **Nature:** 30,043 English 16 kHz single-channel command utterances from 97 speakers, covering 248 phrasings and 31 intents represented by action/object/location slots. The official split is speaker-disjoint: train 23,132 utterances/77 speakers; valid 3,118/10; test 3,793/10. FSC's published license is CC BY-NC-ND 4.0 / academic-research-only terms, despite the older repository data README and this repo's source registry labeling a Zenodo mirror CC BY 4.0. **Resolve this license discrepancy against the specific mirror/version's license before reuse, redistribution, or commercial deployment.** [Dataset description and license](https://fluent.ai/fr/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research) · [official split details](https://github.com/GrindstoneLZX/FluentSpeechCommandsDataset)
- **Project role:** Stage 4 intent classification; FSC commands are also used as realistic non-wake KWS hard negatives in some recipes.
- **Registered root/archive:** `data/raw/fluent_speech_commands`; `fluentai_full.zip`, 1,545,730,387 bytes; MD5 `625d5dfecef850443955a034d5f892b2`, complete_verified.
- **Labeling:** repository code flattens non-`none` action/object/location values into a pipe-separated intent string. It offers an official-manifest builder reading `train_data.csv`, `valid_data.csv`, and `test_data.csv`, retaining transcripts, speaker IDs, and official split labels. A report documents official counts and zero speaker overlap.
- **Evaluation caveat:** earlier local intent manifests were a 90/10 slice of `train_data.csv`, not official validation/test. The reports explicitly identify the official-split metrics separately. For a paper, cite only official-split results as held-out benchmark scores and declare the exact checkpoint/data split.
- **Domain limitation:** FSC is a smart-home command corpus, not an in-car corpus. Noise/TSE-domain FSC variants are generated or transformed experimental data; they are not official FSC benchmarks.

### 7. Vehicle Interior Sound Dataset (VISC)

- **Nature:** 5,980 vehicle-only sound clips, 3–5 s, 48 kHz WAV, eight vehicle classes (bus, minibus, pickup, sports car, jeep, truck, crossover, car); no human voice. [Zenodo record, DOI 10.5281/zenodo.5606504](https://doi.org/10.5281/zenodo.5606504)
- **Registered root/archive:** `data/raw/vehicle_interior_sound`; `VISC_Dataset_SON_full.rar`, 1,228,671,997 bytes; MD5 `a5b26d20784de4e1bfed71fa89a99987`, complete_verified.
- **Project role:** auxiliary vehicle-noise domain material only, not speech supervision. The checked-in CAVEMOVE cabin manifest builder explicitly excludes VISC. No checked-in experiment recipe establishes that VISC was used for an included benchmark, so treat it as acquired/available, not as a training input, unless cache/manifest lineage proves otherwise.
- **Provenance concern:** source page says sounds were collected from YouTube. Review source rights and the dataset's reuse terms before redistribution; do not assume CC BY solely from the repo registry without verifying the Zenodo record's rights field.

### 8. MSWC English and Hey Snips

- **Role in code:** the older generic KWS manifest builder includes at most the first 5,000 WAVs found under `mswc/en` with positive label 1, and scans `hey_snips` into the evaluation manifest as label 0. This is a surprising semantic choice for a negative set: the MSWC label selection is not filtered to the target phrase in this code, and `hey_snips`'s intended negative role needs confirmation against the exact corpus contents.
- **Acquisition:** neither corpus has a `sources.json` record. `configs/paths.example.yaml` only supplies path names. No files/manifests are present in this checkout. Therefore availability, source version, license, target phrase, split design, and actual use are **unverified**.
- **Audit action:** exclude from publication claims until an acquisition manifest records upstream URL/DOI/version, license, archive checksum, exact language/keyword filtering, and row counts by class and split.

### 9. Custom in-car corpus

- The custom manifest builder expects `wake/<speaker>/*.wav`, `negative/<speaker>/*.wav`, and optional `enrollment/<speaker>/*.wav`. It assigns splits by speaker (defaults: 15% validation, 15% test; seed `20261004`) and attaches the first enrollment clip for that speaker where available.
- It is a capability/expected data contract, not proof of an existing corpus. The project report says a complete real driver/passenger continuous corpus is not present. No participant consent, collection protocol, license/ethics record, corpus inventory, or checksums are committed.
- Do not report real custom in-car evaluation until data exists and its collection protocol, annotation, split counts, speaker disjointness and consent/ethics provenance are recorded.

### 10. Non-acquired/restricted noise sources

- **ASR-CabNois:** repo notes 5.08 h, 44.1 kHz stereo and 2.47 GB, with sign-in required; CC BY-NC-ND 4.0. The full download was not obtained. Do not describe it as used.
- **Nexdata 531-hour in-car noise:** commercial license; public repository contains sample/metadata only. Full dataset not redistributed and not used according to project report.
- **SLURP:** appears in a sample path config only; no source registry entry or in-repo use is established.

## Derived datasets and processing lineage

| Derived set / name in code | Source data | Transformation / labels | Audit notes |
|---|---|---|---|
| Local LibriMix bootstrap | LibriSpeech train-clean-100/dev-clean/test-clean | 2 speakers, mono 8 kHz, 4 s, random crops, deterministic seed, target/distractor and mixture files; default 4,000/400/400 | Not official LibriMix; no WHAM; retain exact generated file manifest and per-source speaker/utterance IDs |
| Synthetic multichannel bootstrap | Local mono LibriMix + optional AISHELL noise | 4 channels via gains/delays; optional noise SNR; Gaussian floor | Simulated spatial channels; do not describe as recorded multichannel data |
| CAVEMOVE cabin mixtures | LibriSpeech/LibriMix targets + CAVEMOVE noise/ventilation | Runtime noise addition at selected SNR, typically 16 kHz in transfer config, four channel paths/frontend | Report train/val/test vehicle IDs, noise filename lists, SNR seed and channel mapping |
| GSC `marvin` cabin mixtures | GSC `marvin` wake positives and passenger/non-wake speech + CAVEMOVE noise | Synthetic target-present/absent examples for Stage 1/presence/KWS calibration | Proxy benchmark; archive source clip IDs and mixing code/seed |
| AISHELL target-present/absent presence set | AISHELL-5 channels, near-field references, TextGrid; sometimes GSC target clips | TextGrid-aligned positive crops; target-absent sessions/noise and synthetic target additions in separate recipe | Separate real target-present speech from synthetic additions and target-absent-only evaluation |
| FSC TSE-domain/cache variants | FSC WAVs + TSE outputs/frontend + CAVEMOVE/noise conditions | Cached extracted audio/features, noise conditioning, possible train subsets | Cache-level manifests must point back to original FSC row and extraction checkpoint; keep official test out of tuning |
| KWS hard-negative/hybrid sets | GSC, FSC commands, AISHELL speech, CAVEMOVE, possibly MSWC/Hey Snips | Binary wake/non-wake labels, cache merging and augmentation | These are project-defined datasets; publish composition, counts, and labeling rules, not just source corpus names |

## Stage-to-data map

| Stage | Sources supported by code/reports | Research interpretation |
|---|---|---|
| Stage 1 target-speaker extraction | LibriSpeech-derived two-speaker mixtures; CAVEMOVE cabin noise; AISHELL-5 adaptation diagnostics | Synthetic paired supervision is the principal reproducible target/mixture source. Report real AISHELL results separately from synthetic CAVEMOVE. |
| Stage 2 target presence | CAVEMOVE post-TSE examples; GSC `marvin` target positives; AISHELL-5 target-absent/noise; hard-negative mixtures; bounded AISHELL target-present crops | A presence score is not equivalent to correct identity under arbitrary real cabin conditions. Dataset class balance and operating threshold must be reported. |
| Stage 3 wake-word detection | GSC, extracted/TSE caches, CAVEMOVE augmentation, AISHELL-5 speech-rich negatives, FSC command hard negatives; possible MSWC/Hey Snips legacy paths | Exact word, positive source and all negative-source compositions vary by config; specify them per experiment. |
| Stage 4 intent | FSC official and local/bootstrap splits; CAVEMOVE/TSE-domain transformed FSC | Clean FSC official test and project-generated noisy/TSE tests answer different questions. No fully aligned real in-car command test is established. |

## Reproducibility and audit gaps to resolve before submission

1. **No data files in this checkout.** Attach immutable manifests with paths relative to a declared data root, row counts, archive hashes and derived-file hashes; re-run file-existence and hash checks on the actual experiment machine.
2. **FSC license mismatch.** Registry says CC BY 4.0; upstream publisher describes CC BY-NC-ND 4.0/academic-only restrictions. Verify the exact Zenodo mirror's archive-level license and cite it accurately before any release or commercial use.
3. **AISHELL archive hashes absent.** Record SHA-256 (preferred) for every tarball and extracted inventory. The current archive names/sizes are insufficient to verify identity.
4. **Speech Commands archive license/checksum absent.** Preserve its bundled license/readme and cite archive release/version.
5. **Unregistered datasets/paths.** MSWC, Hey Snips, WHAM, SLURP and custom in-car lack full provenance records. Either add complete records or remove them from claimed experiment inputs.
6. **Bootstrap/official dataset distinction.** Mark locally generated LibriSpeech mixtures as synthetic project data, not official LibriMix. Record exact command, code commit, seed, source file IDs, SNRs, crop offsets and train/validation/test speaker overlap.
7. **Leakage and enrollment handling.** Verify target enrollment clips are separate utterances (and preferably sessions) from target clips. The triplet builder has a fallback that can reuse the target utterance as enrollment.
8. **Noisy-data lineage.** Every cached waveform/embedding should include source row ID, source checksum, sample rate, channel order, crop offsets, resampling method, gain/normalization, noise file/vehicle/condition/SNR, extraction checkpoint hash, and creation code revision.
9. **Split integrity.** Keep official FSC splits immutable; freeze held-out CAVEMOVE vehicle and AISHELL sessions before tuning. Report speaker, session, vehicle and source-recording disjointness as applicable.
10. **VISC/other noise use is unclear.** Presence in an acquisition registry does not establish actual experiment use. Trace cache/manifests or label it “available but not used.”
11. **Model asset provenance.** Record the exact SpEx+ source URL/release, checkpoint SHA-256, license/terms, and whether the checkpoint was trained on LibriSpeech overlapping with this project's train/eval sources.
12. **Ethics and data governance.** For any future custom in-car speech, document consent, participant privacy, vehicle/session metadata, retention, labeling and exclusion rules.

## Minimum experiment record recommended for the paper appendix

For each result, preserve a machine-readable record with: dataset release and license; raw archive SHA-256; source split and source recording/speaker IDs; generated manifest SHA-256 and per-class counts; sample rate, bit depth/channel count and resampling; crop/segment length and crop policy; target/enrollment separation; mixture/noise recipe, SNR distribution and seed; train/validation/test grouping and leakage checks; preprocessing/cache/checkpoint hashes; exclusions; and exact config/code revision. State whether the result is **official benchmark data**, **locally generated synthetic data**, or **real-cabin data**.

## Citation starter list

- LibriSpeech: V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, “LibriSpeech: An ASR corpus based on public domain audio books,” ICASSP 2015. Dataset landing page: [OpenSLR SLR12](https://www.openslr.org/12/).
- LibriMix: J. Cosentino et al., “LibriMix: An Open-Source Dataset for Generalizable Speech Separation,” 2020. [Repository and BibTeX](https://github.com/JorisCos/LibriMix).
- AISHELL-5: Y. Dai et al., “AISHELL-5: The First Open-Source In-Car Multi-Channel Multi-Speaker Speech Dataset for Automatic Speech Diarization and Recognition,” Interspeech 2025. [OpenSLR SLR159](https://www.openslr.org/159/).
- CAVEMOVE: N. Stefanakis et al., “CAVEMOVE: An acoustic database for the study of voice-enabled technologies inside moving vehicles,” dataset release v1, 2024. [Zenodo DOI](https://doi.org/10.5281/zenodo.13594243).
- Fluent Speech Commands: L. Lugosch et al., “Speech Model Pre-training for End-to-End Spoken Language Understanding,” Interspeech 2019. [Dataset description/license](https://fluent.ai/fr/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research).
- Vehicle Interior Sound Dataset: E. Akbal, T. Tuncer, and S. Dogan, “Vehicle Interior Sound Classification Based on Local Quintet Magnitude Pattern and Iterative Neighborhood Component Analysis,” *Applied Artificial Intelligence*, 36(1), 2137653, 2022. [Dataset DOI](https://doi.org/10.5281/zenodo.5606504).
- Speech Commands v2: cite the upstream dataset description/archive and the bundled license/README from the exact archive version used: [TensorFlow dataset documentation](https://www.tensorflow.org/datasets/catalog/speech_commands).

## Repository evidence paths

- Source registry: `data/sources.json`
- Original data notes: `data/README.md`
- Path template: `configs/paths.example.yaml`
- Local mixture recipes: `scripts/data/build_local_librimix.py`, `scripts/data/build_local_multichannel.py`, `scripts/data/build_librimix_triplets.py`
- Cabin/noise recipes: `scripts/data/build_cavemove_cabin_manifests.py`, `scripts/data/build_presence_aishell_target_manifest.py`, `scripts/data/build_presence_gsc_aishell_manifest.py`
- KWS/intent recipes: `scripts/data/build_kws_manifest.py`, `scripts/data/build_kws_hardneg_manifest.py`, `scripts/data/build_intent_manifest.py`, `scripts/data/build_fsc_official_manifests.py`
- High-level reported use/limitations: `reports/project_benchmark_2026-10-03.md`, `reports/base_project_report_2026-10-04.md`
