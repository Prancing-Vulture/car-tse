# A Staged Target-Speaker Voice Assistant for Vehicle Cabins: Repository Walkthrough and Evidence Review

**Document type:** Journal-style technical report and plain-language implementation guide  
**Repository:** `car-tse`  
**Evidence snapshot:** 5 October 2026  
**Intended audience:** readers who need to understand the system without specialist audio or machine-learning vocabulary

## Abstract

This report explains the `car-tse` repository as an end-to-end research prototype for an in-car voice assistant. Its main idea is to process cabin audio in a sequence: use an enrollment recording to extract the intended speaker, decide whether that person is speaking, detect the wake word “marvin”, classify the following command, and map the predicted class to an action. The repository contains data preparation, model training, evaluation, and runtime code for these steps. It uses a pretrained SpEx+ target-speaker extraction model, optional multichannel cabin processing, a target-presence classifier, a wake-word classifier, and either a compact recurrent intent model or a wav2vec 2.0 intent model. The report follows the sound and labels through the system, describes what training teaches each block, summarizes the evidence recorded in the project reports, and distinguishes established results from experiments and unsupported deployment claims. Current evidence shows useful improvement on generated cabin mixtures and strong clean command classification, but does not establish reliable driver-versus-passenger behavior or complete real-cabin wake-to-action accuracy. The purpose is both descriptive and cautionary: code capability, a successful isolated-stage metric, and a validated real-world assistant are different things.

**Keywords:** target-speaker extraction; keyword spotting; in-car voice assistant; audio pipeline; reproducibility; plain-language system description

## 1. Introduction

A car cabin can contain a driver, passengers, road and ventilation noise, and sound played through the speakers. A basic voice assistant hears all of these together. This project tries to answer a more specific question: *Can the system act only when the enrolled person says the wake word, then understand that person's next command?*

The repository addresses this as a cascade of smaller decisions rather than one large model. The cascade is easier to inspect and train stage by stage, but errors can accumulate. If the extractor leaves a passenger audible, later stages may mistake that voice for the driver. If the presence gate is set too high, it may block the driver. If the wake detector fires incorrectly, the system may capture and classify unrelated audio.

This report is a guided account of the code and the checked-in project evidence. It is not a claim that every possible configuration has been run. There are many alternative YAML configurations and experimental checkpoints; the exact model and data used for any result must be taken from that experiment's configuration and saved artifact.

### 1.1 The whole system in one view

```text
Microphones (one or several) + optional speaker-playback reference
                         │
                         ▼
       Cabin frontend (optional echo/spatial processing)
                         │
                         ▼
 Stage 1: extract speech resembling the enrolled voice (SpEx+)
                         │
                         ▼
          Stage 2: is target speech present? (gate)
                         │ yes
                         ▼
       Stage 3: did the target say “marvin”? (wake word)
                         │ yes: arm command capture
                         ▼
    Next accepted speech window → Stage 4: classify intent
                         │
                         ▼
                  Map label to action/reply
```

The diagram describes the intended runtime path. Some experiments omit a stage, use a different model variant, or evaluate generated clips instead of a live stream. The controller is in [in_car_pipeline.py](../src/car_tse/pipeline/in_car_pipeline.py), and the action mapping is in [actions.py](../src/car_tse/pipeline/actions.py).

## 2. Repository and method

### 2.1 How to read the repository

- `src/car_tse/models/` contains the neural-network blocks.
- `src/car_tse/frontend/` contains multichannel cabin input processing.
- `src/car_tse/pipeline/` connects the blocks for inference and maps outputs to actions.
- `src/car_tse/data/` and `src/car_tse/io/` load audio and JSON-lines manifests.
- `src/car_tse/losses/` defines training objectives.
- `scripts/data/` creates manifests and cached audio/features.
- `scripts/train/` trains the separate model stages.
- `scripts/eval/` measures stages and assembled paths.
- `configs/` selects paths, model sizes, thresholds, and training/evaluation settings.
- `checkpoints/`, `data/`, `output/`, and `logs/` are expected to hold large local assets or generated results; availability depends on the machine and run.
- `reports/` records benchmark snapshots and experiment interpretation.

`README.md` describes setup and common commands. `data/DATASET_AUDIT.md` records data provenance caveats. The project is configured through YAML: [config.py](../src/car_tse/config.py) loads a stage file and resolves the data, manifest, checkpoint, and log roots from a paths file. If a local `configs/paths.yaml` is absent, the loader falls back to `configs/paths.example.yaml`; users must check that the resulting paths point to their actual data.

### 2.2 Audio rows and alignment

Most training and evaluation datasets are described by JSON-lines manifests: one JSON object per example. The common fields include mixture audio, enrollment audio, clean target audio, sample rate, speaker ID, whether the target is present, transcript, intent label, optional multichannel paths, optional playback reference, and extra metadata. See [manifest.py](../src/car_tse/io/manifest.py).

The mixture is the sound the system receives. The target is the clean desired speaker used as a training answer. Enrollment is a separate sample used to tell the extractor whose voice to favor. Keeping mixture, target, cabin channels, and playback reference on the same time crop matters: if their crop offsets differ, the training answer no longer lines up with the input. The segment helpers use a shared crop offset for aligned signals; training can select random crops and evaluation uses fixed or centered crops. Short signals are padded with zeros. See [segment.py](../src/car_tse/data/segment.py) and [datasets.py](../src/car_tse/data/datasets.py).

The audio helpers read mono or multichannel WAV and can resample with linear interpolation ([audio.py](../src/car_tse/io/audio.py)). This is a simple resampler; it should not be silently described as a high-quality studio resampling method. A paper should report the exact sample rate, crop length, channel order, resampling, gain treatment, and cache lineage for each result.

## 3. System stages

### 3.1 Cabin frontend: prepare the microphone signals

When enabled, `CabinFrontend` receives a tensor of microphone channels and, optionally, a recording of the audio played by the car. The playback reference is a clue to what sound may have leaked from the car speakers back into the microphones. A small learned branch estimates a correction from the microphones and reference. A second learned branch combines the corrected channels into one signal for the speech extractor. If there is no playback reference, the code supplies silence for that input.

The frontend can also scale the input toward a configured average loudness. Its last layers start at zero, so the initial behavior is a simple mean across microphone channels rather than an arbitrary random transformation. This is a safe starting point for a pretrained speech model. It is still a learned signal-processing front end, not evidence of measured echo-cancellation performance. The repository reports no dedicated real-playback echo benchmark, and the microphone/channel layout must match the configured frontend.

### 3.2 Stage 1: extract the enrolled speaker

The user supplies an enrollment clip, ideally containing the voice the system should follow. Stage 1 receives that clip and the cabin mixture and produces an estimate of the enrolled speaker's speech. In ordinary terms, the enrollment is a voice example; the model compares the sound in each mixture with that voice and tries to keep matching speech while reducing other sound.

The main backend is an adapter around SpEx+, a pretrained target-speaker extraction network. The repository has an 8 kHz vendor path and a 16 kHz transfer path. Stage 1 can run directly on a mono mixture, or after the cabin frontend on multichannel audio. Its result object also contains auxiliary network outputs and speaker-classification information; the runtime principally forwards the extracted waveform.

Training examples pair a mixture, enrollment, and clean target. The main loss rewards the extracted waveform when it resembles the clean target in scale-adjusted signal-to-distortion ratio (SI-SDR). SI-SDR measures how much of the output can be explained by the target compared with leftover error; higher is better. Auxiliary SpEx+ outputs can also contribute to the loss, and an optional speaker classification loss encourages voice identity features to distinguish training speakers. Target-absent examples can instead receive a penalty for producing output energy. These pieces are implemented in [stage1.py](../src/car_tse/losses/stage1.py); training, staged freezing, and optimizer groups are in [train_stage1_spex.py](../scripts/train/train_stage1_spex.py).

The training plan may freeze parts of the pretrained model at first and later allow selected parts to adapt. The cabin frontend can be trained with the extractor. This lets the system learn to handle cabin-like signals while retaining useful pretrained speech behavior. The exact schedule is config-dependent, so the paper should name the chosen YAML file and checkpoint rather than treating all Stage 1 variants as one model.

**Known boundary:** a good SI-SDR score says that the estimated waveform is closer to the clean target by that signal measure. It does not prove that the correct person alone remains, that words are intelligible, or that a command will be understood. Passenger rejection needs speaker-consistent overlapping-speaker tests.

### 3.3 Stage 2: decide whether target speech is present

The extractor may produce some sound even when the enrolled person is absent. Stage 2 is a gate: it examines the extracted waveform and returns a target-present score. If the score is below its configured threshold, the controller stops before wake-word recognition. A threshold is a strictness setting: raising it usually rejects more questionable audio but can also reject real target speech.

Two implementations exist. `PresenceHead` is a small waveform convolution model that compresses the signal into one yes/no score for the whole chunk. `FramePresenceHead` first converts audio to log-mel features (a compact picture of changing sound frequencies), predicts target activity in short frames, and aggregates the top-scoring fraction into a chunk decision. It can preserve evidence of a brief active part inside a longer chunk. Its training labels are weak activity labels derived from the clean target's frame energy; absent examples are all-negative. This means the frame head is trained to locate target activity in the supplied target reference, not to solve every possible driver/passenger identity case by itself.

Training holds Stage 1 fixed in the ordinary presence trainer, runs examples through the extractor, and learns the small head from target-present labels. The loss combines binary classification error with a penalty against energetic extracted outputs for absent examples; the frame variant uses weighted frame classification and a small smoothness term. See [presence_head.py](../src/car_tse/models/presence_head.py), [presence.py](../src/car_tse/losses/presence.py), and [train_stage2_presence.py](../scripts/train/train_stage2_presence.py).

The presence score's operational threshold is critical. Project reports show that a permissive threshold can pass many negative speech examples, while a high threshold can suppress desired speech in a continuous proxy. Therefore isolated accuracy or F1 alone is not enough. A fair gate assessment needs target-present recall and target-absent false accepts at a jointly selected operating point on independent, representative audio.

### 3.4 Stage 3: detect the wake word

The wake word in the current system is “marvin”. The ordinary `KWSModel` means keyword spotting model. It converts a short waveform to log-mel features, processes that sound map with small convolution layers, and assigns class scores. In the newer pipeline configuration this is often a binary decision—wake word or not—though older recipes can use more classes. The model sees the extracted audio, not the raw cabin mixture.

The dataset builder and config decide which clips count as positive wake examples and which count as negative examples. Recipes use sources such as Google Speech Commands `marvin`, ordinary non-wake words, Fluent Speech Commands commands, extracted/TSE caches, CAVEMOVE cabin augmentation, and AISHELL-5 speech-rich negatives. These are not interchangeable datasets: some are generated mixtures or transformed copies, and some are real cabin speech. The exact recipe must be described per checkpoint.

An experimental `TargetSpeakerKWSModel` adds enrollment and extracted-speaker embeddings to the acoustic features. It explicitly gives the wake classifier a voice-comparison signal, which could help if another person says the same wake word. The normal compact KWS model does not itself know the speaker's identity; it relies on Stage 1 and Stage 2 to provide target selectivity. Secondary KWS scores may optionally be fused with the main score by a configured rule.

At runtime, the controller can smooth scores across calls. For example, it can require a number of threshold crossings inside a short window and then wait a few calls before accepting another wake. This helps avoid repeated decisions from one sustained sound, but temporal confirmation does not fix a poorly selective model. `update_kws_state` keeps recent scores, counts threshold crossings, and applies a refractory period. KWS architecture is in [kws_model.py](../src/car_tse/models/kws_model.py); training is in [train_stage3_kws.py](../scripts/train/train_stage3_kws.py).

The useful deployment measure is false accepts per hour together with target wake recall, not only clip-level F1. A test made of isolated one-second clips does not fully reflect a continuous stream where many opportunities exist for a false event.

### 3.5 Runtime state: separate wake from command

The pipeline is stateful. Before a wake, accepted speech is passed to KWS. Once KWS accepts a wake, the state becomes “awaiting command”. The wake frame itself is not classified as a command. On a later accepted target-speech call, the pipeline crops or zero-pads the extracted audio to the configured intent duration and sends it to Stage 4. A timeout measured in incoming calls clears the waiting state if no command arrives. This separation is an important implementation detail: the system does not intentionally treat the word “marvin” as the command.

The controller also caches embeddings where appropriate, can use a separate speaker similarity gate, supports optional playback-reference state, and can clear keyword history when the presence gate rejects a chunk. If the system is awaiting a command, it uses a command-presence threshold and can allow short gaps until timeout. Input lengths and thresholds are configuration parameters. This is chunk-oriented state management, not by itself a complete microphone streaming stack with demonstrated buffer scheduling or hard real-time guarantees.

### 3.6 Stage 4: turn the command into an intent

Stage 4 maps a fixed-length extracted command waveform to one of 31 Fluent Speech Commands (FSC) intent classes in the current main recipe. An intent is the task meaning, such as activating or deactivating a device; it is not a word-for-word transcript. The legacy `IntentClassifier` turns log-mel sound features into a sequence model using convolution and a bidirectional LSTM. The more recent `Wav2VecIntentClassifier` uses a pretrained wav2vec 2.0 speech encoder, averages its frame representations, then passes the result through a small classification head.

The wav2vec path normalizes each fixed-length waveform by subtracting its mean and dividing by its standard deviation (with a lower bound to avoid division by nearly zero). It uses a pretrained encoder checkpoint and a separately trained intent head. Training can cache encoder features to speed head experiments; fine-tuning variants may update the last encoder layers. It is still a large model: prior reports estimate the encoder at roughly 378 MB of checkpoint data. The implementation is in [wav2vec_intent.py](../src/car_tse/models/wav2vec_intent.py), and the training recipe is in [train_stage4_intent_wav2vec.py](../scripts/train/train_stage4_intent_wav2vec.py).

An argmax selects the class with the highest score. The controller maps that class ID through an optional label map to an action string. A `reply:` mapping can also point to a response WAV. The checked-in action mapping is a thin dispatch layer, not a vehicle-control safety system: actual safe vehicle integration, authorization, and action execution behavior are outside the demonstrated model evaluation.

FSC is a command-understanding corpus, not in-car driver/passenger data. High accuracy on its official clean test indicates the classifier can learn those classes under that evaluation; it does not establish the accuracy of the complete chain after extraction, wake detection, and cabin noise.

## 4. Data preparation and training flow

### 4.1 Sources and their roles

The project combines several types of evidence:

| Source | Role in this repository | What it does not establish |
|---|---|---|
| LibriSpeech and locally generated LibriMix-style mixtures | Clean speaker examples and paired target/mixture data for extraction | The local generation is not automatically the official LibriMix benchmark |
| CAVEMOVE | Cabin impulse/noise material; used to create cabin-like mixtures and evaluate synthetic cabin cases | A generated mixture is not a real driver/passenger recording |
| AISHELL-5 | Real multi-microphone in-car speech/noise, including speech-rich negative-stream diagnostics and selected adaptation crops | The evaluated absent-target stream has no enrolled-driver wake positives; Chinese speech differs from many English training clips |
| Google Speech Commands v2 | Short wake-word and non-wake examples, including `marvin` recipes | Not recorded in a car and not a same-driver wake-command interaction corpus |
| Fluent Speech Commands | 31-class intent labels and utterances; official speaker-disjoint splits are available | Not in-car audio; clean official accuracy is not end-to-end cabin accuracy |
| Vehicle Interior Sound material | Possible vehicle-noise augmentation source | Its presence on disk does not prove it was used in a reported experiment |
| Custom in-car manifest format | Supports speaker-grouped wake, negative, and enrollment files | A manifest builder is not evidence that a complete, consented real cabin corpus exists |

The detailed provenance and licensing warnings are in [DATASET_AUDIT.md](../data/DATASET_AUDIT.md) and [sources.json](../data/sources.json). A source being listed or downloaded does not prove it entered a particular checkpoint's training set. Exact manifests and cache lineage should be preserved with results. The repo's own audit flags unresolved provenance or license questions for some mirrored/acquired data; publication and redistribution decisions should use the exact upstream data terms.

### 4.2 Typical training sequence

1. Prepare source data and JSONL manifests with target/enrollment/mixture paths and labels.
2. Create aligned crops, optional multichannel versions, and derived mixtures or caches.
3. Train or adapt Stage 1 on paired target extraction examples.
4. Freeze the chosen Stage 1 checkpoint and use its extracted outputs to train Stage 2 presence.
5. Build wake/non-wake examples in the same extracted-signal domain expected at runtime and train Stage 3.
6. Train Stage 4 on labeled command intent examples, optionally after extraction/noise transformations.
7. Select thresholds and temporal settings on validation data, then assess held-out and continuous tests.

This is a common workflow, not one guaranteed command sequence for every experiment. There are multiple configurations and orchestration scripts, and some workflows described by README may not be included in every checkout state. The four direct trainers accept config files; the config names determine such things as sample rate, segment length, model type, data manifest, checkpoint, optimizer, and training schedule. Evaluation scripts separately choose checkpoint bundles and thresholds.

### 4.3 Avoiding misleading comparisons

Training and test examples should be separated by speaker, recording, session, or vehicle where appropriate. The reports document a speaker-disjoint official FSC split and selected project checks, but not every historical cache or experiment necessarily uses the same split policy. A cached waveform also inherits the model and preprocessing used to create it. Mixing cache versions can create a mismatch between what a head learned and what it receives at runtime. The repository has corrected specific conditioning and manifest issues documented in its benchmark reports; papers should name the exact artifact and preprocessing rather than quote a result without lineage.

## 5. Evaluation measures and current evidence

This section summarizes repository reports available at the evidence snapshot. It is not a fresh re-run. Results should be cited together with the report/config/checkpoint that produced them, and results from different splits should not be compared as if they were one controlled experiment.

### 5.1 Stage-specific evidence

**Stage 1, generated cabin mixtures.** The base status report gives a 400-example held-out CAVEMOVE evaluation: mean SI-SDR rises from −2.41 dB for the input mixture to 8.57 dB for extracted output, a mean improvement of 10.98 dB. This supports useful extraction improvement on that synthetic protocol. It does not by itself measure speaker confusion, intelligibility, or action correctness.

**Stage 1, real-cabin diagnostic.** The same report gives an AISHELL-5 diagnostic around −31.22 dB output SI-SDR for evaluated target-present windows, compared with about −37.31 dB for the far-field input baseline. A later input-scale sweep improved the AISHELL score while leaving CAVEMOVE nearly similar, but also changed false-event behavior in a target-absent stream. This exposes domain and conditioning sensitivity; it is not a single universally superior normalization setting.

**Stage 2.** Reports show a threshold-sensitive gate. On one imbalanced validation split, threshold 0.50 yielded 0% false rejection but 96.25% false-accept rate; a threshold around 0.90 reduced false accepts but also rejected more positives. A framewise head is implemented, but the most important unresolved issue is joint calibration using representative target-present and target-absent continuous audio.

**Stage 3.** Project reports describe strong results on a narrow held-out TSE-domain KWS set, including roughly 97.4% F1 at a selected higher-precision operating point in one benchmark. A broader corrected post-TSE hybrid set has lower reported F1 (about 79.7%), and AISHELL-augmented variants need stream threshold calibration. These are dataset- and threshold-specific numbers; the narrow and broad values are not contradictory if the test distributions differ.

**Stage 4.** The base report records 96.60% top-1 on official FSC validation and 99.31% top-1 on official FSC test for the wav2vec candidate, plus lower scores on local raw cabin/TSE-domain validation sets (about 90.7% and 90.5%). The official clean figures show strong FSC classification under that split, not real cabin command success. The project explicitly does not claim this as a reproduction of a differently architected published 99.7% system.

### 5.2 Whole-chain checks

One AISHELL-5 target-absent continuous check covers 1,200 seconds of real multi-channel speech/noise with approximately 85.5% speech occupancy. In the reported configuration, the full presence-gated path produced zero smoothed events per hour on that stream. This is useful negative-stream evidence, but it contains no target wake word and therefore cannot measure wake recall.

The continuous CAVEMOVE scenes combine clips from different corpora, cabin noise, and synthetic target/passenger roles. The wake and command are not verified to come from the same speaker. In one reported proxy, threshold changes moved target wake recall from zero to one of four; there was no successful final intent. Another passenger-only stress case produced hundreds of false actions per hour equivalent. These small generated proxies are diagnostics, not real-world rates. The cross-corpus speaker mismatch is especially important because Stage 1 is supposed to reject audio from a non-enrolled speaker.

### 5.3 Latency and footprint

The project benchmark reports approximate model-forward time for one second of audio of 69.95 ms on a GPU and 275.34 ms on CPU for Stage 1 + KWS + intent in a particular configuration. These sums omit microphone capture, buffering, resampling, overlap/look-ahead, pipeline scheduling, and action execution. Stage 1 uses a non-causal model with a three-second context in the reported evaluation, so these numbers do not prove end-to-end live latency below 100 ms.

The approximate checkpoint footprint is 423.7 MB, dominated by the wav2vec encoder (about 377.6 MB). This is a research prototype footprint rather than a compact automotive DSP deployment. The repository does not report a validated quantized edge build, thermal/power evaluation, or automotive hardware result.

## 6. Discussion

### 6.1 What the system design contributes

The main engineering strength is the explicit sequence and inspectability. Extraction, presence, wake recognition, and intent can be evaluated independently, and thresholds expose where the cascade blocks or allows an example. The controller also cleanly separates the wake event from later command capture. The multichannel frontend offers a place to learn channel combination and playback-reference conditioning, and the data tooling supports generated cabin stress tests and manifest-driven experiments.

### 6.2 Why end-to-end behavior is harder than adding stage scores

Stage success is conditional. Stage 4 only sees commands that Stage 1 preserved, Stage 2 accepted, and Stage 3 correctly armed. Errors at an early gate prevent later success; leakage can let an unrelated voice through. Multiplying isolated accuracies would still be misleading because stages are not independent and are often tested on different datasets. The system must be measured as one continuous interaction with same-driver enrollment, wake, and command labels, plus labeled passenger speech and cabin conditions.

### 6.3 Claims supported by the current record

- The repository implements a coherent staged research pipeline for target-conditioned cabin voice interaction.
- On a reported generated CAVEMOVE test, Stage 1 substantially improves SI-SDR over the mixture.
- The wav2vec intent candidate performs strongly on the official clean FSC test split.
- The code includes a real AISHELL-5 speech-rich target-absent evaluation path, which is useful for false-trigger investigation.
- The runtime state machine keeps wake detection and post-wake intent classification in separate calls/states.

### 6.4 Claims the record does not support

- Reliable real-world driver-only operation with passengers speaking.
- A measured end-to-end wake-to-correct-action accuracy on same-driver real cabin conversations.
- Production readiness, state-of-the-art status, or low false-trigger rates in general driving.
- Effective real echo cancellation, spatial seat separation, or robustness to wind, open windows, mic failure, and all reverberant conditions.
- Sub-100 ms full-system live latency or deployment on automotive DSP hardware.
- Safe vehicle action execution; the code's action mapping is only a software dispatch representation.

The most important missing benchmark is a consented, well-labeled continuous cabin recording set in which the enrolled driver supplies enrollment, wake word, and commands in the same conditions while passenger speech, road/HVAC noise, and playback are represented. It should report per-speaker/session splits, target wake recall, false accepts per hour, command intent accuracy conditional on a wake, complete wake-to-action success, and end-to-end latency. The threshold should be chosen before final test evaluation.

## 7. Reproducibility guide

To reproduce or extend a result, record at least:

1. Git revision and exact YAML configuration for each stage.
2. Dataset release, source archive checksum, license/provenance, and actual data-root version.
3. Manifest path/hash, row counts, class counts, speaker/session/vehicle split rules, and leakage checks.
4. Audio sample rate, channel order, resampling, crop/segment policy, loudness normalization, noise type/SNR, and random seed.
5. Cache creation code, source manifest, upstream checkpoint hash, and any feature/waveform conditioning.
6. Exact checkpoint files and their hashes, model backend, thresholds, KWS smoothing, and command timeout.
7. Held-out metrics including class-level outcomes, target recall, false accepts per hour, and continuous-stream duration.
8. Whether each test is official benchmark data, locally generated synthetic data, or real cabin data.

Setup instructions and individual commands are summarized in [README.md](../README.md). Typical stage entry points are `scripts/train/train_stage1_spex.py`, `scripts/train/train_stage2_presence.py`, `scripts/train/train_stage3_kws.py`, and `scripts/train/train_stage4_intent_wav2vec.py`; consult each script's `--help` and the selected config. The evaluation programs under `scripts/eval/` have different input and checkpoint requirements. A successful unit test or training run checks software behavior, not scientific validity of the dataset split or the production claim.

## 8. Plain-language glossary

| Term | Simple meaning |
|---|---|
| Enrollment | A short recording that tells the system which person's voice to follow |
| Mixture | The combined sound captured in the cabin |
| Target | The desired speaker's clean speech used as a training answer |
| Target-speaker extraction (TSE) | Keeping speech that resembles the enrolled voice while reducing other sound |
| Frontend | The first processing block that prepares raw microphone signals |
| Log-mel features | A compact picture of how sound energy changes across pitch and time |
| Presence gate | A yes/no-style check that decides whether target speech appears to be present |
| KWS / keyword spotting | Detecting a chosen short word such as “marvin” |
| Intent | The meaning or task category of a spoken command |
| Checkpoint | A saved set of learned model values |
| Manifest | A list of examples and their audio paths and labels |
| Cache | A saved intermediate audio/features result so later runs can reuse it |
| Threshold | A score cutoff controlling how readily a stage accepts an example |
| False accept / false trigger | The system says “yes” when it should have said “no” |
| False reject / miss | The system rejects a real target event |
| SI-SDR | A score comparing an estimated voice signal with a clean target; higher means closer by this measure |
| F1 | A summary that balances how many detected positives are correct with how many positives were found |
| False accepts per hour | The number of false detections during a given amount of continuous audio, scaled to one hour |
| Causal / non-causal | Whether a model can decide using only present/past sound, or also needs future context |
| Speaker-disjoint split | Training and test contain different people, reducing memorization of familiar voices |

## 9. Conclusion

`car-tse` is a research prototype for a target-speaker-aware in-car voice assistant. It starts with microphone audio and a reference voice, estimates that speaker's speech, checks for target activity, listens for “marvin”, captures later command audio, predicts an intent, and maps that intent to an action label. Each stage has its own training scripts, data recipes, configurations, and evaluation tools.

The evidence currently supports a promising extraction result on generated cabin audio and strong clean FSC intent classification. It also exposes the core unresolved problem: target-speaker selectivity and calibrated operation in continuous cabin speech. The repository does not yet establish a reliable real driver/passenger wake-to-action system. A comprehensive paper should present the implemented pipeline and stage-specific evidence, label generated proxies honestly, and treat the missing same-driver real-cabin benchmark as a central limitation rather than infer success from isolated model scores.

## Repository references

- Architecture/runtime: `src/car_tse/pipeline/in_car_pipeline.py`, `src/car_tse/models/stage1_system.py`, `src/car_tse/frontend/cabin_frontend.py`
- Model stages: `src/car_tse/models/presence_head.py`, `src/car_tse/models/kws_model.py`, `src/car_tse/models/wav2vec_intent.py`
- Data and objectives: `src/car_tse/data/datasets.py`, `src/car_tse/data/segment.py`, `src/car_tse/io/manifest.py`, `src/car_tse/losses/`
- Training/evaluation: `scripts/train/`, `scripts/eval/`
- Prior evidence: `reports/base_project_report_2026-10-04.md`, `reports/project_benchmark_2026-10-03.md`, `reports/continuous_pipeline_ablation_2026-10-04.md`, `reports/continuous_full_pipeline_proxy_2026-10-04.md`
- Data notes: `data/DATASET_AUDIT.md`, `data/sources.json`
