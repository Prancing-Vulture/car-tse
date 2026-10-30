# Apples-to-apples comparison plan: SLURP, Paranjape, and Hussein

Date: 2026-10-17

## Important correction before comparing

### SLURP split caveat (verified 2026-10-18)

The official SLURP train/development/test files must be preserved for the
primary benchmark, but they must **not** be described as speaker-disjoint by
default.  The released metadata links recordings to speaker IDs, and an audit
of the official JSONL files shows speaker IDs shared across the published
splits.  Therefore the paper comparison will report two distinct protocols:

1. **Official SLURP protocol:** exact released train/development/test files,
   with close-talk and distant recordings reported separately.  This is the
   apples-to-apples protocol against published SLURP baselines, but it is not
   a speaker-independent generalization test.
2. **Speaker-disjoint stress protocol:** a deterministic derived split made by
   assigning complete speaker IDs to train/development/test.  This is the
   appropriate protocol for claims about unseen drivers, but its scores must
   not be substituted for the official SLURP score.

The current audit found 11,514/2,033/2,974 annotation rows and
50,628/8,690/13,078 real recordings in train/development/test, respectively,
matching the published distribution.  It also found 166/137/142 speaker IDs
present within the individual splits and substantial cross-split overlap.
These counts are recorded from the local official annotations; the audio
archives are stored outside this repository because the released audio is
CC BY-NC 4.0.

The two vehicle-assistant papers should not be treated as SLURP baselines without checking their methods:

- Paranjape et al., *Voice-Based Smart Assistant System for Vehicles using RASA* (arXiv:2312.01642), describes a RASA vehicle assistant for navigation, calls, weather/news, and music. The public abstract does not identify SLURP as its dataset. The workspace handoff records a reported 93.67% intent result on 300 test inputs and the wake word “coffee”; those details must be extracted from the paper's methods/results before quoting them.
- Hussein, Karan & Kurnaz, *Enhancing Driving Control via Speech Recognition Utilizing Influential Parameters in Deep Learning Techniques*, Electronics 14(3):496 (2025), describes a DNN pipeline with noise removal, feature extraction, and classification. The workspace handoff records approximately 687 Google Speech Commands clips and 93% accuracy, but the dataset/split should be verified from the article before using it as a reference.
- SLURP is a separate English spoken-language-understanding resource: about 72k recordings, 18 scenarios, 46 actions, entity annotations, multiple speakers, close-talk and far-field recordings. Its audio is research/non-commercial under the released license, so it must be kept out of any commercial release unless separately licensed.

## Is our current task easier?

There are two different comparisons:

1. **Intent-only difficulty:** our current FSC intent test is easier than SLURP. FSC is a closed, compact, templated 31-intent task and our clean intent score is about 99.3%. SLURP is broader and lexically richer, with more domains, actions, and entity/slot structure. SLURP should therefore be expected to lower intent and slot scores.
2. **Full assistant difficulty:** our current task is harder than either paper's headline classifier result because it requires wake detection, target-speaker attribution, passenger rejection, action suppression, and response timing. The current proxy is still easier than a real car because it is synthetically assembled and lacks a continuous same-driver multi-microphone recording.

The published headline accuracies are therefore not directly comparable to our 79–80% full-pipeline task success. A fair comparison must separate intent recognition from authorization and action execution.

## Comparison tracks

### Track A — reproduce each paper's reported task

For each paper, record the exact dataset, number of classes, preprocessing, train/test split, speaker overlap, wake-word handling, and whether accuracy means intent classification or executed action success. Reproduce the reported model or closest available implementation without adding TSE, passenger speech, or our safety gates. Report accuracy, macro-F1, number of test examples, and latency on the same machine.

Do not import their reported number into our pipeline table unless the protocol is confirmed. If a dataset or split cannot be recovered, report “not reproducible from the paper” rather than silently substituting FSC or SLURP.

### Track B — common SLURP intent benchmark

Use the official SLURP train/development/test split and preserve speaker separation. Evaluate:

1. Clean single-utterance intent classification from the original audio.
2. Far-field/array audio where available, with a fixed 16-kHz preprocessing path.
3. The same utterances after the frozen TSE front end, using an enrollment utterance from the same speaker only where the protocol permits it.
4. A SLURP-trained intent head on the original semantic label inventory; do not map SLURP labels into FSC labels for the headline score.

Report intent accuracy, macro-F1, scenario/action accuracy, entity/slot F1, and exact semantic-frame match. For SLURP's slot metric, use the official implementation and document whether the score is SLU-F1, word-F1, or character-F1.

### Track C — common full-pipeline scenes

Create a separate, frozen 1,000-scene benchmark with:

- one same-speaker enrollment utterance;
- a same-speaker wake word followed by a full SLURP command;
- distinct passenger speech, including passenger wake-word speech;
- non-overlap and overlap conditions;
- no-cabin-noise and controlled cabin-noise variants;
- a fixed passenger identity policy and no test-speaker reuse for tuning;
- one possible action per wake and a full command window determined by endpointing.

For a clean source-controlled version, use a consented same-speaker recording for the wake word or a clearly marked synthetic voice-conversion/TTS source. Do not call synthetic scenes “real-cabin” evidence.

Run the same scene set through:

- our full pipeline;
- no TSE;
- no presence gate;
- no speaker authorization;
- no wake smoothing;
- command recognition without wake gating;
- a conventional ASR-to-NLU baseline;
- a direct speech-to-intent baseline.

Use the same audio, scene order, thresholds, and action policy for every system.

## Metrics and statistical protocol

Primary full-system metrics, in this order:

1. task success / fully correct scene rate;
2. wrong executed action rate;
3. passenger-only false-action rate;
4. false actions per hour;
5. target wake recall;
6. intent accuracy conditioned on an accepted target wake;
7. response latency p50/p95;
8. strict false-wake and false-command rates.

For intent-only comparisons, report exact accuracy, macro-F1, and slot/entity metrics separately. Use paired bootstrap 95% confidence intervals over scenes and McNemar's test for paired exact-intent outcomes. For false actions and false wakes, report event counts, total negative hours, rate per hour, and a one-sided confidence bound; do not report a rate without the exposure time.

## What would be a fair conclusion?

The paper comparison should answer three separate questions:

- Is our intent recognizer competitive on a standard SLU benchmark?
- Does TSE preserve intent accuracy under target/passenger mixtures?
- Does the complete assistant authorize the correct speaker and avoid unintended actions?

Only the third question supports the in-car claim. A higher SLURP or FSC intent score does not compensate for passenger false actions, and a lower SLURP score does not imply the full car assistant is worse if it has stronger speaker authorization and safety behavior.

## Sources to cite and verify

- SLURP paper: https://arxiv.org/abs/2011.13205
- Paranjape et al. vehicle RASA paper: https://arxiv.org/abs/2312.01642
- Hussein, Karan & Kurnaz article: https://doi.org/10.3390/electronics14030496
- Official SLURP repository and license details: https://github.com/pswietojanski/slurp

