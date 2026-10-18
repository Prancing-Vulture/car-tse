# Hussein/Paranjape comparison results

Date: 2026-10-18  
Branch: `multi-speaker-eval`

## Executive result

We ran two clean, intent/command-only comparisons. Both scores are higher
than the headline numbers reported by Hussein et al. and Paranjape et al.,
but neither comparison should be presented as a full in-car pipeline result.

| Comparison | Our result | Paper result | Difference |
|---|---:|---:|---:|
| Speech Commands command classification, Hussein track | **96.75% top-1**, 96.37% macro-F1, 99.25% top-5 | 93% accuracy | **+3.75 percentage points** |
| Clean FSC intent classification, Paranjape track | **99.31% top-1**, 99.24% macro-F1, 99.89% top-5 | 93.67% (281/300) | **+5.64 percentage points** |

The two paper numbers are reported on different datasets, class inventories,
splits, and modality/pipeline definitions. They are reference numbers, not a
shared SOTA benchmark.

## Hussein et al. comparison

Hussein, Karan and Kurnaz describe a speech-command system for driving
control using noise removal, feature extraction, and a DNN classifier. The
paper reports 93% accuracy and approximately 0.75 s response time. Its data
section identifies the standard Speech Commands corpus and describes short
single-word recordings, some with background noise.

The paper's exact reported 687-clip list and 70/30 split are not available in
the public repository/workspace. Therefore the experiment below is a
same-dataset/task-family reproduction on the official public Speech Commands
v2 split, not an exact recreation of their private sample selection.

### Our protocol

- Corpus: local Google Speech Commands v2 release at
  `data/raw/speech_commands_v2/extracted`.
- Classes: all 35 command directories, including `marvin`, not a binary
  wake-word task.
- Official lists: `validation_list.txt` and `testing_list.txt`; all remaining
  recordings are training data.
- Counts: 84,843 train, 9,981 validation, 11,005 test.
- Audio: mono, 16 kHz, one-second input; test audio was not augmented.
- Model: `microsoft/wavlm-base-plus` frozen encoder; mean pooling over valid
  encoder frames; `LayerNorm -> Linear(768,384) -> GELU -> Dropout(0.15) ->
  Linear(384,35)` command head.
- Selection: best validation macro-F1, then one final official test pass.
- Scope: direct command classification only. No TSE, enrollment, presence
  gate, wake smoothing, passenger speech, or vehicle action state machine.

### Result

| Split | Top-1 | Macro-F1 | Top-5 | Examples |
|---|---:|---:|---:|---:|
| Validation, best selection point | 96.84% | 96.59% | 99.15% | 9,981 |
| Official test | **96.75%** | **96.37%** | **99.25%** | 11,005 |

The best checkpoint was selected at epoch 27 using validation macro-F1. The
test score is therefore not used for model selection.

Artifacts:

- Training script: `scripts/train/train_speech_commands_wavlm_cached.py`
- Checkpoint: `D:/datasets/speech_commands/artifacts/checkpoints/speech_commands_wavlm_cached_hussein_track_2026-10-18/best.pt`
- Test metrics: `D:/datasets/speech_commands/artifacts/logs/speech_commands_wavlm_cached_hussein_track_2026-10-18/metrics_official_test.json`
- Confusion matrix: `D:/datasets/speech_commands/artifacts/logs/speech_commands_wavlm_cached_hussein_track_2026-10-18/confusion_official_test.npy`
- Training plot: `reports/figures/speech_commands_wavlm_cached_hussein_track_2026-10-18.png` and `.pdf`

The first attempted end-to-end WavLM fine-tuning run was stopped because the
Windows data-loader path stalled before its first epoch. It produced no
result and was not used. The reported result is from the completed cached
transfer-learning run above.

## Paranjape et al. comparison

Paranjape et al. describe a RASA-based vehicle assistant. Its flow is
speech-to-text, text NLU/intent and entity recognition, dialogue policy, and
application/API actions. The paper reports 281 correctly identified intents
out of 300, or 93.67%. It also reports 0.08 s for intent identification, but
longer interaction/API timings for confirmation and output.

Our existing production-project clean FSC checkpoint was rerun on the
official FSC test manifest:

| Split | Top-1 | Macro-F1 | Top-5 | Examples |
|---|---:|---:|---:|---:|
| Official clean FSC test | **99.31%** | **99.24%** | **99.89%** | 3,793 |

Artifact:

- `reports/artifacts/paranjape_clean_fsc_comparison_2026-10-18.json`

This is a stronger clean audio-intent result numerically, but it is not a
replication of the Paranjape RASA setup: FSC has a different 31-intent
inventory, and our model directly classifies audio rather than evaluating a
speech-recognition-to-RASA text pipeline.

## What this proves—and what it does not

The results support the following limited claims:

1. Our direct-audio command classifier exceeds Hussein et al.'s reported
   93% command accuracy on a larger official Speech Commands test split.
2. Our clean FSC intent classifier exceeds Paranjape et al.'s reported
   93.67% intent accuracy on its official clean test split.
3. The intent/action classifier is not the current bottleneck on clean,
   isolated speech.

They do **not** prove that the complete in-car assistant beats either paper
under their interaction conditions. The papers do not measure the combined
target-speaker extraction, speaker authorization, wake-word acceptance,
passenger rejection, false-action rate, or continuous cabin behavior that
defines our actual product goal.

For the research paper, describe these as two component-level reference
comparisons. Keep the full-pipeline result in a separate table with task
success, wrong-action rate, passenger-only false-action rate, false wakes per
hour, and latency percentiles.

## Sources

- Hussein et al., *Enhancing Driving Control via Speech Recognition Utilizing
  Influential Parameters in Deep Learning Techniques*,
  [Electronics 14(3):496](https://www.mdpi.com/2079-9292/14/3/496).
- Paranjape et al., *Voice-Based Smart Assistant System for Vehicles using
  RASA*, [arXiv:2312.01642](https://arxiv.org/html/2312.01642v1).
