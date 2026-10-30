# Reproducible retraining run (2026-10-14)

## Run protocol

- Environment: `audioml_base`, PyTorch 2.12.0+cu132, CUDA, NVIDIA GeForce RTX 5060 Ti.
- Fixed seed: `20261011` for each training configuration.
- Weights & Biases project: [`narainharreesh-vit/car-tse`](https://wandb.ai/narainharreesh-vit/car-tse).
- AISHELL data is excluded. Stage 1 uses the approved LibriMix `train-100` split and then the CAVEMOVE cabin adaptation data.
- Config and input-manifest SHA-256 values are recorded in each W&B training run. Epoch histories are stored as `metrics_history.jsonl`; `metrics_last.json` is the latest epoch snapshot.
- Checkpoints remain in the local `checkpoints/` tree. Results, evaluation JSON, and publication plots are staged for the repository after each stage completes.

## Tracked metrics and figures

| Stage | Training and validation metrics | Evaluation / plots |
|---|---|---|
| Stage 1 SpEx+ | Train loss and SI-SDR for outputs 1–3; validation output and input SI-SDR; mean, median, p10, p90, and negative-count SI-SDR improvement; absent-target output energy where present. | Train/validation curves and validation SI-SDRi distribution; held-out test metrics after training. |
| Stage 2 presence | Train loss; validation frame and chunk F1, precision, recall, threshold, false-accept rate, and balanced accuracy. | Held-out threshold sweep, ROC and precision-recall curves, plus fixed-threshold metrics. |
| Stage 3 wake word | Train accuracy/loss and sampled class counts; validation accuracy, precision, recall, F1, false positives/negatives, and decision threshold. | Held-out accuracy, precision, recall, F1, ROC-AUC, average precision, confusion matrix, ROC and precision-recall curves. |
| Stage 4 intent | Train loss/accuracy; validation accuracy, top-5 accuracy, macro precision/recall/F1, and learning rates. | Held-out accuracy/top-5 and per-intent precision/recall/F1 with confusion matrix and per-class plots. |

## Stage 1 progress

The first train-100 epoch completed with train loss `-10.0020`, train SI-SDR `8.3309 dB`, validation SI-SDR `8.4295 dB`, and validation SI-SDR improvement `10.3002 dB` over 3,000 validation examples. The W&B history row is committed and visible.

The active resumed run includes the completed epoch-1 metrics and continues at epoch 2 through epoch 6:

- [Stage 1 train-100 live W&B run](https://wandb.ai/narainharreesh-vit/car-tse/runs/ik3eub3b)
- Epoch history: `stage1_spex_train100_resume_rerun_20261014/metrics_history.jsonl`
- Checkpoint: local `checkpoints/stage1_spex_train100_resume_rerun_20261014/best.pt`

