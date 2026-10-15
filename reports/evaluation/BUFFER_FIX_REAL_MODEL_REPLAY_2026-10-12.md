# Buffer fix replay with trained pipeline models — 2026-10-12

## What was run

Two 14-second continuous proxy scenes were built from the existing held-out
Speech Commands wake manifest, official FSC test manifest, and CAVEMOVE cabin
noise. The same fixed scene, seed, model checkpoints, and 0.5-second hop were
used in both runs. The trained Stage 1, Stage 2, Stage 3, and Stage 4 models
were loaded from the `audioml_base` environment on CUDA.

- **Control:** command-presence threshold 0.20; trained Stage 2 ran normally.
- **Forced skip:** same settings, but Stage 2 was forced to reject window 7.
- Both used overall presence threshold 0.50 and KWS threshold 0.29. The lower
  command threshold was an explicit diagnostic setting to let command capture
  continue after wake; it is not being recommended as a production setting.

W&B recorded both runs:

- [Control run](https://wandb.ai/narainharreesh-vit/car-tse/runs/a46u4ato)
- [Forced-rejection run](https://wandb.ai/narainharreesh-vit/car-tse/runs/fpfnyae6)

## Results

| Measurement | Control | Forced rejection |
| --- | ---: | ---: |
| Target wake detected | Yes | Yes |
| Target command intent correct | No | No |
| False wake/action events | 0 / 0 | 0 / 0 |
| Stage 4 decision window | 7 | 8 |
| Stage 4 input source-time span | 2.0–4.0 s | 2.0–4.0 s |
| Labeled command span | 2.1–4.1 s | 2.1–4.1 s |
| Labeled command covered by Stage 4 input | 1.9/2.0 s (95%) | 1.9/2.0 s (95%) |

In the control run, Stage 2 rejected the command-pending windows 3–6. Despite
those rejections, the rolling buffer retained the earlier audio. Stage 4 later
received samples mapped to 2.0–4.0 seconds. In the forced run, Stage 2 also
rejected window 7, delaying Stage 4 by one window; it still received the same
2.0–4.0-second source span. The two saved WAVs contain sample-identical audio
(32,000 samples; maximum sample difference 0.0).

Stage 4 predicted `bring|socks`; the expected label was `activate|lamp` in both
runs. Therefore, this replay demonstrates that the buffer preserves and
delivers command-time audio across gate rejections. It does **not** demonstrate
an end-to-end accuracy improvement. It also shows the fixed two-second Stage 4
crop starts 0.1 seconds before the labeled command and ends 0.1 seconds before
the label ends; this is a separate crop-alignment limit, not audio lost from the
rolling buffer.

## Data limitation

The `marvin` wake and FSC command are from different corpora and are not
verified to be the same speaker. Stage 1 enrollment is a separate GSC recording
from the wake speaker. Thus, the wrong intent cannot be attributed specifically
to buffer behavior, Stage 1 extraction quality, or Stage 4 model accuracy. The
proxy is an engineering check, not a same-speaker performance benchmark.

## Artifacts

- [Control report](./buffer_proxy_control_p050_command020_2026-10-12.json)
- [Forced-rejection report](./buffer_proxy_forced_skip_p050_command020_2026-10-12.json)
- [Control Stage 4 input](./artifacts/buffer_proxy_control_p050_command020_2026-10-12/stage4_input_scene_00_snr_%2B00.0_event_00.wav)
- [Forced-rejection Stage 4 input](./artifacts/buffer_proxy_forced_skip_p050_command020_2026-10-12/stage4_input_scene_00_snr_%2B00.0_event_00.wav)

One preliminary launch used the config's 0.90 presence cutoff and detected no
wake. It was not used in the buffer comparison; the subsequent runs explicitly
used 0.50 and are the results above.
