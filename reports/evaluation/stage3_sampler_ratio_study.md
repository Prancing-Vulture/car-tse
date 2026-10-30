# Stage 3 sampler-ratio check

## Question

Does showing the wake detector more negative examples during fine-tuning reduce its false alarms without losing wakes?

## How the comparison was run

All nine runs started from the same Stage 3 v2 checkpoint. They used the same 1,812 cached Stage-1 training examples (604 wake, 1,208 non-wake), 390 validation examples, learning rate, batch size, epoch limit, early-stopping rule, and fixed 0.29 decision cutoff. Only the sampling rule and random seed changed. Training and checkpoint selection used train/validation data only; the test split was not loaded.

- **Natural:** each training-manifest example once per epoch, shuffled (the dataset's 2:1 negative-to-positive mix).
- **2:1 replacement sample:** draw 1,812 examples with replacement, targeting 2 negatives per positive.
- **4:1 replacement sample:** draw 1,812 with replacement, targeting 4 negatives per positive.

The trainer now records training loss and the actual sampled positive/negative counts each epoch in W&B and the local epoch report. This makes it possible to check that the requested sampling policy actually ran.

## Validation result

Best validation F1 at the fixed 0.29 cutoff, over three seeds:

| Training mix | Seed 20261011 | Seed 20261012 | Seed 20261013 | Mean |
|---|---:|---:|---:|---:|
| Natural 2:1, without replacement | 0.6862 | 0.6862 | 0.6825 | **0.6850** |
| 2:1, with replacement | 0.6844 | 0.6825 | 0.6862 | **0.6844** |
| 4:1, with replacement | 0.6879 | 0.6825 | 0.6839 | **0.6848** |

The starting v2 checkpoint scored 0.6825 on the same validation set and cutoff. The apparent gains are only about 0.002–0.003 F1 on average; the sampler rankings change across seeds. The 4:1 run's best single result (0.6879) did not repeat. This is not evidence that adding more negative samples reliably improves the detector.

## Decision

Keep the v2 checkpoint as the current Stage 3 model. Do not promote a v3 sampler run based on this small and inconsistent validation difference. The next useful improvement needs better event-level negative examples and a larger same-speaker continuous evaluation, not more runs of this same sampler experiment. No test-set metrics are claimed here.

## W&B runs

- Seed 20261011: [natural](https://wandb.ai/narainharreesh-vit/car-tse/runs/gf8bqtqu), [2:1 replacement](https://wandb.ai/narainharreesh-vit/car-tse/runs/f7chyhtw), [4:1 replacement](https://wandb.ai/narainharreesh-vit/car-tse/runs/htebup81)
- Seed 20261012: [natural](https://wandb.ai/narainharreesh-vit/car-tse/runs/cl4y4ifd), [2:1 replacement](https://wandb.ai/narainharreesh-vit/car-tse/runs/jt4e2z7j), [4:1 replacement](https://wandb.ai/narainharreesh-vit/car-tse/runs/uaz1tyt3)
- Seed 20261013: [natural](https://wandb.ai/narainharreesh-vit/car-tse/runs/b3njjb29), [2:1 replacement](https://wandb.ai/narainharreesh-vit/car-tse/runs/2v9bp7cx), [4:1 replacement](https://wandb.ai/narainharreesh-vit/car-tse/runs/tr0qytn4)
