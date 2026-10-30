# Three-way personal-activity candidate

Date: 2026-10-12  
Status: diagnostic candidate; not promoted to the default pipeline

## Motivation

The binary personal-presence head treats passenger speech and silence as the
same negative class. Personal-VAD and personalized-keyphrase systems instead
model whether the enrolled speaker is active, which requires distinguishing
target speech from non-target speech. This experiment adds an opt-in,
speaker-conditioned frame head with three classes:

| Class | Meaning |
|---:|---|
| 0 | non-speech |
| 1 | enrolled target speech |
| 2 | non-target/passenger speech |

The runtime consumes only the class-1 probability, so the existing wake and
command state machine remains unchanged.

## Data and training

`scripts/data/build_personal_activity_manifest.py` preserved the existing
1,208-row GSC-Marvin/AISHELL presence manifest and added 1,200 non-Marvin
Speech Commands clips. Each added row uses an enrollment clip from a different
Marvin speaker, CAVEMOVE cabin noise, and a class-2 label. This is a permitted
training diagnostic, not a real-driver or passenger recording set.

The candidate was trained with frozen Stage 1 output caching for four epochs:

- train rows: 1,926
- validation rows: 482
- final training loss: 0.3373
- validation best chunk F1: 0.697 at class-1 threshold 0.40
- checkpoint: `checkpoints/stage2_personal_activity_gsc_cavemove_ft1/best.pt`

The internal score is not a release metric because the split is public-data
row based and does not contain continuous same-driver sessions.

## End-to-end proxy results

The fixed 16-scene CAVEMOVE proxy used two cabin SNRs (-5 and +10 dB),
passenger speech at -10 dB, 0.5-second hops, and a different FSC utterance
from the command speaker as enrollment. The proxy remains cross-corpus: the
Marvin wake speaker is not verified to be the command speaker.

| Candidate operating point | Wake recall | Overall intent | False wakes/h | False actions/h |
|---|---:|---:|---:|---:|
| Existing binary personal head, 0.15 | 81.25% | 62.50% | 208.93 | 96.43 |
| Three-way activity head, 0.25 | 75.00% | 56.25% | 208.93 | 48.21 |
| Three-way activity head, 0.30 | 25.00% | 18.75% | 208.93 | 80.36 |
| Three-way activity head, 0.40 | 0.00% | 0.00% | 0.00 | 0.00 |

The 0.25 point halves observed false actions on this proxy, but it also loses
some target recall and leaves false wakes very high. The 0.40 point transfers
poorly from internal validation and rejects every proxy target. No operating
point is suitable for promotion.

On the retained 250-file Smart Speaker non-Marvin diagnostic, the 0.25
candidate produced 674.4 merged full-path events/hour versus 728.1/hour for
the retained binary reference. The KWS-only rate remained 1,817.2/hour and
the 14-file sequential action sample remained 105.9 false actions/hour. This
small clean-studio diagnostic therefore shows only a modest false-wake change
and no action-safety improvement.

## Decision

Reject as the default checkpoint. Retain the implementation because it is a
better architecture for the next real-driver experiment: once the same driver
provides enrollment, wake, and command, the class-2 branch can be trained on
passenger speech and passenger “Marvin” directly. Thresholds must then be
selected on tuning sessions using both false-wake/hour and false-action/hour.

## Reproduction

```powershell
python scripts/data/build_personal_activity_manifest.py `
  --base-manifest manifests/train/stage2_presence_gsc_aishell.jsonl `
  --speech-commands-root data/raw/speech_commands_v2/extracted `
  --output manifests/train/stage2_personal_activity_gsc_cavemove.jsonl `
  --max-nontarget 1200

python scripts/train/train_stage2_frame_presence.py `
  --config configs/stage2_personal_activity_gsc_cavemove_ft1.yaml `
  --paths configs/paths.yaml

python scripts/eval/eval_continuous_full_pipeline_proxy.py `
  --config configs/eval_end_to_end_cavemove_personal_activity_operating_diagnostic.yaml `
  --paths configs/paths.yaml --scenes 8 --cabin-snr-db -5 10 `
  --passenger-snr-db -10 --hop-seconds 0.5 `
  --enrollment-source same_speaker_other `
  --output logs/continuous_full_pipeline_proxy_personal_activity_operating_p025_same_speaker_m10_8scenes_2026-10-12.json
```

The complete scene-level logs retain the per-window activity score and action
events for later analysis.

Additional negative diagnostic:

- `logs/smart_speaker_command_negative_250_personal_activity_p025_2026-10-12.json`
