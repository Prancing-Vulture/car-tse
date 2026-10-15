# Smart Speaker open-set threshold sweep — 2026-10-14

## Purpose

Calibrate the current English target-KWS operating point against a complete
non-Marvin continuous-speech corpus before deciding whether another training
run is justified. This is a diagnostic negative-only experiment, not a driver
or real-cabin evaluation.

## Reproduction

```powershell
python scripts/eval/eval_smart_speaker_command_negative.py `
  --config configs/eval_end_to_end_clean_english_identity_targetkws_endpoint08_hardneg_nogate_candidate.yaml `
  --kws-threshold 0.50 `
  --kws-thresholds 0.35 0.40 0.45 0.50 0.55 0.60 `
  --max-files 0 --hop-seconds 0.25 --batch-size 32 `
  --action-sample-files 0 `
  --output logs/smart_speaker_open_set_threshold_sweep_fullwindows_2026-10-14.json
```

The evaluator now computes all requested thresholds from one shared
Stage-1/presence/KWS forward pass. It uses the shared full-context window rule,
so no incomplete zero-padded tail window is counted as an ordinary negative.

## Data contract

- University of Rochester Smart Speaker Command Dataset, 2,110 files.
- 3.1925356 hours total, seven speakers.
- Every clip is scripted as ``Hey Alexa`` plus a command, so it is negative for
  the current ``Marvin`` wake phrase.
- Original audio is clean, single-channel studio speech duplicated to four
  channels for the interface. It is not real-cabin, array, playback, or
  passenger evidence.

## Results

| KWS threshold | Full-path events | Full-path events/hour | KWS-only events | KWS-only events/hour |
|---:|---:|---:|---:|---:|
| 0.35 | 2,958 | 926.54 | 3,110 | 974.15 |
| 0.40 | 2,954 | 925.28 | 3,106 | 972.89 |
| 0.45 | 1,718 | 538.13 | 2,002 | 627.09 |
| 0.50 | 76 | 23.81 | 159 | 49.80 |
| 0.55 | 0 | 0.00 | 0 | 0.00 |
| 0.60 | 0 | 0.00 | 0 | 0.00 |

At the current candidate threshold 0.50, the complete negative corpus still
produces 76 full-path false wakes (23.81/hour). Raising the threshold to 0.55
removes all observed negative events on this corpus, but this is not a release
threshold: the corrected noisy proxy sweep at 0.55 recorded zero target wakes.
The evidence therefore shows a calibration/score-separation problem, not a
safe threshold choice.

The per-file maximum wake-score distribution makes the overlap explicit: the
negative median is 0.4924, the 90th percentile is 0.5043, the 99th percentile
is 0.5096, and the observed maximum is 0.5126. The current 0.50 cutoff is
therefore inside the negative-score tail; 0.55 is above this corpus's maximum,
but also above the useful positive operating region in the noisy proxy.

## Decision

Do not freeze 0.55 and do not claim a zero FAR. The next model improvement
should use a speaker-consistent tuning set containing real ``Marvin`` positives,
same-driver commands, passenger ``Marvin`` negatives, and cabin/playback
conditions. Thresholds must be selected on that tuning split and evaluated on a
separate held-out split. The Smart Speaker corpus remains useful for open-set
regression and should stay in the continuous test suite.

Raw result:

- `logs/smart_speaker_open_set_threshold_sweep_fullwindows_2026-10-14.json`
