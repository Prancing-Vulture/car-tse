# Smart Speaker Command Dataset audit

Source: [University of Rochester Smart Speaker Command Dataset](https://rochester.figshare.com/articles/dataset/Smart_Speaker_Command_Dataset/26417548)  
License: CC BY 4.0 as stated on the official dataset page  
Downloaded archive: `D:/datasets/smart_speaker_command_26417548.zip`  
Archive MD5: `f0561310a234d2be445a5376f08e75c6`  
Extracted directory: `D:/datasets/smart_speaker_command_26417548/SmartSpeakerCommandDataset_RutowskiEtAl_2023`

## Inventory

- 2,110 WAV files
- 7 speakers
- 48 kHz mono source audio
- Scripted phrase format: “Hey Alexa” followed by a command
- Approximate extracted size: 1.1 GB
- Split used for the negative-cache builder: Speaker1–Speaker5 train, Speaker6 validation, Speaker7 test

## Use in this project

The clips are negative for the current wake phrase “Marvin.” They are useful because the audio contains a wake-like phrase followed by a command from one continuous speaker, which exercises the continuous evaluator without inventing a positive Marvin label. They were used to build a target-KWS negative cache and to run a clean-studio hard-negative diagnostic.

They are not evidence of:

- real-cabin performance;
- four-microphone spatial performance;
- passenger rejection;
- same-driver enrollment/wake/command continuity;
- positive Marvin detection;
- production false-accept rate.

Mono clips were duplicated across the configured channels only for API compatibility. That does not create spatial information.

## Reproduction

Cache construction:

```powershell
python scripts/data/build_target_kws_smart_speaker_negative_cache.py `
  --input-root D:/datasets/smart_speaker_command_26417548/SmartSpeakerCommandDataset_RutowskiEtAl_2023 `
  --output-root data/cache/stage3_target_kws_smart_speaker_negative_16k
```

Negative evaluation:

```powershell
python scripts/eval/eval_smart_speaker_command_negative.py `
  --config configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_hardneg_aishell5_neg_safety_splitgate_candidate.yaml `
  --input-root D:/datasets/smart_speaker_command_26417548/SmartSpeakerCommandDataset_RutowskiEtAl_2023 `
  --max-files 250
```

The evaluator records separate KWS-only and full-path diagnostics. Because it resets the streaming state per file and counts positive windows conservatively, these rates should be interpreted as diagnostic pressure, not as a final deduplicated FAR estimate.

## Attribution and publication note

Any redistribution or publication use should preserve the dataset’s CC BY attribution requirements and the official dataset citation. The data should not be described as in-car or multichannel speech.
