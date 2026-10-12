# ANIR multichannel presence-augmentation candidate

Date: 2026-10-11  
Status: completed diagnostic; not promoted

## Motivation

The automotive-systems review identified real vehicle-noise diversity as a
useful improvement, while noting that ANIR does not contain speech, wake, or
command labels. Its V-class archive was therefore converted into a controlled
augmentation source rather than treated as driver evidence.

## Data preparation

- Source archive: `D:\datasets\anir\v_class_noise_files`
- Source checksum: `8CC71CA50F89909D475798761669ABEA788C6C2CA5F1E022480EB63F5022A441`
- Source: 1,302 mono 48-kHz WAV files, 42 conditions
- Derived cache: `D:\datasets\anir\v_class_noise_16k_mc`
- Derived format: 42 four-channel 16-kHz FLOAT WAVs, microphones 0–3
- Manifest: `data/manifests/external/anir_v_class_noise_16k_mc.json`

The cache preserves the four-channel contract and is only an augmentation
input. It does not create labels or turn ANIR into a real-driver evaluation.

## Candidate training

The existing frozen Stage-1 output path was used. The base Stage-2 presence
manifest was expanded with 604 ANIR-noised target-positive copies, then joined
with the existing 1,200 non-wake hard negatives:

- 1,812 rows in the augmented presence manifest;
- 2,409 training rows and 603 internal validation rows after the training
  script's row split;
- personal frame head, 40 mel bins, 64 hidden channels;
- four cached epochs, AdamW, learning rate `5e-5`;
- checkpoint: `checkpoints/stage2_personal_presence_dual_domain_nonwake_anir_ft1/best.pt`.

The internal chunk F1 was 1.000 at threshold 0.40, but this is a row-split
synthetic fit check and not held-out driver evidence.

## End-to-end result

The same 16-scene cross-corpus cabin proxy was used for comparison: 8 scenes,
cabin SNR −5 and +10 dB, passenger SNR −10 dB, 0.5-second hop, and
same-speaker FSC enrollment/command with an unverified cross-corpus Marvin
clip.

| Candidate | Wake recall | Overall intent | False wakes/hour | False actions/hour |
|---|---:|---:|---:|---:|
| Existing personal-presence checkpoint | 81.25% | 62.50% | 208.93 | 96.43 |
| ANIR-augmented personal-presence checkpoint | 81.25% | 62.50% | 208.93 | 96.43 |

ANIR augmentation produced no measurable improvement on this proxy. The high
false-action rate remains dominated by the full cascade's synthetic operating
conditions and cannot be solved by promoting this checkpoint.

## Decision

Keep the ANIR cache and manifest as a reusable noise-diversity asset, but reject
the candidate as a production upgrade. The next training use should be after
the real-driver pilot exists, with ANIR used as an additional noise condition
for speaker-conditioned activity and wake training—not as a substitute for
speaker-consistent recordings.

## Reproduction artifacts

- `scripts/data/build_anir_multichannel_cache.py`
- `scripts/data/build_anir_augmented_presence_manifest.py`
- `data/manifests/external/anir_v_class_noise_16k_mc.json`
- `manifests/train/stage2_presence_gsc_aishell_anir.jsonl`
- `configs/stage2_personal_presence_dual_domain_nonwake_anir_ft1.yaml`
- `configs/eval_end_to_end_cavemove_personal_presence_anir_candidate.yaml`
- `checkpoints/stage2_personal_presence_dual_domain_nonwake_anir_ft1/best.pt`
- `logs/stage2_personal_presence_dual_domain_nonwake_anir_ft1/metrics_last.json`
- `logs/continuous_full_pipeline_proxy_personal_presence_anir_same_speaker_m10_8scenes_2026-10-11.json`
