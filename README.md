# car-tse

SpEx+-based target speaker extraction and in-car voice assistant pipeline (stages: extraction → presence → KWS → audio-to-intent).

## Prerequisites

- Python 3.11+ with PyTorch (see parent `audioml_base` env)
- Corpora on disk under `DATA_ROOT` (no download scripts in this repo)
- Vendored [speaker_extraction_SpEx](https://github.com/xuchenglin28/speaker_extraction_SpEx) under `third_party/`
- Pretrained SpEx+ checkpoint as `best.pt.tar` in `checkpoints/spex_pretrained/` (e.g. from [SpExPlus_LibriSpeech](https://github.com/m4xig1/SpExPlus_LibriSpeech))

## Setup

```powershell
cd car-tse
pip install -e ".[dev]"
.\scripts\setup\check_vendor.ps1
copy configs\paths.example.yaml configs\paths.yaml
# Edit configs\paths.yaml: data_root, checkpoint paths
```

### Vendor (Windows)

If full clone fails on `aux.scp`, fetch `nnet` only:

```powershell
mkdir third_party\speaker_extraction_SpEx
cd third_party\speaker_extraction_SpEx
git init
git remote add origin https://github.com/xuchenglin28/speaker_extraction_SpEx.git
git -c core.protectNTFS=false fetch --depth 1 origin master
git -c core.protectNTFS=false checkout FETCH_HEAD -- nnet decode.py
```

Place `best.pt.tar` in `checkpoints/spex_pretrained/`.

## Data layout (`DATA_ROOT`)

| Subfolder | Corpus |
|-----------|--------|
| `LibriSpeech/` | Enrollment / clean speech |
| `Libri2Mix/` | Mixtures (mix_both) |
| `AISHELL-5/` | In-car multichannel + noise |
| `speech_commands_v2/` | KWS |
| `mswc/en/` | KWS negatives |
| `hey_snips/` | KWS eval negatives |
| `fluent_speech_commands/` | Intent |
| `custom_incar/` | Custom wake + in-car commands |

## Workflow

```powershell
# 1) Build manifests (after data is present)
.\scripts\orchestration\prepare_all.ps1

# 2) Full training + evaluation
.\scripts\orchestration\run_full_study.ps1
```

Individual stages:

```powershell
python scripts/train/train_stage1_spex.py --config configs/stage1_spex_sc.yaml
python scripts/train/train_stage1_spex.py --config configs/stage1_spex_mc.yaml
python scripts/train/train_stage2_presence.py
python scripts/train/train_stage3_kws.py
python scripts/train/train_stage4_intent.py
```

Vehicle test ingest:

```powershell
python scripts/eval/ingest_vehicle_recordings.py --input-dir D:\recordings --enroll D:\enroll.wav
```

## Tests

```powershell
pytest tests/
```

## Reference

- `Abstract.docx` — training curriculum and evaluation protocol
- `fig1_system.png` — architecture and dataset mapping per block
