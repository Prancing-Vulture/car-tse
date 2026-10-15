$ErrorActionPreference = "Stop"

$Stage2RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Stage2Python = "C:\Users\Harreesh\miniconda3\envs\audioml_base\python.exe"
$Stage2Tmp = Join-Path $Stage2RepoRoot "tmp"
$Stage2WandbDir = Join-Path $Stage2RepoRoot "wandb"

Set-Location $Stage2RepoRoot
New-Item -ItemType Directory -Force -Path $Stage2Tmp, $Stage2WandbDir | Out-Null
$env:TEMP = $Stage2Tmp
$env:TMP = $Stage2Tmp
$env:TMPDIR = $Stage2Tmp
$env:WANDB_DIR = $Stage2WandbDir
$env:WANDB_MODE = "online"

& $Stage2Python scripts/train/train_stage2_frame_presence.py `
    --config configs/stage2_frame_presence_gsc_cavemove_pilot.yaml `
    --paths configs/paths.e_research.yaml `
    --run-name stage2_frame_presence_gsc_cavemove_pilot `
    --wandb `
    --wandb-project car-tse `
    --wandb-mode online `
    --wandb-dir $Stage2WandbDir

if ($LASTEXITCODE -ne 0) {
    throw "Stage 2 pilot training failed with exit code $LASTEXITCODE"
}
