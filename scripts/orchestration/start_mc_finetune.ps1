$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$Py = "python"
$Manifest = Join-Path $Root "manifests\train\stage1_mc_triplets.jsonl"
if (-not (Test-Path -LiteralPath $Manifest)) {
    throw "Multichannel training manifest not found: $Manifest"
}

Set-Location $Root
New-Item -ItemType Directory -Force -Path (Join-Path $Root "logs\stage1_spex_mc_finetune") | Out-Null
& $Py "scripts\train\train_stage1_spex.py" `
    --config "configs\stage1_spex_mc_finetune.yaml" `
    --paths "configs\paths.yaml"
if ($LASTEXITCODE -ne 0) { throw "Multichannel fine-tuning failed." }
Write-Host "Multichannel fine-tuning completed."
