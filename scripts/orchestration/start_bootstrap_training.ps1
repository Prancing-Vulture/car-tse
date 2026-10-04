$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$Py = "python"
$Manifest = Join-Path $Root "manifests\train\stage1_triplets.jsonl"
if (-not (Test-Path -LiteralPath $Manifest)) {
    throw "Training manifest not found: $Manifest. Run prepare_local_bootstrap.ps1 first."
}

Set-Location $Root
New-Item -ItemType Directory -Force -Path (Join-Path $Root "logs\stage1_spex_sc_bootstrap") | Out-Null
& $Py "scripts\train\train_stage1_spex.py" `
    --config "configs\stage1_spex_sc_bootstrap.yaml" `
    --paths "configs\paths.yaml"
if ($LASTEXITCODE -ne 0) { throw "Bootstrap Stage 1 training failed." }
Write-Host "Bootstrap Stage 1 training completed."
