$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$Py = "python"
$Manifest = Join-Path $Root "manifests\train\stage2_presence.jsonl"
if (-not (Test-Path -LiteralPath $Manifest)) {
    throw "Presence training manifest not found: $Manifest"
}

Set-Location $Root
New-Item -ItemType Directory -Force -Path (Join-Path $Root "logs\stage2_presence_bootstrap") | Out-Null
& $Py "scripts\train\train_stage2_presence.py" `
    --config "configs\stage2_presence_bootstrap.yaml" `
    --paths "configs\paths.yaml"
if ($LASTEXITCODE -ne 0) { throw "Presence bootstrap training failed." }
Write-Host "Presence bootstrap training completed."
