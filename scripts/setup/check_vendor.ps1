$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$Vendor = Join-Path $Root "third_party\speaker_extraction_SpEx"
$Module = Join-Path $Vendor "nnet\spex_plus.py"

if (-not (Test-Path $Module)) {
    Write-Host "SpEx+ vendor not found at $Vendor"
    Write-Host "Clone on Windows (nnet only):"
    Write-Host "  cd $Vendor"
    Write-Host "  git init"
    Write-Host "  git remote add origin https://github.com/xuchenglin28/speaker_extraction_SpEx.git"
    Write-Host "  git -c core.protectNTFS=false fetch --depth 1 origin master"
    Write-Host "  git -c core.protectNTFS=false checkout FETCH_HEAD -- nnet decode.py"
    exit 1
}

$env:PYTHONPATH = "$Root\src;$Vendor"
python -c "from nnet.spex_plus import SpEx_Plus; print('SpEx_Plus import OK')"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Vendor check passed."
