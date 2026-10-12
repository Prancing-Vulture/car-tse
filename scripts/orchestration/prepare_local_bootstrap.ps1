$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$Py = "python"
$Paths = Join-Path $Root "configs\paths.yaml"

function Run-Step([string]$Script, [string[]]$Arguments) {
    Write-Host "`n>>> $Script $($Arguments -join ' ')"
    & $Py (Join-Path $Root $Script) @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $Script" }
}

Run-Step "scripts\data\verify_datasets.py" @("--paths", $Paths)
Run-Step "scripts\data\build_librispeech_splits.py" @("--paths", $Paths)
Run-Step "scripts\data\build_local_librimix.py" @("--paths", $Paths, "--train-rows", "4000", "--val-rows", "400", "--test-rows", "400")
Run-Step "scripts\data\build_local_multichannel.py" @("--paths", $Paths, "--split", "train", "--split", "val", "--split", "test")

foreach ($Split in @("train", "val", "test")) {
    Run-Step "scripts\data\build_librimix_triplets.py" @("--paths", $Paths, "--split", $Split)
    Run-Step "scripts\data\augment_aishell5_cabin.py" @("--paths", $Paths, "--split", $Split)
    Run-Step "scripts\data\build_multichannel_manifest.py" @("--paths", $Paths, "--split", $Split)
}

foreach ($Split in @("train", "val")) {
    Run-Step "scripts\data\build_target_absent.py" @("--paths", $Paths, "--split", $Split)
}
Run-Step "scripts\data\build_ec_playback_pairs.py" @("--paths", $Paths, "--split", "train")
Run-Step "scripts\data\build_kws_manifest.py" @("--paths", $Paths)
Run-Step "scripts\data\build_intent_manifest.py" @("--paths", $Paths)
Run-Step "scripts\data\check_leakage.py" @()

Write-Host "Local bootstrap data preparation completed."
