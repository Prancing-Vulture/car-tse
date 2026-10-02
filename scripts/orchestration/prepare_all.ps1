$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$Py = "python"

& $Py "$Root\scripts\data\verify_datasets.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Py "$Root\scripts\data\build_librispeech_splits.py"
& $Py "$Root\scripts\data\build_librimix_triplets.py" --split train
& $Py "$Root\scripts\data\build_librimix_triplets.py" --split val
& $Py "$Root\scripts\data\build_librimix_triplets.py" --split test
& $Py "$Root\scripts\data\augment_aishell5_cabin.py" --split train
& $Py "$Root\scripts\data\augment_aishell5_cabin.py" --split val
& $Py "$Root\scripts\data\augment_aishell5_cabin.py" --split test
& $Py "$Root\scripts\data\build_multichannel_manifest.py" --split train
& $Py "$Root\scripts\data\build_multichannel_manifest.py" --split val
& $Py "$Root\scripts\data\build_multichannel_manifest.py" --split test
& $Py "$Root\scripts\data\build_target_absent.py" --split train
& $Py "$Root\scripts\data\build_target_absent.py" --split val
& $Py "$Root\scripts\data\build_ec_playback_pairs.py" --split train
& $Py "$Root\scripts\data\build_kws_manifest.py"
& $Py "$Root\scripts\data\build_intent_manifest.py"
& $Py "$Root\scripts\data\resample_corpus.py" --split train
& $Py "$Root\scripts\data\check_leakage.py"

Write-Host "prepare_all completed."
