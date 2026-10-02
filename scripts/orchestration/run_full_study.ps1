$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$Py = "python"

& "$Root\scripts\orchestration\prepare_all.ps1"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Py "$Root\scripts\train\train_stage1_spex.py" --config "$Root\configs\stage1_spex_sc.yaml"
& $Py "$Root\scripts\eval\eval_stage1_spex.py" --config "$Root\configs\stage1_spex_sc.yaml" --split val

& $Py "$Root\scripts\train\train_stage1_spex.py" --config "$Root\configs\stage1_spex_mc.yaml"
& $Py "$Root\scripts\eval\eval_stage1_spex.py" --config "$Root\configs\stage1_spex_mc.yaml" --split val

& $Py "$Root\scripts\train\train_stage2_presence.py"
& $Py "$Root\scripts\eval\eval_stage2_presence.py" --split val

& $Py "$Root\scripts\train\train_stage3_kws.py"
& $Py "$Root\scripts\eval\eval_stage3_kws.py" --split test

& $Py "$Root\scripts\train\train_stage4_intent.py"
& $Py "$Root\scripts\eval\eval_stage4_intent.py" --split val

& $Py "$Root\scripts\eval\eval_baseline_raw_cabin.py"
& $Py "$Root\scripts\eval\eval_baseline_spex_sc.py"
& $Py "$Root\scripts\eval\eval_proposed_mc.py"
& $Py "$Root\scripts\eval\eval_latency.py"
& $Py "$Root\scripts\eval\eval_end_to_end.py"

Write-Host "run_full_study completed."
