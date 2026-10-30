# Repository validation snapshot — 12 October 2026

## Scope and working-tree condition

This check followed the pipeline failure audit and diagnostic guide. It inspected the current runtime/evaluator and ran the repository tests in the requested `audioml_base` environment. It did not change runtime code, model weights, or SpEx, and it did not launch a model evaluation, training job, sweep, or W&B run.

At the start, `git status --short` showed an already-dirty working tree: 16 modified tracked files and many untracked reports, configs, scripts, tests, and `tmp/` artifacts. In particular, `src/car_tse/pipeline/in_car_pipeline.py`, `tests/test_pipeline_drop.py`, and the new tests were already modified/untracked before validation. Test results therefore describe the current shared worktree; they do not establish which edits introduced behavior relative to a clean baseline. Those changes were preserved.

## Environment and commands

The `conda` command was absent from the PowerShell PATH. The registered environment was available by its explicit executable path:

```powershell
& C:\Users\Harreesh\miniconda3\Scripts\conda.exe env list
& C:\Users\Harreesh\miniconda3\Scripts\conda.exe run -n audioml_base python -c "import sys, torch, pytest; print(sys.executable, torch.__version__, pytest.__version__)"
```

The environment resolved to `C:\Users\Harreesh\miniconda3\envs\audioml_base\python.exe`, Python from that environment, PyTorch `2.12.0+cu132`, and pytest `9.1.1`.

The focused pipeline/evaluator regression run used one CPU thread and passed:

```powershell
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
& C:\Users\Harreesh\miniconda3\Scripts\conda.exe run --no-capture-output -n audioml_base python -m pytest -q tests/test_pipeline_drop.py tests/test_continuous_proxy_attribution.py tests/test_stage3_eval_threshold.py tests/test_stage3_noise_split.py
```

Result: **15 passed in 6.33 s**.

I then collected and ran the full existing test suite with CUDA disabled and the same CPU limits:

```powershell
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:CUDA_VISIBLE_DEVICES=''
& C:\Users\Harreesh\miniconda3\Scripts\conda.exe run --no-capture-output -n audioml_base python -m pytest --collect-only -q tests
& C:\Users\Harreesh\miniconda3\Scripts\conda.exe run --no-capture-output -n audioml_base python -m pytest -q tests
```

Result: **23 collected; 22 passed, 1 skipped in 8.55 s**. The skipped test is `tests/test_spex_adapter.py::test_load_pretrained`, skipped because `checkpoints/spex_pretrained/best.pt.tar` is absent. No failures occurred. Since the suite was run on a pre-existing dirty tree, there is no basis to label a passing or failing result as introduced by this validation or by a particular uncommitted change.

After the standalone matched-diagnostic helper and tests were added, I repeated validation in `audioml_base` with CUDA disabled and one CPU thread:

```powershell
& C:\Users\Harreesh\miniconda3\Scripts\conda.exe run --no-capture-output -n audioml_base python -m py_compile scripts/eval/replay_matched_continuous_diagnostic.py tests/test_matched_continuous_diagnostic.py
& C:\Users\Harreesh\miniconda3\Scripts\conda.exe run --no-capture-output -n audioml_base python -m pytest -q tests
```

`py_compile` succeeded. The latest full suite collected 27 tests and reported **26 passed, 1 skipped in 8.51 s**; the same missing pretrained SpEx checkpoint is the skip. The new matched-diagnostic tests cover manifest identity/range/file checks, observer-vs-baseline output and exact Stage 4 input parity, padded final-window pre-roll mapping, and hook cleanup. No model inference on project data, retraining, threshold sweep, or W&B run was performed.

The post-test `git diff --name-only` still listed only the 16 tracked paths already modified in the initial status; no tracked user file was edited for this validation. The helper, tests, and reports remain untracked additions in the shared tree.

## What the tests cover

`tests/test_pipeline_drop.py` currently covers low-presence rejection, KWS temporal hits, post-wake intent dispatch, duplicate-sample removal for two overlapping timestamped windows, one early pre-roll case with framewise presence, variable-size timeout accumulation, and avoiding intent inference when wake pre-roll is the only speech. `tests/test_continuous_proxy_attribution.py` covers wake and command event attribution and independent speaker-disjoint scene source selection. Stage 3 threshold and noise split tests also pass.

The requested stream edge cases have this coverage status:

| Behavior | Current evidence | Status |
|---|---|---|
| Timestamp overlap / duplicate samples | One two-window regression in `test_pipeline_buffers_only_new_audio_from_overlapping_windows`; intent receives `[0.3, 0.4]` once. | Covered narrowly; not varied for fully contained windows, gaps, backwards timestamps, or resampling. |
| Rejected-gap spans | No runtime test verifies source-time spans or exact samples across a presence-rejected interval. The v10 evaluator recorded appended source segments in its synthetic scenes, but this does not establish what happens on same-speaker runtime audio. | Missing evidence, not a confirmed runtime defect. |
| Early command pre-roll | One framewise-presence test confirms later speech in the wake context can become pre-roll; the scalar-presence path intentionally waits for a later call. | Covered narrowly; no tested wake/command boundary matrix or SpEx audio. |
| Endpoint and timeout on the same frame | No direct regression test. In current code the silence-end inference check runs before timeout reset, so endpoint inference wins when both conditions become true on one call. | Behavior is code-order evidence only; intended precedence is not locked by a test. |
| Switching between timestamped and untimestamped calls | No regression test. `_new_audio_segment` treats an untimestamped call as fully new and leaves the previous timestamp end stored. | Unverified edge case; do not call it a cause without a reproducing trace. |
| Command longer than the Stage 4 window | No duration/truncation test. Both the pipeline and wav2vec model use a fixed 2-second input; excess samples are clipped from the end. | A known fixed-window contract and possible coverage gap for longer commands, not a proven failure for the labeled command set. |

These gaps justify adding targeted tests alongside the first matched-speaker instrumentation, especially for timestamp-mode transitions, rejected time gaps, and endpoint/timeout precedence. They do not demonstrate that those paths caused the existing proxy failures.

## Confirmed measurement problems versus unresolved prerequisites

The audit identifies confirmed historical measurement problems: the v9 capture-time estimate assumed contiguous buffered audio even though the presence gate can skip chunks; v10 superseded that calculation with exact appended source segments. The audit also confirms that the continuous proxy independently selects wake and command examples without matching their speakers. The proxy's 0/8 result is therefore a cross-corpus engineering diagnostic, not a same-driver performance estimate. These conclusions and the v10 artifact are documented in `PIPELINE_ERRORS_AUDIT.md`, `PIPELINE_DIAGNOSTIC_GUIDE.md`, and `diagnostic_v10_command_capture_kws029_two_snr.json`.

The current test run found **no reproducible runtime failure** in its covered cases. It does not test a same enrolled person speaking both wake and command, and it does not show that command words survive SpEx. The parallel data check found no valid same-person replay in the current custom recording directories. A labeled same-speaker continuous recording with source, extracted waveform, every presence decision, exact Stage 4 samples, and output is still a prerequisite to identifying which stage loses the command. No single runtime root cause is established.

## Timing and metric interpretation

The runtime config and v10 config use a 3-second Stage 1 context and a 1-second KWS crop. KWS scoring uses the first one-second crop, while each decision is made from a window containing the full three-second context. The continuous evaluator sets `decision_available_seconds` to the context window end. That timestamp accounts for audio lookahead needed to fill the window, but it does not include measured model compute time. Consequently, its latency is not a wall-clock device latency measurement. The 3-second context also means the KWS evidence from the beginning of a window can only be acted on once the rest of the context is available in this evaluation path; whether that buffering is acceptable needs an explicitly streaming, same-speaker test.

The evaluator's metric names have distinct meanings. `target_intent_correct` asks whether the command event attributed to the target wake and command time has the expected intent. `false_action_events` counts command events not attributed to that target command. A wrong intent emitted for the correctly attributed target command lowers target intent success but is not also counted as a false action by this definition. On target-absent hours, false actions should be measured separately from wrong actions on legitimate target commands. Reports should give target misses, wrong target intents/actions, and unsolicited actions with clear, potentially overlapping event definitions.

The simulated `ActionResult` is a label-map result or `noop_*` placeholder; `actions.py` describes it as a stub. Thus “correct action” in the continuous proxy means the intent dispatch label matched the expected label. It does not verify a real vehicle command or physical execution.

Stage 4 is configured for a 2-second input. The wav2vec wrapper clips longer inputs to the first 2 seconds, pads shorter inputs, then normalizes the fixed-length waveform. That matches the current fixed-length model path but does not show that the command endpoint or a longer utterance is represented completely. The matched recording test should log the exact 2-second tensor and separately assess whether the intended command was truncated.

## W&B status

Evaluation logging is opt-in in the current scripts. The continuous proxy, Stage 3 evaluator, and Stage 4 evaluator expose `--wandb`; Stage 2 also exposes it. Their W&B calls are guarded by `if args.wandb`. The helper `src/car_tse/train/wandb_logging.py` initializes a run, logs scalar metrics and optional confusion/scene/window tables, and can attach the JSON report artifact. Therefore simply having W&B installed or authenticated does not enable logging.

The v10 audit and diagnostic guide cite an existing W&B run URL for the evaluation, which is evidence that this evaluation has a corresponding W&B run. However, the local v10 JSON provenance does not record the command line, a W&B run ID, or a run URL, so the local artifact cannot prove which exact flag invocation created it. The new validation performed no model evaluation, so it did not create a W&B run. For the next model evaluation, pass `--wandb` explicitly (and choose `--wandb-mode online` or `offline`) and retain the run URL or run ID in the local report.

## Evidence files

- `src/car_tse/pipeline/in_car_pipeline.py`: timestamp deduplication, timeout, endpoint, KWS crop, command capture, and fixed-length intent preparation.
- `tests/test_pipeline_drop.py`: existing pipeline stream regressions described above.
- `scripts/eval/eval_continuous_full_pipeline_proxy.py`: decision timestamp, event attribution, metric definitions, and opt-in W&B logging.
- `src/car_tse/models/wav2vec_intent.py` and `src/car_tse/pipeline/actions.py`: Stage 4 input contract and stub action mapping.
- `scripts/eval/eval_stage3_kws.py`, `scripts/eval/eval_stage2_frame_presence.py`, and `scripts/eval/eval_stage4_intent_wav2vec.py`: opt-in evaluator W&B CLI paths.
- `reports/evaluation/PIPELINE_ERRORS_AUDIT.md`, `reports/evaluation/PIPELINE_DIAGNOSTIC_GUIDE.md`, and `reports/evaluation/diagnostic_v10_command_capture_kws029_two_snr.json`: historical measurement corrections and v10 evidence.
