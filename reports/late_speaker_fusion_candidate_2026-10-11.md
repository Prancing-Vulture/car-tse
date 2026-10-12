# Late speaker-verification fusion candidate

Date: 2026-10-11  
Status: diagnostic candidate; not promoted

## Motivation

The pre-existing-systems review recommends treating keyword evidence and
speaker identity as independently inspectable signals. The retained runtime
already has a hard pre-wake SpEx cosine gate, but it does not expose a soft
late-fusion operating point. This experiment adds an opt-in mapping from
SpEx cosine similarity to a speaker probability and combines it with the
target-KWS probability before temporal smoothing.

The default mode remains `speaker_fusion_mode: none`, so the retained
checkpoint and historical behavior are unchanged. The candidate supports
`min`, `product`, and weighted `mean` fusion. The center and scale are
operating-point parameters and are not claimed to be calibrated without
same-driver tuning data.

## Results

The positive study is the existing eight-scene cross-corpus proxy: two target
scenes, four CAVEMOVE cabin SNRs (-5, 0, +10, +20 dB), passenger speech at
-10 dB, and 112 seconds total. The negative study is the 250-file Rochester
Smart Speaker non-Marvin diagnostic (0.335 hours, clean mono studio audio
duplicated across four interface channels).

| Candidate | Positive wake | Positive intent | Positive false actions/h | Smart full-path events/h | Smart KWS-only/h | Decision |
|---|---:|---:|---:|---:|---:|---|
| Retained hard-gate reference | 100.0% | 100.0% | 0.0 | 728.1 | 1817.2 | Reference only |
| Late `min`, center .24, scale .04 | 100.0% | 100.0% | 0.0 | 728.1 | 1817.2 | No improvement |
| Late `min`, center .26, scale .04 | 87.5% | 87.5% | 0.0 | 695.3 | 1817.2 | Reject: recall loss for small negative gain |
| Late `product`, center .24, scale .04 | 25.0% | 25.0% | 0.0 | not run | not run | Reject: severe recall loss |

The center-.24 result is effectively equivalent to the retained hard gate at a
0.50 fused threshold. Tightening the center reduces a small number of Smart
negative events but also removes a target wake at the +20 dB proxy condition.
The product rule is too severe because the raw cosine-to-probability mapping
is not calibrated for this domain.

## Decision and next step

No checkpoint or default configuration was changed. The implementation is
kept as an opt-in research path because it makes the intended two-branch
design testable, but it is not evidence that the current speaker embedding is
a production-quality verifier.

The next valid improvement is to collect permitted same-driver tuning sessions
and train or calibrate a dedicated speaker-verification branch using target
speech, passenger speech, passenger “Marvin,” and no-command intervals. Only
then should the center/scale/fusion rule be selected and frozen before a
speaker-disjoint held-out test.

## Artifacts

- `src/car_tse/pipeline/in_car_pipeline.py`
- `scripts/eval/eval_continuous_full_pipeline_proxy.py`
- `scripts/eval/eval_aishell5_continuous_kws.py`
- `scripts/eval/eval_smart_speaker_command_negative.py`
- `configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_late_fusion_candidate.yaml`
- `logs/continuous_full_pipeline_proxy_late_fusion_min_c024_s004_m10_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_late_fusion_min_c026_s004_m10_2026-10-11.json`
- `logs/continuous_full_pipeline_proxy_late_fusion_product_c024_s004_m10_2026-10-11.json`
- `logs/smart_speaker_command_negative_250_late_fusion_min_c024_s004_2026-10-11.json`
- `logs/smart_speaker_command_negative_250_late_fusion_min_c026_s004_2026-10-11.json`

## Reproduction

```powershell
python scripts/eval/eval_continuous_full_pipeline_proxy.py `
  --config configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_late_fusion_candidate.yaml `
  --paths configs/paths.yaml --scenes 2 --cabin-snr-db -5 0 10 20 `
  --passenger-snr-db -10 --enrollment-source same_speaker_other `
  --speaker-fusion-mode min --speaker-fusion-center 0.26 `
  --speaker-fusion-scale 0.04 `
  --output logs/late_fusion_proxy.json

python scripts/eval/eval_smart_speaker_command_negative.py `
  --config configs/eval_end_to_end_cavemove_target_kws_dual_domain_stage2_late_fusion_candidate.yaml `
  --paths configs/paths.yaml --max-files 250 --hop-seconds 0.25 `
  --speaker-fusion-center 0.26 --action-sample-files 0 `
  --output logs/late_fusion_smart_negative.json
```
