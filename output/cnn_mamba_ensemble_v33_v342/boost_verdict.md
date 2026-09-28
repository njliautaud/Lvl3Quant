# Boosting (a) verdict — ensemble-avg v3.3 + v3.4.2

HC #427 R5 — boosting technique #1 (ensemble averaging).
Generated: $(date)

## Ensemble NPZ
- Path: `/home/jupiter/Lvl3Quant/output/cnn_mamba_ensemble_v33_v342/fold_00_predictions.npz`  (15.77 MB)
- Keys averaged: 32 `pred_*` heads
- Keys copied: 68 (targets / masks / meta)
- Sanity: target_* arrays identical between v3.3 & v3.4.2 (same OOT data)

## Robust-config counts comparison (top-20 configs from each)

| source | preds used        | n_tested | n_robust |
|---|---|---|---|
| v3.3 sweep top configs   | v3.3 SOLO   | 20 | 7 |
| v3.3 sweep top configs   | ENSEMBLE    | 20 | 10 |
| v3.4.2 sweep top configs | v3.4.2 SOLO | 20 | 12 |
| v3.4.2 sweep top configs | ENSEMBLE    | 20 | 11 |

## Top-5 ensemble-boosted robust configs (v3.4.2 top-K basis)

- trial=1110 | 5s/long/passive_at_touch_plus_2 | mean_Sh=20.1 worst_Sh=12.28 fills=77
- trial=1554 | 30s/short/passive_at_touch_plus_2 | mean_Sh=12.03 worst_Sh=0.0 fills=68
- trial=479 | 5s/long/passive_at_touch_plus_2 | mean_Sh=23.4 worst_Sh=11.77 fills=74
- trial=2593 | 30s/short/passive_at_touch_plus_2 | mean_Sh=19.09 worst_Sh=0.0 fills=50
- trial=1207 | 5s/short/passive_at_touch_plus_2 | mean_Sh=15.02 worst_Sh=0.0 fills=34

## Top-5 ensemble-boosted robust configs (v3.3 top-K basis)

- trial=2837 | 5s/short/passive_at_touch_plus_2 | mean_Sh=16.68 worst_Sh=8.09 fills=36
- trial=504 | 5s/short/passive_at_touch_plus_2 | mean_Sh=17.0 worst_Sh=8.06 fills=33
- trial=1683 | 30s/short/passive_at_touch_plus_2 | mean_Sh=17.64 worst_Sh=0.0 fills=61
- trial=1286 | 30s/short/passive_at_touch_plus_2 | mean_Sh=17.54 worst_Sh=7.11 fills=31
- trial=1947 | 5s/short/passive_at_touch_plus_2 | mean_Sh=19.04 worst_Sh=13.35 fills=36

## Interpretation

- If ensemble n_robust > solo n_robust on either basis → ensemble averaging boosts robustness; at least one ensemble-boosted setup qualifies for the HC #426 R1 / HC #427 R5 Friday-5/22 three.
- If ensemble n_robust < solo n_robust → models disagree such that averaging dilutes signal. Try weighted ensemble or rank-averaging next.
- If approximately equal → no harm done; still passes HC #427 R5 "boosting technique tested" gate but offers no new setup.