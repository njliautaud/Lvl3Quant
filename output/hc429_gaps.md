# HC #429 — conditional MFE matrix gaps

Generated: 2026-05-19T09:03:14.997688

- v2 baseline: `/home/jupiter/Lvl3Quant/output/hc417_v2_native_mfe_matrix.csv`
- v3.3 NPZ: `/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz`
- v3.4.2 NPZ: `/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/fold_00_ep1_oot.npz`
- Cost (passive at touch + commission): 0.376 tk

## Gaps / caveats

- **v3.4.2**: NPZ has no `oot_dates` field — date_idx collapsed to single-day. day_conc in this matrix is non-meaningful; aggregate MFE/MAE/net/WR/n are unaffected.

## MAE methodology

- **30s MAE**: true intra-horizon adverse from `target_pred_mae_30s_ticks` (where available).
- **1s/5s/10s MAE**: PROXY = mean magnitude of negative-only signed realized horizon-end moves. This is a LOWER BOUND on true intra-window adverse (true MAE is at least this big). Identical methodology to `output/hc417_v2_native_mfe_matrix.csv` so the matrices are directly comparable.