# Stacker vs Raw Head — Full Market Replay (HC #396)

- Predictions NPZ: `/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz`
- Stacker ckpt: `/home/jupiter/Lvl3Quant/output/meta_mlp_v3_3/stacker_final.pt`
- Best raw-head config from sweep: band=P95 band_frac=0.05 side=long order_type=passive_at_touch_plus_1 cancel_window=40 hold_s=1.0 exit_horizon=1s
- Eval slab: VAL ONLY (last 20% chronological — matches stacker's held-out split)
- Val slab spans original idx [172496, 236864] / n_total=241351

## Headline (VAL SLAB, fair comparison)

| Signal | n_signals | n_filled | fill_rate | ticks/fill | Sharpe | Sortino | PF | WR (%) | day_conc |
|---|---|---|---|---|---|---|---|---|---|
| RAW head     | 631 | 75 | 0.119 | +0.6107 | 2645.7 | 6605.7 | 4.127 | 69.33 | 1.000 |
| META stacker | 591 | 68 | 0.115 | +0.9916 | 1612.3 | 4601.2 | 4.666 | 75.00 | 0.626 |

## Verdict

- Sharpe Δ (stacker − raw) = **-1033.4** → **raw** wins on Sharpe.
- ticks/fill Δ = **+0.3810** (stacker − raw)
- HC #344 (day_conc<=0.20 & n_filled>=30): raw=FAIL  stacker=FAIL

## Caveat
- HC #396 prior agent reported val IC 0.230 (4.8x raw 0.048). This script tests whether that IC boost translates to PnL after queue-position + adverse-selection + commission realism.
- Full-slab numbers are appended for context but are BIASED for the stacker (includes its train rows).