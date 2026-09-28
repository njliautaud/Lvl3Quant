# Sizing Calibrator Replay Verdict (HC #396 follow-up)

Date: 2026-05-16 17:33:49
Source preds: `/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz`
Calibrator: `/home/jupiter/Lvl3Quant/output/meta_mlp_v3_3/sizing/sizing_calibrator_final.pt`
Operational config: head=`fifo_tp8sl5_net` band_frac=0.05 side=long otype=passive_at_touch_plus_1 cw=40 hold=1.0 exit=1s
Val slab: last 20% of mask-valid rows of `target_fifo_tp8sl5_net`.

## Diagnostics
- Calibrator val IC vs `|realized_net|` at FILLED rows only: **+0.1295**
- Calibrator IC vs SIGNED `realized_net` (sanity check; should be ~0 if calibrator only learned magnitude): **+0.1715**

## Per-variant metrics

| variant | n_filled | ticks_total | ticks/fill | Sharpe | Sortino | PF | WR (%) | max_dc | adv_sel_30s | avg_q_pos | cw | day_conc | HC344 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_equal_sized | 75 | 45.80 | 0.6107 | 2645.7 | 6605.7 | 4.127 | 69.33 | 3.26 | 0.720 | 2.000 | 40 | 1.0000 | False |
| B_continuous_clip_0.5_2.0 | 75 | 46.03 | 0.6137 | 2632.2 | 6509.3 | 4.398 | 69.33 | 2.69 | 0.720 | 2.000 | 40 | 1.0000 | False |
| C_quintile_0.5_1.0_1.5 | 75 | 50.80 | 0.6773 | 2615.4 | 7351.3 | 4.846 | 69.33 | 2.07 | 0.720 | 2.000 | 40 | 1.0000 | False |

## Deltas vs baseline (A = equal-sized)

- B (continuous): Sharpe Δ = -0.5% | PF Δ = +6.6% | day_conc 1.0000→1.0000
- C (quintile):   Sharpe Δ = -1.1% | PF Δ = +17.4% | day_conc 1.0000→1.0000

## Verdict
- HC #344 still fails on all three (day_conc > 0.20 on tiny val slab); not a production decision, just a relative-improvement test.
- Sizing improves risk-adjusted returns IFF Sharpe Δ > 0 AND day_conc doesn't worsen. See the table above.