# HC #413 — TP/SL Scalping Backtest Verdict

NPZ: `/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz`
MFE config: `/home/jupiter/Lvl3Quant/output/hc411_regime_agnostic_20260517_215211/mfe_at_confidence_matrix.csv`
Order type: `passive_at_touch` (entry cost = 0.376 ticks)
Seed: 42

## Cells passing HC #408 honesty gate AND realized net > 0

**NONE.** No (model × horizon × side × tier) cell cleared n_fills>=50 AND CI_low_95>0 AND day_conc<=0.20 AND net>0.


## Top 10 by Sharpe√N (whether passing or not)

| cell_id | pass_408 | n_fills | net/fill (t) | Sharpe√N | PF | WR% | day_conc | CI95_lo |
|---|:-:|---:|---:|---:|---:|---:|---:|---:|
| v3.3_30s_long_top05 | N | 91 | +0.253 | 1.92 | 1.64 | 83.5 | 0.348 | -0.008 |
| v3.3_1s_long_top05 | N | 180 | +0.018 | 0.47 | 1.08 | 65.0 | 0.304 | -0.061 |
| v3.3_30s_long_top5 | N | 881 | +0.013 | 0.40 | 1.03 | 76.2 | 0.574 | -0.052 |
| v3.3_5s_long_top1 | N | 318 | -0.017 | -0.37 | 0.95 | 72.3 | 0.488 | -0.112 |
| v3.3_10s_long_top05 | N | 151 | -0.043 | -0.49 | 0.91 | 75.5 | 0.374 | -0.223 |
| v3.3_30s_long_top1 | N | 190 | -0.051 | -0.60 | 0.89 | 78.4 | 0.340 | -0.218 |
| v3.3_5s_long_top05 | N | 165 | -0.092 | -1.44 | 0.76 | 70.3 | 0.396 | -0.222 |
| v3.3_10s_long_top5 | N | 1020 | -0.046 | -1.74 | 0.88 | 75.6 | 0.498 | -0.101 |
| v3.3_1s_long_top1 | N | 366 | -0.060 | -2.29 | 0.77 | 65.3 | 0.351 | -0.110 |
| v3.3_30s_long_top10 | N | 1825 | -0.050 | -2.63 | 0.86 | 76.1 | 0.584 | -0.089 |

## HC compliance
- HC #69: risk-adjusted metrics (Sharpe, Sortino, PF, WR) reported as primary.
- HC #344: day_conc reported; flag when > 0.20.
- HC #397B: canonical FIFO market replay (no midpoint shortcuts).
- HC #408: pass_hc408_honesty requires n_fills>=50, CI_low_95>0, day_conc<=0.20.
- HC #413 rule 3: TP1=0.5·MFE, TP2=1.0·MFE, SL=min(|MAE|,1.5·MFE).