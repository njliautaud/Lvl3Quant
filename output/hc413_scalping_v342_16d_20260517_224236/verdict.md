# HC #413 — TP/SL Scalping Backtest Verdict

NPZ: `output/v342_fold_00_ep1_oot_inference_extended_wrapped.npz`
MFE config: `output/hc411_regime_agnostic_20260517_215211/mfe_at_confidence_matrix.csv`
Order type: `passive_at_touch` (entry cost = 0.376 ticks)
Seed: 42

## Cells passing HC #408 honesty gate AND realized net > 0

**NONE.** No (model × horizon × side × tier) cell cleared n_fills>=50 AND CI_low_95>0 AND day_conc<=0.20 AND net>0.


## Top 10 by Sharpe√N (whether passing or not)

| cell_id | pass_408 | n_fills | net/fill (t) | Sharpe√N | PF | WR% | day_conc | CI95_lo |
|---|:-:|---:|---:|---:|---:|---:|---:|---:|
| v3.4.2_1s_long_top1 | N | 478 | +0.155 | 5.87 | 1.71 | 70.5 | 0.274 | +0.102 |
| v3.4.2_1s_short_top05 | N | 197 | +0.237 | 5.00 | 2.15 | 76.1 | 0.227 | +0.142 |
| v3.4.2_1s_short_top1 | N | 408 | +0.105 | 3.38 | 1.41 | 70.6 | 0.252 | +0.041 |
| v3.4.2_1s_long_top05 | N | 245 | +0.114 | 2.95 | 1.52 | 71.8 | 0.286 | +0.034 |
| v3.4.2_5s_long_top05 | N | 306 | +0.096 | 1.83 | 1.27 | 75.8 | 0.323 | -0.005 |
| v3.4.2_5s_short_top1 | N | 404 | +0.076 | 1.79 | 1.23 | 78.0 | 0.347 | -0.008 |
| v3.4.2_30s_long_top1 | N | 533 | +0.133 | 1.72 | 1.18 | 75.2 | 0.284 | -0.011 |
| v3.4.2_30s_long_top05 | N | 333 | +0.104 | 0.84 | 1.11 | 71.8 | 0.368 | -0.159 |
| v3.4.2_5s_short_top05 | N | 187 | +0.029 | 0.43 | 1.08 | 76.5 | 0.201 | -0.109 |
| v3.4.2_10s_long_top05 | N | 276 | -0.033 | -0.40 | 0.95 | 72.8 | 0.302 | -0.201 |

## HC compliance
- HC #69: risk-adjusted metrics (Sharpe, Sortino, PF, WR) reported as primary.
- HC #344: day_conc reported; flag when > 0.20.
- HC #397B: canonical FIFO market replay (no midpoint shortcuts).
- HC #408: pass_hc408_honesty requires n_fills>=50, CI_low_95>0, day_conc<=0.20.
- HC #413 rule 3: TP1=0.5·MFE, TP2=1.0·MFE, SL=min(|MAE|,1.5·MFE).