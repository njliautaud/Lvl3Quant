# HC #413 — TP/SL Scalping Backtest Verdict

NPZ: `output/v342_ep1_eval/fold_00_ep1_oot_wrapped.npz`
MFE config: `output/hc411_regime_agnostic_20260517_215211/mfe_at_confidence_matrix.csv`
Order type: `passive_at_touch` (entry cost = 0.376 ticks)
Seed: 42

## Cells passing HC #408 honesty gate AND realized net > 0

**NONE.** No (model × horizon × side × tier) cell cleared n_fills>=50 AND CI_low_95>0 AND day_conc<=0.20 AND net>0.


## Top 10 by Sharpe√N (whether passing or not)

| cell_id | pass_408 | n_fills | net/fill (t) | Sharpe√N | PF | WR% | day_conc | CI95_lo |
|---|:-:|---:|---:|---:|---:|---:|---:|---:|
| v3.4.2_1s_long_top1 | N | 173 | +0.256 | 6.46 | 2.62 | 78.6 | 0.262 | +0.183 |
| v3.4.2_1s_short_top1 | N | 263 | +0.170 | 4.63 | 1.79 | 75.3 | 0.351 | +0.102 |
| v3.4.2_1s_long_top05 | N | 79 | +0.240 | 3.74 | 2.49 | 79.7 | 0.376 | +0.115 |
| v3.4.2_1s_short_top05 | N | 131 | +0.209 | 3.73 | 2.06 | 77.1 | 0.307 | +0.096 |
| v3.4.2_30s_long_top1 | N | 155 | +0.296 | 2.36 | 1.56 | 81.3 | 0.337 | +0.059 |
| v3.4.2_1s_short_top5 | N | 1281 | +0.028 | 2.12 | 1.14 | 75.0 | 0.365 | +0.002 |
| v3.4.2_5s_short_top1 | N | 253 | +0.101 | 1.91 | 1.32 | 79.1 | 0.449 | -0.000 |
| v3.4.2_1s_long_top5 | N | 907 | +0.028 | 1.79 | 1.14 | 71.6 | 0.399 | -0.002 |
| v3.4.2_5s_long_top05 | N | 95 | +0.117 | 1.27 | 1.35 | 76.8 | 0.357 | -0.059 |
| v3.4.2_5s_short_top05 | N | 127 | +0.047 | 0.59 | 1.14 | 78.7 | 0.449 | -0.108 |

## HC compliance
- HC #69: risk-adjusted metrics (Sharpe, Sortino, PF, WR) reported as primary.
- HC #344: day_conc reported; flag when > 0.20.
- HC #397B: canonical FIFO market replay (no midpoint shortcuts).
- HC #408: pass_hc408_honesty requires n_fills>=50, CI_low_95>0, day_conc<=0.20.
- HC #413 rule 3: TP1=0.5·MFE, TP2=1.0·MFE, SL=min(|MAE|,1.5·MFE).