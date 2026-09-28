# HC #413 — TP/SL Scalping Backtest Verdict

NPZ: `/home/jupiter/Lvl3Quant/output/hc417_v2_full_oot_wrapped_for_hc413.npz`
MFE config: `/home/jupiter/Lvl3Quant/output/hc417_v2_native_mfe_matrix.csv`
Order type: `passive_at_touch` (entry cost = 0.376 ticks)
Seed: 42

## Cells passing HC #408 honesty gate AND realized net > 0

| cell_id | n_fills | net/fill (t) | Sharpe√N | PF | WR% | day_conc | CI95_lo |
|---|---:|---:|---:|---:|---:|---:|---:|
| v2_1s_short_top05 | 639 | +0.274 | 12.77 | 2.92 | 84.8 | 0.132 | +0.232 |

## Top 10 by Sharpe√N (whether passing or not)

| cell_id | pass_408 | n_fills | net/fill (t) | Sharpe√N | PF | WR% | day_conc | CI95_lo |
|---|:-:|---:|---:|---:|---:|---:|---:|---:|
| v2_1s_short_top05 | Y | 639 | +0.274 | 12.77 | 2.92 | 84.8 | 0.132 | +0.232 |

## HC compliance
- HC #69: risk-adjusted metrics (Sharpe, Sortino, PF, WR) reported as primary.
- HC #344: day_conc reported; flag when > 0.20.
- HC #397B: canonical FIFO market replay (no midpoint shortcuts).
- HC #408: pass_hc408_honesty requires n_fills>=50, CI_low_95>0, day_conc<=0.20.
- HC #413 rule 3: TP1=0.5·MFE, TP2=1.0·MFE, SL=min(|MAE|,1.5·MFE).