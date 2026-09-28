# longer_horizon_v1 — VERDICT
Generated: 2026-06-05T09:53:07.552875

## (a) WHAT RAN — N trades / N days per family

| family | n_trade_rows | n_days |
|---|---|---|
| A | 98941 | 28 |
| A_inv | 98941 | 28 |
| B1 | 90824 | 28 |
| B2 | 2648 | 28 |
| B3 | 1257 | 28 |
| C | 84 | 14 |
| C_short | 84 | 14 |

## (b) TOP 5 CELLS by per-day Sharpe (ES market RT cost — HEADLINE)

 family               signal side_label  horizon_m      pct  n_trades  n_days  trades_per_day  mean_daily_usd    sharpe   sortino        pf   wr_day  day_conc  regime_gap  passes_gates
      C        C_dayclf_top5       long       9999 0.172414         5       5             1.0     1982.800000 21.981680       NaN 19.716254 0.800000  0.274099    0.000000         False
      C        C_dayclf_top5       long        120 0.172414         5       5             1.0      575.300000  7.959613 51.661812  4.347103 0.600000  0.426044    0.000000         False
C_short C_dayclf_top10_SHORT      short         60 0.344828         9       9             1.0      313.355556  6.319021 11.971443  2.953318 0.555556  0.303585    0.992009         False
C_short C_dayclf_top15_SHORT      short         60 0.517241        14      14             1.0      191.728571  4.089672  5.703055  1.999330 0.642857  0.215089    1.337325         False
      C       C_dayclf_top10       long       9999 0.344828         9       9             1.0      502.244444  3.361276  5.081980  1.684713 0.666667  0.205502    1.722205         False

## (c) CELLS PASSING ALL GATES (per family separately)

NONE — no cell passes the HC #428 R1 gates at ES market RT execution cost in this window.

## (d) VERDICT — does longer-horizon execution work?

**NO — no cell passes the HC #428 R1 gates at retail-grade execution (ES market RT) in this window.**

Best Sharpe achieved: 21.98 (C / C_dayclf_top5 / horizon=9999m / side=long / pct=0.17; $/day=1982.80, PF=19.72, WR_day=0.80).

Best $/day: 14281.14 (B1 / B1_mom_30m / horizon=9999m / side=momentum).

- **Family A** closest cell (v342_30s/h=30m/predicted): blocked by PF=1.22≤1.4, WR_day=0.36≤0.55, regime_gap=1.93>0.5.
- **Family A_inv** closest cell (v342_30s_INV/h=30m/inverted): blocked by Sharpe=0.37≤1.5, PF=1.02≤1.4, regime_gap=1.44>0.5.
- **Family B1** closest cell (B1_mom_15m/h=9999m/momentum): blocked by PF=1.15≤1.4, WR_day=0.39≤0.55, regime_gap=0.59>0.5.
- **Family B2** closest cell (B2_meanrev_z2/h=15m/meanrev): blocked by Sharpe=-0.18≤1.5, PF=0.99≤1.4, regime_gap=1.57>0.5, day_conc=1.00>0.7.
- **Family B3** closest cell (B3_atr_k1.0/h=15m/breakout): blocked by PF=1.20≤1.4, WR_day=0.40≤0.55, regime_gap=1.55>0.5, n_days=5<20, n_trades=32<50.
- **Family C** closest cell (C_dayclf_top5/h=9999m/long): blocked by n_days=5<20, n_trades=5<50.
- **Family C_short** closest cell (C_dayclf_top10_SHORT/h=60m/short): blocked by regime_gap=0.99>0.5, n_days=9<20, n_trades=9<50.

## (e/f) RECOMMENDED CONFIG / NEXT LANE

Since nothing survives at retail execution costs in this 32-day Feb-Apr 2026 window, the microstructure-research lane should shift to:

1. **DIFFERENT SAMPLE WINDOW** — Try 2025 H2 (Aug-Dec) or 2024 high-vol regimes where signal-to-noise may be materially different.
2. **DIFFERENT ASSET CLASS** — ES at retail cost (~1 tick spread + $4.70 RT comm) leaves only ~0.6 ticks/trade for net edge AFTER costs; even a 70%-WR/1.5-tick-edge system clears just ~0.2 net ticks. Consider: NQ (lower relative cost % of move), CL (wider ranges per session), or ZN/ZB (lower volatility but tight book).
3. **PROFESSIONAL EXECUTION** — passive queue with rebates and adverse-selection-aware ordering could shift the calculus, but only if a serious queue simulator is built.

## (informational) PASSIVE EXEC TOP 5 — for context (NOT used for headline gates)

 family               signal side_label  horizon_m      pct  n_trades  n_days  mean_daily_usd    sharpe        pf   wr_day  day_conc  regime_gap  passes_gates
      C        C_dayclf_top5       long       9999 0.172414         5       5      995.300000 22.068152 20.070703 0.800000  0.274225    0.000000         False
      C        C_dayclf_top5       long        120 0.172414         5       5      291.550000  8.067530  4.455203 0.600000  0.427017    0.000000         False
C_short C_dayclf_top10_SHORT      short         60 0.344828         9       9      160.577778  6.476313  3.041675 0.666667  0.304205    0.970128         False
C_short C_dayclf_top15_SHORT      short         60 0.517241        14      14       99.764286  4.256050  2.054073 0.714286  0.215059    1.313648         False
      A             v342_30s  predicted         15 0.010000      2364      28     1296.712500  3.421461  1.200606 0.500000  0.144735    0.313489         False

Note: passive numbers apply a 50% gross-PnL discount to account for adverse selection on filled-only-when-favorable; this is a coarse proxy, not a rigorous queue simulation.
