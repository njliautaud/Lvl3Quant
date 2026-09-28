# Trade Tape Imbalance v1 — Report

**Generated:** 2026-05-22T14:03:14
**OOT range:** 20260223 .. 20260429  (weekday trading days)
**N dates evaluated:** 48
**Wall time:** 185.4s

## Hypothesis
Aggressor-side trade-tape imbalance (large sweeps vs small absorptive
trades) is predictive of forward 5s/10s mid-price moves and is
independent of BBO-level queue/OFI features in v3.4.2.

## Top features by |median per-date Spearman| vs realized 10s tick move
| Feature | Median IC | Mean IC | Std | Frac days >0 |
|---|---:|---:|---:|---:|
| `tape_imbalance_10s__y10` | -0.0357 | -0.0303 | 0.0273 | 0.13 |
| `large_aggressor_flow_10s__y10` | -0.0296 | -0.0342 | 0.0354 | 0.13 |
| `large_aggressor_flow_5s__y10` | -0.0261 | -0.0322 | 0.0327 | 0.15 |
| `tape_imbalance_5s__y10` | -0.0248 | -0.0263 | 0.0236 | 0.17 |
| `large_aggressor_flow_1s__y10` | -0.0217 | -0.0242 | 0.0232 | 0.15 |
| `small_aggressor_flow_1s__y10` | -0.0153 | -0.0116 | 0.0241 | 0.28 |
| `tape_imbalance_1s__y10` | -0.0135 | -0.0132 | 0.0203 | 0.23 |
| `small_aggressor_flow_5s__y10` | -0.0104 | -0.0117 | 0.0321 | 0.34 |
| `sweep_count_1s__y10` | -0.0101 | -0.0069 | 0.0275 | 0.38 |
| `sweep_count_5s__y10` | -0.0090 | -0.0084 | 0.0311 | 0.43 |

## Headline strategy: top/bottom-5% tape_imbalance_10s combined long+short, passive cost 0.376t

- Pooled net (per-event mean):  **-0.4725 ticks**
- Profit days ratio: **0.11** (5 / 47)
- Day Sharpe (annualized): **-9.67**
- Day concentration: **0.11**  (cap 0.70)
- Long pooled net:  -0.4669 ticks
- Short pooled net: -0.4781 ticks
- Best side: **long**

## Regime stratification (annualized day-Sharpe)
- green-day Sharpe: -9.89
- red-day   Sharpe: -16.79
- flat-day  Sharpe: -5.74
- regime imbalance (|green-red|/max): **0.41**  (cap 0.50)

## Sweep vs Absorption (large vs small aggressor flow)
- SWEEP (large) — |median IC|=0.0296 vs absorption 0.0078

## Leave-one-day-out stability — best feature (tape_imbalance_10s__y10)
- Base median IC: -0.0357
- LOO median IC range: [-0.0357, -0.0356],  std = 0.0000

## Deploy gates (HC #428)
- ACCEPT requires: pooled_net >= +0.10 AND profit_days >= 0.60 AND Sharpe >= 0.30 AND day_conc <= 0.70 AND regime_imbalance <= 0.50
- PARTIAL requires: pooled_net >= +0.05 AND profit_days >= 0.55

## VERDICT: **REJECT**

## Honest caveats
- This is **label-level (mid-price) P&L**, not FIFO market replay. HC #74 says FIFO is canonical for any production decision. If verdict is PASS/PARTIAL the next step is a FIFO fill-sim run.
- Sample is 48 trading days. Statistical robustness modest.
- Regime classifier uses mean(labels_10s) as proxy for ES close-to-close direction (no external SPX close fetched here).
- Passive cost (0.376 ticks) assumes both legs fill at the resting price without queue jumping; in reality top-decile tape-imbalance moments are exactly when the queue runs you over. Real FIFO net likely worse.