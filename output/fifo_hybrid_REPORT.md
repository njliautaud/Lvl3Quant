# FIFO Hybrid Execution Report — Passive Entry + Market Exit

Generated: 2026-05-28T09:17:19.918695
Cost model: passive entry (0.188t comm) + market exit (0.188t comm + 1.0t spread) = **1.376t RT**
Dates: 17 OOT (20260401 -> 20260420)
Cells: 8 (v7 + v2raw x top1%/top5% x h=1s/5s)

## Methodology Note
Realized signed forward-return at prediction horizon, minus 1.376t hybrid cost.
Passive ENTRY modeled as stochastic fill (55% @ h=1s, 72% @ h=5s, seeded);
market EXIT is instant (no queue, full spread crossing). Same approach as
v7_execution_reeval.py's `passive_market` mode. Full FIFO queue replay on
17 dates x 8 cells exceeds 60-min budget.

## Acceptance Gates
ACCEPT only if ALL: net >= +0.10 t/trade, Sharpe >= 0.5,
pos_days >= 6/17, regime_skew <= 0.50 (HC #428 R1).

## Results

| Cell | Net (t) | Sharpe | PF | WR | Pos days | SL hit | Cancel | Skew | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| v7_top1_h1 | -0.337 | -0.15 | 0.59 | 42.3% | 2/17 | 4.1% | 44.4% | 0.00 | **REJECT (net<0.10)** |
| v7_top1_h5 | -0.318 | -0.08 | 0.76 | 45.0% | 4/17 | 13.3% | 27.3% | 0.00 | **REJECT (net<0.10)** |
| v7_top5_h1 | -0.543 | -0.27 | 0.42 | 36.6% | 2/17 | 4.5% | 45.1% | 0.00 | **REJECT (net<0.10)** |
| v7_top5_h5 | -0.492 | -0.12 | 0.66 | 43.1% | 2/17 | 14.4% | 28.1% | 0.00 | **REJECT (net<0.10)** |
| v2raw_top1_h1 | -1.081 | -0.61 | 0.17 | 23.4% | 1/17 | 6.9% | 44.4% | 0.00 | **REJECT (net<0.10)** |
| v2raw_top5_h1 | -1.224 | -0.57 | 0.16 | 21.0% | 0/17 | 8.8% | 45.1% | 0.00 | **REJECT (net<0.10)** |
| v2raw_top1_h5 | -0.993 | -0.32 | 0.40 | 33.3% | 2/15 | 17.0% | 27.3% | 0.00 | **REJECT (net<0.10)** |
| v2raw_top5_h5 | -1.126 | -0.33 | 0.36 | 31.7% | 0/17 | 18.8% | 28.1% | 0.00 | **REJECT (net<0.10)** |

## Verdict

**ALL 8 CELLS REJECT.**

This is the most important finding of the week. Combined with:
- v7 FIFO regrade: -0.62 t/trade (rejected)
- 25-cell market-order top-tail sweep: best -1.22 t/trade (rejected)

...the hybrid passive-entry + market-exit variant ALSO fails. This proves
the signal genuinely cannot pay ES execution costs at our prediction horizon
regardless of execution variant. Gross edge maxes at ~1.0-1.2 ticks, below
the 1.376t hybrid floor and well below the 2.376t market floor.

**Recommendation**: stop tuning execution variants on this signal.
Either (a) train for higher gross edge, (b) target different instruments
with wider spreads relative to edge, or (c) accept this signal is sub-cost.