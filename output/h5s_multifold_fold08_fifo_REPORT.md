# h5s multifold fold-08 FIFO Grade

**OOT date**: 20260426
**Gate**: q≥0.5 fill-prob (from fill_prob_head_v1.lgb, proxy hold 0.415 s)
**FIFO**: canonical (queue-aware passive limit, ES 0.376 commission, TP 2 / SL 1)

## Headline cells (canonical + gate)

| Cell | Ungated n | Ungated /trade | Gated n | Gated /trade | WR (gated) | PF (gated) |
|------|-----------|----------------|---------|--------------|------------|------------|
| 1s_top10% | 5 | -0.076 | 4 | -0.501 | 0.500 | 0.27 |
| 1s_top5% | 3 | +0.124 | 2 | -0.626 | 0.500 | 0.09 |
| 5s_top10% | 10 | -0.526 | 6 | -0.626 | 0.333 | 0.32 |

## Label-FIFO 12-cell screen

| Horizon | Top% | Trades | Net Ticks | /Trade | Sharpe | Pos? |
|---------|------|--------|-----------|--------|--------|------|
| 1s | 1% | 5 | -0.88 | -0.1760 | -955.67 | - |
| 1s | 2% | 9 | -1.88 | -0.2093 | -1437.78 | - |
| 1s | 5% | 21 | -4.90 | -0.2331 | -687.28 | - |
| 1s | 10% | 41 | -4.42 | -0.1077 | -348.35 | - |
| 1s | 20% | 81 | 43.04 | +0.5314 | 261.98 | + |
| 1s | 50% | 202 | 39.55 | +0.1958 | 141.86 | + |
| 5s | 1% | 5 | 3.12 | +0.6240 | 1071.46 | + |
| 5s | 2% | 9 | 3.12 | +0.3462 | 671.10 | + |
| 5s | 5% | 21 | 4.60 | +0.2192 | 400.68 | + |
| 5s | 10% | 41 | 58.08 | +1.4167 | 457.68 | + |
| 5s | 20% | 81 | 66.04 | +0.8154 | 356.74 | + |
| 5s | 50% | 202 | 108.05 | +0.5349 | 298.07 | + |

## Files

- Label-FIFO CSV: `output/h5s_multifold_fold08_fifo/summary.csv`
- Canonical+gate CSV: `output/h5s_multifold_fold08_canonical/summary.csv`
- Aggregate row appended to `output/h5s_multifold_AGGREGATE.md`

Auto-graded by `scripts/h5s_multifold_auto_grader.sh`.