# h5s multifold fold-01 FIFO Grade

**OOT date**: 20260413
**Gate**: q≥0.5 fill-prob (from fill_prob_head_v1.lgb, proxy hold 0.415 s)
**FIFO**: canonical (queue-aware passive limit, ES 0.376 commission, TP 2 / SL 1)

## Headline cells (canonical + gate)

| Cell | Ungated n | Ungated /trade | Gated n | Gated /trade | WR (gated) | PF (gated) |
|------|-----------|----------------|---------|--------------|------------|------------|
| 1s_top10% | 1127 | -0.599 | 585 | -0.614 | 0.311 | 0.32 |
| 1s_top5% | 536 | -0.567 | 406 | -0.503 | 0.342 | 0.41 |
| 5s_top10% | 1686 | -0.584 | 868 | -0.607 | 0.266 | 0.39 |

## Label-FIFO 12-cell screen

| Horizon | Top% | Trades | Net Ticks | /Trade | Sharpe | Pos? |
|---------|------|--------|-----------|--------|--------|------|
| 1s | 1% | 227 | 83.15 | +0.3663 | 524.01 | + |
| 1s | 2% | 453 | 146.67 | +0.3238 | 493.31 | + |
| 1s | 5% | 1133 | 300.99 | +0.2657 | 415.90 | + |
| 1s | 10% | 2265 | 509.36 | +0.2249 | 319.73 | + |
| 1s | 20% | 4530 | 910.72 | +0.2010 | 276.97 | + |
| 1s | 50% | 11324 | 1106.18 | +0.0977 | 136.71 | + |
| 5s | 1% | 227 | 205.15 | +0.9037 | 749.33 | + |
| 5s | 2% | 453 | 248.67 | +0.5489 | 459.93 | + |
| 5s | 5% | 1133 | 504.49 | +0.4453 | 368.88 | + |
| 5s | 10% | 2265 | 864.86 | +0.3818 | 301.73 | + |
| 5s | 20% | 4530 | 1179.72 | +0.2604 | 197.05 | + |
| 5s | 50% | 11324 | 1608.68 | +0.1421 | 102.46 | + |

## Files

- Label-FIFO CSV: `output/h5s_multifold_fold01_fifo/summary.csv`
- Canonical+gate CSV: `output/h5s_multifold_fold01_canonical/summary.csv`
- Aggregate row appended to `output/h5s_multifold_AGGREGATE.md`

Auto-graded by `scripts/h5s_multifold_auto_grader.sh`.