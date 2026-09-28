# h5s multifold fold-02 FIFO Grade

**OOT date**: 20260414
**Gate**: q≥0.5 fill-prob (from fill_prob_head_v1.lgb, proxy hold 0.415 s)
**FIFO**: canonical (queue-aware passive limit, ES 0.376 commission, TP 2 / SL 1)

## Headline cells (canonical + gate)

| Cell | Ungated n | Ungated /trade | Gated n | Gated /trade | WR (gated) | PF (gated) |
|------|-----------|----------------|---------|--------------|------------|------------|
| 1s_top10% | 886 | -0.666 | 459 | -0.698 | 0.316 | 0.23 |
| 1s_top5% | 431 | -0.637 | 297 | -0.607 | 0.360 | 0.28 |
| 5s_top10% | 1453 | -0.656 | 987 | -0.662 | 0.258 | 0.35 |

## Label-FIFO 12-cell screen

| Horizon | Top% | Trades | Net Ticks | /Trade | Sharpe | Pos? |
|---------|------|--------|-----------|--------|--------|------|
| 1s | 1% | 211 | 81.66 | +0.3870 | 872.53 | + |
| 1s | 2% | 421 | 150.70 | +0.3580 | 633.13 | + |
| 1s | 5% | 1051 | 284.32 | +0.2705 | 528.56 | + |
| 1s | 10% | 2102 | 549.15 | +0.2613 | 473.30 | + |
| 1s | 20% | 4204 | 997.80 | +0.2373 | 443.23 | + |
| 1s | 50% | 10508 | 1384.99 | +0.1318 | 239.84 | + |
| 5s | 1% | 211 | 99.16 | +0.4700 | 600.40 | + |
| 5s | 2% | 421 | 226.20 | +0.5373 | 623.10 | + |
| 5s | 5% | 1051 | 392.32 | +0.3733 | 371.94 | + |
| 5s | 10% | 2102 | 687.15 | +0.3269 | 335.41 | + |
| 5s | 20% | 4204 | 1267.80 | +0.3016 | 317.12 | + |
| 5s | 50% | 10508 | 2120.99 | +0.2018 | 217.20 | + |

## Files

- Label-FIFO CSV: `output/h5s_multifold_fold02_fifo/summary.csv`
- Canonical+gate CSV: `output/h5s_multifold_fold02_canonical/summary.csv`
- Aggregate row appended to `output/h5s_multifold_AGGREGATE.md`

Auto-graded by `scripts/h5s_multifold_auto_grader.sh`.