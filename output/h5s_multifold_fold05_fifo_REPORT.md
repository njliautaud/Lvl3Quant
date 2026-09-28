# h5s multifold fold-05 FIFO Grade

**OOT date**: 20260417
**Gate**: q≥0.5 fill-prob (from fill_prob_head_v1.lgb, proxy hold 0.415 s)
**FIFO**: canonical (queue-aware passive limit, ES 0.376 commission, TP 2 / SL 1)

## Headline cells (canonical + gate)

| Cell | Ungated n | Ungated /trade | Gated n | Gated /trade | WR (gated) | PF (gated) |
|------|-----------|----------------|---------|--------------|------------|------------|
| 1s_top10% | 1470 | -0.600 | 877 | -0.574 | 0.317 | 0.37 |
| 1s_top5% | 683 | -0.633 | 595 | -0.619 | 0.297 | 0.34 |
| 5s_top10% | 2112 | -0.647 | 1098 | -0.626 | 0.256 | 0.39 |

## Label-FIFO 12-cell screen

| Horizon | Top% | Trades | Net Ticks | /Trade | Sharpe | Pos? |
|---------|------|--------|-----------|--------|--------|------|
| 1s | 1% | 279 | 108.10 | +0.3874 | 594.23 | + |
| 1s | 2% | 558 | 265.19 | +0.4753 | 720.61 | + |
| 1s | 5% | 1395 | 637.98 | +0.4573 | 666.83 | + |
| 1s | 10% | 2790 | 929.46 | +0.3331 | 487.56 | + |
| 1s | 20% | 5580 | 1476.42 | +0.2646 | 373.65 | + |
| 1s | 50% | 13949 | 2350.18 | +0.1685 | 235.33 | + |
| 5s | 1% | 279 | 189.10 | +0.6778 | 591.52 | + |
| 5s | 2% | 558 | 310.69 | +0.5568 | 435.45 | + |
| 5s | 5% | 1395 | 523.98 | +0.3756 | 282.90 | + |
| 5s | 10% | 2790 | 1044.46 | +0.3744 | 283.72 | + |
| 5s | 20% | 5580 | 1680.42 | +0.3012 | 227.61 | + |
| 5s | 50% | 13949 | 2283.18 | +0.1637 | 118.53 | + |

## Files

- Label-FIFO CSV: `output/h5s_multifold_fold05_fifo/summary.csv`
- Canonical+gate CSV: `output/h5s_multifold_fold05_canonical/summary.csv`
- Aggregate row appended to `output/h5s_multifold_AGGREGATE.md`

Auto-graded by `scripts/h5s_multifold_auto_grader.sh`.