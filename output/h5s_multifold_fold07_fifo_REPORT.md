# h5s multifold fold-07 FIFO Grade

**OOT date**: 20260420
**Gate**: q≥0.5 fill-prob (from fill_prob_head_v1.lgb, proxy hold 0.415 s)
**FIFO**: canonical (queue-aware passive limit, ES 0.376 commission, TP 2 / SL 1)

## Headline cells (canonical + gate)

| Cell | Ungated n | Ungated /trade | Gated n | Gated /trade | WR (gated) | PF (gated) |
|------|-----------|----------------|---------|--------------|------------|------------|
| 1s_top10% | 1159 | -0.571 | 600 | -0.517 | 0.333 | 0.41 |
| 1s_top5% | 570 | -0.615 | 477 | -0.629 | 0.298 | 0.32 |
| 5s_top10% | 1820 | -0.588 | 1503 | -0.581 | 0.277 | 0.41 |

## Label-FIFO 12-cell screen

| Horizon | Top% | Trades | Net Ticks | /Trade | Sharpe | Pos? |
|---------|------|--------|-----------|--------|--------|------|
| 1s | 1% | 257 | 152.37 | +0.5929 | 893.43 | + |
| 1s | 2% | 514 | 198.24 | +0.3857 | 604.45 | + |
| 1s | 5% | 1285 | 430.34 | +0.3349 | 503.34 | + |
| 1s | 10% | 2570 | 926.68 | +0.3606 | 567.43 | + |
| 1s | 20% | 5140 | 1327.36 | +0.2582 | 391.87 | + |
| 1s | 50% | 12850 | 1882.90 | +0.1465 | 212.74 | + |
| 5s | 1% | 257 | 163.87 | +0.6376 | 611.24 | + |
| 5s | 2% | 514 | 226.74 | +0.4411 | 428.71 | + |
| 5s | 5% | 1285 | 442.84 | +0.3446 | 325.05 | + |
| 5s | 10% | 2570 | 828.68 | +0.3224 | 277.75 | + |
| 5s | 20% | 5140 | 1670.36 | +0.3250 | 275.93 | + |
| 5s | 50% | 12850 | 2930.40 | +0.2280 | 184.88 | + |

## Files

- Label-FIFO CSV: `output/h5s_multifold_fold07_fifo/summary.csv`
- Canonical+gate CSV: `output/h5s_multifold_fold07_canonical/summary.csv`
- Aggregate row appended to `output/h5s_multifold_AGGREGATE.md`

Auto-graded by `scripts/h5s_multifold_auto_grader.sh`.