# h5s multifold fold-09 FIFO Grade

**OOT date**: 20260427
**Gate**: q≥0.5 fill-prob (from fill_prob_head_v1.lgb, proxy hold 0.415 s)
**FIFO**: canonical (queue-aware passive limit, ES 0.376 commission, TP 2 / SL 1)

## Headline cells (canonical + gate)

| Cell | Ungated n | Ungated /trade | Gated n | Gated /trade | WR (gated) | PF (gated) |
|------|-----------|----------------|---------|--------------|------------|------------|
| 1s_top10% | 903 | -0.681 | 719 | -0.700 | 0.288 | 0.25 |
| 1s_top5% | 440 | -0.718 | 382 | -0.737 | 0.272 | 0.23 |
| 5s_top10% | 1599 | -0.601 | 1237 | -0.577 | 0.283 | 0.41 |

## Label-FIFO 12-cell screen

| Horizon | Top% | Trades | Net Ticks | /Trade | Sharpe | Pos? |
|---------|------|--------|-----------|--------|--------|------|
| 1s | 1% | 237 | 65.89 | +0.2780 | 615.27 | + |
| 1s | 2% | 474 | 129.78 | +0.2738 | 616.91 | + |
| 1s | 5% | 1184 | 350.32 | +0.2959 | 552.75 | + |
| 1s | 10% | 2368 | 691.13 | +0.2919 | 555.14 | + |
| 1s | 20% | 4736 | 1189.76 | +0.2512 | 499.80 | + |
| 1s | 50% | 11840 | 1828.16 | +0.1544 | 290.57 | + |
| 5s | 1% | 237 | 88.89 | +0.3751 | 424.11 | + |
| 5s | 2% | 474 | 181.78 | +0.3835 | 445.63 | + |
| 5s | 5% | 1184 | 409.82 | +0.3461 | 351.77 | + |
| 5s | 10% | 2368 | 863.63 | +0.3647 | 393.32 | + |
| 5s | 20% | 4736 | 1339.76 | +0.2829 | 305.35 | + |
| 5s | 50% | 11840 | 2296.66 | +0.1940 | 206.79 | + |

## Files

- Label-FIFO CSV: `output/h5s_multifold_fold09_fifo/summary.csv`
- Canonical+gate CSV: `output/h5s_multifold_fold09_canonical/summary.csv`
- Aggregate row appended to `output/h5s_multifold_AGGREGATE.md`

Auto-graded by `scripts/h5s_multifold_auto_grader.sh`.