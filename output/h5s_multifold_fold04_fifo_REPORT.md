# h5s multifold fold-04 FIFO Grade

**OOT date**: 20260416
**Gate**: q≥0.5 fill-prob (from fill_prob_head_v1.lgb, proxy hold 0.415 s)
**FIFO**: canonical (queue-aware passive limit, ES 0.376 commission, TP 2 / SL 1)

## Headline cells (canonical + gate)

| Cell | Ungated n | Ungated /trade | Gated n | Gated /trade | WR (gated) | PF (gated) |
|------|-----------|----------------|---------|--------------|------------|------------|
| 1s_top10% | 1182 | -0.637 | 958 | -0.635 | 0.314 | 0.30 |
| 1s_top5% | 582 | -0.653 | 507 | -0.654 | 0.302 | 0.29 |
| 5s_top10% | 1804 | -0.619 | 1453 | -0.600 | 0.269 | 0.40 |

## Label-FIFO 12-cell screen

| Horizon | Top% | Trades | Net Ticks | /Trade | Sharpe | Pos? |
|---------|------|--------|-----------|--------|--------|------|
| 1s | 1% | 252 | 114.75 | +0.4553 | 714.83 | + |
| 1s | 2% | 504 | 195.50 | +0.3879 | 659.07 | + |
| 1s | 5% | 1260 | 538.24 | +0.4272 | 714.03 | + |
| 1s | 10% | 2519 | 926.86 | +0.3679 | 639.21 | + |
| 1s | 20% | 5038 | 1583.71 | +0.3144 | 538.33 | + |
| 1s | 50% | 12595 | 2243.78 | +0.1781 | 302.02 | + |
| 5s | 1% | 252 | 151.75 | +0.6022 | 578.43 | + |
| 5s | 2% | 504 | 222.50 | +0.4415 | 429.70 | + |
| 5s | 5% | 1260 | 622.74 | +0.4942 | 466.14 | + |
| 5s | 10% | 2519 | 1031.36 | +0.4094 | 400.58 | + |
| 5s | 20% | 5038 | 1648.21 | +0.3272 | 308.92 | + |
| 5s | 50% | 12595 | 2658.28 | +0.2111 | 196.37 | + |

## Files

- Label-FIFO CSV: `output/h5s_multifold_fold04_fifo/summary.csv`
- Canonical+gate CSV: `output/h5s_multifold_fold04_canonical/summary.csv`
- Aggregate row appended to `output/h5s_multifold_AGGREGATE.md`

Auto-graded by `scripts/h5s_multifold_auto_grader.sh`.