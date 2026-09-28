# h5s multifold fold-06 FIFO Grade

**OOT date**: 20260419
**Gate**: q≥0.5 fill-prob (from fill_prob_head_v1.lgb, proxy hold 0.415 s)
**FIFO**: canonical (queue-aware passive limit, ES 0.376 commission, TP 2 / SL 1)

## Headline cells (canonical + gate)

| Cell | Ungated n | Ungated /trade | Gated n | Gated /trade | WR (gated) | PF (gated) |
|------|-----------|----------------|---------|--------------|------------|------------|
| 1s_top10% | 19 | -0.481 | 17 | -0.435 | 0.294 | 0.52 |
| 1s_top5% | 8 | -0.501 | 7 | -0.519 | 0.286 | 0.47 |
| 5s_top10% | 40 | -0.563 | 24 | -0.855 | 0.167 | 0.24 |

## Label-FIFO 12-cell screen

| Horizon | Top% | Trades | Net Ticks | /Trade | Sharpe | Pos? |
|---------|------|--------|-----------|--------|--------|------|
| 1s | 1% | 8 | 98.99 | +12.3740 | 1121.33 | + |
| 1s | 2% | 16 | 96.48 | +6.0303 | 753.10 | + |
| 1s | 5% | 39 | 99.34 | +2.5471 | 489.39 | + |
| 1s | 10% | 77 | 36.05 | +0.4682 | 114.02 | + |
| 1s | 20% | 154 | 8.60 | +0.0558 | 18.72 | + |
| 1s | 50% | 384 | 16.62 | +0.0433 | 21.26 | + |
| 5s | 1% | 8 | 102.49 | +12.8115 | 941.30 | + |
| 5s | 2% | 16 | 101.48 | +6.3428 | 653.67 | + |
| 5s | 5% | 39 | 55.84 | +1.4317 | 216.11 | + |
| 5s | 10% | 77 | 51.55 | +0.6695 | 130.18 | + |
| 5s | 20% | 154 | 41.60 | +0.2701 | 71.84 | + |
| 5s | 50% | 384 | -57.38 | -0.1494 | -52.99 | - |

## Files

- Label-FIFO CSV: `output/h5s_multifold_fold06_fifo/summary.csv`
- Canonical+gate CSV: `output/h5s_multifold_fold06_canonical/summary.csv`
- Aggregate row appended to `output/h5s_multifold_AGGREGATE.md`

Auto-graded by `scripts/h5s_multifold_auto_grader.sh`.