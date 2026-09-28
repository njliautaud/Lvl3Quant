# h5s multifold fold-03 FIFO Grade

**OOT date**: 20260415
**Gate**: q≥0.5 fill-prob (from fill_prob_head_v1.lgb, proxy hold 0.415 s)
**FIFO**: canonical (queue-aware passive limit, ES 0.376 commission, TP 2 / SL 1)

## Headline cells (canonical + gate)

| Cell | Ungated n | Ungated /trade | Gated n | Gated /trade | WR (gated) | PF (gated) |
|------|-----------|----------------|---------|--------------|------------|------------|
| 1s_top10% | 1029 | -0.679 | 729 | -0.658 | 0.321 | 0.27 |
| 1s_top5% | 476 | -0.706 | 343 | -0.711 | 0.294 | 0.23 |
| 5s_top10% | 1726 | -0.660 | 1142 | -0.601 | 0.276 | 0.39 |

## Label-FIFO 12-cell screen

| Horizon | Top% | Trades | Net Ticks | /Trade | Sharpe | Pos? |
|---------|------|--------|-----------|--------|--------|------|
| 1s | 1% | 246 | 142.50 | +0.5793 | 1151.69 | + |
| 1s | 2% | 492 | 283.51 | +0.5762 | 1100.28 | + |
| 1s | 5% | 1230 | 555.02 | +0.4512 | 859.70 | + |
| 1s | 10% | 2459 | 852.42 | +0.3467 | 617.03 | + |
| 1s | 20% | 4918 | 1407.83 | +0.2863 | 531.27 | + |
| 1s | 50% | 12294 | 1944.46 | +0.1582 | 287.55 | + |
| 5s | 1% | 246 | 175.00 | +0.7114 | 817.56 | + |
| 5s | 2% | 492 | 295.01 | +0.5996 | 721.86 | + |
| 5s | 5% | 1230 | 630.52 | +0.5126 | 579.81 | + |
| 5s | 10% | 2459 | 1064.42 | +0.4329 | 467.83 | + |
| 5s | 20% | 4918 | 1746.33 | +0.3551 | 384.46 | + |
| 5s | 50% | 12294 | 2755.46 | +0.2241 | 231.55 | + |

## Files

- Label-FIFO CSV: `output/h5s_multifold_fold03_fifo/summary.csv`
- Canonical+gate CSV: `output/h5s_multifold_fold03_canonical/summary.csv`
- Aggregate row appended to `output/h5s_multifold_AGGREGATE.md`

Auto-graded by `scripts/h5s_multifold_auto_grader.sh`.