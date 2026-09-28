# Stacked Confluence Validation v1

**Dates**: 48 OOT days (20260224 to 20260429)
**Total predictions**: 2,313,722
**Cost model**: 0.376 ticks RT (passive-passive)

## Results Comparison

| Config | Trades | Days | Mean tk | WR | PF | Sharpe | Sortino | Green% | Regime |
|--------|--------|------|---------|----|----|--------|---------|--------|--------|
| signal_top_5pct | 55686 | 46 | +0.3344 | 59.4% | 1.63 | 26.9 | 309.6 | 91% | PASS |
| signal_top_5pct_ofi | 30193 | 46 | +0.4421 | 60.6% | 1.97 | 30.9 | inf | 98% | PASS |
| signal_top_5pct_meta30 | 20736 | 46 | +0.3586 | 58.5% | 1.60 | 17.6 | 125.3 | 87% | PASS |
| signal_top_5pct_meta30_ofi | 10477 | 46 | +0.5050 | 59.5% | 2.02 | 17.9 | 1225.5 | 91% | PASS |
| signal_top_10pct | 111378 | 46 | +0.2768 | 58.2% | 1.51 | 27.4 | 11362.0 | 96% | FAIL |
| signal_top_10pct_ofi | 59707 | 46 | +0.3751 | 59.2% | 1.79 | 31.3 | inf | 98% | FAIL |
| signal_top_10pct_meta30 | 51695 | 46 | +0.3090 | 58.7% | 1.55 | 25.7 | 279.2 | 89% | PASS |
| signal_top_10pct_meta30_ofi | 26944 | 46 | +0.4201 | 59.7% | 1.86 | 28.2 | 3142.6 | 91% | PASS |
