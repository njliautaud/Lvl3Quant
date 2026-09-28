# Hybrid Entry/Exit Analysis v1

**OOT Days Analyzed:** 48
**Date Range:** 20260224 to 20260429

## Cost Scenarios
- PURE MARKET: 1.376 ticks (spread both sides + commission)
- HYBRID: 0.876 ticks (passive entry, market exit)
- AGGRESSIVE HYBRID: 0.688 ticks (passive entry + passive exit)

## Short Signal Results (10s horizon)

| Percentile | Avg Realized | Cost Scenario | Net Ticks | Win Rate | Sharpe |
|------------|-------------|---------------|-----------|----------|--------|
| top_1pct | 0.958 | PURE_MARKET | -0.418 | 46.0% | -0.87 |
| top_1pct | 0.958 | HYBRID | 0.082 | 52.8% | 0.17 |
| top_1pct | 0.958 | AGGRESSIVE_HYBRID | 0.270 | 52.8% | 0.56 |
| top_2pct | 0.890 | PURE_MARKET | -0.486 | 45.0% | -1.16 |
| top_2pct | 0.890 | HYBRID | 0.014 | 51.7% | 0.03 |
| top_2pct | 0.890 | AGGRESSIVE_HYBRID | 0.202 | 51.7% | 0.48 |
| top_3pct | 0.841 | PURE_MARKET | -0.535 | 44.4% | -1.35 |
| top_3pct | 0.841 | HYBRID | -0.035 | 51.2% | -0.09 |
| top_3pct | 0.841 | AGGRESSIVE_HYBRID | 0.153 | 51.2% | 0.39 |
| top_5pct | 0.766 | PURE_MARKET | -0.610 | 43.8% | -1.63 |
| top_5pct | 0.766 | HYBRID | -0.110 | 50.8% | -0.30 |
| top_5pct | 0.766 | AGGRESSIVE_HYBRID | 0.078 | 50.8% | 0.21 |
| top_10pct | 0.670 | PURE_MARKET | -0.706 | 43.0% | -1.98 |
| top_10pct | 0.670 | HYBRID | -0.206 | 50.2% | -0.58 |
| top_10pct | 0.670 | AGGRESSIVE_HYBRID | -0.018 | 50.2% | -0.05 |

## Per-Day Profitability (10s horizon)

- Top 1% HYBRID: 19/46 days profitable (41%)
- Top 3% HYBRID: 17/46 days profitable (37%)
- Top 5% HYBRID: 13/46 days profitable (28%)

## Fill Rate Sensitivity (10s, HYBRID cost)

- top_1pct random_50pct: realized=0.958, net=0.082
- top_1pct worst_50pct: realized=-3.091, net=-3.967
- top_1pct best_50pct: realized=5.006, net=4.130
- top_1pct worst_25pct: realized=-5.991, net=-6.867
- top_3pct random_50pct: realized=0.841, net=-0.035
- top_3pct worst_50pct: realized=-2.834, net=-3.710
- top_3pct best_50pct: realized=4.516, net=3.640
- top_3pct worst_25pct: realized=-5.415, net=-6.291
- top_5pct random_50pct: realized=0.766, net=-0.110
- top_5pct worst_50pct: realized=-2.830, net=-3.706
- top_5pct best_50pct: realized=4.362, net=3.486
- top_5pct worst_25pct: realized=-5.362, net=-6.238
