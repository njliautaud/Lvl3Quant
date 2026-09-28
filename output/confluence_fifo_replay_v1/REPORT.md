# Confluence FIFO Replay v1: CNN-Mamba v2 x PatchTST

Days analyzed: 32
Total aligned events: 1,388,095
FIFO configs: tp4sl3, tp8sl5

Proxy baseline (from confluence_stacking_v1):
  Both top 2% short: +1.19 avg realized ticks (log_ret proxy)
  Hybrid breakeven: 0.876 ticks

## TP4SL3 Results

### Both top 1% [UNPROFITABLE]
  Events: 298, Filled: 74 (24.8%)
  Net ticks (filled): -0.396
  Gross ticks: -0.020
  TP hit rate: 39.2%
  Daily Sharpe: -2.14
  Daily Sortino: -1.57
  Profitable days: 32.0%
  Avg fills/day: 3.0

### Both top 2% [UNPROFITABLE]
  Events: 895, Filled: 243 (27.2%)
  Net ticks (filled): -0.269
  Gross ticks: +0.107
  TP hit rate: 38.7%
  Daily Sharpe: -4.64
  Daily Sortino: -3.55
  Profitable days: 21.4%
  Avg fills/day: 8.7

### Both top 5% [UNPROFITABLE]
  Events: 4,344, Filled: 1,257 (28.9%)
  Net ticks (filled): -0.614
  Gross ticks: -0.238
  TP hit rate: 32.5%
  Daily Sharpe: -6.83
  Daily Sortino: -5.97
  Profitable days: 13.8%
  Avg fills/day: 43.3

### Both top 10% [UNPROFITABLE]
  Events: 15,388, Filled: 4,299 (27.9%)
  Net ticks (filled): -0.584
  Gross ticks: -0.208
  TP hit rate: 31.9%
  Daily Sharpe: -5.80
  Daily Sortino: -6.48
  Profitable days: 21.9%
  Avg fills/day: 134.3

### Both top 20% [UNPROFITABLE]
  Events: 58,910, Filled: 16,014 (27.2%)
  Net ticks (filled): -0.547
  Gross ticks: -0.171
  TP hit rate: 33.8%
  Daily Sharpe: -18.85
  Daily Sortino: -12.36
  Profitable days: 6.2%
  Avg fills/day: 500.4

## TP8SL5 Results

### Both top 1% [UNPROFITABLE]
  Events: 298, Filled: 74 (24.8%)
  Net ticks (filled): -1.455
  Gross ticks: -1.095
  TP hit rate: 16.2%
  Daily Sharpe: -4.43
  Daily Sortino: -3.66
  Profitable days: 20.0%
  Avg fills/day: 3.0

### Both top 2% [UNPROFITABLE]
  Events: 895, Filled: 243 (27.2%)
  Net ticks (filled): -0.946
  Gross ticks: -0.578
  TP hit rate: 18.5%
  Daily Sharpe: -3.60
  Daily Sortino: -3.16
  Profitable days: 25.0%
  Avg fills/day: 8.7

### Both top 5% [UNPROFITABLE]
  Events: 4,344, Filled: 1,257 (28.9%)
  Net ticks (filled): -1.330
  Gross ticks: -0.959
  TP hit rate: 14.2%
  Daily Sharpe: -6.89
  Daily Sortino: -5.40
  Profitable days: 10.3%
  Avg fills/day: 43.3

### Both top 10% [UNPROFITABLE]
  Events: 15,388, Filled: 4,299 (27.9%)
  Net ticks (filled): -1.290
  Gross ticks: -0.919
  TP hit rate: 14.1%
  Daily Sharpe: -7.60
  Daily Sortino: -6.95
  Profitable days: 18.8%
  Avg fills/day: 134.3

### Both top 20% [UNPROFITABLE]
  Events: 58,910, Filled: 16,014 (27.2%)
  Net ticks (filled): -1.303
  Gross ticks: -0.933
  TP hit rate: 15.4%
  Daily Sharpe: -29.44
  Daily Sortino: -13.77
  Profitable days: 3.1%
  Avg fills/day: 500.4

## Verdict

tp4sl3: top 2% confluence FAILS FIFO at -0.269 net ticks (fill rate 27.2%, 243 fills)

tp8sl5: top 2% confluence FAILS FIFO at -0.946 net ticks (fill rate 27.2%, 243 fills)