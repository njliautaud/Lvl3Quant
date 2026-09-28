# Macro-Exposure GA Report (smoke)

Evals: **55**

Backtest span: **2023-01-03 → 2024-12-30**


Tiers picked from the GA Pareto/eval pool.


## Conservative

- CAGR: **11.02%**
- Max drawdown: **2.83%**
- Worst month: **-1.52%**
- Sortino: **2.18**
- Sharpe: **1.88**
- Avg leverage: **0.27x**
- Turnover/year: **1.8**
- Time long / flat / short: **55% / 45% / 0%**
- Basket: SPY 22% / QQQ 41% / IWM 36%
- Cadence: **biweekly**, allow_short: **False**, max_leverage: **1.0x**
- Long/short strength: **0.5x / 1.0x**
- Long/short thresholds: **1.27 / -0.55** (flat band 0.22)
- Top features (|weight|):
    - w_spy_above_200dma: +1.88
    - w_mom_6m: +1.49
    - w_vix_pct: -1.36
    - w_yield_2s10s: +1.10
    - w_naaim: -0.96

![Conservative equity](equity_conservative.png)

## Balanced

- CAGR: **28.19%**
- Max drawdown: **14.55%**
- Worst month: **-5.81%**
- Sortino: **1.85**
- Sharpe: **1.45**
- Avg leverage: **0.99x**
- Turnover/year: **5.4**
- Time long / flat / short: **76% / 17% / 7%**
- Basket: SPY 8% / QQQ 79% / IWM 12%
- Cadence: **biweekly**, allow_short: **True**, max_leverage: **1.25x**
- Long/short strength: **1.25x / 0.5x**
- Long/short thresholds: **0.17 / 0.09** (flat band 0.48)
- Top features (|weight|):
    - w_yield_2s10s: -1.84
    - w_vix_ts_slope: +1.75
    - w_spy_above_200dma: +1.70
    - w_mom_6m: -1.55
    - w_aaii_bullbear: -1.45

![Balanced equity](equity_balanced.png)

## Aggressive

- CAGR: **28.19%**
- Max drawdown: **14.55%**
- Worst month: **-5.81%**
- Sortino: **1.85**
- Sharpe: **1.45**
- Avg leverage: **0.99x**
- Turnover/year: **5.4**
- Time long / flat / short: **76% / 17% / 7%**
- Basket: SPY 8% / QQQ 79% / IWM 12%
- Cadence: **biweekly**, allow_short: **True**, max_leverage: **1.25x**
- Long/short strength: **1.25x / 0.5x**
- Long/short thresholds: **0.17 / 0.09** (flat band 0.48)
- Top features (|weight|):
    - w_yield_2s10s: -1.84
    - w_vix_ts_slope: +1.75
    - w_spy_above_200dma: +1.70
    - w_mom_6m: -1.55
    - w_aaii_bullbear: -1.45

![Aggressive equity](equity_aggressive.png)
