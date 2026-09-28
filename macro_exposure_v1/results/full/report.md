# Macro-Exposure GA Report (full)

Evals: **3240**

Backtest span: **2010-01-04 → 2026-06-04**


Tiers picked from the GA Pareto/eval pool.


## Conservative

- CAGR: **5.57%**
- Max drawdown: **4.01%**
- Worst month: **-2.42%**
- Sortino: **1.13**
- Sharpe: **1.22**
- Avg leverage: **0.21x**
- Turnover/year: **5.1**
- Time long / flat / short: **41% / 59% / 0%**
- Basket: SPY 16% / QQQ 70% / IWM 14%
- Cadence: **weekly**, allow_short: **False**, max_leverage: **1.25x**
- Long/short strength: **0.5x / 0.5x**
- Long/short thresholds: **1.34 / -0.82** (flat band 0.13)
- Top features (|weight|):
    - w_aaii_bullbear: +1.97
    - w_yield_2s10s: -1.91
    - w_vix_pct: -1.88
    - w_naaim: -1.86
    - w_mom_3m: -1.79

![Conservative equity](equity_conservative.png)

## Balanced

- CAGR: **20.38%**
- Max drawdown: **12.38%**
- Worst month: **-9.72%**
- Sortino: **1.56**
- Sharpe: **1.41**
- Avg leverage: **0.75x**
- Turnover/year: **12.0**
- Time long / flat / short: **60% / 40% / 0%**
- Basket: SPY 49% / QQQ 38% / IWM 13%
- Cadence: **weekly**, allow_short: **False**, max_leverage: **1.25x**
- Long/short strength: **1.25x / 1.0x**
- Long/short thresholds: **0.08 / -0.23** (flat band 0.38)
- Top features (|weight|):
    - w_aaii_bullbear: -1.88
    - w_naaim: -1.86
    - w_mom_3m: -1.79
    - w_vix_ts_slope: +1.68
    - w_yield_2s10s: -1.52

![Balanced equity](equity_balanced.png)

## Aggressive

- CAGR: **25.84%**
- Max drawdown: **27.68%**
- Worst month: **-20.09%**
- Sortino: **1.30**
- Sharpe: **1.12**
- Avg leverage: **1.14x**
- Turnover/year: **5.8**
- Time long / flat / short: **76% / 24% / 0%**
- Basket: SPY 1% / QQQ 83% / IWM 16%
- Cadence: **weekly**, allow_short: **False**, max_leverage: **1.5x**
- Long/short strength: **1.5x / 1.0x**
- Long/short thresholds: **-0.83 / -1.33** (flat band 1.06)
- Top features (|weight|):
    - w_naaim: -1.86
    - w_vix_ts_slope: +1.68
    - w_spy_above_200dma: +1.31
    - w_breadth: -1.27
    - w_yield_2s10s: +0.91

![Aggressive equity](equity_aggressive.png)
