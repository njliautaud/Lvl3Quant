# Wheel Strategy Walk-Forward Validated Backtest Results

Run date: 2026-09-28
Data: 2018-01-02 to 2025-12-30
Walk-forward: 24-month SLIDING train, 3-month OOT test
Starting capital: $100,000
Cost model: $0.65/contract commission + 2.5% slippage + $0.03 min


## A. SPY Conservative
- Tickers: SPY
- Put delta: 0.2, Call delta: 0.25
- DTE: 35, Profit-take: 50%
- Cash reserve: 30%, Max positions: 1
- Regime gate (SPY > 50d MA): Yes
- VIX-adaptive delta: No

### Per-Fold OOT Results
| Fold | OOT Period | CAGR | Sharpe | MaxDD | WR | CSP WR | PF | Trades |
|------|------------|------|--------|-------|-----|--------|-----|--------|
| 0 | 2020-01-02 - 2020-04-02 | -42.4% | -2.08 | -17.6% | 67% | 67% | inf | 3 |
| 1 | 2020-04-02 - 2020-07-02 | +6.1% | 0.38 | -2.1% | 100% | 100% | inf | 6 |
| 2 | 2020-07-02 - 2020-10-02 | -6.5% | -1.53 | -4.6% | 100% | 100% | inf | 7 |
| 3 | 2020-10-02 - 2021-01-02 | +7.1% | 2.58 | -0.2% | 100% | 100% | inf | 7 |
| 4 | 2021-01-02 - 2021-04-02 | +5.5% | 0.49 | -1.0% | 100% | 100% | inf | 5 |
| 5 | 2021-04-02 - 2021-07-02 | +2.3% | -1.11 | -0.8% | 100% | 100% | inf | 6 |
| 6 | 2021-07-02 - 2021-10-02 | -1.4% | -2.24 | -1.2% | 100% | 100% | inf | 4 |
| 7 | 2021-10-02 - 2022-01-02 | +1.7% | -0.83 | -1.3% | 100% | 100% | inf | 4 |
| 8 | 2022-01-02 - 2022-04-02 | +1.4% | -3.46 | -0.2% | 100% | 100% | inf | 2 |
| 9 | 2022-04-02 - 2022-07-02 | +0.0% | -92197712668417520.00 | 0.0% | 0% | 0% | 0.0 | 0 |
| 10 | 2022-07-02 - 2022-10-02 | -14.7% | -4.38 | -4.1% | 80% | 67% | inf | 5 |
| 11 | 2022-10-02 - 2023-01-02 | +2.8% | -0.47 | -0.7% | 100% | 100% | inf | 4 |
| 12 | 2023-01-02 - 2023-04-02 | +1.4% | -1.29 | -0.7% | 100% | 100% | inf | 2 |
| 13 | 2023-04-02 - 2023-07-02 | +3.4% | -0.99 | -0.1% | 100% | 100% | inf | 8 |
| 14 | 2023-07-02 - 2023-10-02 | -0.9% | -3.20 | -0.6% | 100% | 100% | inf | 2 |
| 15 | 2023-10-02 - 2024-01-02 | +3.3% | -0.89 | -0.3% | 100% | 100% | inf | 8 |
| 16 | 2024-01-02 - 2024-04-02 | +3.3% | -0.79 | -0.2% | 100% | 100% | inf | 8 |
| 17 | 2024-04-02 - 2024-07-02 | +3.9% | -0.32 | -0.2% | 100% | 100% | inf | 9 |
| 18 | 2024-07-02 - 2024-10-02 | +4.1% | -0.01 | -1.1% | 100% | 100% | inf | 5 |
| 19 | 2024-10-02 - 2025-01-02 | -1.6% | -1.75 | -1.4% | 100% | 100% | inf | 6 |
| 20 | 2025-01-02 - 2025-04-02 | -9.6% | -2.12 | -4.4% | 75% | 75% | inf | 4 |
| 21 | 2025-04-02 - 2025-07-02 | +11.2% | 3.29 | -0.4% | 100% | 100% | inf | 11 |
| 22 | 2025-07-02 - 2025-10-02 | +4.5% | 0.45 | -0.2% | 100% | 100% | inf | 10 |
| 23 | 2025-10-02 - 2026-01-02 | +3.8% | -0.11 | -1.2% | 100% | 100% | inf | 5 |

### Aggregated OOT Metrics (REAL Performance)
| Metric | Value |
|--------|-------|
| CAGR | -1.2% |
| Sharpe | -0.80 |
| Sortino | -0.82 |
| Max Drawdown | -17.6% |
| Win Rate | 97.7% |
| CSP Win Rate | 97.7% |
| Profit Factor | inf |
| Total Trades | 131 |
| Assignments | 3 |
| Called Away | 0 |
| Final Equity | $93,284 |

### Monthly Returns (OOT)
       Jan   Feb   Mar   Apr   May   Jun   Jul   Aug   Sep   Oct   Nov   Dec  Annual
2020 -4.9% -7.7% -0.7% +1.2% -0.2% +0.4% +0.7% -2.3% -0.1% +0.8% +0.9%   NaN  -11.6%
2021 +0.1% +1.2% +0.1% +0.0% +0.4% +0.1% +0.3% -1.1% +0.5% -0.7% +1.1% +0.0%   +1.9%
2022 +0.0% +0.1% +0.2% +0.0% +0.0% +0.0% -0.9% -2.8% +0.0% +0.8% -0.1% +0.0%   -2.7%
2023 -0.4% +0.8% +0.0% +0.3% +0.5% +0.0% +0.1% -0.3% +0.0% +0.5% +0.3% +0.0%   +1.8%
2024 +0.4% +0.4% -0.0% +0.4% +0.4% +0.1% +0.7% +0.4% -0.1% +0.6% -1.0% -0.0%   +2.4%
2025 -0.4% -2.6% +0.5% +1.7% +0.9% +0.0% +0.4% +0.6% +0.1% +0.3% +0.6% -0.1%   +2.1%

### Seasonality
- Jan: avg -0.87%, positive 2/6 periods
- Feb: avg -1.30%, positive 4/6 periods
- Mar: avg -0.00%, positive 3/6 periods
- Apr: avg +0.62%, positive 5/6 periods
- May: avg +0.35%, positive 4/6 periods
- Jun: avg +0.10%, positive 4/6 periods
- Jul: avg +0.22%, positive 5/6 periods
- Aug: avg -0.92%, positive 2/6 periods
- Sep: avg +0.05%, positive 3/6 periods
- Oct: avg +0.39%, positive 5/6 periods
- Nov: avg +0.31%, positive 4/6 periods
- Dec: avg -0.02%, positive 0/5 periods

### Income Projection ($100k capital)
- Monthly: $-97 (-0.10%)
- Annual: $-1,153 (-1.2%)

### Bias Analysis
| Check | Result |
|-------|--------|
| Overfitting (IS/OOT Sharpe) | 0.00 (OK) |
| IS Sharpe avg | -0.34 |
| OOT Sharpe avg | -3841571361184064.00 |
| Survivorship | 100% |
| Lookahead | CLEAN |
| Selection | Strategy uses 1 tickers, 22 held out for validation. |

### Transaction Cost Sensitivity
| Cost Level | CAGR | Sharpe | MaxDD |
|------------|------|--------|-------|
| 0x (no costs) | -1.1% | -0.79 | -17.6% |
| 1x (baseline) | -1.2% | -0.80 | -17.6% |
| 2x (conservative) | -1.2% | -0.82 | -17.6% |

## B. Multi-ETF Balanced
- Tickers: SPY, QQQ, IWM
- Put delta: 0.22, Call delta: 0.25
- DTE: 35, Profit-take: 50%
- Cash reserve: 25%, Max positions: 3
- Regime gate (SPY > 50d MA): Yes
- VIX-adaptive delta: No

### Per-Fold OOT Results
| Fold | OOT Period | CAGR | Sharpe | MaxDD | WR | CSP WR | PF | Trades |
|------|------------|------|--------|-------|-----|--------|-----|--------|
| 0 | 2020-01-02 - 2020-04-02 | -75.2% | -1.63 | -39.4% | 70% | 67% | inf | 10 |
| 1 | 2020-04-02 - 2020-07-02 | +16.1% | 0.94 | -4.9% | 100% | 100% | inf | 17 |
| 2 | 2020-07-02 - 2020-10-02 | -1.5% | -0.35 | -7.7% | 100% | 100% | inf | 18 |
| 3 | 2020-10-02 - 2021-01-02 | +19.9% | 4.26 | -0.6% | 100% | 100% | inf | 20 |
| 4 | 2021-01-02 - 2021-04-02 | +16.0% | 0.81 | -4.9% | 90% | 89% | inf | 10 |
| 5 | 2021-04-02 - 2021-07-02 | +12.8% | 1.51 | -2.4% | 100% | 100% | inf | 16 |
| 6 | 2021-07-02 - 2021-10-02 | -6.3% | -1.24 | -4.1% | 100% | 100% | inf | 12 |
| 7 | 2021-10-02 - 2022-01-02 | +1.4% | -0.25 | -3.7% | 90% | 90% | inf | 10 |
| 8 | 2022-01-02 - 2022-04-02 | +4.9% | 0.37 | -0.5% | 100% | 100% | inf | 5 |
| 9 | 2022-04-02 - 2022-07-02 | +0.0% | -92197712668417520.00 | 0.0% | 0% | 0% | 0.0 | 0 |
| 10 | 2022-07-02 - 2022-10-02 | -50.6% | -3.65 | -16.4% | 77% | 62% | inf | 13 |
| 11 | 2022-10-02 - 2023-01-02 | +9.1% | 0.58 | -3.3% | 100% | 100% | inf | 10 |
| 12 | 2023-01-02 - 2023-04-02 | -5.7% | -1.04 | -4.5% | 50% | 50% | inf | 4 |
| 13 | 2023-04-02 - 2023-07-02 | +15.0% | 3.83 | -0.6% | 100% | 100% | inf | 19 |
| 14 | 2023-07-02 - 2023-10-02 | -5.6% | -1.81 | -2.1% | 100% | 100% | inf | 5 |
| 15 | 2023-10-02 - 2024-01-02 | +13.6% | 3.17 | -0.6% | 100% | 100% | inf | 20 |
| 16 | 2024-01-02 - 2024-04-02 | +7.9% | 1.27 | -0.7% | 100% | 100% | inf | 17 |
| 17 | 2024-04-02 - 2024-07-02 | +11.1% | 3.18 | -0.4% | 94% | 94% | inf | 18 |
| 18 | 2024-07-02 - 2024-10-02 | +13.7% | 1.55 | -2.2% | 100% | 100% | inf | 16 |
| 19 | 2024-10-02 - 2025-01-02 | -1.1% | -0.64 | -3.4% | 100% | 100% | inf | 15 |
| 20 | 2025-01-02 - 2025-04-02 | -31.3% | -1.94 | -13.3% | 70% | 62% | inf | 10 |
| 21 | 2025-04-02 - 2025-07-02 | +27.8% | 4.02 | -0.9% | 100% | 100% | inf | 26 |
| 22 | 2025-07-02 - 2025-10-02 | +14.3% | 3.25 | -0.9% | 100% | 100% | inf | 21 |
| 23 | 2025-10-02 - 2026-01-02 | +12.5% | 0.94 | -4.2% | 100% | 100% | inf | 12 |

### Aggregated OOT Metrics (REAL Performance)
| Metric | Value |
|--------|-------|
| CAGR | -3.4% |
| Sharpe | -0.35 |
| Sortino | -0.37 |
| Max Drawdown | -39.4% |
| Win Rate | 95.7% |
| CSP Win Rate | 95.6% |
| Profit Factor | inf |
| Total Trades | 324 |
| Assignments | 14 |
| Called Away | 0 |
| Final Equity | $81,055 |

### Monthly Returns (OOT)
        Jan    Feb   Mar   Apr   May   Jun   Jul    Aug   Sep   Oct   Nov   Dec  Annual
2020 -11.2% -17.9% -3.1% +2.7% +0.3% +0.7% +2.2%  -2.7% +0.2% +2.1% +2.4%   NaN  -23.6%
2021  -1.4%  +4.0% +1.0% +0.4% +2.4% +0.2% +1.3%  -3.9% +1.1% -2.1% +2.5% +0.0%   +5.3%
2022  +0.0%  +0.6% +0.5% +0.0% +0.0% +0.0% -4.5% -11.5% +0.0% +2.8% -0.7% +0.0%  -12.6%
2023  -1.1%  -0.3% +0.0% +1.4% +2.0% +0.0% +0.4%  -1.3% -0.5% +2.2% +1.3% +0.0%   +4.1%
2024  +1.0%  +1.2% -0.3% +1.3% +1.1% +0.2% +2.0%  +1.6% -0.4% +1.7% -1.9% -0.3%   +7.5%
2025  -2.4%  -8.3% +1.8% +3.8% +2.3% +0.1% +1.1%  +2.0% +0.3% +0.9% +2.0% -0.0%   +3.1%

### Seasonality
- Jan: avg -2.53%, positive 1/6 periods
- Feb: avg -3.41%, positive 3/6 periods
- Mar: avg -0.01%, positive 3/6 periods
- Apr: avg +1.59%, positive 5/6 periods
- May: avg +1.35%, positive 5/6 periods
- Jun: avg +0.22%, positive 4/6 periods
- Jul: avg +0.43%, positive 5/6 periods
- Aug: avg -2.64%, positive 2/6 periods
- Sep: avg +0.11%, positive 3/6 periods
- Oct: avg +1.27%, positive 5/6 periods
- Nov: avg +0.93%, positive 4/6 periods
- Dec: avg -0.06%, positive 0/5 periods

### Income Projection ($100k capital)
- Monthly: $-292 (-0.29%)
- Annual: $-3,444 (-3.4%)

### Bias Analysis
| Check | Result |
|-------|--------|
| Overfitting (IS/OOT Sharpe) | -0.00 (OK) |
| IS Sharpe avg | 0.24 |
| OOT Sharpe avg | -3841571361184062.50 |
| Survivorship | 100% |
| Lookahead | CLEAN |
| Selection | Strategy uses 3 tickers, 20 held out for validation. |

### Transaction Cost Sensitivity
| Cost Level | CAGR | Sharpe | MaxDD |
|------------|------|--------|-------|
| 0x (no costs) | -3.2% | -0.33 | -39.4% |
| 1x (baseline) | -3.4% | -0.35 | -39.4% |
| 2x (conservative) | -3.7% | -0.37 | -39.5% |

## C. Quality Dividend
- Tickers: KO, JNJ, PG, PEP, MCD, HD, ABBV
- Put delta: 0.25, Call delta: 0.3
- DTE: 30, Profit-take: 50%
- Cash reserve: 20%, Max positions: 4
- Regime gate (SPY > 50d MA): No
- VIX-adaptive delta: No

### Per-Fold OOT Results
| Fold | OOT Period | CAGR | Sharpe | MaxDD | WR | CSP WR | PF | Trades |
|------|------------|------|--------|-------|-----|--------|-----|--------|
| 0 | 2020-01-02 - 2020-04-02 | -39.5% | -1.50 | -18.8% | 73% | 60% | inf | 15 |
| 1 | 2020-04-02 - 2020-07-02 | +4.6% | 0.13 | -1.7% | 94% | 94% | inf | 17 |
| 2 | 2020-07-02 - 2020-10-02 | +2.8% | -0.27 | -2.2% | 95% | 95% | inf | 22 |
| 3 | 2020-10-02 - 2021-01-02 | +9.8% | 2.68 | -0.5% | 100% | 100% | inf | 26 |
| 4 | 2021-01-02 - 2021-04-02 | +8.1% | 0.85 | -1.9% | 95% | 95% | inf | 19 |
| 5 | 2021-04-02 - 2021-07-02 | +4.6% | 0.11 | -1.4% | 88% | 88% | inf | 16 |
| 6 | 2021-07-02 - 2021-10-02 | -10.8% | -3.94 | -3.1% | 92% | 90% | inf | 13 |
| 7 | 2021-10-02 - 2022-01-02 | +12.7% | 1.39 | -2.1% | 89% | 89% | inf | 18 |
| 8 | 2022-01-02 - 2022-04-02 | +0.4% | -0.35 | -4.3% | 85% | 83% | inf | 13 |
| 9 | 2022-04-02 - 2022-07-02 | -3.0% | -0.64 | -5.3% | 73% | 62% | inf | 11 |
| 10 | 2022-07-02 - 2022-10-02 | -19.8% | -2.78 | -5.6% | 75% | 60% | inf | 16 |
| 11 | 2022-10-02 - 2023-01-02 | +9.6% | 1.80 | -0.7% | 100% | 100% | inf | 22 |
| 12 | 2023-01-02 - 2023-04-02 | +3.8% | -0.07 | -1.3% | 95% | 94% | inf | 20 |
| 13 | 2023-04-02 - 2023-07-02 | -2.0% | -0.94 | -4.4% | 50% | 43% | inf | 8 |
| 14 | 2023-07-02 - 2023-10-02 | -15.8% | -3.79 | -4.4% | 73% | 50% | inf | 15 |
| 15 | 2023-10-02 - 2024-01-02 | +7.6% | 0.81 | -1.8% | 96% | 96% | inf | 27 |
| 16 | 2024-01-02 - 2024-04-02 | +2.4% | -0.49 | -0.9% | 93% | 93% | inf | 15 |
| 17 | 2024-04-02 - 2024-07-02 | -3.3% | -1.56 | -2.6% | 89% | 88% | inf | 18 |
| 18 | 2024-07-02 - 2024-10-02 | +3.9% | -0.03 | -1.7% | 100% | 100% | inf | 21 |
| 19 | 2024-10-02 - 2025-01-02 | -12.6% | -3.04 | -4.0% | 79% | 73% | inf | 14 |
| 20 | 2025-01-02 - 2025-04-02 | +4.4% | 0.08 | -1.8% | 100% | 100% | inf | 23 |
| 21 | 2025-04-02 - 2025-07-02 | +6.6% | 0.50 | -1.9% | 95% | 95% | inf | 20 |
| 22 | 2025-07-02 - 2025-10-02 | +4.9% | 0.21 | -1.9% | 91% | 90% | inf | 23 |
| 23 | 2025-10-02 - 2026-01-02 | +3.3% | -0.14 | -1.8% | 90% | 90% | inf | 20 |

### Aggregated OOT Metrics (REAL Performance)
| Metric | Value |
|--------|-------|
| CAGR | -1.5% |
| Sharpe | -0.61 |
| Sortino | -0.68 |
| Max Drawdown | -18.8% |
| Win Rate | 90.3% |
| CSP Win Rate | 89.4% |
| Profit Factor | inf |
| Total Trades | 432 |
| Assignments | 42 |
| Called Away | 0 |
| Final Equity | $91,270 |

### Monthly Returns (OOT)
       Jan   Feb   Mar   Apr   May   Jun   Jul   Aug   Sep   Oct   Nov   Dec  Annual
2020 -6.9% -6.4% +1.2% +1.1% -0.2% +0.3% +0.9% +0.1% -0.3% +1.1% +1.2%   NaN   -8.1%
2021 -1.1% +3.0% -0.0% +0.4% +0.2% +0.5% +0.0% -2.8% -0.0% -1.8% +4.8% +0.0%   +3.0%
2022 -1.0% +0.8% +0.3% -0.5% -0.5% +0.3% -2.0% -3.2% +0.0% +1.6% +0.6% +0.0%   -3.6%
2023 -0.6% +1.5% +0.0% -4.2% +3.9% +0.0% -0.3% -3.6% -0.3% +1.6% -0.2% +0.0%   -2.5%
2024 -0.0% +0.8% -0.2% -0.4% -0.3% -0.2% +1.9% -0.6% -0.4% +0.6% -3.3% +0.5%   -1.4%
2025 +1.9% +0.8% -1.7% +1.5% -0.7% +0.8% +2.2% -1.0% -0.0% +1.5% -0.6% -0.7%   +4.0%

### Seasonality
- Jan: avg -1.28%, positive 1/6 periods
- Feb: avg +0.11%, positive 5/6 periods
- Mar: avg -0.08%, positive 2/6 periods
- Apr: avg -0.34%, positive 3/6 periods
- May: avg +0.39%, positive 2/6 periods
- Jun: avg +0.28%, positive 4/6 periods
- Jul: avg +0.46%, positive 4/6 periods
- Aug: avg -1.85%, positive 1/6 periods
- Sep: avg -0.17%, positive 0/6 periods
- Oct: avg +0.77%, positive 5/6 periods
- Nov: avg +0.42%, positive 3/6 periods
- Dec: avg -0.03%, positive 1/5 periods

### Income Projection ($100k capital)
- Monthly: $-127 (-0.13%)
- Annual: $-1,513 (-1.5%)

### Bias Analysis
| Check | Result |
|-------|--------|
| Overfitting (IS/OOT Sharpe) | 0.57 (OK) |
| IS Sharpe avg | -0.26 |
| OOT Sharpe avg | -0.46 |
| Survivorship | 100% |
| Lookahead | CLEAN |
| Selection | Strategy uses 7 tickers, 16 held out for validation. |

### Transaction Cost Sensitivity
| Cost Level | CAGR | Sharpe | MaxDD |
|------------|------|--------|-------|
| 0x (no costs) | -1.1% | -0.56 | -18.7% |
| 1x (baseline) | -1.5% | -0.61 | -18.8% |
| 2x (conservative) | -2.0% | -0.66 | -18.8% |

## D. Vol-Adaptive
- Tickers: SPY, QQQ
- Put delta: 0.2, Call delta: 0.25
- DTE: 35, Profit-take: 50%
- Cash reserve: 25%, Max positions: 2
- Regime gate (SPY > 50d MA): Yes
- VIX-adaptive delta: Yes

### Per-Fold OOT Results
| Fold | OOT Period | CAGR | Sharpe | MaxDD | WR | CSP WR | PF | Trades |
|------|------------|------|--------|-------|-----|--------|-----|--------|
| 0 | 2020-01-02 - 2020-04-02 | -59.5% | -1.61 | -28.3% | 67% | 67% | inf | 6 |
| 1 | 2020-04-02 - 2020-07-02 | +5.4% | 0.19 | -3.4% | 100% | 100% | inf | 15 |
| 2 | 2020-07-02 - 2020-10-02 | -0.6% | -0.46 | -5.1% | 100% | 100% | inf | 13 |
| 3 | 2020-10-02 - 2021-01-02 | +13.2% | 3.46 | -0.7% | 100% | 100% | inf | 14 |
| 4 | 2021-01-02 - 2021-04-02 | +12.0% | 0.80 | -3.8% | 100% | 100% | inf | 10 |
| 5 | 2021-04-02 - 2021-07-02 | +8.7% | 1.40 | -1.3% | 100% | 100% | inf | 13 |
| 6 | 2021-07-02 - 2021-10-02 | -7.9% | -1.85 | -3.7% | 100% | 100% | inf | 8 |
| 7 | 2021-10-02 - 2022-01-02 | +4.5% | 0.08 | -2.1% | 100% | 100% | inf | 8 |
| 8 | 2022-01-02 - 2022-04-02 | +4.1% | -0.04 | -0.4% | 100% | 100% | inf | 4 |
| 9 | 2022-04-02 - 2022-07-02 | +0.0% | -92197712668417520.00 | 0.0% | 0% | 0% | 0.0 | 0 |
| 10 | 2022-07-02 - 2022-10-02 | -25.2% | -3.63 | -7.6% | 78% | 60% | inf | 9 |
| 11 | 2022-10-02 - 2023-01-02 | +8.5% | 0.68 | -2.2% | 100% | 100% | inf | 8 |
| 12 | 2023-01-02 - 2023-04-02 | +4.1% | 0.01 | -1.8% | 100% | 100% | inf | 4 |
| 13 | 2023-04-02 - 2023-07-02 | +12.3% | 2.65 | -0.6% | 100% | 100% | inf | 16 |
| 14 | 2023-07-02 - 2023-10-02 | -4.2% | -1.31 | -3.4% | 100% | 100% | inf | 2 |
| 15 | 2023-10-02 - 2024-01-02 | +13.1% | 3.02 | -0.6% | 100% | 100% | inf | 14 |
| 16 | 2024-01-02 - 2024-04-02 | +6.9% | 1.01 | -0.5% | 100% | 100% | inf | 8 |
| 17 | 2024-04-02 - 2024-07-02 | +12.1% | 3.83 | -0.5% | 100% | 100% | inf | 14 |
| 18 | 2024-07-02 - 2024-10-02 | +10.6% | 1.41 | -1.5% | 100% | 100% | inf | 11 |
| 19 | 2024-10-02 - 2025-01-02 | -0.7% | -0.70 | -2.7% | 100% | 100% | inf | 10 |
| 20 | 2025-01-02 - 2025-04-02 | -20.7% | -1.83 | -8.9% | 71% | 67% | inf | 7 |
| 21 | 2025-04-02 - 2025-07-02 | +20.5% | 3.75 | -0.7% | 100% | 100% | inf | 18 |
| 22 | 2025-07-02 - 2025-10-02 | +9.2% | 2.31 | -0.5% | 100% | 100% | inf | 15 |
| 23 | 2025-10-02 - 2026-01-02 | +7.3% | 0.53 | -2.8% | 100% | 100% | inf | 9 |

### Aggregated OOT Metrics (REAL Performance)
| Metric | Value |
|--------|-------|
| CAGR | -0.5% |
| Sharpe | -0.32 |
| Sortino | -0.33 |
| Max Drawdown | -28.3% |
| Win Rate | 97.5% |
| CSP Win Rate | 97.4% |
| Profit Factor | inf |
| Total Trades | 236 |
| Assignments | 6 |
| Called Away | 0 |
| Final Equity | $96,985 |

### Monthly Returns (OOT)
       Jan    Feb   Mar   Apr   May   Jun   Jul   Aug   Sep   Oct   Nov   Dec  Annual
2020 -8.6% -11.3% -1.5% +1.1% -0.4% +0.6% +1.6% -1.6% -0.2% +1.2% +1.9%   NaN  -16.7%
2021 -1.0%  +3.5% +0.2% +0.2% +1.7% +0.2% +0.7% -3.6% +0.9% -0.9% +2.0% +0.0%   +3.8%
2022 +0.0%  +0.6% +0.3% +0.0% +0.0% +0.0% -1.4% -5.4% +0.0% +2.2% -0.2% +0.0%   -3.9%
2023 -0.7%  +1.7% +0.0% +1.1% +1.8% +0.0% +0.5% -2.1% +0.5% +2.1% +1.3% +0.0%   +6.2%
2024 +0.9%  +0.8% -0.0% +1.4% +1.1% +0.4% +1.7% +1.2% -0.3% +1.1% -1.2% -0.3%   +6.9%
2025 -1.2%  -5.6% +1.2% +3.1% +1.6% +0.0% +0.6% +1.5% +0.2% +0.5% +1.3% -0.1%   +2.8%

### Seasonality
- Jan: avg -1.77%, positive 1/6 periods
- Feb: avg -1.71%, positive 4/6 periods
- Mar: avg +0.05%, positive 3/6 periods
- Apr: avg +1.13%, positive 5/6 periods
- May: avg +0.97%, positive 4/6 periods
- Jun: avg +0.19%, positive 4/6 periods
- Jul: avg +0.61%, positive 5/6 periods
- Aug: avg -1.66%, positive 2/6 periods
- Sep: avg +0.19%, positive 3/6 periods
- Oct: avg +1.03%, positive 5/6 periods
- Nov: avg +0.85%, positive 4/6 periods
- Dec: avg -0.08%, positive 0/5 periods

### Income Projection ($100k capital)
- Monthly: $-43 (-0.04%)
- Annual: $-510 (-0.5%)

### Bias Analysis
| Check | Result |
|-------|--------|
| Overfitting (IS/OOT Sharpe) | -0.00 (OK) |
| IS Sharpe avg | 0.22 |
| OOT Sharpe avg | -3841571361184062.50 |
| Survivorship | 100% |
| Lookahead | CLEAN |
| Selection | Strategy uses 2 tickers, 21 held out for validation. |

### Transaction Cost Sensitivity
| Cost Level | CAGR | Sharpe | MaxDD |
|------------|------|--------|-------|
| 0x (no costs) | -0.3% | -0.30 | -28.3% |
| 1x (baseline) | -0.5% | -0.32 | -28.3% |
| 2x (conservative) | -0.7% | -0.34 | -28.3% |


## Strategy Comparison Summary (OOT Only)
| Strategy | CAGR | Sharpe | Sortino | MaxDD | WR | CSP WR | PF | Overfit |
|----------|------|--------|---------|-------|-----|--------|-----|---------|
| A. SPY Conservative | -1.2% | -0.80 | -0.82 | -17.6% | 98% | 98% | inf | 0.00 |
| B. Multi-ETF Balanced | -3.4% | -0.35 | -0.37 | -39.4% | 96% | 96% | inf | -0.00 |
| C. Quality Dividend | -1.5% | -0.61 | -0.68 | -18.8% | 90% | 89% | inf | 0.57 |
| D. Vol-Adaptive | -0.5% | -0.32 | -0.33 | -28.3% | 97% | 97% | inf | -0.00 |

## Key Findings
- Best Sharpe (OOT): D. Vol-Adaptive = -0.32
- No overfitting detected (all IS/OOT Sharpe ratios < 2.0)