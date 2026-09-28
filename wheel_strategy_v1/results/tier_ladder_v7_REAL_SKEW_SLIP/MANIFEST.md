# v7_REAL_SKEW_SLIP — Final honest backtest

**Stack:**
- Real IV from DoltHub (51% of rows; 49% modeled fallback for pre-2019 + ARM/SHOP)
- Skew model (slope -0.10, curvature +0.20 in log(K/S)/sqrt(T))
- Slippage model (2.5% of premium per leg, $0.03/share floor — applies to all opens & closes)
- Macro regime overlay HC #555 (25% of days masked as risk-off)
- Full wheel (profit-take 0.65, roll DTE 1, allows assignment)

**Per-tier realized metrics, 2020-2025, $100k each:**

| Tier         | CAGR   | Sharpe | MaxDD  | Win Rate | Trades |
|--------------|-------:|-------:|-------:|---------:|-------:|
| Conservative |  3.33% |  0.85  |  -5.9% |   93.0%  |   388  |
| Balanced     | 10.54% |  1.02  | -11.8% |   89.6%  |   570  |
| Income       |  1.01% |  0.14  | -23.0% |   86.7%  |   533  |
| Aggressive   |  7.53% |  0.36  | -39.0% |   83.5%  |   565  |
| Turbo        |  7.00% |  0.54  | -28.0% |   80.5%  |   728  |

**Recommended blended ladder (35% Conservative / 35% Balanced / 25% Aggressive / 5% Turbo; Income dropped):**

| Metric           | Value      |
|------------------|-----------:|
| Starting capital | $1,000,000 |
| Ending equity    | $1,525,999 |
| Total return     | +52.6%     |
| CAGR             | 7.30%      |
| Sharpe           | 0.76       |
| Sortino          | 0.55       |
| Max drawdown     | -16.8%     |

**Yearly:** 2021 +20.5% | 2022 -8.1% | 2023 -1.3% | 2024 +20.1% | 2025 +26.5%

**Engine flags (in wheel_engine.py):**
- USE_SKEW = True
- USE_SLIPPAGE = True / SLIPPAGE_FRAC = 0.025 / SLIPPAGE_MIN_TICKS = 0.03

To reproduce:
```
cp data/cache/iv_features_real_blend.parquet data/cache/iv_features_modeled.parquet
python -m strategy.tier_runner --modeled --full-wheel --regime-overlay \
   --start 2020-01-01 --end 2025-12-31 \
   --out results/tier_ladder_v7_REAL_SKEW_SLIP
cp data/cache/iv_features_modeled.MODELED_BACKUP.parquet data/cache/iv_features_modeled.parquet
```
