# ETF rotation sweep

Output: `/home/jupiter/Lvl3Quant/output/macro_picker/etf_rotation_sweep_20260608_121557`

| Run | Sharpe | Calmar | CAGR | MaxDD | Deploy |
|---|---|---|---|---|---|
| hold10_long_only | 1.90 | 3.79 | 31.4% | -8.3% | Y |
| hold10_long_short | 1.57 | 1.99 | 24.4% | -12.3% | Y |
| hold21_long_only | 1.46 | 1.88 | 22.7% | -12.1% | Y |
| hold5_long_only | 0.68 | 0.77 | 8.4% | -10.9% | Y |
| hold21_long_short | 0.61 | 0.58 | 7.8% | -13.6% | Y |
| hold5_long_short | -0.19 | -0.13 | -3.2% | -25.2% | N |

Deploy gate: Calmar >= 1.0