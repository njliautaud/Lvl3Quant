# ETF rotation sweep

Output: `/home/jupiter/Lvl3Quant/output/macro_picker/etf_rotation_sweep_20260608_121218`

| Run | Sharpe | Calmar | CAGR | MaxDD | Deploy |
|---|---|---|---|---|---|
| hold10_long_only | 1.65 | 2.98 | 26.7% | -9.0% | Y |
| hold21_long_only | 1.63 | 2.45 | 25.6% | -10.4% | Y |
| hold10_long_short | 1.19 | 1.42 | 17.6% | -12.4% | N |
| hold5_long_only | 0.78 | 0.95 | 9.9% | -10.4% | Y |
| hold21_long_short | 0.69 | 0.67 | 9.1% | -13.7% | Y |
| hold5_long_short | 0.36 | 0.22 | 3.9% | -17.6% | N |

Deploy gate: Calmar >= 1.0