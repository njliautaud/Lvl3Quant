# ETF rotation sweep

Output: `/home/jupiter/Lvl3Quant/output/macro_picker/etf_rotation_sweep_20260608_120509`

| Run | Sharpe | Calmar | CAGR | MaxDD | Deploy |
|---|---|---|---|---|---|
| hold5_long_short | 1.78 | 1.20 | 117.3% | -98.1% | Y |
| hold5_long_only | 1.67 | 0.46 | 44.7% | -96.2% | N |
| hold10_long_short | 0.92 | -0.90 | -89.9% | -99.4% | N |
| hold21_long_only | 0.56 | -0.98 | -98.3% | -99.9% | N |
| hold21_long_short | 0.08 | -1.00 | -99.6% | -99.9% | N |
| hold10_long_only | -0.13 | -1.00 | -99.7% | -100.0% | N |

Deploy gate: Calmar >= 1.0