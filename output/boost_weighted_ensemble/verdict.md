# Boosting (c) verdict — weighted ensemble v3.3 / v3.4.2

HC #427 R5 boosting technique #3. Sweep `w33*v3.3 + w342*v3.4.2` weights;
score against v3.4.2 top-20 best_configs.

## Baselines (from boost (a))
- v3.4.2 SOLO: 12/20 robust
- ensemble 0.5/0.5: 11/20 robust

## Sweep results

| w33 | w342 | n_robust | mean(worst_day_Sharpe) |
|----:|-----:|---------:|-----------------------:|
| 0.4 | 0.6 | 11 | 6.26 |
| 0.3 | 0.7 | 9 | 7.83 |
| 0.2 | 0.8 | 12 | 6.77 |
| 0.6 | 0.4 | 10 | 7.29 |
| 0.1 | 0.9 | 11 | 6.30 |

## Winning weight: **w33=0.2, w342=0.8** (n_robust=12)