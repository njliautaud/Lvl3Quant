# PASS 8 — Top5% Loose-Band Validation

Candidate: S_Top5% × agree_15 × no_reversal × golden-ToD × vol_mid

## T1. Per-Day Breakdown

| day | n_fill | mean_t | WR% | Sharpe | total_t |
|---|---:|---:|---:|---:|---:|
| 0 | 0 | nan | nan | nan | 0.00 |
| 1 | 10 | -0.550 | 40.0 | -0.53 | -5.50 |
| 2 | 239 | 0.653 | 62.8 | 3.79 | 156.00 |
| 3 | 13 | 1.308 | 61.5 | 1.81 | 17.00 |
| 4 | 0 | nan | nan | nan | 0.00 |

## T2. Leave-One-Day-Out CV

| Holdout | n_fill | mean_t | WR% | Sharpe | total_t |
|---|---:|---:|---:|---:|---:|
| day 0 | 0 | nan | nan | nan | 0.00 |
| day 1 | 61 | 0.811 | 65.6 | 2.09 | 49.50 |
| day 2 | 245 | 0.667 | 62.9 | 3.92 | 163.50 |
| day 3 | 8 | 1.500 | 62.5 | 1.71 | 12.00 |
| day 4 | 0 | nan | nan | nan | 0.00 |

## T3. ToD Bucket Coverage

| Bucket | n_fill | mean_t | WR% | Sharpe |
|---|---:|---:|---:|---:|
| 11:1800-12:00 | 197 | 0.734 | 65.5 | 3.68 |
| 12:1800-13:00 | 42 | 0.274 | 50.0 | 0.92 |
| 13:1800-14:00 | 23 | 0.500 | 52.2 | 0.80 |

## T4. Permutation Test (1000 random sign-shuffles)

- Observed Sharpe: **3.845**
- Null mean: 3.662
- Null 95th: 4.895
- Null 99th: 5.350
- **p-value = 0.3900**

## T5. Half-Split

| Split | n_fill | mean_t | WR% | Sharpe |
|---|---:|---:|---:|---:|
| first_2_days | 10 | -0.550 | 40.0 | -0.53 |
| last_3_days | 252 | 0.687 | 62.7 | 4.10 |

## T6. Equity Curve & Drawdown

- Total trades: 262
- Total PnL: **167.5 ticks = $2094**
- Max drawdown: **-30.0 ticks = $-375**
- Calmar proxy (total / |maxDD|): 5.58

## T7. WITHOUT ToD filter (any time)

- n=1053, mean_t=0.268, WR=56.4%, Sharpe=3.03

## T8. Top5% RAW (no confluence at all)

- n=2587, mean_t=0.202, WR=55.7%, Sharpe=3.51

## T9. Band Sensitivity (full filters + golden ToD)

| Band | n_fill | mean_t | WR% | Sharpe |
|---|---:|---:|---:|---:|
| Top0.5 | 19 | 1.289 | 68.4 | 2.70 |
| Top1.0 | 48 | 0.688 | 62.5 | 1.92 |
| Top2.0 | 106 | 0.533 | 63.2 | 2.07 |
| Top3.0 | 165 | 0.564 | 61.2 | 2.78 |
| Top5.0 | 262 | 0.639 | 61.8 | 3.85 |
| Top10.0 | 447 | 0.538 | 60.4 | 4.05 |
| Top20.0 | 651 | 0.552 | 60.5 | 4.93 |

## T10. Per-Day Dollars (assuming 1 contract/fill)

| day | n | total_t | total_$ | max_dd_t |
|---|---:|---:|---:|---:|
| 0 | 0 | 0.00 | $0 | 0.00 |
| 1 | 10 | -5.50 | $-69 | -9.50 |
| 2 | 239 | 156.00 | $1950 | -30.00 |
| 3 | 13 | 17.00 | $212 | -4.00 |
| 4 | 0 | 0.00 | $0 | 0.00 |

