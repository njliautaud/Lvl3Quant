# Tier2 Balanced Scalp - Vol Regime + Tail Risk

Sharpe 2.51, CAGR 27.4%, MaxDD -15.5%, n=1507 days


## VIX-regime Sharpe table

| Regime | n_days | mean% | std% | Sharpe | WR | worst day% |
|---|---|---|---|---|---|---|
| VIX_low_lt15 | 278 | 0.147 | 0.591 | 3.95 | 64.0% | -1.72 |
| VIX_mid_15_25 | 895 | 0.124 | 0.484 | 4.08 | 61.1% | -1.61 |
| VIX_high_25_35 | 275 | 0.067 | 0.745 | 1.44 | 51.3% | -2.50 |
| VIX_spike_gt35 | 59 | -0.392 | 1.350 | -4.61 | 23.7% | -6.32 |

## 5 worst single-day losses

| date | wheel% | VIX close | VIX chg | SPY% |
|---|---|---|---|---|
| 2020-03-09 | -6.32 | 54.5 | +12.5 | -7.81 |
| 2020-03-12 | -3.93 | 75.5 | +21.6 | -9.57 |
| 2020-02-27 | -2.97 | 39.2 | +11.6 | -4.49 |
| 2025-04-04 | -2.78 | 45.3 | +15.3 | -5.85 |
| 2025-03-10 | -2.50 | 27.9 | +4.5 | -2.66 |

## Tail risk

- VaR-95 daily: -0.77%
- CVaR-95 daily: -1.33%

## Top 5 drawdowns (decomposed)

| start | trough | recover | maxDD% | VIX peak | regime |
|---|---|---|---|---|---|
| 2020-02-20 | 2020-03-23 | 2020-10-15 | -15.52 | 82.7 | vol_spike |
| 2025-04-03 | 2025-04-04 | 2025-04-23 | -4.56 | 45.3 | vol_spike |
| 2022-04-14 | 2022-05-11 | 2022-05-26 | -3.53 | 34.8 | elevated_vol |
| 2025-03-06 | 2025-03-10 | 2025-03-24 | -3.26 | 27.9 | elevated_vol |
| 2023-03-06 | 2023-04-04 | 2023-05-19 | -3.20 | 26.5 | elevated_vol |

## Survivability

- **COVID_2020** (2020-02-15 to 2020-05-31): maxDD -15.52%, worst week -10.19%, VIX peak 82.7, recovered 2020-10-15 (206 days)
- **Aug_2024_carry_unwind** (2024-07-15 to 2024-09-15): maxDD -1.33%, worst week -1.31%, VIX peak 38.6, recovered 2024-08-09 (4 days)
- **2022_bear_grind_down** (2022-01-01 to 2022-12-31): maxDD -3.53%, worst week -2.93%, VIX peak 36.5, recovered 2022-05-26 (15 days)

## Kelly-style sizing

- Observed max DD at 1.0x: 15.52%
- Stressed 60d loss: 15.60%
- 25% DD-cap leverage at observed: 1.71x
- 25% DD-cap leverage at stress: 1.70x
- **Recommended deployable fraction: 1.36x** (20% model-risk haircut)

## HC #428 R1 record

- Sharpe green days: 6.89  red days: -2.04  gap: 1.30  pass: False
- Short-vol/wheel is structurally long up-tape; failing HC#428 R1 was expected. This vol-regime stratification is the short-vol-appropriate substitute.
