# Macro-Exposure v1 — Balanced Tier Walk-Forward OOS Validation

**Fixed chromosome** (NO re-optimization): the Balanced tier row from `results/full/pareto.parquet` (CAGR=20.38%, DD=12.38%, Sortino=1.56 in-sample).

**Walk-forward**: 1-year OOT folds stepped every 6 months from 2015-01-01. Pooled OOS uses non-overlapping folds (every other fold).

**Regime label per fold**: SPY CAGR over OOT window. bull > +10%, bear < -5%, chop in between.

**Gates (task brief)**: pooled CAGR >= 10%, pooled DD <= 18%, no regime Sortino < 0.5, hit rate >= 65%.

## Pooled OOS metrics

- CAGR: **21.54%**
- Max drawdown: **12.38%**
- Sortino: **1.55**
- Sharpe: **1.43**
- Worst month: **-8.83%**
- N trading days pooled: **2771**
- Fold hit rate (positive CAGR): **100.0%**  (n_folds=22)

## Per-fold table

| fold | OOT start | OOT end | regime | SPY CAGR | fold CAGR | fold DD | fold Sortino | fold Sharpe |
|---|---|---|---|---|---|---|---|---|
| 0 | 2015-01-01 | 2015-12-31 | chop | +1.3% | +3.62% | 7.08% | 0.39 | 0.37 |
| 1 | 2015-07-01 | 2016-06-30 | chop | +3.1% | +0.53% | 8.09% | 0.05 | 0.11 |
| 2 | 2016-01-01 | 2016-12-31 | bull | +13.5% | +9.42% | 8.26% | 0.81 | 0.86 |
| 3 | 2016-07-01 | 2017-06-30 | bull | +17.5% | +23.87% | 5.84% | 2.60 | 1.99 |
| 4 | 2017-01-01 | 2017-12-31 | bull | +20.9% | +26.02% | 4.05% | 2.98 | 2.44 |
| 5 | 2017-07-01 | 2018-06-30 | bull | +14.2% | +22.99% | 12.38% | 1.45 | 1.32 |
| 6 | 2018-01-01 | 2018-12-31 | bear | -5.3% | +6.53% | 12.38% | 0.42 | 0.49 |
| 7 | 2018-07-01 | 2019-06-30 | chop | +10.0% | +25.44% | 2.90% | 2.49 | 2.67 |
| 8 | 2019-01-01 | 2019-12-31 | bull | +31.1% | +34.81% | 8.13% | 2.45 | 2.15 |
| 9 | 2019-07-01 | 2020-06-30 | chop | +6.4% | +50.57% | 8.13% | 2.38 | 1.87 |
| 10 | 2020-01-01 | 2020-12-31 | bull | +17.2% | +75.93% | 7.34% | 3.18 | 2.61 |
| 11 | 2020-07-01 | 2021-06-30 | bull | +39.9% | +29.21% | 4.95% | 2.36 | 2.44 |
| 12 | 2021-01-01 | 2021-12-31 | bull | +30.5% | +8.31% | 2.14% | 1.76 | 1.96 |
| 13 | 2021-07-01 | 2022-06-30 | bear | -11.1% | +4.13% | 4.90% | 0.39 | 0.77 |
| 14 | 2022-01-01 | 2022-12-31 | bear | -18.7% | +4.03% | 10.99% | 0.36 | 0.31 |
| 15 | 2022-07-01 | 2023-06-30 | bull | +18.3% | +36.64% | 10.99% | 2.54 | 1.53 |
| 16 | 2023-01-01 | 2023-12-31 | bull | +26.8% | +38.03% | 10.23% | 3.06 | 2.09 |
| 17 | 2023-07-01 | 2024-06-30 | bull | +24.5% | +15.52% | 7.18% | 1.55 | 1.51 |
| 18 | 2024-01-01 | 2024-12-31 | bull | +25.6% | +25.00% | 6.36% | 1.62 | 1.85 |
| 19 | 2024-07-01 | 2025-06-30 | bull | +14.7% | +30.81% | 10.46% | 2.00 | 1.77 |
| 20 | 2025-01-01 | 2025-12-31 | bull | +17.9% | +21.58% | 10.46% | 1.64 | 1.37 |
| 21 | 2025-07-01 | 2026-06-04 | bull | +25.4% | +31.28% | 12.34% | 2.36 | 1.71 |

## Per-regime stratification

| regime | n_folds | avg CAGR | avg DD | avg Sortino | avg Sharpe | hit rate |
|---|---|---|---|---|---|---|
| bull | 15 | +28.63% | 8.07% | 2.16 | 1.84 | 100.0% |
| bear | 3 | +4.90% | 9.43% | 0.39 | 0.52 | 100.0% |
| chop | 4 | +20.04% | 6.55% | 1.33 | 1.25 | 100.0% |

## Verdict

**FAIL**

Failure reasons:
- bear regime Sortino 0.39 < 0.5

## In-sample reference

- CAGR 20.38%, DD 12.38%, Sortino 1.56, Sharpe 1.41 (2010-2026 full sample).
