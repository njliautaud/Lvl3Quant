# ES->SPY Lead-Lag (raw cross-correlation)

**Grid**: 50ms  **Lag range**: -2000..2000ms step 50ms  **Days**: 9/9

## Headline

- Overall peak correlation **0.1496** at lag **100 ms** (ES leads SPY).
- Best positive (ES-leads-SPY) lag: **100 ms**, corr **0.1496**.
- Best negative (SPY-leads-ES) lag: **-50 ms**, corr **0.0396**.

## Per-day peak ES->SPY lag

| date | peak_lag_ms | peak_corr |
|---|---|---|
| 20260302 | 50 | 0.1408 |
| 20260303 | 50 | 0.1598 |
| 20260304 | 50 | 0.1387 |
| 20260305 | 100 | 0.1476 |
| 20260306 | 50 | 0.1558 |
| 20260309 | 100 | 0.1617 |
| 20260310 | 50 | 0.1539 |
| 20260311 | 100 | 0.1452 |
| 20260312 | 50 | 0.1446 |

## Top-10 lags by overall correlation

| rank | lag_ms | corr |
|---|---|---|
| 1 | 100 | 0.1496 |
| 2 | 50 | 0.1495 |
| 3 | 150 | 0.1455 |
| 4 | 0 | 0.1406 |
| 5 | 200 | 0.1282 |
| 6 | -50 | 0.0396 |
| 7 | -100 | 0.0190 |
| 8 | 250 | 0.0133 |
| 9 | -150 | 0.0097 |
| 10 | -200 | 0.0043 |

## SPY-leads-ES windows of interest (neg-lag corr > 50% of pos-lag corr)

None found. SPY does NOT consistently lead ES in any intraday window. Negative-lag correlations are uniformly far weaker than positive-lag.


## Verdict

ES leads SPY by ~100 ms with peak correlation 0.150. This matches the prior that ES futures are the S&P price-discovery venue. No window shows SPY leading ES nontrivially — typical positive-lag asymmetry is preserved everywhere.
