# v3.3 PRICE-PATH BLOCK — HC #363 deliverable 6 (per HC #361)

Source: `fold_00_predictions.npz` (5 OOT days, 241,351 events).
Note: `target_log_ret_*` empirically already in tick units; no log-return conversion applied.
All metrics computed on FIFO-fillable samples (mask_fifo_tp4sl3_net=1). NaN entries filtered.

## p_reversal_60s SHORT Top0.1% (HC357 #2, JOINT #1) (n=61)

### Time-exit P&L (SHORT, after 0.376t commission)

| Exit | n | Mean ticks | Median | Sharpe | Win-rate | P25 / P75 |
|---|---|---|---|---|---|---|
|   1s | 61 | -1.687 | -1.376 | -0.904 | 14.8% | -2.38 / -0.38 |
|   5s | 61 | -2.294 | -2.376 | -0.807 | 11.5% | -3.38 / -0.38 |
|  10s | 60 | -1.909 | -2.376 | -0.436 | 21.3% | -4.63 / -0.38 |
|  30s | 60 | -1.876 | -1.376 | -0.258 | 27.9% | -6.38 / +0.87 |
| **tp4sl3 FIFO** | 61 | +0.885 | n/a | +0.258 | n/a | n/a |

### Price-path characteristics (30s window, on FIFO-fillable+MFE-labeled)

| Metric | Value |
|---|---|
| Short-side favorable excursion (mean / median / p75) | 7.50 / 4.00 / 5.50 ticks |
| Short-side adverse excursion (mean / median / p75) | 4.62 / 4.00 / 6.00 ticks |
| Favorable / Adverse ratio (mean) | 1.63 |
| Time to long-MFE peak (mean / median, sec) | 12.6 / 8.0 |
| Realized vol 30s (mean) | 7.38 ticks |

**Optimal time-exit (best Sharpe): 30s → Sharpe -0.258, mean -1.876 ticks, win-rate 27.9%.**

## fifo_tp8sl5_net SHORT Top0.1% (HC357 #1, FIFO #2) (n=72)

### Time-exit P&L (SHORT, after 0.376t commission)

| Exit | n | Mean ticks | Median | Sharpe | Win-rate | P25 / P75 |
|---|---|---|---|---|---|---|
|   1s | 72 | -0.140 | +0.124 | -0.061 | 50.0% | -1.38 / +1.62 |
|   5s | 72 | -1.890 | -2.376 | -0.423 | 25.0% | -4.38 / -0.13 |
|  10s | 72 | -3.209 | -2.876 | -0.738 | 25.0% | -6.38 / -0.13 |
|  30s | 72 | -9.626 | -10.376 | -2.192 | 0.0% | -11.63 / -7.38 |
| **tp4sl3 FIFO** | 72 | +0.882 | n/a | +0.274 | n/a | n/a |

### Price-path characteristics (30s window, on FIFO-fillable+MFE-labeled)

| Metric | Value |
|---|---|
| Short-side favorable excursion (mean / median / p75) | 13.23 / 14.00 / 16.50 ticks |
| Short-side adverse excursion (mean / median / p75) | 7.33 / 7.50 / 12.50 ticks |
| Favorable / Adverse ratio (mean) | 1.80 |
| Time to long-MFE peak (mean / median, sec) | 3.5 / 1.8 |
| Realized vol 30s (mean) | 14.80 ticks |

**Optimal time-exit (best Sharpe): 1s → Sharpe -0.061, mean -0.140 ticks, win-rate 50.0%.**

## log_ret_60s_q50 SHORT Top1% (HC357 #3, low-σ) (n=599)

### Time-exit P&L (SHORT, after 0.376t commission)

| Exit | n | Mean ticks | Median | Sharpe | Win-rate | P25 / P75 |
|---|---|---|---|---|---|---|
|   1s | 599 | -0.490 | -0.376 | -0.190 | 36.6% | -1.38 / +1.12 |
|   5s | 599 | -0.633 | -0.376 | -0.102 | 44.4% | -3.38 / +2.62 |
|  10s | 599 | -1.089 | -1.376 | -0.131 | 40.1% | -7.38 / +6.12 |
|  30s | 599 | +1.332 | +2.624 | +0.082 | 61.6% | -3.38 / +9.62 |
| **tp4sl3 FIFO** | 599 | -0.448 | n/a | -0.132 | n/a | n/a |

### Price-path characteristics (30s window, on FIFO-fillable+MFE-labeled)

| Metric | Value |
|---|---|
| Short-side favorable excursion (mean / median / p75) | 9.31 / 8.00 / 16.00 ticks |
| Short-side adverse excursion (mean / median / p75) | 6.32 / 6.50 / 9.00 ticks |
| Favorable / Adverse ratio (mean) | 1.47 |
| Time to long-MFE peak (mean / median, sec) | 12.9 / 8.8 |
| Realized vol 30s (mean) | 11.44 ticks |

**Optimal time-exit (best Sharpe): 30s → Sharpe +0.082, mean +1.332 ticks, win-rate 61.6%.**

## log_ret_10s_q50 SHORT Top0.1% (FIFO #3) (n=62)

### Time-exit P&L (SHORT, after 0.376t commission)

| Exit | n | Mean ticks | Median | Sharpe | Win-rate | P25 / P75 |
|---|---|---|---|---|---|---|
|   1s | 62 | +2.124 | +0.624 | +0.167 | 58.1% | -0.38 / +1.62 |
|   5s | 62 | +2.358 | +0.624 | +0.180 | 61.3% | -0.38 / +2.62 |
|  10s | 62 | +1.293 | +0.124 | +0.093 | 50.0% | -1.38 / +3.62 |
|  30s | 62 | -0.481 | +0.624 | -0.035 | 53.2% | -1.25 / +4.37 |
| **tp4sl3 FIFO** | 62 | +0.048 | n/a | +0.015 | n/a | n/a |

### Price-path characteristics (30s window, on FIFO-fillable+MFE-labeled)

| Metric | Value |
|---|---|
| Short-side favorable excursion (mean / median / p75) | 6.78 / 4.50 / 10.25 ticks |
| Short-side adverse excursion (mean / median / p75) | 7.00 / 5.50 / 9.25 ticks |
| Favorable / Adverse ratio (mean) | 0.97 |
| Time to long-MFE peak (mean / median, sec) | 17.2 / 16.7 |
| Realized vol 30s (mean) | 9.07 ticks |

**Optimal time-exit (best Sharpe): 5s → Sharpe +0.180, mean +2.358 ticks, win-rate 61.3%.**

## fifo_tp4sl3_net SHORT Top0.5% (largest n_fills survivor) (n=408)

### Time-exit P&L (SHORT, after 0.376t commission)

| Exit | n | Mean ticks | Median | Sharpe | Win-rate | P25 / P75 |
|---|---|---|---|---|---|---|
|   1s | 408 | -0.097 | -0.376 | -0.050 | 46.8% | -1.38 / +1.62 |
|   5s | 408 | -0.606 | -0.376 | -0.164 | 41.4% | -2.38 / +1.62 |
|  10s | 408 | -1.151 | -0.376 | -0.273 | 42.6% | -4.38 / +1.87 |
|  30s | 408 | -2.798 | -1.376 | -0.318 | 41.4% | -10.38 / +4.62 |
| **tp4sl3 FIFO** | 408 | +0.555 | n/a | +0.170 | n/a | n/a |

### Price-path characteristics (30s window, on FIFO-fillable+MFE-labeled)

| Metric | Value |
|---|---|
| Short-side favorable excursion (mean / median / p75) | 8.62 / 8.50 / 12.50 ticks |
| Short-side adverse excursion (mean / median / p75) | 10.98 / 10.50 / 16.50 ticks |
| Favorable / Adverse ratio (mean) | 0.79 |
| Time to long-MFE peak (mean / median, sec) | 13.4 / 10.9 |
| Realized vol 30s (mean) | 13.62 ticks |

**Optimal time-exit (best Sharpe): 1s → Sharpe -0.050, mean -0.097 ticks, win-rate 46.8%.**

## Cross-cell summary — best time-exit horizon per cell

| Cell | Best exit | Sharpe | Mean t | Win-rate | n |
|---|---|---|---|---|---|
| p_reversal_60s Top0.1% (HC357 #2, JOINT #1) | 30s | -0.258 | -1.876 | 27.9% | 60 |
| fifo_tp8sl5_net Top0.1% (HC357 #1, FIFO #2) | 1s | -0.061 | -0.140 | 50.0% | 72 |
| log_ret_60s_q50 Top1% (HC357 #3, low-σ) | 30s | +0.082 | +1.332 | 61.6% | 599 |
| log_ret_10s_q50 Top0.1% (FIFO #3) | 5s | +0.180 | +2.358 | 61.3% | 62 |
| fifo_tp4sl3_net Top0.5% (largest n_fills survivor) | 1s | -0.050 | -0.097 | 46.8% | 408 |

## Interpretation guide

- **Win-rate > 55% AND mean > 0 AND Sharpe > 0.15** → cell is robustly tradeable at that horizon.
- **Favorable/Adverse ratio > 1.2** → the price-path is asymmetric in our favor; supports use of trailing stops or wider TP.
- **Time-to-MFE < 10s** → exit fast; ride the edge to peak then bail before mean reversion.
- If FIFO Sharpe ≫ best time-exit Sharpe → the tp4sl3 levels are picking off the FAVORABLE slice; FIFO is over-optimistic about path realizability.
