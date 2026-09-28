# HC #444 R2 — Per-Day Cross-Model Agreement Filter

Cross-model filter: keep v2 fills only on days where v3.4.2 AND v3.3 both lean SHORT (or both LONG, depending on config side) at the 1s horizon.

- **meansign** filter: both model's day-mean pred_log_ret_1s has matching sign.
- **majority** filter: both models have >50% of predictions in matching direction.

Cross-model day-counts: short-meansign=6, short-majority=3, long-meansign=14, long-majority=9 out of 32 common dates.

## Results (sorted by meansign net tk/fill)

| config | side | base n / mean_tk_net / day% | meansign n / mean_tk_net / pf / wr / day% / sharpe | majority n / mean_tk_net / pf / wr / day% / sharpe |
|---|---|---|---|---|
| v342_lshort_5s_ensemble_50_50 | long | 3982 / -0.490 / 6% | 2753 / -0.494 / 0.56 / 48.6% / 8% / -14.56 | 1526 / -0.492 / 0.57 / 48.9% / 0% / -10.61 |
| v2_short_1s_top0.5_baseline | short | 2676 / -0.549 / 9% | 377 / -0.496 / 0.72 / 50.4% / 0% / -3.14 | 149 / -0.547 / 0.63 / 47.7% / 0% / -2.82 |
| v342_long_1s_top0.5_tp1.0_sl0.5_h1.5_c1.0_passive_at_touch | long | 1728 / -0.529 / 0% | 1268 / -0.523 / 0.67 / 48.7% / 0% / -6.98 | 755 / -0.517 / 0.68 / 49.0% / 0% / -5.18 |
| v342_long_5s_top0.5_for_ensemble | long | 2645 / -0.532 / 6% | 1741 / -0.528 / 0.66 / 48.2% / 8% / -8.47 | 985 / -0.508 / 0.70 / 49.5% / 0% / -5.55 |
| hc443_h5_hold5_top0p5_short | short | 2625 / -0.667 / 0% | 444 / -0.550 / 0.74 / 22.5% / 0% / -2.71 | 213 / -0.538 / 0.76 / 23.9% / 0% / -1.76 |
| hc443_band_top1_short | short | 4189 / -0.667 / 6% | 906 / -0.559 / 0.70 / 28.5% / 17% / -4.56 | 447 / -0.473 / 0.83 / 32.0% / 33% / -1.66 |
| hc443_multih_confluence_top30_v3 | short | 99154 / -0.622 / 3% | 16853 / -0.599 / 0.64 / 27.5% / 0% / -24.75 | 4885 / -0.591 / 0.65 / 28.4% / 0% / -13.00 |
| hc443_chase_sl05_tp3_h15 | short | 2431 / -0.702 / 0% | 434 / -0.602 / 0.64 / 27.2% / 0% / -4.02 | 200 / -0.559 / 0.69 / 30.5% / 0% / -2.24 |
| hc442_v2_canon_c1 | short | 2431 / -0.702 / 0% | 434 / -0.602 / 0.64 / 27.2% / 0% / -4.02 | 200 / -0.559 / 0.69 / 30.5% / 0% / -2.24 |
| v2_short_1s_top0.5_realtime_sl_HC437_47day_bracketparams | short | 3455 / -0.650 / 0% | 623 / -0.604 / 0.54 / 47.0% / 0% / -7.49 | 297 / -0.578 / 0.58 / 48.8% / 0% / -4.56 |
| hc443_band_top5_short | short | 19774 / -0.643 / 8% | 4191 / -0.612 / 0.62 / 27.0% / 0% / -13.16 | 1643 / -0.575 / 0.67 / 28.7% / 0% / -6.86 |
| hc443_multih_confluence_top10_v3b | short | 27568 / -0.636 / 8% | 5595 / -0.625 / 0.60 / 26.5% / 0% / -16.14 | 2088 / -0.606 / 0.63 / 27.5% / 0% / -9.07 |
| hc443_multih_1s5s_top10_v3 | short | 29995 / -0.637 / 5% | 6103 / -0.630 / 0.60 / 26.2% / 0% / -17.27 | 2220 / -0.609 / 0.62 / 27.5% / 0% / -9.54 |
| hc442_v2_canon_c10 | short | 3455 / -0.689 / 0% | 623 / -0.634 / 0.59 / 26.3% / 0% / -5.68 | 297 / -0.609 / 0.62 / 27.6% / 0% / -3.57 |
| hc443_long_top05_baseline | long | 2933 / -0.744 / 3% | 432 / -0.833 / 0.37 / 16.2% / 0% / -9.23 | 290 / -0.845 / 0.35 / 16.2% / 0% / -8.00 |
| hc443_wider_sl2_tp3 | short | 2431 / -1.021 / 0% | 434 / -0.849 / 0.55 / 42.2% / 0% / -5.39 | 200 / -0.772 / 0.59 / 44.0% / 0% / -3.16 |
| hc443_tight_sl1_tp2_h05 | short | 2076 / -1.013 / 0% | 356 / -0.881 / 0.38 / 34.0% / 0% / -8.54 | 156 / -0.752 / 0.47 / 41.7% / 0% / -4.23 |
| hc443_market_entry_tp3 | short | 4299 / -1.052 / 14% | 765 / -1.036 / 0.16 / 9.7% / 0% / -25.33 | 357 / -0.977 / 0.21 / 11.5% / 0% / -14.19 |
| hc443_upside_sl3_tp8_h10 | short | 2431 / -1.101 / 6% | 434 / -1.084 / 0.65 / 36.4% / 0% / -4.04 | 200 / -0.837 / 0.76 / 37.5% / 0% / -1.72 |
| hc442_primary_cancel10s_match | short | 2102 / -0.561 / 0% | 0 / nan / nan / nan% / nan% / nan | 0 / nan / nan / nan% / nan% / nan |
| hc442_primary_canonical_revalidation | short | 1613 / -0.586 / 0% | 0 / nan / nan / nan% / nan% / nan | 0 / nan / nan / nan% / nan% / nan |
| v2_short_1s_top0.5_baseline_HC437_REPRO_3day_HC413TP | short | 160 / -0.895 / 0% | 0 / nan / nan / nan% / nan% / nan | 0 / nan / nan / nan% / nan% / nan |
| v2_short_1s_top0.5_baseline_HC437_REPRO_3day | short | 160 / -0.646 / 0% | 0 / nan / nan / nan% / nan% / nan | 0 / nan / nan / nan% / nan% / nan |
| v342_short_10s_top0.5_t2831_R2fix | short | 1427 / -0.602 / 27% | 0 / nan / nan / nan% / nan% / nan | 0 / nan / nan / nan% / nan% / nan |
| v342_short_5s_top0.5_t1422_R2fix | short | 1337 / -0.748 / 7% | 0 / nan / nan / nan% / nan% / nan | 0 / nan / nan / nan% / nan% / nan |

## Verdict

**NO CONFIG SURVIVES PER-DAY CROSS-MODEL FILTER**. Per-signal join would not rescue any config (strictly stricter filter). Ship HC #444 R4 fallback.