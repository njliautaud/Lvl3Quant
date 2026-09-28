# HC #415 Multi-Gate Sweep Verdict

**NPZ**: `output/v342_ep1_eval/fold_00_ep1_oot_wrapped.npz`

**Label col**: `target_fifo_tp4sl3_net` (canonical FIFO realized net)

**N samples**: 241351  |  **N day chunks**: 5

**Total cells evaluated**: 476

**Cells passing HC #415 rule 2**: 0

**Wall time**: 35.1s

## Acceptance gate (HC #415 rule 2)
- per_day_pass_rate >= 0.8
- per_day_pass_rate_strict >= 0.7
- max_single_day_pnl_share <= 0.4
- n_fills >= 50, CI_low_95 > 0, day_conc <= 0.2

## Top 20 cells (ranked by Sortino, gate-pass first)

```
                              cell_id  n_fills  n_days_with_fills  realized_net_tk_per_fill  sortino_sqrtN  sharpe_sqrtN       pf        wr  per_day_pass_rate  per_day_pass_rate_strict  max_single_day_pnl_share  day_conc  ci_low_95_net  pass_hc415_rule2
        log_ret_5s_long_top01_vol_mid        7                  3                  0.624000   5.653897e+06      0.441235 1.431280 57.142857           0.666667                  0.666667                  0.370504  0.370504      -2.376000             False
      fifo_tp4sl3_short_top01_vol_mid        7                  4                  0.624000   5.653897e+06      0.441235 1.431280 57.142857           0.500000                  0.500000                  0.419427  0.419427      -2.376000             False
       fifo_tp4sl3_short_top1_rev_low        7                  4                  0.624000   5.653897e+06      0.441235 1.431280 57.142857           0.500000                  0.500000                  0.417127  0.417127      -2.376000             False
  fifo_tp8sl5_short_top01_hconfluence       28                  4                  0.124000   2.651969e+06      0.184092 1.073460 50.000000           0.500000                  0.500000                  0.357820  0.357820      -1.126000             False
    fifo_tp8sl5_short_top01_hconf+mfe       28                  4                  0.124000   2.651969e+06      0.184092 1.073460 50.000000           0.500000                  0.500000                  0.357820  0.357820      -1.126000             False
 fifo_tp8sl5_short_top01_hconf+direct       28                  4                  0.124000   2.651969e+06      0.184092 1.073460 50.000000           0.500000                  0.500000                  0.357820  0.357820      -1.126000             False
        fifo_tp4sl3_long_top1_vol_mid        9                  4                  0.401778   9.642669e-01      0.367505 1.328608 55.555556           0.750000                  0.750000                  0.404197  0.404197      -1.653778             False
  log_ret_1s_long_top01_hconf+mfe+rev        9                  3                 -0.542667   0.000000e+00     -0.472430 0.710664 44.444444           0.333333                  0.333333                  0.469541  0.469541      -2.598222             False
        log_ret_1s_long_top05_vol_mid       57                  5                 -0.893544   0.000000e+00     -2.034057 0.580931 36.842105           0.000000                  0.000000                  0.245661  0.245661      -1.709333             False
       log_ret_1s_short_top05_rev_low        8                  4                 -0.751000   0.000000e+00     -0.586321 0.644076 37.500000           0.500000                  0.250000                  0.490983  0.490983      -2.501000             False
     log_ret_1s_short_top05_hconf+rev        8                  4                 -0.751000   0.000000e+00     -0.586321 0.644076 37.500000           0.500000                  0.250000                  0.490983  0.490983      -2.501000             False
 log_ret_1s_short_top05_hconf+mfe+rev        7                  3                 -1.376000   0.000000e+00     -1.065845 0.429384 28.571429           0.333333                  0.000000                  0.666667  0.666667      -3.376000             False
log_ret_5s_long_top01_uncertain_tight        5                  3                 -2.476000   0.000000e+00     -2.751111 0.083235 20.000000           0.333333                  0.333333                  0.461581  0.461581      -3.376000             False
        log_ret_5s_long_top01_rev_low        5                  3                 -2.476000   0.000000e+00     -2.751111 0.083235 20.000000           0.333333                  0.333333                  0.461581  0.461581      -3.376000             False
      log_ret_5s_long_top01_hconf+rev        5                  3                 -2.476000   0.000000e+00     -2.751111 0.083235 20.000000           0.333333                  0.333333                  0.461581  0.461581      -3.376000             False
   log_ret_5s_long_top01_hconf+uncert        5                  3                 -2.476000   0.000000e+00     -2.751111 0.083235 20.000000           0.333333                  0.333333                  0.461581  0.461581      -3.376000             False
  log_ret_5s_long_top01_hconf+mfe+rev        5                  3                 -2.476000   0.000000e+00     -2.751111 0.083235 20.000000           0.333333                  0.333333                  0.461581  0.461581      -3.376000             False
        log_ret_5s_long_top05_vol_mid       30                  4                 -0.992666   0.000000e+00     -1.671498 0.535732 36.666667           0.250000                  0.250000                  0.511376  0.511376      -2.076000             False
       log_ret_10s_long_top01_vol_mid        8                  3                  0.124000   0.000000e+00      0.093735 1.073460 50.000000           0.666667                  0.333333                  0.532025  0.532025      -2.501000             False
        log_ret_10s_long_top1_vol_mid       51                  4                 -0.738745   0.000000e+00     -1.571457 0.640002 39.215686           0.250000                  0.250000                  0.404129  0.404129      -1.611539             False
```

