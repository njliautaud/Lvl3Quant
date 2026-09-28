# HC #415 Multi-Gate Sweep Verdict

**NPZ**: `output/v342_ep1_eval/fold_00_ep1_oot_wrapped.npz`

**Label col**: `target_fifo_tp8sl5_net` (canonical FIFO realized net)

**N samples**: 241351  |  **N day chunks**: 5

**Total cells evaluated**: 476

**Cells passing HC #415 rule 2**: 0

**Wall time**: 34.0s

## Acceptance gate (HC #415 rule 2)
- per_day_pass_rate >= 0.8
- per_day_pass_rate_strict >= 0.7
- max_single_day_pnl_share <= 0.4
- n_fills >= 50, CI_low_95 > 0, day_conc <= 0.2

## Top 20 cells (ranked by Sortino, gate-pass first)

```
                               cell_id  n_fills  n_days_with_fills  realized_net_tk_per_fill  sortino_sqrtN  sharpe_sqrtN       pf        wr  per_day_pass_rate  per_day_pass_rate_strict  max_single_day_pnl_share  day_conc  ci_low_95_net  pass_hc415_rule2
   fifo_tp8sl5_short_top01_hconfluence       28                  4                  1.713286       9.339259      1.395189 1.722600 53.571429           0.750000                  0.750000                  0.347692  0.347692      -0.610375             False
     fifo_tp8sl5_short_top01_hconf+mfe       28                  4                  1.713286       9.339259      1.395189 1.722600 53.571429           0.750000                  0.750000                  0.347692  0.347692      -0.610375             False
  fifo_tp8sl5_short_top01_hconf+direct       28                  4                  1.713286       9.339259      1.395189 1.722600 53.571429           0.750000                  0.750000                  0.347692  0.347692      -0.610375             False
    fifo_tp8sl5_long_top01_hconfluence       26                  5                  1.508615       6.469304      2.242580 3.148084 61.538462           0.800000                  0.600000                  0.591450  0.591450       0.238904             False
      fifo_tp8sl5_long_top01_hconf+mfe       26                  5                  1.508615       6.469304      2.242580 3.148084 61.538462           0.800000                  0.600000                  0.591450  0.591450       0.238904             False
   fifo_tp8sl5_long_top01_hconf+uncert       26                  5                  1.508615       6.469304      2.242580 3.148084 61.538462           0.800000                  0.600000                  0.591450  0.591450       0.238904             False
      fifo_tp8sl5_short_top01_baseline       69                  5                  0.971826       5.706388      1.208506 1.348959 49.275362           0.600000                  0.600000                  0.569256  0.569256      -0.615130             False
      fifo_tp8sl5_short_top01_mfe_room       69                  5                  0.971826       5.706388      1.208506 1.348959 49.275362           0.600000                  0.600000                  0.569256  0.569256      -0.615130             False
  fifo_tp8sl5_short_top01_direct_agree       69                  5                  0.971826       5.706388      1.208506 1.348959 49.275362           0.600000                  0.600000                  0.569256  0.569256      -0.615130             False
fifo_tp8sl5_long_top01_uncertain_tight       34                  5                  1.241647       4.889752      2.069132 2.541405 61.764706           0.600000                  0.600000                  0.718896  0.718896       0.094588             False
        fifo_tp8sl5_long_top01_rev_low       25                  4                  1.064000       4.474083      1.716055 2.456737 60.000000           0.750000                  0.500000                  0.638787  0.638787      -0.176000             False
      fifo_tp8sl5_long_top01_hconf+rev       24                  4                  1.061500       4.373388      1.642128 2.395181 58.333333           0.750000                  0.500000                  0.665149  0.665149      -0.167667             False
  fifo_tp8sl5_long_top01_hconf+mfe+rev       24                  4                  1.061500       4.373388      1.642128 2.395181 58.333333           0.750000                  0.500000                  0.665149  0.665149      -0.167667             False
      fifo_tp8sl5_short_top05_baseline      296                  5                  0.314054       1.860982      0.763231 1.098485 45.945946           0.400000                  0.400000                  0.651747  0.651747      -0.519177             False
      fifo_tp8sl5_short_top05_mfe_room      296                  5                  0.314054       1.860982      0.763231 1.098485 45.945946           0.400000                  0.400000                  0.651747  0.651747      -0.519177             False
  fifo_tp8sl5_short_top05_direct_agree      296                  5                  0.314054       1.860982      0.763231 1.098485 45.945946           0.400000                  0.400000                  0.651747  0.651747      -0.519177             False
       fifo_tp8sl5_long_top01_baseline       61                  5                  0.621967       1.287065      0.982712 1.400312 55.737705           0.800000                  0.600000                  0.657765  0.657765      -0.568590             False
       fifo_tp8sl5_long_top01_mfe_room       61                  5                  0.621967       1.287065      0.982712 1.400312 55.737705           0.800000                  0.600000                  0.657765  0.657765      -0.568590             False
 log_ret_1s_long_top01_uncertain_tight       10                  3                  0.224000       0.517306      0.127360 1.102377 50.000000           0.666667                  0.666667                  0.493600  0.493600      -2.926000             False
    log_ret_1s_long_top01_hconf+uncert       10                  3                  0.224000       0.517306      0.127360 1.102377 50.000000           0.666667                  0.666667                  0.493600  0.493600      -2.926000             False
```

