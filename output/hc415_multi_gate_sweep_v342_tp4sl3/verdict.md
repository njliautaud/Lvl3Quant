# HC #415 Multi-Gate Sweep Verdict — label=tp4sl3

**NPZ**: `output/v342_fold_00_ep1_oot_inference_extended.npz` (v3.4.2 16d, 516,569 samples, n_valid=80840)

**Label col**: `target_fifo_tp4sl3_net` (CANONICAL FIFO realized net per HC #74/#397B)

**Total cells evaluated**: 469

**Cells passing HC #415 rule 2 (all-OOT-stability)**: 0

**Cells passing HC #408 honesty alone**: 0

**Cells with positive realized_net**: 60 / 469

## Key observations

- The realized label has **mean = -0.438 ticks** across all valid samples. The fixed TP/SL config in this label is a STRUCTURAL LOSER at the population level — meaning these particular (TP=4tk, SL=3tk) and (TP=8tk, SL=5tk) configs lose money in canonical FIFO replay regardless of model gating.

- Best signal+gate combos extract positive net/fill on small samples, but **CI_low_95 < 0** for ALL of them — sample sizes (6-86 fills) too small to claim statistical positive expectancy with 95% confidence.

- Best n_days_with_fills is 6-7 out of 16 — gates filter most days away entirely, violating HC #415 min_n_days_with_fills>=10.

- This is an HONEST NEGATIVE: with FIXED TP/SL labels, the v3.4.2 16d (10d-trained, book-head NOT YET ACTIVATED) cannot produce a regime-stable promotable cell.

## What this does NOT rule out

1. **v3.4.2 60d with book-head fixed** (training NOW on Neptune, NPZ lands ~22:00 ET 5/18). Book-head was inactive in this 16d run. 60d sliding window + active book features = different game.
2. **Per-cell MFE-derived TP/SL** (HC #411 mfe_at_confidence_matrix). The HC #413 backtester uses per-cell TP/SL from predicted MFE — not the fixed-grid tp4sl3/tp8sl5 used here. May change picture.
3. **Shorter holds + scalping exits**: model alpha decays fast (HC `signal characteristics`); the FIFO labels here assume full TP/SL horizons. Multi-output gating + tighter-hold scalping could differ.

## Top-10 by Sortino (n_fills>=50; for diagnostic)

```
                             cell_id  n_fills  n_days_with_fills  realized_net_tk_per_fill  sortino_sqrtN  per_day_pass_rate  per_day_pass_rate_strict  max_single_day_pnl_share  ci_low_95_net  pass_hc415_rule2
     log_ret_30s_long_top01_baseline       82                  6                  0.038634   1.449796e+06           0.500000                  0.333333                  0.341459      -0.729658             False
  log_ret_30s_long_top01_hconfluence       82                  6                  0.038634   1.449796e+06           0.500000                  0.333333                  0.341459      -0.729658             False
     log_ret_30s_long_top01_mfe_room       82                  6                  0.038634   1.449796e+06           0.500000                  0.333333                  0.341459      -0.729658             False
    log_ret_30s_long_top01_hconf+mfe       82                  6                  0.038634   1.449796e+06           0.500000                  0.333333                  0.341459      -0.729658             False
    fifo_tp8sl5_short_top01_baseline       86                  7                  0.042605   8.191146e+05           0.428571                  0.428571                  0.277568      -0.689953             False
    fifo_tp8sl5_short_top01_mfe_room       86                  7                  0.042605   8.191146e+05           0.428571                  0.428571                  0.277568      -0.689953             False
fifo_tp8sl5_short_top01_direct_agree       86                  7                  0.042605   8.191146e+05           0.428571                  0.428571                  0.277568      -0.689953             False
    fifo_tp8sl5_short_top05_baseline      420                  9                  0.007333   6.289133e+05           0.444444                  0.222222                  0.269117      -0.326000             False
    fifo_tp8sl5_short_top05_mfe_room      420                  9                  0.007333   6.289133e+05           0.444444                  0.222222                  0.269117      -0.326000             False
fifo_tp8sl5_short_top05_direct_agree      420                  9                  0.007333   6.289133e+05           0.444444                  0.222222                  0.269117      -0.326000             False
```

## Top-10 by realized_net (all, for diagnostic)

```
                             cell_id  n_fills  n_days_with_fills  realized_net_tk_per_fill  sortino_sqrtN  per_day_pass_rate  per_day_pass_rate_strict  max_single_day_pnl_share  ci_low_95_net  pass_hc415_rule2
      fifo_tp4sl3_long_top05_vol_mid        6                  3                  0.874000       1.427236           0.666667                  0.666667                  0.391699      -1.626000             False
      fifo_tp4sl3_long_top01_rev_low       20                  4                  0.474000       0.000000           0.500000                  0.500000                  0.409628      -0.926000             False
 fifo_tp8sl5_short_top01_hconfluence       62                  6                  0.349807       0.000000           0.333333                  0.333333                  0.455872      -0.440516             False
   fifo_tp8sl5_short_top01_hconf+mfe       62                  6                  0.349807       0.000000           0.333333                  0.333333                  0.455872      -0.440516             False
fifo_tp8sl5_short_top01_hconf+direct       62                  6                  0.349807       0.000000           0.333333                  0.333333                  0.455872      -0.440516             False
  log_ret_30s_long_top1_direct_agree       14                  4                  0.302572       1.198122           0.500000                  0.500000                  0.363846      -1.376000             False
  log_ret_30s_long_top1_hconf+direct       14                  4                  0.302572       1.198122           0.500000                  0.500000                  0.363846      -1.376000             False
  log_ret_1s_long_top01_direct_agree       21                  4                  0.076381       0.464357           0.500000                  0.500000                  0.387289      -1.256952             False
  log_ret_1s_long_top01_hconf+direct       21                  4                  0.076381       0.464357           0.500000                  0.500000                  0.387289      -1.256952             False
  log_ret_1s_long_top05_direct_agree       21                  4                  0.076381       0.464357           0.500000                  0.500000                  0.387289      -1.256952             False
```
