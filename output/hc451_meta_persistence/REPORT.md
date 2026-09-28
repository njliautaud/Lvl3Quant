# HC #451 — Meta-Persistence Filter FIFO Eval Report

Engine: canonical `FIFOReplayEngine` (HC #74). Cost: ES_RT_COMMISSION_TICKS = 0.376 (netted in `pnl_ticks_net`).

## Meta-classifier OOT (validation) summary

- AUC_val = **0.5547**  (baseline 0.500)
- Base positive rate (val) = 0.177
- LightGBM best_iter = 75
- Train days = 25 (20260223..20260406)
- Val days   = 7 (20260407..20260414)
- Top features by gain: tod_frac (38405), abs_pred_1s (33562), pct_rank_abs_1s_within_day (11034), pred_5s_minus_10s (10813), mfe_minus_mae_30s (9132)

Meta-vs-raw-confidence overlap (val): top_1pct: Jaccard=0.0003, meta∈conf=0.1%, top_2pct: Jaccard=0.0062, meta∈conf=1.2%, top_5pct: Jaccard=0.0439, meta∈conf=8.4%

## Canonical FIFO replay results

| cell | side | conf_top_pct | meta_top_pct | geom | r2_compliant | n_signals_pre_fifo | n | n_days | fills_per_day | mean_tk_net | sharpe_ann | pf | wr | day_positive_pct | day_conc | regime_delta_norm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B_short_conf0p5_meta0p3_wide | short | 0.0050 | 0.3000 | wide_h10s | N | 1314 | 1176 | 31 | 37.9355 | -0.1719 | -2.4420 | 0.9016 | 0.4668 | 0.4516 | 0.1040 | 0.1509 |
| C_short_conf1_meta0p2_mid | short | 0.0100 | 0.2000 | mid_h5s | N | 1655 | 1393 | 30 | 46.4333 | -0.3031 | -8.1899 | 0.7704 | 0.4221 | 0.3333 | 0.1091 | 0.2150 |
| B_short_conf0p5_meta0p3_mid | short | 0.0050 | 0.3000 | mid_h5s | N | 1314 | 1117 | 31 | 36.0323 | -0.3039 | -8.4099 | 0.7722 | 0.4226 | 0.2581 | 0.1055 | 0.2187 |
| C_short_conf1_meta0p2_strict | short | 0.0100 | 0.2000 | strict_h1s | Y | 1655 | 1113 | 28 | 39.7500 | -0.3540 | -20.7986 | 0.4729 | 0.5103 | 0.0000 | 0.1328 | 0.0381 |
| B_short_conf0p5_meta0p3_strict | short | 0.0050 | 0.3000 | strict_h1s | Y | 1314 | 902 | 28 | 32.2143 | -0.3633 | -18.9589 | 0.4644 | 0.5055 | 0.1071 | 0.1014 | 0.1512 |
| F_long_conf0p5_meta0p3_strict | long | 0.0050 | 0.3000 | strict_h1s | Y | 1490 | 878 | 30 | 29.2667 | -0.3680 | -21.7445 | 0.4589 | 0.5011 | 0.0333 | 0.0800 | 0.1918 |
| E_short_metaonly0p05_strict | short | nan | 0.0500 | strict_h1s | Y | 30349 | 20829 | 32 | 650.9062 | -0.3859 | -30.3055 | 0.4435 | 0.4944 | 0.0000 | 0.0630 | 0.2860 |
| D_short_conf2_meta0p1_strict | short | 0.0200 | 0.1000 | strict_h1s | Y | 1955 | 1300 | 29 | 44.8276 | -0.3914 | -14.5086 | 0.4380 | 0.4908 | 0.0000 | 0.1621 | 0.0166 |
| A_baseline_short_strict | short | 0.0050 | nan | strict_h1s | Y | 4415 | 2931 | 30 | 97.7000 | -0.4364 | -23.6870 | 0.3997 | 0.4684 | 0.0333 | 0.0943 | 0.1271 |

## Verdict

**NO FRIDAY CANDIDATE**: no config achieves net > 0, day_pct >= 60%, regime_delta_norm <= 0.50 simultaneously.

Meta-filter alone is **insufficient** to flip the v3.4.2 short-side FIFO baseline to profitable. Per HC #451 R5 deeper path, this points to needing a full multi-head retrain — the predictor's confidence does not concentrate on the 78.8%-persistent subset of events, and a meta-classifier built ON TOP OF the existing predictor cannot recover that information.
