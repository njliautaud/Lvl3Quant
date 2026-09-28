# HC #413 — TP/SL Scalping Backtest Verdict (CNN-Mamba v2 full-OOT, v2-NATIVE MFE)

**NPZ**: `/home/jupiter/Lvl3Quant/output/hc417_v2_full_oot_wrapped_for_hc413.npz` (n=1,464,715 samples, 36 OOT dates)
**Model tag**: v2 (CNN-Mamba v2, fold_10_best.pt)
**MFE config**: `hc417_v2_native_mfe_matrix.csv` — V2-NATIVE (computed from v2 predictions, replaces v3.4.2 borrowed config)
**Order type**: passive_at_touch (cost = 0.376 ticks)
**Canonical FIFO market replay** (HC #74/#377/#397B)

## HC #408-passing cells (n_fills>=50, day_conc<=0.20, CI95lo>0, net>0)

| cell_id | n_fills | net/fill (tk) | Sharpe√N | Sortino√N | PF | WR% | day_conc | CI95_lo (tk) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v2_1s_short_top05 | 639 | +0.274 | 12.77 | 707.57 | 2.92 | 84.8 | 0.132 | +0.232 |
| v2_1s_short_top1 | 1290 | +0.223 | 15.69 | 11334.59 | 2.58 | 84.0 | 0.109 | +0.195 |
| v2_5s_short_top05 | 698 | +0.177 | 5.96 | 28.02 | 1.71 | 84.2 | 0.152 | +0.119 |
| v2_5s_short_top1 | 1433 | +0.146 | 7.54 | 41.12 | 1.62 | 84.1 | 0.136 | +0.109 |
| v2_10s_short_top1 | 1453 | +0.079 | 3.71 | 17.58 | 1.28 | 83.3 | 0.150 | +0.037 |
| v2_1s_short_top5 | 6410 | +0.022 | 3.38 | nan | 1.10 | 74.9 | 0.098 | +0.009 |
| v2_1s_short_top10 | 12919 | +0.013 | 3.06 | 3.44 | 1.07 | 62.3 | 0.117 | +0.005 |

## HC #415 rule 2 (per_day_pass_rate>=0.80, n_days_with_fills>=10)

**Cells passing HC #415 rule 2: 5 / 7 HC408+net>0 candidates**

| cell_id | per_day_pass_rate | n_days_with_fills | day_conc_abs | pass_hc415_rule2 |
|---|---:|---:|---:|:-:|
| v2_1s_short_top05 | 1.000 | 25 | 0.131 | YES |
| v2_1s_short_top1 | 1.000 | 25 | 0.109 | YES |
| v2_5s_short_top1 | 0.962 | 26 | 0.137 | YES |
| v2_5s_short_top05 | 0.880 | 25 | 0.151 | YES |
| v2_10s_short_top1 | 0.800 | 25 | 0.143 | YES |
| v2_1s_short_top10 | 0.655 | 29 | 0.122 | no |
| v2_1s_short_top5 | 0.621 | 29 | 0.102 | no |

## Bottom line

- HC #408 passing AND net>0: **7** cells
- HC #415 rule 2 passing: **5** cells
- Best: `v2_1s_short_top05` net/fill=+0.274 tk, n=639, Sharpe√N=12.77, CI95lo=+0.232

## Caveats

- Uses v2-NATIVE MFE matrix (Caveat A of HC #417 falsification resolved).
- Still no v3.4.2-style 30s head in v2 NPZ; only 1s/5s/10s horizons are evaluated.
- MAE proxy at 1s/5s/10s = mean magnitude of negative-only signed realized move (same proxy as HC #411).