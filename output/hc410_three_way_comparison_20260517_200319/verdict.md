# HC #410 — Three-way comparison (v2 vs v3.3 vs v3.4.2)

Generated: 2026-05-17T20:03:21.688173
Output dir: `/home/jupiter/Lvl3Quant/output/hc410_three_way_comparison_20260517_200319`

## Headline
- **Recommended model: v3.4.2**
- Promoted cells per model (n>=50, day_conc<=0.20, CI_low_95(net)>0):
  - v2: 0 cells
  - v3.3: 19 cells
  - v3.4.2: 27 cells
- Avg net_tk_per_fill across PROMOTED cells:
  - v3.3: 0.4499 tk/fill
  - v3.4.2: 0.5591 tk/fill
- Per-cell wins (v3.3 vs v3.4.2 on net_tk_per_fill across all 16 cells = 4h x 2side x 4tier - lookups):
  - v3.3 wins: 4
  - v3.4.2 wins: 28

## Top 3 promoted cells per model
### v2
- (no promoted cells for v2)

### v3.3
- 30s/long/Top0.5: n=3357, net=1.125 tk, WR=54.1%, day_conc=0.15, Sharpe/fill=0.106
- 30s/long/Top1: n=6715, net=0.887 tk, WR=52.9%, day_conc=0.17, Sharpe/fill=0.084
- 10s/long/Top0.5: n=3362, net=0.862 tk, WR=56.9%, day_conc=0.13, Sharpe/fill=0.142

### v3.4.2
- 30s/long/Top0.5: n=3771, net=1.762 tk, WR=55.1%, day_conc=0.13, Sharpe/fill=0.130
- 30s/long/Top1: n=7542, net=1.353 tk, WR=53.8%, day_conc=0.13, Sharpe/fill=0.112
- 10s/long/Top0.5: n=3780, net=0.989 tk, WR=55.1%, day_conc=0.13, Sharpe/fill=0.135

## v2 IC (continuous pred vs discrete label {0,0.5,1})
- IC_1s = 0.0133
- IC_5s = -0.0020
- IC_10s = 0.0168

## Per-cell winners (v3.3 vs v3.4.2)
| horizon | side | conf_tier | winner | net (tk) | promoted | n |
|---|---|---|---|---|---|---|
| 1s | long | Top0.5 | **v3.4.2** | 0.6789 | True | 3789 |
| 1s | long | Top1 | **v3.4.2** | 0.5265 | True | 7579 |
| 1s | long | Top5 | **v3.4.2** | 0.3274 | True | 37896 |
| 1s | long | Top10 | **v3.4.2** | 0.2349 | True | 75792 |
| 1s | short | Top0.5 | **v3.4.2** | 0.8006 | True | 3789 |
| 1s | short | Top1 | **v3.4.2** | 0.5103 | True | 7579 |
| 1s | short | Top5 | **v3.4.2** | 0.2967 | True | 37896 |
| 1s | short | Top10 | **v3.4.2** | 0.2342 | True | 75792 |
| 5s | long | Top0.5 | **v3.4.2** | 0.8093 | True | 3784 |
| 5s | long | Top1 | **v3.4.2** | 0.6405 | True | 7568 |
| 5s | long | Top5 | **v3.4.2** | 0.3962 | True | 37842 |
| 5s | long | Top10 | **v3.4.2** | 0.2689 | True | 75684 |
| 5s | short | Top0.5 | **v3.4.2** | 0.7704 | True | 3784 |
| 5s | short | Top1 | **v3.4.2** | 0.5297 | True | 7568 |
| 5s | short | Top5 | **v3.4.2** | 0.2228 | True | 37842 |
| 5s | short | Top10 | **v3.4.2** | 0.1267 | True | 75684 |
| 10s | long | Top0.5 | **v3.4.2** | 0.9888 | True | 3780 |
| 10s | long | Top1 | **v3.4.2** | 0.8100 | True | 7561 |
| 10s | long | Top5 | **v3.4.2** | 0.4542 | True | 37806 |
| 10s | long | Top10 | **v3.3** | 0.3130 | True | 67240 |
| 10s | short | Top0.5 | **v3.4.2** | 0.3835 | True | 3780 |
| 10s | short | Top1 | **v3.4.2** | 0.3385 | True | 7561 |
| 10s | short | Top5 | **v3.4.2** | 0.1411 | True | 37806 |
| 10s | short | Top10 | **v3.3** | 0.0266 | False | 67240 |
| 30s | long | Top0.5 | **v3.4.2** | 1.7619 | True | 3771 |
| 30s | long | Top1 | **v3.4.2** | 1.3526 | True | 7542 |
| 30s | long | Top5 | **v3.4.2** | 0.7151 | True | 37714 |
| 30s | long | Top10 | **v3.4.2** | 0.4658 | True | 75429 |
| 30s | short | Top0.5 | **v3.3** | 0.1003 | False | 3357 |
| 30s | short | Top1 | **v3.3** | 0.0413 | False | 6715 |
| 30s | short | Top5 | **v3.4.2** | -0.0149 | False | 37714 |
| 30s | short | Top10 | **v3.4.2** | -0.1432 | False | 75429 |

## Concrete next step
- **Resume v3.4.2 training to convergence**, then wire the top-3 promoted cells into Razer paper trader.

## Limitations & caveats
- **v2 label encoding**: labels are discrete {0, 0.5, 1.0} (down / unclear / up), NOT continuous tick returns. Therefore:
  - `mfe_mean_tk`, `net_tk_per_fill`, `sharpe_per_fill`, `ci_low_95_net` are **NaN** for v2.
  - `wr_pct` for v2 is the directional hit-rate on decisive samples (excluding the 0.5 'unclear' bin) — NOT comparable apples-to-apples to v3.3/v3.4.2 wr_pct (which is sign of horizon-end log-return).
  - `promote` is always False for v2 (no tick-level net available).
- **v3.4.2 dates**: 5d NPZ covers 02-23..02-27, 11d_ext covers 03-01..03-15 (11 RTH days). Date attribution uses uniform chunking per the HC #408 convention since `oot_dates` is not stored in the v3.4.2 NPZs.
- **MAE**: only available at the 30s horizon (from `target_pred_mae_30s_ticks`); other horizons report NaN.
- **Cost model**: net = realized - 0.376 ticks (commission only, HC #405 fill-price framing; passive entry assumed). Market-cross would subtract additional 1.0 tick.
- **OOT-period overlap**: v3.3 = 15 RTH days in March 2026; v3.4.2 = 16 RTH days (Feb 23 - Mar 15). v2 = 96 days. Periods are NOT identical, so cross-model numbers are indicative not strictly head-to-head.
- **Confidence ranking**: top-k by |signed pred|, NOT by predicted realized vol. This is the simplest comparable ranking across all three models.