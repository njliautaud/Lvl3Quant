# HC #450 R5 — Canonical FIFO Replay (definitive go/no-go)

Engine: `alpha_discovery.deep_models.fifo_market_replay.FIFOReplayEngine` (HC #74).
Cost: ES_RT_COMMISSION_TICKS = 0.376 (already netted in engine `pnl_ticks_net`).
Order type: passive limit at touch. TP = 1.0 tk. SL = 1.0 tk.
HC #428 R2 bounds: horizon = 1.0s, hold_s = 1.5, cancel_s = 1.0.
Regime labels: ES close-minus-open ticks; |Δ|<4 = flat, ≥+4 = green, ≤-4 = red (canonical per `output/regime_labels/oot_dates_regime.parquet`).

## Results (one row per cell, sorted by mean_tk_net desc)

| cell | ema_n | n | n_days | mean_tk_net | sharpe_ann | sortino_ann | pf | wr | day_positive_pct | day_conc | sharpe_green | sharpe_red | sharpe_flat | regime_delta_ratio | hc428_r1_pass | overall_pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B5_patchtst_long_ema40 | 40 | 5303 | 32 | -0.4087 | -18.8714 | -19.0191 | 0.4165 | 0.4763 | 0.0312 | 0.0939 | -22.0778 | -15.8183 | nan | 0.2835 | Y | N |
| B4_patchtst_long_ema20 | 20 | 5150 | 34 | -0.4177 | -18.2934 | -18.4011 | 0.4093 | 0.4728 | 0.0294 | 0.0897 | -19.5765 | -15.5552 | nan | 0.2054 | Y | N |
| B3_patchtst_long_ema10 | 10 | 4918 | 34 | -0.4187 | -18.1261 | -18.1261 | 0.4090 | 0.4738 | 0.0000 | 0.0896 | -19.5220 | -16.5424 | -11.6170 | 0.1526 | Y | N |
| B2_patchtst_long_ema4 | 4 | 4595 | 35 | -0.4380 | -18.4549 | -18.5739 | 0.3924 | 0.4640 | 0.0286 | 0.0863 | -21.2264 | -16.2254 | -11.4198 | 0.2356 | Y | N |
| A5_v2_short_ema40 | 40 | 1799 | 31 | -0.4413 | -18.5873 | -18.5873 | 0.3928 | 0.4664 | 0.0000 | 0.1035 | -22.6724 | -22.3364 | nan | 0.0148 | Y | N |
| A4_v2_short_ema20 | 20 | 2472 | 31 | -0.4482 | -24.0962 | -24.0962 | 0.3870 | 0.4616 | 0.0000 | 0.0822 | -33.7201 | -24.9477 | nan | 0.2602 | Y | N |
| B1_patchtst_long_raw | 1 | 4214 | 34 | -0.4610 | -19.7455 | -20.2312 | 0.3753 | 0.4518 | 0.0588 | 0.0827 | -22.6338 | -17.8935 | -15.9507 | 0.2094 | Y | N |
| A3_v2_short_ema10 | 10 | 3084 | 31 | -0.4840 | -27.3939 | -27.3939 | 0.3595 | 0.4426 | 0.0000 | 0.0869 | -42.7999 | -25.1638 | nan | 0.4121 | Y | N |
| A1_v2_short_raw | 1 | 4338 | 34 | -0.4911 | -24.7948 | -25.4450 | 0.3546 | 0.4396 | 0.0294 | 0.0655 | -31.0852 | -19.8878 | nan | 0.3602 | Y | N |
| A2_v2_short_ema4 | 4 | 3708 | 33 | -0.5015 | -24.0416 | -24.6332 | 0.3468 | 0.4358 | 0.0303 | 0.0729 | -32.2872 | -19.7186 | nan | 0.3893 | Y | N |

## Verdict

**NO CELL PASSES HC #428.** Closest-to-pass and the reason it failed:
- `B5_patchtst_long_ema40`: mean_tk_net=-0.4087, sharpe_ann=-18.87, day%=3%. FAILED: unprofitable after commission, only 32 OOT days (<40)
- `B4_patchtst_long_ema20`: mean_tk_net=-0.4177, sharpe_ann=-18.29, day%=3%. FAILED: unprofitable after commission, only 34 OOT days (<40)
- `B3_patchtst_long_ema10`: mean_tk_net=-0.4187, sharpe_ann=-18.13, day%=0%. FAILED: unprofitable after commission, only 34 OOT days (<40)
- `B2_patchtst_long_ema4`: mean_tk_net=-0.4380, sharpe_ann=-18.45, day%=3%. FAILED: unprofitable after commission, only 35 OOT days (<40)
- `A5_v2_short_ema40`: mean_tk_net=-0.4413, sharpe_ann=-18.59, day%=0%. FAILED: unprofitable after commission, only 31 OOT days (<40)

## Smoothing impact (EMA-N vs raw within each model)

**CNN-Mamba v2 short (raw mean_tk_net = -0.4911):**
- EMA-4: mean_tk_net = -0.5015 (delta = -0.0104) — smoothed beats raw? NO
- EMA-10: mean_tk_net = -0.4840 (delta = +0.0072) — smoothed beats raw? YES
- EMA-20: mean_tk_net = -0.4482 (delta = +0.0429) — smoothed beats raw? YES
- EMA-40: mean_tk_net = -0.4413 (delta = +0.0498) — smoothed beats raw? YES
**PatchTST long (raw mean_tk_net = -0.4610):**
- EMA-4: mean_tk_net = -0.4380 (delta = +0.0229) — smoothed beats raw? YES
- EMA-10: mean_tk_net = -0.4187 (delta = +0.0423) — smoothed beats raw? YES
- EMA-20: mean_tk_net = -0.4177 (delta = +0.0432) — smoothed beats raw? YES
- EMA-40: mean_tk_net = -0.4087 (delta = +0.0522) — smoothed beats raw? YES