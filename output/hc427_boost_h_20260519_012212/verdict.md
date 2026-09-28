# Boosting (h) verdict — volatility-regime gating v3.3 / v3.4.2

HC #427 R5 boosting technique (h). Generated: 20260519_012212

## Hypothesis

HC #411 verdict: signals are REGIME-FRAGILE — sub-window stability fails. Counter-measure: identify regime and choose model per-regime. Vol regime = the most natural axis (chop vs trend). Uses v3.4.2's own `pred_pred_realized_vol_30s_ticks` head (jointly trained alongside directional heads — same role as the LGBM vol model in execution).

## Regime definition

Per-sample vol proxy = `pred_pred_realized_vol_30s_ticks` (v3.4.2). Terciles computed PER-DAY (no look-ahead across days). Per-day cutpoints:

| day_idx | q33 | q67 | n |
|---:|---:|---:|---:|
| 0 | 7.562 | 8.062 | 49585 |
| 1 | 7.219 | 7.375 | 47600 |
| 2 | 7.094 | 7.344 | 28342 |
| 3 | 7.719 | 8.062 | 55459 |
| 4 | 7.469 | 8.000 | 60365 |

Total bucket counts (low/mid/high/missing): 84088/91171/66092/0

## Scheme results (v3.4.2 sweep top-20 basis, LOO across 5 OOT days)

| scheme | w33(low,mid,high) | n_robust / 20 | top-trial worst-day Sh | top-trial mean Sh | top fills | top PF |
|---|---|---:|---:|---:|---:|---:|
| baseline_v342 | (0.0, 0.0, 0.0) | 12 | 7.25 | 17.2 | 57 | 0.0 |
| baseline_uniform | (0.5, 0.5, 0.5) | 11 | 12.28 | 20.1 | 77 | 0.0 |
| v33_chop_v342_trend | (1.0, 0.5, 0.0) | 8 | 8.19 | 17.07 | 79 | 0.0 |
| v342_chop_v33_trend | (0.0, 0.5, 1.0) | 16 | 4.51 | 14.76 | 79 | 0.0 |
| v33_low_v342_else | (1.0, 0.0, 0.0) | 9 | 6.68 | 13.41 | 76 | 0.0 |
| v342_low_v33_else | (0.0, 1.0, 1.0) | 13 | 0.0 | 14.4 | 85 | 0.0 |
| blended_low_pure_hi | (0.5, 0.5, 0.0) | 10 | 4.12 | 14.42 | 73 | 0.0 |
| pure_hi_v33 | (0.0, 0.0, 1.0) | 13 | 0.0 | 11.78 | 75 | 0.0 |

## Verdict

✅ **POSITIVE** — scheme `v342_chop_v33_trend` produced **16/20 robust** vs baseline v3.4.2 SOLO 12/20 (Δ = +4).

Top-3 robust configs under winning scheme:
- trial=1110 | 5s/long/passive_at_touch_plus_2 | mean_Sh=14.76 worst_Sh=4.51 fills=79 pf=0.0 prof_days=4
- trial=1554 | 30s/short/passive_at_touch_plus_2 | mean_Sh=15.18 worst_Sh=0.0 fills=88 pf=0.0 prof_days=5
- trial=479 | 5s/long/passive_at_touch_plus_2 | mean_Sh=13.51 worst_Sh=4.07 fills=77 pf=0.0 prof_days=4

## Net-new robust trials (boost-h only)

Trials newly LOO-robust under any boost-h scheme but NOT in v3.4.2 SOLO (12) and NOT trial 2142 (boost-f net-new):
`[563, 1296, 2326]` (count = 3)

## HC #427 R5 boost-counter (updated)

- (a) mean ensemble v3.3+v3.4.2 — ✅ POSITIVE on v3.3 basis (+43% n_robust)
- (b) meta-LGBM gate — ❌ NEGATIVE
- (c) weighted-ensemble sweep — ⚪ NULL
- (f) confidence-conditional ensemble — ⚪ NULL (+1 net-new trial 2142)
- (h) volatility-regime gating — see verdict above (+3 net-new)
