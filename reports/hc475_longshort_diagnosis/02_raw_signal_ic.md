# 02 — Raw Signal IC (Long-Side vs Short-Side)

**Attribution to this root cause (signal asymmetry): ~10%.** The model is mildly LONG-side competent and short-side near-zero — but it predicts MORE longs than shorts in raw output. This contradicts the trade-level bias entirely. Signal asymmetry is NOT the driver of the short-fill ratio.

## Punchline

At every trading horizon (1s/5s/10s) the model's LONG-side IC is meaningfully positive while the SHORT-side IC is near zero or slightly negative. The model also outputs ~2.2× more long predictions than short predictions at those horizons. By HC #475 R2's "both-sides competency" rule, **the short side is the weaker side**, not the stronger one. So if anything, the model has earned the right to trade longs more than shorts — yet at fill-time the ratio is reversed.

## Methodology

- Per-head IC = Pearson correlation between predicted value and realized target return, restricted to the relevant sub-sample.
- LONG-side IC: correlation computed only over events where `pred > 0`.
- SHORT-side IC: correlation computed only over events where `pred < 0`.
- All masked-as-valid events used. No threshold or confluence filtering at this stage.

## Overall + signed IC by horizon

| Horizon | Overall IC | IC long-side | IC short-side | n_long_pred | n_short_pred | L/S ratio |
|---|---|---|---|---|---|---|
| 1s  | +0.092 | **+0.0345** | +0.0057 | 618,599 | 281,855 | 2.19 |
| 5s  | +0.046 | **+0.0224** | −0.0022 | 617,791 | 281,458 | 2.19 |
| 10s | +0.037 | **+0.0217** | −0.0100 | 614,972 | 283,465 | 2.17 |
| 30s | −0.009 | +0.0051 | −0.0116 | 246,337 | 650,013 | 0.38 |

Three things stand out:

1. **At 1s, 5s, 10s — the head with proven trading edge — long-side IC dominates short-side IC by 4–6×.** Short-side IC at 5s and 10s is negative (anti-edge).
2. **The model emits 2.17× more long predictions than short predictions** at the 10s horizon (614,972 long vs 283,465 short). This is consistent with ES's structural long drift — the model has correctly learned that "up" is the more common direction.
3. **At 30s, the sign flips**: more short predictions (650,013) than long (246,337), and overall IC turns slightly negative — this head is the weakest and probably should not be a trade input. Note: this is `pred_log_ret_30s`, not the quantile heads.

## Magnitude (tail) asymmetry — where the short bias actually lives

The model's PREDICTED-MAGNITUDE distribution is severely negatively skewed at the trading horizons. p99 of the negative tail is **larger** than p99 of the positive tail:

| Horizon | pos_p99 magnitude | neg_p99 magnitude | ratio neg/pos |
|---|---|---|---|
| 1s  | 2.31e-1 | 2.79e-1 | **1.21** |
| 5s  | 2.14e-1 | 3.40e-1 | **1.59** |
| 10s | 2.64e-1 | 3.50e-1 | **1.33** |
| 30s | 4.53e+0 | 2.50e+0 | 0.55 |

**Translation**: when the model is wrong it's wrong bigger in the negative direction. When you select "top X% by absolute magnitude" (what every confluence config does), the negative tail dominates the selection.

Concretely — top-5% by |magnitude|:

| Head | share_short in top-5% by |mag| |
|---|---|
| pred_log_ret_1s  | 98.2% |
| pred_log_ret_5s  | 100.0% |
| pred_log_ret_10s | 98.7% |
| pred_log_ret_30s | 1.6% (inverted) |

This is the smoking gun for the FIFO bias. The trade triggers don't fire on "model is bullish" — they fire on "model has a large-magnitude prediction", and the large-magnitude tail is almost entirely shorts at 1s/5s/10s.

## Auxiliary heads tell the same story

- `pred_pred_mfe_30s_ticks` mean = **−1.54** (negative predicted upside excursion). Target mean = +7.52. Model is systematically underpredicting upside.
- `pred_pred_mae_30s_ticks` mean = +0.97 (positive predicted downside magnitude). Target mean = −7.58.
- The model has compressed both excursion predictions toward "small upside, modest downside" — and the relative-magnitude bias amplifies in the tails.

## Per-day IC at 10s (decay check, HC #472)

IC_long > IC_short on **14 of 16 days**. Only 20260301 (1,518 events — pre-market light day) and 20260306 (red day) show short-side IC matching long-side IC. No decay trend toward short-side competency over the window. Full per-day breakdown in `per_day_ic_10s.parquet`.

## HC #475 R2 competency check

Rule: long-side IC and short-side IC must both be > 0.5× the better-side IC. At the trading horizon (10s):
- Better side: |IC_long| = 0.0217
- Worse side: |IC_short| = 0.0100
- Ratio: 0.46 — **FAILS the 0.5× competency bar by a hair on the short side**, and the short side has the WRONG SIGN (negative) which is worse than the magnitude check suggests.

So per HC #475 R2 the v3.4.2 multi-head model is **REJECTED for production "trade both sides" use until retrained**. A short-only deploy is only justified if the short-side IC is the strong side — here it's the weak side, so even short-only is dubious without further analysis.

## Conclusion

Raw signal IC says the model is BETTER at long-side prediction than short-side prediction. The short-fill bias is NOT explained by "the model found a real short edge". The short bias comes from how the prediction-MAGNITUDE distribution is shaped (negatively skewed tails) interacting with threshold policies that select on |magnitude|. That's a **threshold-attribution problem** (next report), not a signal-edge problem.
