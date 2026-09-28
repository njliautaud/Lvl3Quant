# HC #411 — Regime-agnostic signal verdict

Generated: 2026-05-17T21:52:13.393370
Output dir: `/home/jupiter/Lvl3Quant/output/hc411_regime_agnostic_20260517_215211`

Models analyzed: v3.3 (15d OOT), v3.4.2 (16d OOT)
Sub-windows per model: 4 contiguous date chunks
Honesty gate per window: n_fills>=50 AND day_conc<=0.20 AND CI_low_95(net)>0
Cost: net = realized - 0.376 ticks (HC #405 commission-only)

## Sub-window date bounds
### v3.3
- window 0: dates [20260301 ... 20260304] (indices 0..3, n_dates=4)
- window 1: dates [20260305 ... 20260310] (indices 4..7, n_dates=4)
- window 2: dates [20260311 ... 20260316] (indices 8..11, n_dates=4)
- window 3: dates [20260317 ... 20260319] (indices 12..14, n_dates=3)

### v3.4.2
- window 0: dates [20260223 ... 20260226] (indices 0..3, n_dates=4)
- window 1: dates [20260227 ... 20260303] (indices 4..7, n_dates=4)
- window 2: dates [20260304 ... 20260309] (indices 8..11, n_dates=4)
- window 3: dates [20260310 ... 20260315] (indices 12..15, n_dates=4)

## MFE / MAE table at confidence (units: ticks, full-OOT aggregate)

MFE = mean favorable-direction realized horizon-end move (signed by side).
MAE = at 30s: mean of `target_pred_mae_30s_ticks` (true realized adverse).
       at 1s/5s/10s: proxy = mean magnitude of negative-only realized moves (no intra-horizon MAE available in NPZ).

### v3.3

| horizon | side | Top0.5 MFE | MAE | net | Top1 MFE | MAE | net | Top5 MFE | MAE | net | Top10 MFE | MAE | net |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1s | long | 0.777 | 0.317 | 0.401 | 0.678 | 0.375 | 0.302 | 0.638 | 0.426 | 0.262 | 0.592 | 0.438 | 0.216 |
| 1s | short | 0.273 | 0.518 | -0.103 | 0.294 | 0.534 | -0.082 | 0.425 | 0.516 | 0.049 | 0.468 | 0.488 | 0.091 |
| 5s | long | 1.082 | 0.919 | 0.706 | 0.878 | 1.010 | 0.502 | 0.739 | 1.193 | 0.363 | 0.643 | 1.239 | 0.267 |
| 5s | short | 0.317 | 1.114 | -0.059 | 0.276 | 1.188 | -0.100 | 0.443 | 1.203 | 0.067 | 0.465 | 1.224 | 0.089 |
| 10s | long | 1.238 | 1.521 | 0.862 | 1.085 | 1.537 | 0.709 | 0.808 | 1.809 | 0.432 | 0.689 | 1.881 | 0.313 |
| 10s | short | 0.389 | 1.520 | 0.013 | 0.273 | 1.678 | -0.103 | 0.377 | 1.727 | 0.001 | 0.403 | 1.773 | 0.027 |
| 30s | long | 1.501 | -4.972 | 1.125 | 1.263 | -4.966 | 0.887 | 0.928 | -4.919 | 0.552 | 0.777 | -4.764 | 0.401 |
| 30s | short | 0.476 | -2.879 | 0.100 | 0.417 | -2.855 | 0.041 | 0.228 | -3.513 | -0.148 | 0.177 | -3.825 | -0.199 |

### v3.4.2

| horizon | side | Top0.5 MFE | MAE | net | Top1 MFE | MAE | net | Top5 MFE | MAE | net | Top10 MFE | MAE | net |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1s | long | 1.055 | 0.420 | 0.679 | 0.902 | 0.364 | 0.526 | 0.703 | 0.354 | 0.327 | 0.611 | 0.387 | 0.235 |
| 1s | short | 1.177 | 0.486 | 0.801 | 0.886 | 0.491 | 0.510 | 0.673 | 0.406 | 0.297 | 0.610 | 0.403 | 0.234 |
| 5s | long | 1.185 | 1.106 | 0.809 | 1.016 | 1.113 | 0.640 | 0.772 | 1.174 | 0.396 | 0.645 | 1.221 | 0.269 |
| 5s | short | 1.146 | 1.262 | 0.770 | 0.906 | 1.171 | 0.530 | 0.599 | 1.133 | 0.223 | 0.503 | 1.177 | 0.127 |
| 10s | long | 1.365 | 1.939 | 0.989 | 1.186 | 1.844 | 0.810 | 0.830 | 1.833 | 0.454 | 0.687 | 1.894 | 0.311 |
| 10s | short | 0.759 | 1.926 | 0.384 | 0.715 | 1.708 | 0.339 | 0.517 | 1.682 | 0.141 | 0.401 | 1.758 | 0.025 |
| 30s | long | 2.138 | -5.906 | 1.762 | 1.729 | -4.859 | 1.353 | 1.091 | -3.530 | 0.715 | 0.842 | -3.319 | 0.466 |
| 30s | short | 0.212 | -5.079 | -0.164 | 0.415 | -4.914 | 0.039 | 0.361 | -4.843 | -0.015 | 0.233 | -4.910 | -0.143 |

## Regime-agnostic winners

Bidirectional regime-stable cells (BOTH long AND short pass all 4 sub-windows):
- **v3.3: 0 bidirectional regime-stable cells**
- **v3.4.2: 0 bidirectional regime-stable cells**

Unidirectional regime-stable cells (one side passes all 4 sub-windows):
- v3.3 / long: 0 cells
- v3.3 / short: 0 cells
- v3.4.2 / long: 0 cells
- v3.4.2 / short: 0 cells

### Bidirectional regime-stable cells

**NO bidirectional regime-stable cell exists in either model.**

No (model x horizon x conf_tier) cell has BOTH long AND short sides passing the HC #408 honesty gate in ALL 4 sub-windows simultaneously.

### Unidirectional regime-stable cells

**NO unidirectional regime-stable cell exists in either model.**

No single (model x horizon x side x conf_tier) cell passes the HC #408 honesty gate in ALL 4 sub-windows.

## TP / SL recommendations

Convention: TP = mean MFE (favorable); SL = -1.5 x mean MAE (1.5x buffer over historical adverse).
TP/SL units = ticks. Net cost (commission) = 0.376 tk per round trip.

## Honest assessment

**NO regime-stable cell of ANY kind was found** — neither bidirectional nor unidirectional cells pass the honesty gate in all 4 sub-windows.

This means: every promoted aggregate-OOT result in HC #410 was carried by 1-2 favorable sub-windows and would have failed in the other(s). DO NOT deploy these models to paper trader on signal-only basis.

Next research directions:
1. **Investigate sub-window failure modes** — read `regime_stability_matrix.csv` and identify which windows broke each candidate. Is it always the same date range?
2. **Extend OOT** — 4 contiguous chunks of ~4 days each have wide CIs; with 60+ days of OOT the gate may become passable.
3. **Add execution layer** — perhaps the raw horizon-end signal isn't tradeable but a smart-execution agent (Neptune RL/MLP) can lift expectancy by adaptive exit.
4. **Re-examine commission assumption** — if a market maker rebate path exists (passive fills only, with negative commission), the bar drops.

## Limitations & caveats

- **Sub-window count = 4**. With 15-16 OOT days that's ~4 days per window — wide CIs. Promote-gate may be too strict at this granularity.
- **MAE at 1s/5s/10s is a proxy** (mean of negative-only realized horizon-end moves) — NOT a true intra-horizon adverse-excursion. The 30s MAE is the true `target_pred_mae_30s_ticks`. Treat non-30s MAE numbers as a lower bound on true intra-window adverse.
- **Confidence ranking is in-window**: within each sub-window the top-k is taken from that window's samples (NOT a global top-k restricted to the window). This is the correct semantic for 'would I have traded this in real time inside this regime'.
- **No v2 in this analysis** — v2 has discrete labels {0, 0.5, 1.0} and cannot produce tick-level MFE/MAE/net.
- **Cost model**: net = realized - 0.376 (commission only, passive entry assumed). Market-cross adds +1.0 tick spread.
- **OOT periods are short and unique per model** — v3.3 = 15 days in March 2026, v3.4.2 = 16 days Feb 23-Mar 15. The regimes covered are NOT exhaustive — even a bidirectional regime-stable cell here may fail on out-of-sample regimes not represented (e.g. extreme vol events).
- **Independence assumption in CI**: 95% CI uses iid normal approximation per sub-window. Intra-day autocorrelation likely inflates CI tightness (real CIs are wider).