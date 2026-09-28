# HC #437 Bug 2 — Phase C: FIFO Realtime-SL vs HC #413 Bracket Mode Comparison

Both runs are the SAME v2 1s short top0.5% signal set (5,569 selected signals over
46 signal days), same TP=0.9564/SL=0.5686 magnitudes, same passive_at_touch entry,
hold_s=10, cancel_s=10. Only the EXIT logic differs:

- `realtime_sl`: intra-event price tracking. Every MBO event after fill, check if
  bid/ask crosses TP or SL threshold. First touch wins.
- `hc413_bracket`: evaluate at horizon checkpoints (1s, 5s, 10s) ONLY using the
  realized signed log-return from the v2 NPZ `labels` array. Priority: SL → TP2 → TP1.

## Aggregate side-by-side
| metric | bracket | realtime_sl | Δ (b - r) |
|---|---|---|---|
| n_fills | 3,452 | 3,455 | — |
| mean_gross_tk | +0.4923 | n/a | — |
| **mean_net_tk** | **+0.1163** | **-0.2745** | **+0.391** |
| **PF** | **1.45** | **0.48** | +0.97 |
| **WR%** | **72.05** | **43.97** | +28.08 |
| **Sh(√N)** | **+10.24** | **-21.32** | +31.56 |
| exits | tp2:2188 sl:935 tp1:299 ts:30 | sl:1934 tp:1516 max_hold:5 | — |

## Interpretation

The two modes produce **diametrically opposite verdicts on the same signals
with the same threshold magnitudes**:

- Under bracket exits (HC #413 methodology): mild positive edge (+0.116 tk/fill,
  PF 1.45). 72% WR with TP2-heavy exit mix.
- Under realtime SL (HC #432 / live trading reality): catastrophic loss
  (-0.274 tk/fill, PF 0.48). 44% WR with 56% SL hits.

The delta (+0.391 tk/fill, i.e. ~$4.89 per ES contract per fill) is entirely
explained by the SL-touch-before-horizon mechanic: under realtime tracking, an
intra-event excursion below -SL fires the stop EVEN IF the price recovers to
TP-territory by the next horizon checkpoint. Under bracket, the stop only fires
if the price is below -SL AT the 1/5/10s checkpoint itself.

**This is the HC #437 Bug 2 finding restated quantitatively: HC #413's
published edge is NOT a live-tradeable edge. It is a methodology artifact of
checkpoint-only exit evaluation. The realtime stop, which is what an actual
limit-bracket order will be subject to at the exchange, kills the edge entirely.**

## Regime stratification (per HC #428 R1)
| mode | regime | n_days | n_fills | net_tk | PF | WR% | Sh√N |
|---|---|---|---|---|---|---|---|
| bracket | green | 17 | 1,915 | +0.128 | 1.51 | 72.95 | +8.49 |
| bracket | red | 15 | 1,537 | +0.102 | 1.37 | 70.92 | +5.89 |
| realtime_sl | green | 17 | 1,916 | -0.257 | 0.50 | 45.09 | -14.85 |
| realtime_sl | red | 15 | 1,539 | -0.296 | 0.46 | 42.56 | -15.39 |

Bracket: regime delta |Sh_g - Sh_r|/max = 0.306 → REGIME-AGNOSTIC PASS.
Realtime_sl: regime delta = 0.035 → REGIME-AGNOSTIC PASS (catastrophic uniformly).

## Per-day delta table
| date | regime | n_b | net_b | n_r | net_r | Δnet | PF_b | PF_r | WR_b | WR_r |
|------|--------|-----|-------|-----|-------|------|------|------|------|------|
| 20260224 | green | 90 | +0.2309 | 90 | -0.2499 | +0.4808 | 2.08 | 0.51 | 76.7 | 45.6 |
| 20260226 | red | 107 | +0.2668 | 109 | -0.2730 | +0.5398 | 2.37 | 0.48 | 79.4 | 44.0 |
| 20260301 | green | 1 | -0.9446 | 1 | -0.9446 | +0.0000 | 0.00 | 0.00 | 0.0 | 0.0 |
| 20260303 | green | 231 | +0.1604 | 232 | -0.2544 | +0.4148 | 1.62 | 0.51 | 72.3 | 45.3 |
| 20260305 | green | 199 | +0.2585 | 199 | -0.2472 | +0.5057 | 2.30 | 0.52 | 78.9 | 45.7 |
| 20260306 | red | 197 | +0.1836 | 197 | -0.3098 | +0.4934 | 1.75 | 0.44 | 73.6 | 41.6 |
| 20260309 | green | 199 | +0.1742 | 199 | -0.3699 | +0.5441 | 1.69 | 0.37 | 73.4 | 37.7 |
| 20260310 | red | 166 | +0.0694 | 166 | -0.3658 | +0.4352 | 1.22 | 0.38 | 66.3 | 38.0 |
| 20260311 | red | 105 | +0.0721 | 105 | -0.3491 | +0.4212 | 1.23 | 0.39 | 66.7 | 39.0 |
| 20260312 | green | 160 | +0.1038 | 160 | -0.3155 | +0.4193 | 1.35 | 0.43 | 68.8 | 41.2 |
| 20260313 | red | 128 | +0.0562 | 128 | -0.4680 | +0.5242 | 1.17 | 0.28 | 65.6 | 31.2 |
| 20260316 | green | 74 | +0.1436 | 74 | -0.2636 | +0.4072 | 1.59 | 0.49 | 74.3 | 44.6 |
| 20260317 | green | 95 | +0.1874 | 95 | -0.2062 | +0.3936 | 2.06 | 0.58 | 80.0 | 48.4 |
| 20260318 | red | 125 | +0.2353 | 125 | -0.1431 | +0.3784 | 2.33 | 0.68 | 80.8 | 52.8 |
| 20260319 | red | 77 | +0.1296 | 77 | -0.2910 | +0.4206 | 1.56 | 0.46 | 75.3 | 42.9 |
| 20260401 | red | 125 | -0.0276 | 125 | -0.2614 | +0.2338 | 0.91 | 0.50 | 65.6 | 44.8 |
| 20260402 | green | 101 | -0.0247 | 101 | -0.3255 | +0.3008 | 0.93 | 0.42 | 64.4 | 40.6 |
| 20260405 | red | 1 | -0.9446 | 1 | -0.9446 | +0.0000 | 0.00 | 0.00 | 0.0 | 0.0 |
| 20260406 | red | 54 | -0.0852 | 54 | -0.3515 | +0.2663 | 0.78 | 0.39 | 59.3 | 38.9 |
| 20260407 | red | 100 | +0.0494 | 100 | -0.1668 | +0.2162 | 1.16 | 0.64 | 68.0 | 51.0 |
| 20260408 | red | 102 | +0.0428 | 102 | -0.1971 | +0.2399 | 1.15 | 0.59 | 69.6 | 49.0 |
| 20260409 | green | 73 | -0.1339 | 73 | -0.1925 | +0.0586 | 0.68 | 0.60 | 56.2 | 49.3 |
| 20260410 | red | 63 | +0.0623 | 63 | -0.1700 | +0.2323 | 1.23 | 0.63 | 69.8 | 50.8 |
| 20260413 | green | 84 | +0.1184 | 84 | -0.1821 | +0.3005 | 1.48 | 0.61 | 73.8 | 50.0 |
| 20260414 | green | 76 | +0.2389 | 76 | -0.2222 | +0.4611 | 2.55 | 0.55 | 82.9 | 47.4 |
| 20260415 | green | 107 | +0.0798 | 107 | -0.1465 | +0.2263 | 1.32 | 0.68 | 72.0 | 52.3 |
| 20260416 | green | 90 | +0.0404 | 90 | -0.2888 | +0.3292 | 1.15 | 0.46 | 70.0 | 43.3 |
| 20260417 | green | 118 | +0.1464 | 118 | -0.3630 | +0.5094 | 1.68 | 0.38 | 77.1 | 38.1 |
| 20260420 | red | 92 | +0.0767 | 92 | -0.2650 | +0.3417 | 1.28 | 0.49 | 70.7 | 44.6 |
| 20260422 | red | 95 | +0.1544 | 95 | -0.4470 | +0.6014 | 1.78 | 0.30 | 79.0 | 32.6 |
| 20260424 | green | 108 | -0.0618 | 108 | -0.1391 | +0.0773 | 0.82 | 0.69 | 63.9 | 52.8 |
| 20260427 | green | 109 | +0.1698 | 109 | -0.1793 | +0.3491 | 1.95 | 0.62 | 78.9 | 50.5 |
