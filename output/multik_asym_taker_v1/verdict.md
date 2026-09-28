# Multi-K Asymmetric TAKER Walk — Verdict
Generated 2026-05-31 16:45 ET. Script: scripts/multik_asym_taker_walk_v1.py

## Setup
- OOT window: 20260227..20260429, 16 folds (one all-zeros book day dropped), **266,063 events**.
- Horizon h=10s; hold cap 15s (HC #428 R2 compliant, p90 realized MFE = 5.0t).
- Predictions: 3-head (adverse, MFE, toxicity) trained at h=10s on canonical OOT (`tmp_tox_data/`).
- TAKER cost only: **1.376 ticks** (commission 0.376 + 1.0 spread crossing).
- **Ordered first-passage MBO walk**: records ts of first time mid travels +K vs −S from entry; exit = whichever hits first; if neither, exit at mid-change at hold cap.

## Grid (360 cells total)
- TP K in {2, 3, 4, 5} ticks (with extension K=1 to support reversal-cell SL=1)
- SL S in {1, 2, 3, 4} ticks
- Asymmetric only: TP > SL
- Conviction Q in {0.5, 1, 2.5, 5, 10, 20} %
- Gates: composite (tox_rank − adv_rank), toxicity_only, mfe_only
- **Two directions tested**: continuation (trade WITH predicted side) AND reversal (trade AGAINST predicted side, i.e. invert prediction)

## Headline result

**0 of 360 cells positive net of taker cost. 0 cells PF > 1. 0 cells beat their break-even conditional WR. Zero cells within even 5 percentage points of BE.**

| Direction | Best Sharpe cell | cond_WR | BE | gap | mean_net | PF |
|---|---|---|---|---|---|---|
| Continuation | mfe_only Q=0.5% K=5/S=4 | 0.443 | 0.597 | −0.165 | −1.39t | 0.53 |
| Reversal | mfe_only Q=1% K=5/S=4 | 0.447 | 0.597 | −0.150 | −1.34t | 0.54 |

## Continuation vs Reversal head-to-head (best cond_WR per K/S)
| K/S | Continuation cond_WR | Reversal cond_WR |
|---|---|---|
| 2/1 | 0.327 | 0.316 |
| 3/1 | 0.239 | 0.224 |
| 3/2 | 0.403 | 0.394 |
| 4/1 | 0.177 | 0.178 |
| 4/2 | 0.334 | 0.325 |
| 4/3 | 0.441 | 0.427 |
| 5/1 | 0.142 | 0.149 |
| 5/2 | 0.277 | 0.275 |
| 5/3 | 0.380 | 0.369 |
| 5/4 | 0.443 | 0.447 |

**Continuation and reversal produce nearly identical cond_WRs at every K/S.** The model's predicted side has essentially no informational bite on the ordered first-passage direction question — flipping the side does NOT rescue the framework.

## Key finding: barrier hit asymmetry is structural noise, not predicted-side mis-direction

Universe-level (no gating) hit rates:
- TP2: 58.6%, TP3: 42.2%, TP4: 30.6%, TP5: 22.4%
- SL1: 82.0%, SL2: 59.3%

Reading: 82% of trades experience a 1-tick adverse move WITHIN the 15s hold window — irrespective of trade direction. This is just the natural microstructure noise around ES futures at the event-trigger timestamps. The TP-hit-rates decay roughly geometrically with K (the random-walk pattern). Conviction gating (Q=0.5% on highest mfe) only slightly skews these rates and does NOT push cond_WR above BE for any (K,S) cell.

## Universe-level mean PnL by cell (no gating, n=263k)
| K/S | cond_WR | BE | mean_net (taker) | PF |
|---|---|---|---|---|
| 2/1 | 0.275 | 0.792 | −1.53t | 0.10 |
| 3/1 | 0.184 | 0.594 | −1.57t | 0.15 |
| 4/1 | 0.130 | 0.475 | −1.59t | 0.17 |
| 5/1 | 0.094 | 0.396 | −1.61t | 0.18 |
| 3/2 | 0.365 | 0.675 | −1.46t | 0.26 |
| 4/2 | 0.275 | 0.563 | −1.49t | 0.29 |
| 5/2 | 0.208 | 0.482 | −1.51t | 0.30 |

Even at the most asymmetric (5/1, BE=39.6%) where the user expected "extremely achievable", actual cond_WR is 9.4% — the gap is **−30 percentage points**. Not close.

## HC #428 R1 regime gate (best cell)
- Greens 1, Reds 15 (mean-tick sign per day)
- Sharpe_green +0.61 / Sharpe_red −2.91 → asymmetry **1.21** (FAIL, > 0.50 cap)
- Day-concentration 0.13 (PASS, < 0.70 cap)
- **R1 GATE FAIL** — universally red strategy

## Verdict: NO-GO (definitive)

Multi-K asymmetric TAKER first-passage is structurally dead at h=10s for both continuation and reversal interpretations of the model's predicted side. This is consistent with all prior tonight's K-framework results at h=1s and h=10s under maker and taker, symmetric and asymmetric. **The K-framework is exhausted.**

What this run uniquely adds vs prior nights:
1. First **ordered first-passage** walk (prior runs used magnitudes, which hid which barrier hit first).
2. First test of **K up to 5** with proper +K/−1 and +K/−2 asymmetry (user's specific request).
3. First explicit **side-flip / reversal** evaluation alongside continuation.

All three angles confirm: the predicted-side does NOT generate exploitable ordered motion at K-barriers under taker cost.

## What this rules out (definitively)
1. K=4/S=1, K=5/S=1 (user's "extremely achievable" cells): cond_WR 13-17% vs BE 40-48% — gap −25 to −34pp.
2. K=5/S=4 (smallest required gap): cond_WR 44.3% vs BE 59.7% — gap −15pp. Closest cell. Still NO-GO.
3. Reversal hypothesis: side-flip produces NEAR-IDENTICAL cond_WRs to continuation. The "predicted side" is not anti-predictive; it's just non-predictive for K-barrier first-passage.
4. Conviction gating (any Q from 0.5% to 20%): does not lift any cell above BE.

## Recommended next steps (TAKER agenda)

The K-framework has now been exhausted across both horizons, both directions, full grid. The signal IS real (toxicity IC 0.77, composite WR lift ~10pp on the +1-ADV<1 binary question per session_state). What the signal does NOT do is generate ordered multi-tick directional moves at event-aligned timestamps before microstructure noise hits 1-2 ticks adverse.

**Three viable next moves on the TAKER agenda:**

(A) **Entry-delay / reversion-wait gate**: defer entry until the mid has touched and re-crossed the entry-side -1t threshold. Hypothesis: predictions fire on micro-pop events; the genuine directional move begins AFTER the immediate reversion completes. Cheap to test on existing 263k walks (post-process, no MBO recompute). Recommended FIRST.

(B) **Quantile-regression head on REALIZED first-passage TP-time vs SL-time**: train an XGB to predict P(TP_K hits before SL_S | features) directly, then gate trades on that probability rather than on the proxy heads (adverse / MFE / toxicity). This is a direct probabilistic gate against the actual decision question. Razer XGB job; modest cost.

(C) **RL execution policy** (per yesterday's pivot): continuous action space (post / cancel / reprice / taker-exit) on the live prediction stream. Captures partial-tick edge that K-framework buckets to 0. Highest implementation cost, biggest potential payoff.

**Recommendation**: I'll dispatch (A) next — it's a post-process on per_trade_walks.parquet, ~5 min wall, and directly tests whether the entry-timing micro-noise IS the entire problem. If (A) shows positive cond_WR for any reasonable delay, escalate to FIFO grade. If (A) also fails, escalate (B) to Razer.

## Top 5 cells by Sharpe
```
   direction          gate  Q_pct  K  S  n_total  n_trades  n_tp  n_sl  n_hold  n_hit  cond_WR_tp_vs_sl  WR_pos_net  mean_ticks_net  median_ticks_net       PF     Sharpe  sum_ticks_net  n_days  n_days_positive
continuation      mfe_only    0.5  5  4     1330      1330   577   726      27   1303          0.443    0.436       -1.390            -5.376 0.531    -11.44       -1848.08      16                1
    reversal toxicity_only    0.5  5  4     1330      1330   542   716      72   1258          0.431    0.430       -1.450            -5.376 0.509    -12.12       -1928.58      16                1
    reversal      mfe_only    0.5  5  4     1330      1330   561   742      27   1303          0.431    0.425       -1.493            -5.376 0.506    -12.33       -1985.58      16                0
continuation toxicity_only    0.5  5  4     1330      1330   544   717      69   1261          0.431    0.423       -1.478            -5.376 0.503    -12.37       -1966.08      16                0
continuation      mfe_only    0.5  5  3     1330      1330   498   812      20   1310          0.380    0.375       -1.334            -4.376 0.504    -12.61       -1773.58      16                1
```

## Outputs
- `per_trade_walks.parquet` — 266,063 rows; tp{1..5}_dt_ns and sl{1..5}_dt_ns columns, full first-passage record
- `cells_summary.parquet` — 360 cells with metrics + direction (continuation / reversal)
- `regime_gate.json` — fold-level diagnostics + R1 regime gate on best cell
- `verdict.md` — this file
