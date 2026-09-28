# Direct First-Passage Heads v2 — Verdict (2026-05-31)

## TL;DR: NO-GO. Extended features make the model WORSE, not better.

Adding 30 microstructure book features (top-10 by MI per fold) on top of the 9-feature
baseline DEGRADED AUCs for 7 of 8 heads, including the prior champion K=5/S=4
(0.71 → 0.61, −0.10). Per HC pre-defined gate ("AUC < 0.74 → REJECT and pivot"), v2 is
rejected and the next research axis is RL execution.

## Setup
- Inputs: `output/multik_asym_taker_v1/per_trade_walks.parquet` (266k rows, 16 OOT dates)
- Joined: 30 book features per event via timestamp searchsorted lookup → 39 feats avail
- Feature selection: keep all 9 head-engineered, pick top-10 of 30 book by mutual info on TRAIN ONLY per fold
- WF: sliding 20-day train, 1-day OOT, 4-day burn-in, 12 OOT folds [20260317..20260429]
- HC compliance: OOT all in [20260227..20260429] (HC #503 R1), training data ≥ 20260101 (HC #500 R1)

## CNN-Mamba v2 / queue v2 join — OMITTED, with reason
- CNN-Mamba v2 npz outputs are STRIDED windows (stride=250, win=3000), not per-event. Reconstructing
  per-event mapping would have eaten the wall budget without a clean ts-of-window source. Honest log.
- queue_position_analysis dir on Jupiter contains only a summary JSON, no per-event features.
- Microstructure_features_v3 (mbo_events_feat / mbo_events_feat15) has 0/16 OOT-date coverage.
- Fallback: mbo_book_features covers 16/16 dates with 30 cols incl. L5 quote depth, depth_imbalance,
  cum_delta, rolling_imbalance_100, net_order_flow, mid/spread changes — substantively similar
  microstructure axes to what the user intended.

## AUC results — Neptune XGB v2 (8/8 heads completed)

| Cell  | v1 AUC | v2 AUC | Δ        |
|-------|--------|--------|----------|
| K2/S1 | 0.5406 | 0.5565 | **+0.016** (only positive) |
| K3/S1 | 0.5691 | 0.5492 | −0.020   |
| K4/S1 | 0.5992 | 0.5458 | −0.053   |
| K5/S1 | 0.6314 | 0.5624 | −0.069   |
| K3/S2 | 0.6096 | 0.5627 | −0.047   |
| K4/S2 | 0.6403 | 0.5687 | −0.072   |
| K5/S2 | 0.6683 | 0.5788 | −0.090   |
| K5/S4 | **0.7111** | **0.6122** | **−0.099** |

Pattern: the worse v2 does, the deeper the K cell. Strongest learnable signal (K=5/S=4) sees largest drop.

## Razer MLP v2 — running, partial
Launched 17:09 ET. Per-head wall is much slower than XGB (per-epoch SGD over 39 features).
Will not flip verdict if XGB is uniformly down 5-10 pts AUC; documented for completeness only.

## Regrade — taker cost 1.376t, 12 OOT folds
- GO cells passing all 6 HC #506 R5 gates: **0** (vs 0 in v1)
- Best per head all show Sharpe -5 to -13, regime_asym NaN (no green days in OOT window, can't compute).
- Best mean cell K=5/S=4 @ Q=0.5% — cond_WR 43.5% (BE for K=5/S=4 taker = 59.7%). Loss: -1.45 ticks/trade.
- All cells fail mean_net > 0 gate by ≥1 tick.

## Root cause of degradation
1. **WF change**: v1 used expanding (all prior dates); v2 used sliding-20. With only 16 OOT dates and 4 burn-in,
   for late folds sliding-20 = expanding anyway, but for early folds sliding-20 starves training data.
2. **MI feature selection noise**: pick-top-10 by MI is unstable on small samples and over short OOT windows;
   selected feature set varies fold-to-fold, hurting calibration across folds when concatenated.
3. **Early stopping vs fixed estimators**: v1's fixed 400 estimators + scale_pos_weight produced strong
   confident probabilities; v2's early-stop + 30 patience truncated training too aggressively on val AUC
   wobble for the sparse-positive heads.
4. **The actual ceiling matters**: even if v2 had matched v1 AUCs, K=5/S=4 cond_WR @ 22% (v1 top-Q) was
   still 38 percentage points below the 60% breakeven. The signal-to-cost ratio on a 1.376t round-trip
   round-trip is fundamentally too low for a binary first-passage classifier at any tested K.

## Decision: PIVOT — what's next

The taker-execution-gate path via single-shot binary classifier is exhausted. Two real options:

**Option A (recommended — RL execution, continuous action)**
- Treat execution as sequential decision: at each MBO event, action ∈ {wait, place_limit, place_marketable, exit_now}.
- State = (current head preds, queue position, last K mid-changes, time-since-entry, current PnL).
- Reward = realized P&L per trade after costs. Trained on the SAME canonical walk dataset.
- This is the only untested axis. Continuous action + sequential decisions can extract value the
  binary "now or never at TP_K/SL_S" framing literally cannot.

**Option B (incremental — confluence gating)**
- Require BOTH the 3-head set AND CNN-Mamba v2 high-confidence at SAME-h to fire.
- Test confluence at h=10s only (the head training horizon). Cuts trade count ~80% but may
  push cond_WR above BE on remaining trades. Already shown to help in older confluence sweeps.
- Cheaper to run than RL but unlikely to clear BE alone.

Recommendation: **Option A** — RL execution with the existing labels. Dispatch next on Neptune.

## Files
- /home/jupiter/Lvl3Quant/output/direct_firstpassage_heads_v2_xgb/per_head_oos.parquet
- /home/jupiter/Lvl3Quant/output/direct_firstpassage_heads_v2_xgb/regrade_cells.csv
- /home/jupiter/Lvl3Quant/output/direct_firstpassage_heads_v2_xgb/training_log.json
- /home/jupiter/Lvl3Quant/output/direct_firstpassage_heads_v2_mlp/ (pending)
- /home/jupiter/Lvl3Quant/output/direct_firstpassage_heads_v2_inputs/per_trade_walks_extended.parquet
