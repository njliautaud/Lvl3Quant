# CNN-Mamba v3.2 — Long-Context, Book-Memory, Regime-Aware Multi-Head Trainer

**Author:** Claude (head of quant)
**Date:** 2026-05-11
**Authority:** DIRECTIVES.md HC #283, HC #293 (alpha-first heads), HC #294 (skip v3.1 → straight to v3.2)
**Supersedes:** v3.1 incremental plan from HC #292

---

## 1. Motivation & High-Level Diff vs v3

v3 sees only the most recent **~30-60s of order flow** (1500 native MBO events) and outputs **7 strategy-baked heads** (3 directional + 4 FIFO TP/SL net+hit). It works (Ep1 OOT IC_1s=0.256, IC_5s=0.128, IC_10s=0.089 — matches v2 within 5-7 bp), but it can't see:

- **Session-scale context** (S/R levels, prior-session H/L/C/VWAP, where today's volume has accumulated)
- **Vol regime** (high-vol open vs lunch chop vs close drive)
- **Book memory** (resting-order persistence, cancel/replace velocity, trade aggression history beyond a few seconds)

v3.2 fixes all three while keeping v3's proven CNN-Mamba backbone — Mamba SSMs scale **linearly** with sequence length, so we can stack a wider context without an attention-style cost blowup.

### Architectural diff (one-line summary)

| Capability | v3 | v3.2 |
|---|---|---|
| Context | 1500 native MBO events (~30-60s) | **Three parallel tiers** — 1500 native + 1500 @100ms (~2.5min) + 1500 @1Hz (~25min) |
| Inputs | 25 event features | **25 event + 4 PatchTST + 10 book-history + 12 session-context + 3 vol-regime/TOD = 54 features** |
| Heads | 7 (3 dir + 4 FIFO) | **23+ alpha-first** (directional 1s/5s/10s/30s/60s/5min, p_up, quantiles, MFE/MAE/reversal/vol) + legacy FIFO at λ=0.1 |
| Warmstart | v2 fold_10_best.pt | **v3 fold_00_best.pt** (strict=False) |
| Folds | 10 weekly, anchor 2026-02-23 | **10 weekly, anchor 2026-02-23** (unchanged so we can directly compare per-fold OOT IC to v3) |

---

## 2. Multi-Resolution Context (Tier 1/2/3)

Each tier is processed by its **own Mamba branch** (parameters not shared — different temporal dynamics at each resolution). Outputs are concatenated then fed into the trunk.

### Tier 1 — RECENT (fine grain)
- **Window:** 1500 raw MBO events at native cadence
- **Coverage:** ~30-60s of microstructure (queue dynamics, immediate aggression)
- **Features per event:** 25 smart_v3 + 4 PatchTST (forward-filled) + 10 book-history = **39**
- **Backbone:** unchanged from v3 (CNNMamba, d_model=128, n_layers=4)

### Tier 2 — MEDIUM (100ms buckets)
- **Window:** 1500 buckets of 100ms each
- **Coverage:** ~2.5 min of recent order-flow accumulation
- **Features per bucket:** rolled-up event stats — 25 event aggregates (sum/mean over bucket) + 4 PatchTST (held flat) + 10 book-snapshot at bucket close = **39**
- **Backbone:** identical CNNMamba class, fresh params (`branch_t2_mamba`)

### Tier 3 — LONG (1Hz snapshots)
- **Window:** 1500 snapshots at 1 Hz
- **Coverage:** ~25 min of session-scale regime
- **Features per snapshot:** 12 session-context features (S/R distance, prior session H/L/C/VWAP, accumulation/distribution, VPOC/VAH/VAL distance) + 3 vol-regime/TOD bin one-hot = **15**
- **Backbone:** identical CNNMamba class, fresh params (`branch_t3_mamba`); smaller (n_layers=2, d_model=64) because lower information density

### Fusion

```
emb_t1 = T1_backbone(window_t1)      # (B, d_model=128)
emb_t2 = T2_backbone(window_t2)      # (B, d_model=128)
emb_t3 = T3_backbone(window_t3)      # (B, d_model=64)
emb = concat([emb_t1, emb_t2, emb_t3])  # (B, 320)
trunk_out = Trunk(emb)               # (B, trunk_dim=192)
preds = {head: head_module(trunk_out) for head in HEADS}
```

---

## 3. Input Features

### 3.1 Tier 1 features (39 per event)

| Idx | Feature | Source |
|---|---|---|
| 0-24 | smart_v3 (price/qty/imbalance/queue/persistence...) | existing `mbo_events_smart_v3/` |
| 25-27 | PatchTST pred_1s/5s/10s (forward-filled, rank-normed) | existing `mbo_events_smart_v3_pt_pred/` |
| 28 | has_pt_pred mask | existing pt_pred |
| 29-38 | Book-history (top-10 levels, imbalance, resting persistence, cancel/replace velocity, trade-aggression ratio) | **NEW** — derived in `mbo_book_history.py` from raw MBO |

### 3.2 Tier 2 features (39 per 100ms bucket)

Bucket-level aggregates of Tier 1 features. Same 39 columns, but each is a 100ms-window summary (mean for continuous, sum for counts).

### 3.3 Tier 3 features (15 per 1s snapshot)

Session-context features. **These do NOT exist yet as precomputed parquet**. v3.2 uses ZEROS as placeholders so the model still trains while we backfill (TODO in code). Once the precompute pipeline runs, swap to real values.

| Idx | Feature | Description |
|---|---|---|
| 0 | dist_to_today_s_R | distance (ticks) from current mid to nearest support / resistance built today |
| 1 | dist_to_pdh | distance to prior-day high |
| 2 | dist_to_pdl | distance to prior-day low |
| 3 | dist_to_pdc | distance to prior-day close |
| 4 | dist_to_session_vwap | distance to today's VWAP |
| 5 | volume_above_vwap_pct | accumulation/distribution proxy |
| 6 | dist_to_vpoc | distance to today's volume POC |
| 7 | dist_to_vah | volume area high |
| 8 | dist_to_val | volume area low |
| 9 | session_high_tested_pct | % of session high tested |
| 10 | session_low_tested_pct | % of session low tested |
| 11 | tod_minutes_from_open | regime bin proxy (continuous) |
| 12 | vol_regime_low | one-hot from LGBM vol model |
| 13 | vol_regime_med | one-hot |
| 14 | vol_regime_high | one-hot |

### 3.4 Book-history features (NEW — derived in trainer)

Derived from the same raw MBO that smart_v3 uses, so no new data dependency:

| Idx | Feature | Description |
|---|---|---|
| 0 | book_imbalance_top10 | sum(bid_top10) / (sum(bid_top10) + sum(ask_top10)) |
| 1 | book_imbalance_top5 | same, top-5 |
| 2 | resting_order_persistence_5s | fraction of top-5 orders unchanged in last 5s |
| 3 | resting_order_persistence_10s | same, 10s |
| 4 | cancel_replace_velocity_1s | count of cancel-then-replace events / 1s |
| 5 | cancel_replace_velocity_5s | same, 5s |
| 6 | trade_aggression_ratio_1s | aggressor_buys / total_trades |
| 7 | trade_aggression_ratio_5s | same, 5s |
| 8 | book_pressure_5s | (delta_bid_size - delta_ask_size) / total_size, rolling |
| 9 | order_size_skew | mean(top_bid_size) - mean(top_ask_size), rolling |

Some of these (idx 0-1) are already in smart_v3 features. Where overlap exists we keep the v3 version and skip the dup.

---

## 4. Output Heads (Alpha-First)

Per HC #293(B), heads describe **what the signal actually is** (path, distribution, time-evolution) — not strategy parameters.

### Directional regression (MSE, λ=1.0)
- `log_ret_1s`, `log_ret_5s`, `log_ret_10s`, `log_ret_30s`, **`log_ret_60s`**, **`log_ret_5min`**

### Directional probability (BCE on sign, λ=0.5)
- `p_up_5s`, `p_up_10s`, `p_up_30s`, `p_up_60s`

### Quantiles (Pinball, λ=0.5)
- `log_ret_10s_q10/q50/q90`, `log_ret_30s_q10/q50/q90`, **`log_ret_60s_q10/q50/q90`**

### Path heads (Huber, λ=1.0)
- `pred_mfe_30s_ticks`, `pred_mae_30s_ticks`
- **`pred_mfe_60s_ticks`**, **`pred_mae_60s_ticks`**

### Time-evolution (Huber, λ=0.5)
- `pred_time_to_mfe_secs`

### Reversal probability (BCE, λ=0.5)
- `p_reversal_15s`, `p_reversal_30s`, **`p_reversal_60s`**

### Vol prediction (Huber, λ=0.5)
- `pred_realized_vol_30s_ticks`

### Legacy aux (λ=0.1, per HC #293(F))
- `fifo_tp4sl3_net`, `fifo_tp8sl5_net`, `fifo_tp4sl3_hit_tp`, `fifo_tp8sl5_hit_tp`

**Total heads:** 28 (vs v3.1's 23, vs v3's 7).

For heads where labels don't exist yet (60s, 5min, 60s quantiles/reversal/MFE/MAE), they are MASKED OUT at training time (mask=0 → no gradient). Will activate once alpha_labels parquet is backfilled.

---

## 5. Training Plan

| Setting | Value |
|---|---|
| Folds | 10 weekly |
| Anchor | 2026-02-23 (Mon) — same as v3 so we can directly compare |
| Train window | 60 trading days SLIDING (HC #0) |
| OOT window | 5 trading days (Mon-Fri) |
| Window size | 1500 native MBO events (Tier 1); 1500 × 100ms (Tier 2); 1500 × 1s (Tier 3) |
| Stride | 250 events (Tier 1); Tiers 2/3 follow Tier 1 timestamps |
| Batch size | 96 (smaller than v3's 128 because Tier 2+3 inflate memory ~1.7x) |
| LR | same as v3 (3e-4 with warmup-cosine) |
| Epochs per fold | 5 |
| Warmup steps | 1000 |
| Grad clip | 1.0 |
| Mixed precision | bf16 |
| Optimizer | AdamW, wd=1e-4 |

### Warmstart
- Load `output/cnn_mamba_v3_smart_v3_fifo/fold_00_best.pt` into the **Tier 1 branch backbone + trunk**.
- Strict=False — Tier 2/3 branches init fresh, new heads init Xavier.
- Log how many tensors loaded vs init-random.

---

## 6. Falsification Criterion at Ep 1 OOT

v3.2 must show at Ep 1 OOT **at least ONE** of the following improvements over v3 Ep 1 OOT baseline (IC_1s=0.256, IC_5s=0.128, IC_10s=0.089):

1. ≥+0.01 absolute on any of `IC_5s`, `IC_10s`, `IC_30s`
2. ≥+0.05 absolute on Spearman correlation between predicted MFE-MAE and realized MFE-MAE (path-shape capture)
3. Non-degenerate quantile calibration on `log_ret_10s` quantiles — pred q10/q50/q90 should bracket realized at ~10%/50%/90% within ±5%

If NONE of these are hit at Ep 1, the run is FAILED. Kill, write a postmortem, and revert to v3 fold_00_best.pt as the production weights. Do not let it train for 5 epochs hoping it converges — wasted GPU.

---

## 7. Open Items / Caveats

- **Session-context features are placeholders (zeros)** at launch. Tier 3 branch will train but won't add signal until backfilled. TODO log emitted in trainer.
- **Tier 2 bucket aggregation is implemented in trainer** (not precomputed). This adds ~15% to dataset load time but avoids a separate parquet pipeline.
- **Book-history features are derived on-the-fly from raw MBO** (same raw NPZ as smart_v3). Adds ~10% to load time.
- **60s and 5min labels do not exist yet** — heads exist with mask=0 (no gradient). Backfill in alpha_labels generator.

---

## 8. MLflow Logging

- Tracking URI: `http://jupiter:5000` (Tailscale)
- Experiment: `CNNMamba_v3_2_long_context`
- Tag: `cnn_mamba_v3.2`
- Hyperparams logged: all of §5 plus head set, tier window sizes, warmstart path, feature counts per tier.
- Ep 1 OOT metrics: `ic_log_ret_{1s,5s,10s,30s}`, `corr_mfe_30s`, `corr_mae_30s`, `quantile_calib_{10s,30s}`.
