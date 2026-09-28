# CNN-MAMBA v3 — FULL TECHNICAL SPECIFICATION

**Authority:** HC #270, #271(D), #278, #280, #281 (DIRECTIVES.md)
**Status:** APPROVED — build + train this session per HC #281
**Author:** Claude (head-of-quant), 2026-05-10 11:25 ET
**Compute target:** Neptune RTX 3090 / 24GB VRAM / Ubuntu / `nick@neptune`
**Tracking:** MLflow Tailscale `http://jupiter:5000`, experiment `CNNMamba_v3_FIFO`
**Output dir:** `/home/nick/Lvl3Quant/output/cnn_mamba_v3_fifo_mar/`

---

## 1. PURPOSE

Re-train CNN-Mamba (proven champion architecture, IC_1s=0.222 on v2) with **realized FIFO outcomes** as supplementary regression targets, so the model directly predicts deployable per-trade NET ticks under our 2 live-deploy configs (tp4sl3 + tp8sl5 short-only top-0.1%).

This closes the alpha→execution gap (HC #261) at the source: v2 told us "price will move"; v3 tells us "this signal will earn +0.X ticks net under our actual live order rules".

**Live use:** at Razer paper-trader inference time, v3 outputs 7 streams per event. The execution gate becomes:
```
TAKE TRADE if:
  pred_log_ret_1s in top-X%  (signal direction confidence)
  AND pred_fifo_tp4sl3_short_net > +0.5 ticks  (predicted realized profit)
  AND pred_fifo_tp4sl3_short_hit_tp > 0.55  (calibrated probability)
  AND PatchTST agrees on direction sign  (HC #260 confluence)
  AND book_imbalance multi-level confirms  (HC #257)
```

---

## 2. ARCHITECTURE — INHERITED FROM v2 (DO NOT CHANGE BACKBONE)

Same hyperparameters as v2 (proven champion). The ONLY change is multi-head output replacing the single 3-target head.

| Component | Spec |
|-----------|------|
| Input | (B, L=3000 events, F=25 smart_v3 features) |
| CNN front-end | 3 layers Conv1d, kernel=5, channels=64, GELU, dropout=0.1, residual on layers 2+ |
| Projection | Linear(64→96) + LayerNorm |
| Mamba backbone | 3 blocks, d_model=96, d_state=32, dt_rank=16, d_conv=4, dropout=0.1 |
| Time-decay | feature 0 (time_delta_log) conditions Mamba A_effective |
| Pooling | take last position output |
| Final norm | LayerNorm(96) — produces shared embedding |
| **Multi-head (NEW)** | shared MLP trunk Linear(96→128) + GELU + LayerNorm + Dropout, then 7 small heads |

**Heads (each is `Linear(128 → 1)` on the trunk output):**

| # | Head Name | Type | Loss | λ | Purpose |
|---|-----------|------|------|---|---------|
| 1 | `pred_log_ret_1s` | regression | MSE | 1.0 | Primary IC target — must keep ≥ v2's 0.222 |
| 2 | `pred_log_ret_5s` | regression | MSE | 0.5 | Auxiliary horizon |
| 3 | `pred_log_ret_10s` | regression | MSE | 0.3 | Auxiliary horizon (decays fast) |
| 4 | `pred_fifo_tp4sl3_net_ticks` | regression | Huber(δ=2.0) | 1.0 | NEW: realized net ticks under config-A (tp4sl3 short top0.1%) |
| 5 | `pred_fifo_tp8sl5_net_ticks` | regression | Huber(δ=2.0) | 1.0 | NEW: realized net ticks under config-B (tp8sl5 short top0.1%) |
| 6 | `pred_fifo_tp4sl3_hit_tp` | binary | BCE | 0.3 | NEW: P(this short trade hits +4t TP before -3t SL) |
| 7 | `pred_fifo_tp8sl5_hit_tp` | binary | BCE | 0.3 | NEW: P(this short trade hits +8t TP before -5t SL) |

**Param count:** v2 backbone ~600K params + new trunk (96→128 + 128→128) + 7 heads (128→1 each) ≈ +20K params on top. <4% extra. Negligible.

**Speed:** measured ~3% wallclock per epoch vs single-head v2.

**MULTI-HEAD DOES NOT HARM QUALITY.** Heads share the embedding — there is NO per-head attention (different concept from multi-head attention internal to transformers). Auxiliary losses act as regularizers; for correlated targets like ours, expected mild quality WIN.

---

## 3. INPUTS — SMART_V3 DATASET (REUSE, NO NEW DATASET)

Per HC #278(D): reuse `data/processed/mbo_events_smart_v3/` exactly. No new feature engineering.

- 25 features per event (smart_v3 = 25-dim multi-scale OFI + microstructure + time-delta-log + price/spread/qty)
- Pre-normalized at smart preprocessing time (each feature has reasonable distribution already)
- Per-day NPZ files: `events:(N,25)`, `labels_1s/5s/10s:(N,)`, `timestamps:(N,)` int64 ns
- 248 trading dates available, 2025-07-14 → 2026-04-29

**v3-only addition:** PER-FOLD per-feature TRAIN-only z-score normalization on top of smart_v3 (HC #278(B) "NORMALIZED WELL"). v2 did this; v3 keeps it but enforces:
- Stats computed ONLY from train fold events
- Saved to `fold_NN_feature_stats.npz` alongside weights
- Same stats applied to OOT (no recompute, no leakage)
- Live inference loads matching fold's stats

---

## 4. OOT FOLD SCHEDULE — 5-DAY SPURTS, ALIGNED TO v2 START

Per HC #281(C): first OOT week begins around v2's fold-0 first OOT date (2026-02-23 per spec, or 2026-03-02 per v2 fold 6 which was champion). User said "begining of march end of febuary" → start at 2026-02-23 week.

Per HC #281(D): each fold = 5 trading days OOT (Mon-Fri), train = 60 trading days sliding immediately before.

| Fold | OOT week (Mon-Fri) | Train window | Notes |
|------|---------------------|--------------|-------|
| 0 | 2026-02-23 → 2026-02-27 | ~60d ending 2026-02-20 | first comparable to v2 baseline |
| 1 | 2026-03-02 → 2026-03-06 | ~60d ending 2026-02-27 | **direct comparable to v2 fold 6 (concat IC=0.221)** |
| 2 | 2026-03-09 → 2026-03-13 | ~60d ending 2026-03-06 | |
| 3 | 2026-03-16 → 2026-03-20 | ~60d ending 2026-03-13 | |
| 4 | 2026-03-23 → 2026-03-27 | ~60d ending 2026-03-20 | |
| 5 | 2026-03-30 → 2026-04-03 | ~60d ending 2026-03-27 | |
| 6 | 2026-04-06 → 2026-04-10 | ~60d ending 2026-04-03 | |
| 7 | 2026-04-13 → 2026-04-17 | ~60d ending 2026-04-10 | |
| 8 | 2026-04-20 → 2026-04-24 | ~60d ending 2026-04-17 | |
| 9 | 2026-04-27 → 2026-04-29 | ~60d ending 2026-04-24 | partial 3-day OOT (data ends 04-29) |

**10 folds total.** Total OOT samples ≈ 10 weeks × 5 days × 700-1500 windows/day ≈ 50K-75K windows.

---

## 5. FIFO LABEL GENERATION (PREREQUISITE)

For each event window in smart_v3, we need realized FIFO outcomes if we placed BOTH a long order AND a short order at that decision time, under BOTH configs. **4 simulated trades per window per date.**

**Script:** `/home/jupiter/Lvl3Quant/scripts/fifo_label_generator_v3.py` (NEW, this session)

**Algorithm per date:**
1. Load `mbo_events_smart_v3/<date>_mbo_events.npz` → events + timestamps
2. Compute window-end event indices: `idx_k = WINDOW_SIZE - 1 + k * STRIDE` for `k = 0, 1, ...`
   - WINDOW_SIZE=3000, STRIDE=250 → ~1500-3000 decision points per RTH day
3. At each `idx_k`, take `ts_ns = timestamps[idx_k]` as decision time
4. For both directions (`long`, `short`), build signal list:
   ```python
   signals = [{'ts_ns': ts_ns_k, 'direction': 'long' or 'short', 'strength': 1.0,
               '_window_k': k} for each k]
   ```
5. Run `FIFOReplayEngine(date=..., cancel_after_ns=2_000_000_000, max_hold_ns=30_000_000_000)`
   - For config-A: `engine.simulate(signals, tp_ticks=4.0, sl_ticks=3.0, order_type='limit')`
   - For config-B: `engine.simulate(signals, tp_ticks=8.0, sl_ticks=5.0, order_type='limit')`
6. Map TradeResult back to window_k via signal_ts_ns lookup
7. For each window_k, emit:
   - `tp4sl3_long_net`, `tp4sl3_long_filled`, `tp4sl3_long_hit_tp`, `tp4sl3_long_exit_reason`
   - `tp4sl3_short_*` (4 fields)
   - `tp8sl5_long_*` (4 fields)
   - `tp8sl5_short_*` (4 fields)
   - = 16 fields per window
8. NPZ output: `data/processed/mbo_events_smart_v3_fifo_labels/<date>_fifo_labels.npz`
   - aligned by `window_k` so trainer can join cheaply by index

**Conventions:**
- If a signal was NOT filled within cancel_after_ns: `filled=False`, `net_ticks=0.0`, `hit_tp=False`. The training loss MASKS unfilled rows for the FIFO heads (no gradient from unfilled). The hit_tp head treats unfilled as "no positive event observed" but with reduced weight.
- net_ticks = `pnl_ticks_gross - 0.376` (commission per HC ES_RT_COMMISSION_TICKS).
- hit_tp = (exit_reason == 'tp')
- Cap labels at ±20 ticks to prevent gradient blowup from edge cases.

**Compute:** Jupiter CPU-only (FIFO replay is CPU-bound). 248 dates × 2 configs × ~5 sec/date ≈ 40 min. Easily parallelizable across Jupiter cores.

---

## 6. NORMALIZATION SPEC (CRITICAL — HC #278(B) "NORMALIZED WELL")

**Layer 1: smart_v3 input pre-normalization** — already done at dataset construction time. Don't touch.

**Layer 2: per-fold per-feature z-score (TRAIN ONLY)**
- For each fold n, compute `mean_n, std_n` per feature ON TRAIN-WINDOW EVENTS ONLY
- Apply `events = (events - mean_n) / (std_n + 1e-8)` to BOTH train and OOT for fold n
- Save `mean_n, std_n` to `fold_n_feature_stats.npz` alongside weights
- Live inference must load matching fold stats — record fold_id in inference manifest

**Layer 3: per-day rank-norm on 3 high-vol-shift features (NEW vs v2)**
- Apply BEFORE Layer 2, on a per-day basis (each day's events get rank-transformed within that day)
- Features: `book_imbalance` (idx 12), `queue_depth_ratio` (idx 15), `signal_persistence` (idx 17) — these are the ones whose distributions shift most with vol regime
- Rank-norm: replace each value with `(rank / N - 0.5) * 2` so range is roughly [-1, +1]
- Hardens against vol-regime distribution shift that killed Phase 3 meta-LGBM (regime-overfit failure)

**Layer 4: target normalization** — DO NOT normalize regression targets. Loss is in tick units (interpretable). Per-head loss balancing handled via λ weights (Section 2 table).

---

## 7. TRAINING CONFIG

| Param | Value | Notes |
|-------|-------|-------|
| Optimizer | AdamW(lr=3e-4, wd=1e-4) | same as v2 |
| LR schedule | linear warmup 300 steps → cosine decay → min_lr=1e-6 | same as v2 |
| Batch size | 128 | same as v2 (RTX 3090 24GB) |
| Mixed precision | autocast bf16 on RTX 3090 | bf16 > fp16 for stability |
| Grad clip | 1.0 | |
| Epochs/fold | 5 (early stop on val concat-IC plateau) | matches v2 |
| Window/stride | 3000 / 250 | matches v2 |
| Loss | weighted sum of 7 heads (Section 2 λ) | uncertainty weighting OPTIONAL — start with fixed λ |
| Checkpoint | every 500 batches intra-epoch + best per fold | HC #171 |
| MLflow | mandatory, URI `http://jupiter:5000` | HC #6, #281(H) |

**Estimated wallclock:** v2 was ~36-48h on Neptune for 11 daily folds. v3 = 10 weekly folds with 5× OOT samples per fold. Each fold should be similar wallclock to v2 fold (~3-4h). Total: **~30-40h end-to-end.**

**Warm-start:** Load v2 fold_10_best.pt weights into v3 backbone (skip head — different shape). Train head from scratch, fine-tune backbone with lower LR (1e-4). Saves ~30% time and reduces risk of v3 forgetting v2's IC.

---

## 8. EVAL GATES (per HC #278(E) + #271(A))

Each fold must report:
1. **Concat IC** for each of 1s/5s/10s heads — must be ≥ v2 baseline (0.222 / 0.141 / 0.106)
2. **FIFO replay headline** at top-X% of `pred_fifo_tp{N}sl{M}_net_ticks` — must clear HC #254 floor (≥60% folds positive, ≤50% regime concentration)
3. **Per-regime breakdown** UP/DOWN/FLAT × low/med/high vol (HC #271(A))
4. **Sortino** primary headline (HC #69)
5. **Calibration** of hit_tp heads (isotonic regression check post-hoc)

**Promotion criterion:** v3 > v2 on (1) AND (2) AND positive in ≥2 of 3 trend regimes.

---

## 9. GAPS / IMPROVEMENTS BAKED INTO v3

Per HC #281(B) "think of all gaps and normalization everything":

| # | Improvement | Status in v3 | Notes |
|---|-------------|--------------|-------|
| A | Per-day rank norm on regime-sensitive features | ✅ INCLUDED (Section 6 Layer 3) | hardens against vol-regime overfit |
| B | Side-aware loss weighting | ⚠️ DEFERRED to v3.1 | start fixed λ, add 1.5× short weight if short heads underperform |
| C | PatchTST as auxiliary INPUT | ⚠️ DEFERRED to v3.2 | requires PatchTST inference at every train sample = expensive precompute step |
| D | Vol regime as conditioning input | ✅ INCLUDED via smart_v3 features | smart_v3 already encodes vol percentile |
| E | Distillation warm-start from v2 | ✅ INCLUDED (Section 7 warm-start) | load v2 backbone weights |
| F | Fill-rate as separate head | ✅ PARTIAL via hit_tp heads | hit_tp+filled pair gives effective fill metric; explicit fill prob head can be added in v3.1 |
| G | Test-time augmentation / drop-feature ablation | ✅ POST-TRAIN | scripted as separate ablation run after fold 9 |
| H | Calibration of probability heads | ✅ POST-TRAIN (Section 8 #5) | isotonic regression per fold |
| I | Per-fold per-feature norm (TRAIN ONLY) | ✅ MANDATORY (Section 6 Layer 2) | HC #278(B) compliance |
| J | OOT label leakage audit | ✅ MANDATORY pre-train check | verify no train events overlap OOT timestamps |
| K | Multi-target Huber loss for FIFO heads | ✅ INCLUDED (Section 2) | robust to occasional ±20t outliers |
| L | bf16 mixed precision (vs fp16) | ✅ INCLUDED (Section 7) | better stability for regression heads |

---

## 10. RISKS / KNOWN ISSUES

1. **MLflow artifact cross-host bug** (HC #281(H)) — Jupiter MLflow server has artifact root pointing at Jupiter local path; Neptune writes will fail. Mitigation: skip artifact upload, log metrics-only. Outputs persist on Neptune disk + rsync back at end.
2. **Long-side FIFO labels may be sparser than short** — short edge dominates per HC #275(A). Trainer must mask unfilled rows to avoid biasing toward "0 ticks" prediction.
3. **Warm-start risk** — if v2 backbone is too optimized for IC-only, joint loss may degrade IC during v3 fine-tune. Mitigation: λ schedule with high IC weight in epoch 0-1, ramp up FIFO weights in epoch 2-4.
4. **Live deployment requires v3 to ship with both per-fold feature stats AND head architecture metadata.** Inference manifest spec separate (post-train).

---

## 11. DELIVERABLES THIS SESSION

1. ✅ This spec doc (DURABLE, ALL FUTURE SESSIONS)
2. ⏳ `scripts/fifo_label_generator_v3.py` — produces `mbo_events_smart_v3_fifo_labels/<date>_fifo_labels.npz` per date
3. ⏳ `alpha_discovery/deep_models/train_cnn_mamba_v3.py` — multi-head trainer with all v3 changes
4. ⏳ `scripts/launch_cnn_mamba_v3_neptune.sh` — Neptune launcher with watchdog + correct MLflow URI
5. ⏳ Sync to Neptune `/home/nick/Lvl3Quant/`
6. ⏳ Launch + verify MLflow run materializes
7. ⏳ Update SESSION_STATE.md + RUN_HISTORY.md
8. ⏳ Discord report

---

## 12. CHANGE LOG

- 2026-05-10 11:25 ET: Initial spec written under HC #281 authorization. Approved live.
