# CNN-Mamba v3.2.1 Priority-1 Build Status

**Date**: 2026-05-12 18:10 ET
**Built by**: autonomous session (per HC #299 deliverable + HC #300A no-permission-asking)
**Source spec**: `docs/cnn_mamba_v3.2.1_design_memo.md` (HC #299)
**Source audit**: `docs/cnn_mamba_v3.2_normalization_audit_20260512.md` (HC #298)

## Status: ✅ PRIORITY-1 CODE COMPLETE, py_compile OK, instantiation + forward-pass smoke tested

NOT YET DONE (downstream): parquet rebuild on Jupiter, full smoke test on Neptune with real data, training launch. These are gated on (a) Jupiter T2/T3 v3.2 backfill completing, (b) v3.2 fold 0 Ep 1 OOT verdict.

---

## Files

| Path | Purpose | LOC | md5 |
|---|---|---|---|
| `alpha_discovery/features/build_v3_2_1_tier_features.py` | T2+T3 feature builder with HC #298 normalization + 5 missing T2 + LGBM vol + session phase | 1406 (+605 vs v3.2) | `697011e60e6ba832419d11ac4d282e0f` |
| `alpha_discovery/deep_models/train_cnn_mamba_v3_2_1.py` | Trainer with `CNNMambaV321` (event-type embedding + ZSCORE_EXCLUDE mask) | 1981 (+91 vs v3.2) | `021c2d00cd5c52f5ff68a89c8294ae37` |

Both pass `python -m py_compile`. Both pass `importlib.util.spec_from_file_location` smoke load.

---

## Priority-1 Item Coverage

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | Apply 19 HC #298 normalization fixes | ✅ DONE | Builder writes log1p for counts, sign-preserved log-z for signed_volume, secs/23400, dist/100. Dataloader applies `T3_ZSCORE_APPLY_MASK` (excludes 24/31 T3 cols from z-score). |
| 2 | Restore 5 missing T2 features | ✅ DONE | `microprice_change_ticks`, `avg_spread_in_bucket_ticks`, `n_top_of_book_changes`, `avg_top5_depth_volume`, `large_order_count`. T2 = 19 features. Builder reconstructs L2 book state from MBO add/cancel/trade events. |
| 3 | Add LGBM vol predictions as T3 features | ✅ DONE | `lgbm_vol_pred_5min`, `lgbm_vol_pred_30min` columns added. Builder loads `output/vol_lgbm_v3/*_models.pkl`. T3 = 31 features. |
| 4 | Embed event-type as 8-dim learnable in T1 | ✅ DONE | `nn.Embedding(8, 8)` in `CNNMambaV321`. Forward extracts col 0 from T1 raw input, embeds, concats with cols 1..38. Adapter resized: `Linear(46 -> 25)`. Identity init for the 24 continuous event cols preserves v3 warmstart compatibility (load_v3_warmstart unchanged). |
| 5 | Add session-phase one-hot to T3 | ✅ DONE | `phase_open` / `phase_morning` / `phase_lunch` / `phase_afternoon` (4 binary cols). Existing `is_lunch_lull` / `is_close_hour` kept for backward compat. |

---

## Forward-pass smoke test

```
CNNMambaV321 instantiated OK
  Total params: 756,267
  event_type_emb: torch.Size([8, 8])
  t1_adapter: in=46, out=25
  t2_adapter: in=19, out=25
  t3_adapter: in=31, out=25
  Forward pass OK, 32 heads, sample shape: torch.Size([2])
```

Dummy batch: B=2, T1=(2,1500,39) with event_type IDs in [0,7], T2=(2,1500,19), T3=(2,1500,31). All 32 heads emit valid scalar outputs per example.

---

## What is NOT done (deferred to next phase)

1. **Parquet rebuild on Jupiter** — `build_v3_2_1_tier_features.py` needs to run for 60+ training dates. Jupiter is currently busy with v3.2 T2/T3 backfill (PID 4041655, on date 2025-10-22, ~1h remaining). Once that finishes, dispatch v3.2.1 builder to backfill on Jupiter. ETA: 8-12h CPU work, can run overnight.
2. **Full smoke test on Neptune with real data** — only synthetic-tensor forward pass verified. Real parquet ingest path needs validation once Step 1 produces actual data.
3. **Training launch on Neptune** — gated on v3.2 fold 0 Ep 1 OOT verdict (ETA ~21:30 ET tonight). If v3.2 passes gate, v3.2.1 may not launch (continue v3.2 fold 1+). If v3.2 fails gate, v3.2.1 launches immediately.
4. **`load_v3_2_warmstart` method** — current `load_v3_warmstart` only handles v3 ckpts. A `load_v3_2_warmstart` for chaining v3.2 → v3.2.1 would let v3.2.1 inherit T2/T3 backbone weights from a completed v3.2 fold. Not strictly required (random init also works) but would save ~1 epoch of training. Add when v3.2 fold completes.
5. **Priority-2 items (memo §3 P2)** — cross-asset (NQ/VIX), macro calendar, FiLM fusion, T3 architecture split. These are bigger lifts deferred until v3.2.1 baseline is validated.

---

## Architectural decisions made during build

1. **Option A path (separate v3.2.1 files)** chosen over flag-toggled v3.2 extension. Justification: v3.2 in production (fold 0 active on Neptune), clean delta, easy rollback.
2. **Output dirs disjoint from v3.2** to prevent clobber: `data/processed/tier{2,3}_*_v3_2_1/`, `output/cnn_mamba_v3_2_1_long_context/`.
3. **MLflow experiment**: `CNNMamba_v3_2_1_long_context` (new).
4. **N_T1 split** — raw input still 39-d (dataloader untouched), POST-EMBED 46-d (only inside model). `t1_adapter` widened from 39→25 to 46→25; identity init preserved for 24 continuous event cols (backbone warmstart still works); 8 new embed cols + 12 extras (4 PT + 10 book — note v3.2 had 13 extras but I count 22 here actually) get small random projection. v3 warmstart still applies via existing `load_v3_warmstart`.
5. **T3_ZSCORE_APPLY_MASK** — precomputed boolean array (31,) computed at module load time. True count = 7 (only the 5 path-memory features + 2 LGBM vol pred get z-scored; the other 24 are passthrough per HC #298).
6. **HC #300 + #301 ckpt+resume code preserved verbatim** — cache_size=1, gc.collect after eviction, RNG ByteTensor cast, 500-batch full-state intra-ckpt, absolute-path resume. Same OOM-resistance as v3.2.

---

## Next actions (when v3.2 verdict lands)

If v3.2 fails gate at ~21:30 ET:
```bash
# Step 1: rebuild T2/T3 parquets for training dates (on Jupiter)
cd /home/jupiter/Lvl3Quant
python3 -m alpha_discovery.features.build_v3_2_1_tier_features \
    --start-date 2025-12-01 --end-date 2026-04-29 \
    > logs/build_v3_2_1_$(date +%Y%m%d_%H%M).log 2>&1 &

# Step 2: launch trainer on Neptune (after parquet rebuild)
ssh nick@neptune "cd /home/nick/Lvl3Quant && \
    V32_BATCH_SIZE=64 V32_EPOCHS=5 PYTHONPATH=/home/nick/Lvl3Quant \
    nohup python3 -u alpha_discovery/deep_models/train_cnn_mamba_v3_2_1.py \
        --data-dir /home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3 \
        --fifo-label-dir /home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3_fifo_labels \
        --alpha-label-dir /home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3_alpha_labels \
        --pt-pred-dir /home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3_pt_pred \
        --tier2-dir /home/nick/Lvl3Quant/data/processed/tier2_orderflow_v3_2_1 \
        --tier3-dir /home/nick/Lvl3Quant/data/processed/tier3_session_v3_2_1 \
        --output-dir /home/nick/Lvl3Quant/output/cnn_mamba_v3_2_1_long_context \
        --warmstart-ckpt /home/nick/Lvl3Quant/output/cnn_mamba_v3_smart_v3_fifo/fold_00_best.pt \
        --n-folds 1 > logs/cnn_mamba_v3_2_1_$(date +%Y%m%d_%H%M).log 2>&1 &"
```

If v3.2 passes gate at ~21:30 ET:
- v3.2.1 code sits unused on disk. No compute waste.
- Continue v3.2 Ep 2-5 + dispatch fold 1.
- v3.2.1 launch deferred to v3.2 fold 1 verdict (24h+) or later iteration.
