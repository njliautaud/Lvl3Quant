# Harness v3 Risk Report — Friday Deliverable (HC #448 R2)

**Date**: 2026-05-22
**Author**: Claude (head-of-quant)
**Scope**: Resolve risk on v3.4.2 live-feed feasibility + draft v3.3 inference adapter

---

## Task A — Are v3.4.2's book-tensor features live-available on Razer?

**VERDICT: BLOCKED**. v3.4.2 cannot ship on the live harness Friday. The
gap is larger than just `book_pyramid`: the entire v3.x dual-trunk input
contract is not produced by the current live feature pipeline.

### Evidence

- `live_trading/streaming_features_smart_v3.py:80` declares
  `N_FEATURES = 25`. The module emits exactly **25 raw event features per
  MBO event** — the same set that fed CNN-Mamba v2. No other channel is
  emitted by the live builder.
- The v3.2/v3.3/v3.4.2 model input contract requires **three event tiers**
  (`train_cnn_mamba_v3_2.py:170-173`):
  `N_T1_FEATURES = 39` (= 25 event + 4 PatchTST preds + 10 book-history),
  `N_T2_FEATURES = 14` (per-100ms order flow),
  `N_T3_FEATURES = 25` (per-1Hz session context).
  T2 and T3 have **no live producer** anywhere in `live_trading/` (grep of
  `events_t1|events_t2|events_t3|N_T1_FEATURES|N_PT_FEATURES|N_BOOK_HISTORY`
  across the whole directory returns no matches).
- v3.4.2 additionally requires `batch["book_pyramid"]` of shape
  `(B, T_book, 5_levels, 4_features)`
  (`dispatch_v34_2_fixedmtl.py:321`,
  `train_cnn_mamba_v3_4.py:131-132,292`). The offline OOT path builds this
  from `DEFAULT_BOOK_FEATURES_DIR` parquet via
  `SmartV34DualTrunkDataset` / `collate_v34`
  (`dispatch_v34_2_fixedmtl.py:93-94`, `v342_run_oot_inference.py:64-73`).
  None of that machinery exists on Razer — only `lgbm_book_inference.py`
  references "book" at all, and it uses a different feature schema.
- The live paper trader still imports the legacy `cnn_mamba_v2_inference.py`
  (the v3 adapter shim files exist on Razer but are not yet wired in).

### Gap, quantified

| Channel             | Required (per sample)               | Live produces        | Gap                 |
| ------------------- | ----------------------------------- | -------------------- | ------------------- |
| events_t1           | (1500, 39)                          | (1500, 25) only      | 14 cols (PT + book-history) |
| events_t2           | (1500, 14)                          | not produced         | full                |
| events_t3           | (1500, 25)                          | not produced         | full                |
| book_pyramid (v3.4.2 only) | (T_book, 5, 4)               | not produced         | full                |

Soft-degrade (zero-padding everything missing) is possible — the v3.3
adapter does it — but predictions become calibrated for cold-start
conditions only. Not a production path.

---

## Task B — v3.3 inference adapter

**Drafted**: `/home/jupiter/Lvl3Quant/staging/harness_v3/v3_3_inference.py`
**LOC**: 270 lines
**Status**: DRAFT — passes structural review against
`train_cnn_mamba_v3_2.CNNMambaV32`. Not yet runtime-tested against a real
ckpt.

### Design decisions

- Reuses `CNNMambaV32` directly (v3.3 only swapped the loss, model class
  is identical).
- Accepts both a structured dict (`events_t1`/`events_t2`/`events_t3`) AND
  a flat `(L, 25)` numpy array (live `streaming_features_smart_v3` output)
  with zero-pad fallback for the missing T1/T2/T3 columns.
- Z-scores per tier using `feature_stats.npz` keys
  `mean_t1`/`std_t1`/`mean_t2`/`std_t2`/`mean_t3`/`std_t3` when present.
- Exposes only the four HC #428 R2-valid heads
  (`pred_log_ret_{1s,5s,10s,30s}`) as trade signals; surfaces `60s`/`5min`
  under a `diag_` prefix for monitoring.
- Returns NaN + `reason` on shape mismatch or forward failure (matches
  the v2 adapter convention).

### Key risks

1. **The (L, 25) fallback is unvalidated**. Backtest-equivalent
   predictions only come from the structured-dict path. Until live
   builds T2/T3 + PT-pred + book-history columns, v3.3 in the harness
   runs in soft-degrade mode.
2. **Razer already has a near-identical `v3_3_inference.py`** at
   `C:\Users\claude\Lvl3Quant\live_trading\v3_3_inference.py`. Confirm
   which is canonical before deploying — the staging draft adds the
   25-column fallback path that Razer's copy lacks.
3. **`feature_stats.npz` key contract** depends on the dataset's
   `SKIP_NORMALIZE` flag at train time. Adapter handles missing keys
   gracefully (pass-through) but predictions will be uncalibrated.

---

## Friday harness path — recommendation

**SHIP 2 CHANNELS**: v2 (production) + v3.3 (soft-degrade, shadow only).

Rationale: v3.4.2 needs the book_pyramid pipeline; v3.3 needs T2/T3 +
PT-pred + book-history columns. Both gaps are pre-existing pipeline work,
not Friday work. Trying to land the full v3 input contract on Razer
today risks corrupting the live v2 stack on the same day. The conservative
path:

1. Wire the staging `v3_3_inference.py` into the harness as a
   **shadow-only** channel (predictions logged, NOT routed to orders).
2. Keep v2 as the sole trading channel for the weekend.
3. File a follow-on for next week: extend `streaming_features_smart_v3`
   to emit the missing 14 T1 columns + T2 + T3 + book_pyramid. That
   unblocks BOTH v3.3 (full-fidelity) and v3.4.2.

**Estimated remaining effort for ship-2-channels path**: 2-3 hours
(adapter test on Jupiter against a real v3.3 ckpt; scp to Razer;
shadow-mode wiring in `paper_trading_*` runner; verify NaN-on-startup
behavior; 1 RTH session validation).

**Estimated effort for ship-3-channels path**: 1-2 days minimum
(implement live T2/T3 producers, wire book_features stream into Razer's
recorder, regression-test against offline OOT). Not safe to attempt
Friday.
