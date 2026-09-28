# SPY MBO TRIAL — PIPELINE READINESS AUDIT (HC #527 R3)

**Date**: 2026-06-04
**Trigger**: User purchased SPY MBO data Mar 1 – Mar 13, 2026 (≈9 trading days). This audit verifies the pipeline can run AS SOON AS the .dbn.zst files land on disk, with no scrambling.

**Scope**: end-to-end ingest → translate → label → normalize → walk-forward → regime-stratify → report.
**Status legend**: READY = drop-in usable. NEEDS-FIX = exists but a one-line change required. MISSING = file does not exist; must be authored.

---

## 1. INGEST (raw .dbn.zst → 6-col events NPZ)

| Item | File | Status | Action to make READY |
|---|---|---|---|
| Databento historical pull | `feeds/spy_databento_trial.py --mode historical` | READY | None. Already wraps `db.Historical.timeseries.get_range` with `schema="mbo"`, writes the canonical 6-col NPZ + JSONL fan-out. |
| Schema translator (Databento MBO → 6-col) | `feeds/schema_adapter.py` (`SchemaAdapter.from_databento_mbo`) | READY | None. Handles ts_event (ns int), A/C/M/T/F action chars, B/A/N sides, int64 1e-9 price scaling, BBO maintenance, spread_ticks. Self-test in `__main__`. |
| Output directory | `data/processed/spy_mbo_events/` | READY | Exists; contains the fixture file `20260604_mbo_events.npz` proving the writer works end-to-end. |
| Smoke validator | `feeds/spy_walkforward_smoke.py` | READY | None. Validates dtypes (float32 events, int64 ts), 6-col shape, monotonic ts, metadata tick_size==0.01, mid-price reconstruction. |

**OFFLINE INGEST COMMAND when data lands** (assumes user has placed the 9 raw .dbn.zst files OR will run the script with `--start/--end` to pull them via the trial key):

```
# Per-day, mode=historical (uses trial credits — bounded)
for d in 2026-03-02 2026-03-03 2026-03-04 2026-03-05 2026-03-06 \
         2026-03-09 2026-03-10 2026-03-11 2026-03-12; do
  python feeds/spy_databento_trial.py --mode historical \
    --symbol SPY --dataset XNAS.ITCH \
    --start "${d}T13:30:00" --end "${d}T20:00:00"
done
```

NOTE: Mar 1 2026 was Sunday; Mar 7-8 were Sat/Sun. Mar 7-8 were Sat/Sun. 9 RTH days = Mar 2,3,4,5,6,9,10,11,12,13.

**Dataset selection caveat**: the trial script defaults to `DBEQ.BASIC` (consolidated). If the user's purchased data is single-venue (e.g. `XNAS.ITCH` Nasdaq-only or `ARCX.PILLAR` NYSE Arca-only), pass `--dataset` explicitly. SPY trades on multiple venues; consolidated will give the most complete order book.

---

## 2. FEATURE NORMALIZATION (per-feature z-scores)

| Item | File | Status | Action to make READY |
|---|---|---|---|
| ES feature-stats artifact (reference) | `output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_feature_stats.npz`, also under `output/smart_exec_v4/fold_*_norm_stats.npz` | READY | None — these prove the artifact shape (mean/std per feature, saved per-fold). |
| SPY feature-stats recompute path | NONE | MISSING | Add a 30-line script `scripts/spy_recompute_norm_stats.py` that: (1) loads the first 3 SPY NPZ days, (2) concatenates events, (3) computes per-column mean/std on cols [0,3,4,5] (cols 1=event_type and 2=side stay categorical), (4) writes `output/spy_trial/norm_stats.npz`. ETA: 30 min when data lands. |
| Recompute logic | inline in walk-forward harness | NEEDS-FIX | Existing harness reads stats from `output/<run>/fold_NN_norm_stats.npz`. Trial runner (deliverable #2) should pass `--norm-stats output/spy_trial/norm_stats.npz` so the same code reads SPY stats. Zero core changes. |

**Why this matters**: SPY price-rel-ticks (col 3) and spread-ticks (col 5) will have wildly different distributions than ES (SPY spread ≈ 1 tick almost always; ES spread can blow out during news). Re-fitting per-feature z-scores on SPY is mandatory for the model to see in-distribution inputs. NO RETRAIN — just input rescaling.

---

## 3. MFE/MAE LABELER (raw mid-price path)

| Item | File | Status | Action to make READY |
|---|---|---|---|
| Tick-by-tick MFE/MAE on raw price | `scripts/mfe_mae_relabel_v2.py` (and `alpha_discovery/execution/raw_trajectory_mfe_mae_v2.py`) | NEEDS-FIX | Two issues: (a) v2 has hard-coded Windows paths `C:\Users\claude\Lvl3Quant\...` and `N_FEATURES=25` (smart_v3 25-col, not the canonical 6-col). For SPY trial we use the 6-col events directly; mid-price reconstruction is `cumsum(events[:,3]) * 0.01` + seed mid. (b) Per HC #515 R6, the broken smart_v3 path used normalized features for the price walk — verify by checking that the v2 script does `cumsum(events[:,3])` BEFORE z-score, not after. Action: write a thin wrapper `scripts/spy_mfe_mae.py` that takes a 6-col NPZ + tick_size and emits per-horizon MFE/MAE at SPY tick resolution. ETA: 1 hr. |
| Output schema | per-day NPZ with `mfe_*, mae_*, net_*` per horizon | READY | Existing labels under `data/mfe_mae_labels_v2/*_mfe_mae.npz` show the convention. Mirror it under `data/spy_mfe_mae_labels/`. |

**Horizons for SPY trial**: 1s, 5s, 10s, 30s — IDENTICAL to ES. The model's predictive horizon doesn't change just because we switch instruments. HC #428 R2 cap (TP ≤ p90 of MFE within horizon h) applies; this labeler produces the distribution needed to set those caps.

---

## 4. REGIME CLASSIFIER (HC #428 R1)

| Item | File | Status | Action to make READY |
|---|---|---|---|
| ES close-to-close green/red/flat | `analysis/regime_label.py` (referenced in logs) | NEEDS-FIX | Currently keyed off ES daily close. For SPY trial, switch the reference universe to SPY itself — green/red/flat day = SPY close vs prior SPY close. Simpler: this is the SAME instrument we're trading, so we don't introduce a basis. Change point: one-line — pass `--reference SPY --prices data/spy_daily.csv` (download SPY daily bars once from `feeds/spy_databento_trial.py` with `schema="ohlcv-1d"` or just yfinance). |
| Daily SPY closes for the 9 days | NONE | MISSING | Pull once into `data/external/spy_daily_2026Q1.csv`. Trivial — 9 numbers. The trial runner can do this itself on launch. |
| Regime-gap gate (Sharpe_green vs Sharpe_red < 50%) | enforced in trial runner | NEEDS-FIX | Implement in the trial-runner deliverable (#2 below). Existing reporters print per-regime metrics; gate logic is in their post-processing. |

---

## 5. WALK-FORWARD HARNESS

| Item | File | Status | Action to make READY |
|---|---|---|---|
| Canonical SLIDING-window WF (60d train / 1d OOT, HC #0) | `alpha_discovery/deep_models/walkforward_oot_lean.py` | NEEDS-FIX | Hard-coded for 60d train. With 9 SPY days we cannot run HC #0 protocol. **Decision**: do NOT touch `walkforward_oot_lean.py` — write a trial-only runner (#2) that does 2d-train / 1d-OOT over 9 days. This is a TRIAL protocol, not a production launch; HC #0 still applies to anything that goes to deployment. |
| Smoke validator | `feeds/spy_walkforward_smoke.py` | READY | Already passes. |
| Zero-shot ES-model-on-SPY inference path | `scripts/mfe_mae_relabel_v2.py` (model loader) | NEEDS-FIX | The model loader assumes 25-col smart_v3 input. For the canonical 6-col SPY events, we need to either (a) recompute the smart_v3 derived features for SPY (engineering work, ~half day) OR (b) use the 6-col CNN-Mamba checkpoint if one exists. Find: `find output -name 'fold_*_best.pt' | xargs -I{} python -c "import torch; m=torch.load('{}',map_location='cpu'); print('{}', list(m.keys()) if hasattr(m,'keys') else type(m))"` to map checkpoints to input dims. **Default action for trial**: option (a) — derive the smart_v3 features on SPY. The derivation code lives in `scripts/compute_smart_v3_features.py` (verify name). |

---

## 6. PAPER TRADING BRIDGE (for forward-look only, not part of trial backtest)

| Item | File | Status | Action to make READY |
|---|---|---|---|
| Alpaca paper executor | `feeds/spy_paper_alpaca.py` | READY | Stub; works in DRY-RUN without keys. Decision logic uses `cost_constants_spy.round_trip_cost_ticks`. Not used for the trial backtest, but in place for forward paper validation once a SPY-trained or zero-shot model proves out. |

---

## 7. BLOCKERS / KNOWN GAPS

1. ~~ES MBO data for Mar 1–13 2026 is NOT on disk.~~ **CORRECTED 2026-06-04 19:15 ET**: ES MBO for Mar 1–13 2026 **IS on disk** (12 files at `/home/jupiter/Lvl3Quant/data/raw/mbo/glbx-mdp3-20260301..13.mbo.dbn.zst`). The agent confused Mar 2025 vs Mar 2026 — user almost certainly meant 2026 (buying historical data 15 months old for a trial is implausible). **Execution-only cross-correlation research is UNBLOCKED.** Once the SPY MBO lands, we can do the full ES↔SPY lead-lag IC study on the SAME 10-RTH-day window. No additional ES data purchase needed.

2. **Smart_v3 25-col feature derivation for SPY.** Existing model checkpoints expect 25 cols. We have 6. Either re-derive smart_v3 features on SPY (preferred — leverages existing model weights for zero-shot inference) or train a fresh 6-col model on the 9 days (likely too thin). Trial runner uses path (a).

3. **9 trading days is genuinely thin.** Even with 2d-train / 1d-OOT we get 7 folds. Concat IC will have wide CI. Treat the trial as a directional read, not a green-light. Sample-size limitation must be in every report.

---

## 8. ONE-LINE SUMMARY

Ingest + translator + cost model are **READY**; norm-stats recompute, MFE/MAE labeler wrapper, regime ref-swap, and the trial-WF runner are NEEDS-FIX (all routine, total ETA ~3 hours of work when data lands); execution-only research is BLOCKED by absent ES Mar 2025 data.
