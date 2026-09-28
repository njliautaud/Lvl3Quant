# V2 Inference Pipeline Audit — HC #355 Anomaly Investigation

**Date**: 2026-05-14
**Author**: Claude (Opus 4) — Jupiter CPU
**Script**: `scripts/v3_3_research/v2_inference_pipeline_audit.py`
**Output dir**: `output/v2_inference_pipeline_audit_20260514/`

---

## VERDICT: **HYPOTHESIS A — PIPELINE BUG** (high confidence)

The v2 IC collapse on Mar 6+ is **NOT** signal decay. It is a window-size mismatch in
the `cnn_mamba_v2_bulk_oot/` daily-inference pipeline that produced corrupted
predictions for every date in that folder.

**v2 alpha is intact.** CLAUDE.md / DIRECTIVES.md model-status for CNN-Mamba v2
(IC_1s=0.222 canonical) is correct. The all-OOT-dates sweep that triggered this
investigation is invalid.

## Single most important piece of evidence

```
Checkpoint   fold_10_best.pt arch.window_size = 1000     (sha256 300e338d…)
bulk_oot     n_windows=81589 metadata window_size = 3000  (Mar 6 NPZ)

Re-run inference on Mar 6 with the CHECKPOINT'S CORRECT window_size=1000:
  IC_1s = +0.216   pred mean = +0.039   pred std = 0.337     <-- matches canon
Stored bulk_oot @ window_size=3000:
  IC_1s = +0.010   pred mean = -0.111   pred std = 0.262     <-- corrupted

Pairwise pearson on identical window indices (rerun W=1000 vs stored W=3000):
  1s = -0.040   5s = -0.050   10s = -0.022                   <-- ~uncorrelated
```

## How we got here

1. `train_cnn_mamba_v2.py` default `WINDOW_SIZE = 3000`.
2. `fold_10_best.pt` was trained with the `EVENT_WINDOW_SIZE=1000` override — its
   `arch` dict records `window_size=1000`.
3. `live_trading_linux/cnn_mamba_v2_inference.py` correctly reads `arch.window_size`
   from the checkpoint. The live stack is fine.
4. The **bulk_oot generator script** (May 3) ran with `EVENT_WINDOW_SIZE=3000`
   (trainer default), ignoring `arch.window_size`. All 46 per-day NPZs (Mar 6 –
   Apr 24) were produced with this mismatch.
5. **Fold-source NPZs** (`cnn_mamba_v2_smart_v3_mar/fold_*_oot_predictions.npz`) were
   produced inside training, so window sizes match. Those are correct.

`cnn_mamba_v2_all_oot/` mixes them: fold_00..10 are symlinks to the (correct)
`smart_v3_mar/` fold predictions; 2026*_predictions.npz are symlinks into the
(broken) `cnn_mamba_v2_bulk_oot/`. That's why the all-OOT-dates sweep showed a
sharp Feb 23 → Mar 5 plateau (good) → Mar 6 cliff (bad).

## Supporting evidence (full table in `diff_table.csv`)

### Fold-source predictions (training-time OOT, GOOD)
| File | OOT date | n_rows | pred_1s mean | pred_1s std | IC_1s |
|------|----------|-------:|-------------:|------------:|------:|
| fold_07_oot_predictions.npz | 20260303 | 39602 | +0.079 | 0.367 | +0.187 |
| fold_08_oot_predictions.npz | 20260304 | 22654 | +0.021 | 0.353 | +0.265 |
| fold_09_oot_predictions.npz | 20260305 | 37189 | +0.021 | 0.368 | +0.236 |
| fold_10_oot_predictions.npz | 20260224 | 23692 | +0.022 | 0.341 | +0.303 |

### Bulk_oot predictions (per-day inference, BAD)
| Date | n_rows | pred_1s mean | pred_1s std | IC_1s |
|------|-------:|-------------:|------------:|------:|
| 20260306 | 81589 | −0.111 | 0.262 | +0.010 |
| 20260309 | 69283 | −0.112 | 0.259 | +0.003 |
| 20260310 | 65669 | −0.110 | 0.258 | +0.005 |
| 20260311 | 58565 | −0.109 | 0.259 | +0.013 |
| 20260313 | 61002 | −0.111 | 0.262 | +0.018 |
| 20260317 | 29969 | −0.121 | 0.279 | +0.020 |
| 20260320 | 65982 | −0.112 | 0.270 | +0.002 |

Mean shift +0.02 → −0.11 (systematic, NOT regime-change-like).
Std compression 0.35 → 0.26 (model output squashed).
IC 0.22-0.30 → 0.00-0.02 (≈ noise).

### Re-run inference on Mar 6 (script Step 3, 2500 sampled windows)

Using fold_10_best.pt + fold_09_feature_stats.npz (SKIP_NORMALIZE),
window_size=1000 (per checkpoint arch), stride=250, USE_DERIVED_FEATURES=False.

| Horizon | Re-run W=1000 | Stored W=3000 |
|---------|---------------|----------------|
| 1s mean / std / IC | +0.039 / 0.337 / **+0.216** | −0.111 / 0.262 / +0.010 |
| 5s mean / std / IC | +0.072 / 0.354 / +0.079 | −0.001 / 0.226 / −0.011 |
| 10s mean / std / IC | +0.044 / 0.369 / +0.048 | −0.053 / 0.292 / +0.025 |

Re-run IC_1s = +0.216 on Mar 6 (NOT a training date) confirms the model still
has alpha after Mar 5. Only the bulk_oot pipeline output is wrong.

## Recommended next action

**Delete `output/cnn_mamba_v2_bulk_oot/` and regenerate per-day predictions with
window_size read from `arch.window_size` of `fold_10_best.pt` (= 1000).** Audit
`output/patchtst_bulk_oot/` for the same bug. Any downstream code that consumed
the bulk_oot predictions (SAC/PPO/supervised-exec v4, FIFO replay/backtest,
meta-LGBM merges, v32_* analyses, v2_all_oot_profitability sweep) must be re-run.

### Proposed minimal-edit fix (NOT implemented per HC #307D)

The bulk_oot generator should read `arch.window_size` from the checkpoint. The fix
already exists in `live_trading_linux/cnn_mamba_v2_inference.py` lines 104-106:

```python
if "window_size" in arch:
    self.window_size = int(arch["window_size"])
```

Generator script was not located (may be ad-hoc / unsaved). Either recover from git
history or write a new minimal one that imitates the in-training OOT path.

## Downstream contamination

References to `cnn_mamba_v2_bulk_oot/`:
- alpha_discovery/execution/{train_sac_v7,train_ppo_v7,train_supervised_exec_v4,fifo_time_exit_backtest}.py
- scripts/{validate_via_fifo_replay,merge_meta_lgbm_features,aggregate_v2_oot_for_v3_compare}.py
- scripts/v3_3_research/v2_all_oot_profitability_sweep.py and v32_*.py family

Any quantitative claim built on these pipelines from May 3 onward is suspect.
Fold-source numbers in CLAUDE.md (IC_1s=0.222) remain valid.

## Files produced

- REPORT.md (this file)
- diff_table.csv — 57 rows
- 20260306_rerun_predictions.npz
- run.log
