# HC #477 R3 — Alpha Label Generator Diagnosis

**Date**: 2026-05-21
**Author**: alpha-redev kickoff agent (Opus 4)
**Status**: DIAGNOSIS COMPLETE — fix spec ready for user GO/NO-GO
**Prerequisite for**: HC #475 R4 alpha redev

---

## Section 1 — Files involved

| Role | Path |
|------|------|
| Label generator (producer) | `/home/jupiter/Lvl3Quant/scripts/generate_v3_1_alpha_labels.py` |
| Output dir (124 NPZs) | `/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_alpha_labels/` |
| Consumer (trainer) | `/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/train_cnn_mamba_v3_2.py` |
| Downstream OOT NPZ (where bug surfaces) | `/home/jupiter/Lvl3Quant/output/hc432_v342_47day_validation/fold_00_ep1_oot_inference_47day_hc432.npz` |
| HC #477 audit | `/home/jupiter/Lvl3Quant/scripts/hc477_alpha_label_audit.py` + `reports/hc477_alpha_label_audit/summary.md` |

---

## Section 2 — Root cause

**ONE root cause, two failure modes:**

### Root cause (single)
The label generator `generate_v3_1_alpha_labels.py` is **scoped to a 30-second forward window only**. It does not compute any horizon longer than 30s. Specifically:

- Module-level horizon constants defined: `WIN_NS_5S`, `WIN_NS_10S`, `WIN_NS_15S`, `WIN_NS_30S` (lines 45-48). **NO 60S, NO 5MIN.**
- Two-pointer index computation: `j_5s`, `j_10s`, `j_15s`, `j_30s` (lines 251-254). **NO `j_60s`, NO `j_5min`.**
- MFE/MAE windowed deque computed once, against `j_30s` (lines 283, 285). **No 60s MFE/MAE pass.**
- Realized vol uses fixed W=30 seconds (line 184). **No 60s vol.**
- `np.savez_compressed(out_path, ...)` (lines 326-338) writes exactly these keys:
  ```
  log_ret_30s
  p_up_5s, p_up_10s, p_up_30s
  mfe_30s_ticks, mae_30s_ticks
  time_to_mfe_secs
  p_reversal_15s, p_reversal_30s
  realized_vol_30s_ticks
  ```
  **NO `log_ret_60s`, NO `log_ret_5min`, NO `mfe_60s_ticks`, NO `mae_60s_ticks`, NO `p_reversal_60s`, NO `p_up_60s`.**

This matches the audit summary perfectly: all 124 files have `log_ret_60s` and `log_ret_5min` flagged MISSING (not ALL_ZERO).

### Failure mode 1 — silent zero-fill in OOT NPZ (visible bug)
The trainer (`train_cnn_mamba_v3_2.py` lines 1102-1114, 1130-1148, 1158-1164) gracefully handles missing keys via `_alpha_at()` returning `np.nan`, then writes:
```python
targets["log_ret_60s"] = np.float32(0.0); masks["log_ret_60s"] = np.float32(0.0)
```
The downstream 47-day OOT NPZ stores **target=0.0 for every row, mask=0 for every row** for: `log_ret_60s`, `log_ret_5min`, `pred_mfe_60s_ticks`, `pred_mae_60s_ticks`, `p_reversal_60s`, `p_up_60s`.

When HC #475 diagnostics read `target_log_ret_60s` *without* applying `mask_log_ret_60s`, the column appears as 900K zeros, exactly as reported. When mask IS applied (`y = y[m]`), the array is empty — equally useless, but at least not misleading.

### Failure mode 2 — silent training degradation (hidden bug, larger blast radius)
Since masks are all-zero for these heads, the training loss for `log_ret_60s` / `log_ret_5min` / `mfe_60s` / `mae_60s` / `p_reversal_60s` / `p_up_60s` is multiplied by zero **for every sample on every day**. Effective loss contribution = 0. These heads receive **no gradient signal** and produce **untrained outputs**.

The `LOSS_LAMBDA` dict (lines 224-230) assigns lambdas of 1.0, 1.0, 0.5, 0.5, etc. to these heads, but mask-zero short-circuits everything. The model has been advertising 6+ heads it never actually learned.

This explains HC #475 R3 finding: "60s only 0.4% positive" — that's not signal, that's the model emitting near-constant noise around its initialization bias, then quantile-thresholded confluence rules read meaningless predictions.

---

## Section 3 — Severity assessment

| Aspect | Affected? | Notes |
|--------|-----------|-------|
| 30s targets | NO | `log_ret_30s` flagged VALID in 118/124 files (6 ALL_NAN = legitimate short-session days) |
| 1s / 5s / 10s targets | NO (in trainer) | Trainer sources these from `ev["labels_1s/5s/10s"]` in upstream `mbo_events_smart_v3`, not from alpha NPZ. Audit script flagged them MISSING because it only looked at alpha NPZ. **No actual problem.** |
| 30s MFE/MAE | NO | Generated correctly, VALID |
| **60s MFE/MAE** | **YES — never produced** | Trainer gets `np.nan`, masks out, head never trains |
| **log_ret_60s** | **YES — never produced** | Same as above |
| **log_ret_5min** | **YES — never produced** | Same as above |
| **p_reversal_60s** | **YES — never produced** | Same as above |
| **p_up_60s** | **YES — never produced** | Same as above |
| Train NPZs vs OOT NPZs | BOTH affected | Same generator output feeds both training folds and OOT inference. Every v3.2/v3.3/v3.4/v3.4.2 model trained against this directory has the same 6 dead heads. |
| Quantile heads (`log_ret_60s_q10/q50/q90`) | YES — derived from dead head | Quantile heads use the 60s regression head as basis; if base is untrained, quantiles are noise too |
| MLflow run integrity | Compromised for any run logging "60s IC" or "5min IC" metrics | Those metrics are computed on near-constant model outputs, not real predictions |
| Live inference (`v3_4_2_inference.py`) | DEGRADED if it uses 60s/5min heads downstream | Need to audit which production policies read these heads |

**Severity = HIGH.** Six heads of the multi-head model have been training as no-ops since v3.1 was deployed (timeline of the alpha-label dir). Any confluence rule, FIFO sweep, or RL policy that reads `pred_log_ret_60s`, `pred_log_ret_5min`, `pred_mfe_60s_ticks`, etc., has been consuming noise. **This must be fixed before HC #475 R4 alpha redev** — otherwise the redev re-trains 6 heads on the same broken pipeline and produces the same useless outputs.

---

## Section 4 — Fix specification

### 4A — Code change (DO NOT IMPLEMENT WITHOUT USER GO)

**File**: `/home/jupiter/Lvl3Quant/scripts/generate_v3_1_alpha_labels.py`

**Change 1 — add horizon constants (lines 45-48)**
```python
# BEFORE
WIN_NS_5S  = 5  * SEC_NS
WIN_NS_10S = 10 * SEC_NS
WIN_NS_15S = 15 * SEC_NS
WIN_NS_30S = 30 * SEC_NS

# AFTER
WIN_NS_5S    = 5   * SEC_NS
WIN_NS_10S   = 10  * SEC_NS
WIN_NS_15S   = 15  * SEC_NS
WIN_NS_30S   = 30  * SEC_NS
WIN_NS_60S   = 60  * SEC_NS
WIN_NS_5MIN  = 300 * SEC_NS
```

**Change 2 — add 60s/5min two-pointer indices (insert after line 254)**
```python
j_60s  = _two_pointer_indices(ts_ns, WIN_NS_60S)
j_5min = _two_pointer_indices(ts_ns, WIN_NS_5MIN)
```

**Change 3 — compute `log_ret_60s` and `log_ret_5min` (insert after line 265)**

Follow the same fallback pattern used for 30s at lines 261-265:
```python
log_ret_60s = np.full(N, np.nan, dtype=np.float32)
valid60 = j_60s < N
idx60 = np.where(valid60)[0]
log_ret_60s[idx60] = (mids_clean[j_60s[idx60]] - mids_clean[idx60]).astype(np.float32)

log_ret_5min = np.full(N, np.nan, dtype=np.float32)
valid5m = j_5min < N
idx5m = np.where(valid5m)[0]
log_ret_5min[idx5m] = (mids_clean[j_5min[idx5m]] - mids_clean[idx5m]).astype(np.float32)
```
Note: `j_60s < N` for the last ~60s of each day will be False (end-of-day), so those rows correctly stay NaN. The trainer's `_alpha_at` → `np.nan` → `mask=0` path then correctly skips them — no zero-fill leak.

**Change 4 — compute 60s MFE/MAE (insert after line 295)**
```python
_, max_in_win_60s, argmax_in_win_60s = _compute_min_max_windowed(
    mids_for_max, j_60s, return_argmax=True)
min_proper_60s, _ = _compute_min_max_windowed(mids_for_min, j_60s, return_argmax=False)

mfe_60s = max_in_win_60s - mids_clean
mae_60s = min_proper_60s - mids_clean
mfe_60s = np.where(np.isnan(mids_clean) | np.isinf(mfe_60s), np.nan, mfe_60s).astype(np.float32)
mae_60s = np.where(np.isnan(mids_clean) | np.isinf(mae_60s), np.nan, mae_60s).astype(np.float32)
```

**Change 5 — add `p_up_60s` (insert near line 270)**
```python
p_up_60s = (log_ret_60s > 0).astype(np.float32)
p_up_60s[np.isnan(log_ret_60s)] = np.nan
```

**Change 6 — add `p_reversal_60s` (insert near line 319)**
```python
reversal_60s = ((min_proper_60s < mids_clean) & (max_in_win_60s > mids_clean)).astype(np.float32)
reversal_60s[np.isnan(mids_clean) | np.isinf(min_proper_60s) | np.isinf(max_in_win_60s)] = np.nan
```

**Change 7 — update savez (lines 326-338)** — add the 6 new keys:
```python
log_ret_60s=log_ret_60s,
log_ret_5min=log_ret_5min,
p_up_60s=p_up_60s,
mfe_60s_ticks=mfe_60s,
mae_60s_ticks=mae_60s,
p_reversal_60s=reversal_60s,
```

**Total LOC delta**: ~30 lines added, 0 removed, 1 schema bump (output NPZ key set grows by 6).

### 4B — Dependent artifacts requiring regeneration

| Artifact | Action | Count |
|----------|--------|-------|
| Alpha label NPZs | Regenerate ALL with `--overwrite` | 124 files |
| Training fold checkpoints (v3.2/v3.3/v3.4/v3.4.2) | Optional: retrain from scratch to recover the 6 dead heads. OR mark current checkpoints as "5-head valid, 6-head dead" and only use the validated heads in production policies. | ~20 fold ckpts |
| OOT inference NPZs (hc432_v342_47day_validation/) | Regenerate after retrain | 1-5 files |
| HC #471 / HC #432 / HC #475 FIFO sweep results | Re-run after OOT regen | several reports |
| MLflow runs touting 60s / 5min IC | Mark deprecated in MLflow tags | ~50 runs (audit needed) |

### 4C — Regeneration cost estimate

- **Label-gen runtime**: process_one_date is dominated by the numba windowed deque + 1s vol cumsum. Adding 60s + 5min windows roughly doubles deque work (longer windows = more elements pushed per signal). Estimate: **~2x current per-file runtime**.
- **Current per-file runtime** (rough estimate from script structure + 33MB outputs for ~7K-event days): order of 30-90 seconds per day on a single core.
- **124 files × ~2 minutes each ÷ N workers** = on Jupiter (use `--workers 8`): **~30-45 minutes wall clock** to regenerate all alpha labels.
- **Retraining v3.4.2 on regenerated labels**: separate decision, NOT required for HC #475 R4 alpha redev (which retrains from scratch anyway). If retraining: ~24-48 hrs on Razer/Neptune per the historical training logs.
- **OOT re-inference**: ~1-2 hrs after retrain.

**Bottom line: 30-45 min of pure data-prep compute on Jupiter unblocks HC #475 R4.** Retraining is a separate downstream cost.

### 4D — Risk callouts

1. **Backward compatibility**: any code reading the alpha-label NPZ via `d.files` and iterating keys will see 6 new keys. Should be safe since consumers use explicit `d[key]` access, not `d.files` iteration. But: HC #477 audit script itself reads explicit keys, so it will need re-running to confirm fix landed. (NOT a code change — just rerun the script.)
2. **No schema-version bump needed in NPZ filename** (no version embedded). Recommend: bump output dir name to `mbo_events_smart_v3_alpha_labels_v2/` so old files are preserved for rollback and any code path still pointing to old dir errors loudly rather than silently mixing schemas. **Trainer constant `DEFAULT_ALPHA_LABEL_DIR` at line 103 of train_cnn_mamba_v3_2.py would need a one-line update.**
3. **Live trading host (Razer)**: `live_trading_linux/v3_4_2_inference.py` references `log_ret_60s` in its head set. If the live model has dead 60s heads, the live confluence rules may rely on noise. Audit live config before retraining live weights.
4. **End-of-day NaN propagation**: with 5min window, the last 5 minutes of each session day will have `j_5min == N` and produce NaN — about ~2-3% of samples per day go from "trainable on this head" to "masked out". Acceptable per HC #428 R2 (model horizon must match trade horizon — if there's no 5min forward window, there's no 5min trade).
5. **`labels_30s` source priority**: the generator currently prefers `ev["labels_30s"]` from upstream events file (line 257-258) over recomputing from mids. The 60s/5min fix should use mid-based computation only (no upstream fallback), since upstream `mbo_events_smart_v3` doesn't carry those keys either. Spec above does this correctly.
6. **Numba cache invalidation**: the `_windowed_minmax_argmax_njit` cache will be invalidated on first run after change (signature unchanged, but cache may rebuild). One-time ~10s JIT warm-up cost.

---

## Section 5 — Additional concerns found during audit

### 5.1 — Quantile heads inherit deadness
`train_cnn_mamba_v3_2.py` lines 216-221 derive `log_ret_60s_q10/q50/q90` from `log_ret_60s`. With the base head untrained, the quantile heads are also untrained. Any FIFO confluence rule using `pred_log_ret_60s_q50` (e.g., trip07/trip09 from hc475_pred_vs_label_asymmetry.py lines 260-261) is reading noise. After fix, these heads will train correctly without code changes.

### 5.2 — `mfe_30s` valid, `mfe_60s` dead — asymmetric path-head training
`pred_mfe_30s_ticks` and `pred_mae_30s_ticks` train normally (sourced from valid alpha labels). `pred_mfe_60s_ticks` and `pred_mae_60s_ticks` train as no-ops. Any execution policy using 60s MFE/MAE predictions is broken. HC #428 R2 ("MFE-within-horizon, TP ≤ p90 of realized MFE within horizon h") is impossible to enforce for any horizon > 30s today — the data simply doesn't exist.

### 5.3 — Naming consistency
Trainer expects `mfe_60s_ticks` / `mae_60s_ticks` (lines 1132-1133). Generator currently produces `mfe_30s_ticks` / `mae_30s_ticks`. Naming is consistent: `{stat}_{horizon}_ticks`. Fix spec uses the same convention. **No misnamed-key bug.** Audit `log_ret_60sec` vs `log_ret_60s` etc. — all consumers use `log_ret_60s`, no typo collisions found.

### 5.4 — Day-boundary windowing IS handled correctly for 30s
For 30s, last 30s of session day have `j_30s == N`, which `_compute_min_max_windowed` handles correctly (skip when `j_end <= i+1`), producing NaN MFE/MAE. The trainer then masks those samples. Audit confirms `log_ret_30s` ALL_NAN in 6/124 files — these are short-session days (holidays / half-days) where the entire session is < 30s after the last signal, not a windowing bug. **No day-boundary leak hypothesis confirmed for 30s; the audit-script category "ALL_ZERO" was a hypothesis that turned out not to apply here. The actual failure mode is MISSING (key never emitted), not zero-fill.**

### 5.5 — Sign-handling: nothing wrong in the label generator
HC #475 noted "15/21 heads sign-miscalibrated". Reviewed the generator: signs are computed straightforwardly as `mid[j] - mid[i]` (line 264) and `mid[k] - mid[i]` (in deque). No sign-flip bugs in label code. The 15/21 issue is in the **model** (loss-function asymmetry per HC #475 R4 sign-balanced loss recommendation), not the labels. This diagnosis does not address that — it's a separate workstream and the HC #475 R4 alpha redev is the right vehicle.

### 5.6 — `labels_1s` / `labels_5s` / `labels_10s` audit-flag is a FALSE POSITIVE
HC #477 audit script flagged these MISSING in 124/124 files because it only inspected the alpha NPZ. Trainer sources these from upstream `ev["labels_1s/5s/10s"]` in `mbo_events_smart_v3/`. Confirmed valid via trainer lines 726-728. **No fix needed**, but the audit script's interpretation column could be improved (separate issue, not blocking).

### 5.7 — `realized_vol_30s` is the only volatility output
No `realized_vol_60s_ticks` or `realized_vol_5min_ticks`. If post-fix HC #475 R4 wants volatility-conditioning at 60s/5min horizons, that needs a separate spec. Not blocking.

### 5.8 — No constant outputs detected in other columns
Per audit `per_file.jsonl`, the VALID-flagged keys (30s family) have non-zero std and reasonable min/max ranges across all 118 valid-day files. No silent zero-collapse in other columns.

---

## Section 6 — Verification plan post-fix

1. Re-run `scripts/hc477_alpha_label_audit.py` → expect MISSING column to drop to 0 for `log_ret_60s` and `log_ret_5min` (VALID column should be ~118, ALL_NAN ~6 matching short-session days).
2. Spot-check one regenerated NPZ: load `log_ret_60s`, verify std > 0, mean near 0, n_finite > 90% of N.
3. Spot-check end-of-day NaN propagation: last ~60s of signals should have `log_ret_60s == NaN`.
4. Quick sanity on a trainer mini-run: load one fold, verify `masks["log_ret_60s"]` is now 1.0 for the vast majority of samples (not all-zero as before).
5. Only after #1-4 pass: green-light HC #475 R4 alpha redev.

---

## END OF DIAGNOSIS

Awaiting user GO signal to dispatch the fix (per HC #393 / R4 — this falls in the "data-prep upstream change with cascading consumer impact" bucket, which is borderline-routine but worth one confirm given the regeneration of 124 files and downstream model retraining implications).
