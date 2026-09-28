# Friday Harness Extension Plan — v3.4.2 + v3.3 Inference Channels on Razer

**Status**: NOT STARTED. Identified 2026-05-20 07:15 ET. Has been deferred across ≥4 sessions.
**Friday deadline**: 2026-05-22 EOD. Owner: next active session.
**Why this matters**: Per HC #443 R1 / HC #448 R2, the Friday deliverable is a live-data-collection harness recording v2 + v3.4.2 + v3.3 predictions side-by-side. Right now only v2 is running. Without the other two channels, the post-Friday closest-to-profit research has no live multi-model corpus.

---

## Current state (verified 2026-05-20 07:15 ET)

**On Razer (`C:\Users\claude\Lvl3Quant`):**
- MBO recorder PID 1436 — alive, recording every 5 min ✓
- `paper_trading_v2_1s_short_top05.py` PID 33424 (shadow_v2) — alive, scoring 114 signals/day, intentionally 0 fills under ultra-tight top-0.5% gate ✓
- Watchdog PID 32980 supervising ✓
- Models present: `cnn_mamba_v2_smart_v3_mar/fold_10_best.pt` + `patchtst_smart_v3_mar/fold_16_best.pt`
- **Models MISSING**: `cnn_mamba_v3_4_2_fixedmtl/`, `cnn_mamba_v3_3_uncertainty_weighted/`

**On Neptune (`/home/nick/Lvl3Quant/output/`):**
- v3.4.2 clean retrain running PID 346447, MLflow `841d8089...` — ETA ~5-6 hours from 06:52 ET, so done ~12:30 ET
- Existing v3.4.2 backups in `cnn_mamba_v3_4_2_fixedmtl/` (DO NOT use — corrupted by v3.4.3 work, wait for retrain)
- `cnn_mamba_v3_3_uncertainty_weighted/fold_00_intra_ckpt.pt` — known-good v3.3 weights ✓

**On Jupiter (`/home/jupiter/Lvl3Quant/output/`):**
- `cnn_mamba_v3_3_uncertainty_weighted/` — mirror of v3.3 weights ✓
- `cnn_mamba_v3_4_2_fixedmtl/` — check sync status (may also be corrupted; rebuild from Neptune after retrain)

---

## Execution sequence (run on the next session, ~12:30 ET when retrain completes)

### Step 1 — Verify Neptune retrain completed cleanly (~12:30 ET)

```bash
ssh nick@neptune "ls -la /home/nick/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/fold_00_intra_ckpt.pt /home/nick/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/fold_00_feature_stats.npz /home/nick/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/fold_00_ep1_oot.npz && tail -20 /home/nick/Lvl3Quant/logs/v342_clean_retrain_20260520_065140.log"
```

Expected: log ends with `>>> v3.4.2 DONE`; intra_ckpt mtime ~12:30 ET; OOT IC printed (target IC_1s ≥ 0.20).

If fail: read log for error, retry. Do not proceed until clean.

### Step 2 — scp v3.4.2 weights Neptune → Razer

```bash
# v3.4.2 checkpoint + stats
ssh nick@neptune "scp /home/nick/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/fold_00_intra_ckpt.pt /home/nick/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/fold_00_feature_stats.npz claude@razer:C:/Users/claude/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/"
```

If Razer dir doesn't exist:
```bash
ssh claude@razer "powershell -Command \"New-Item -ItemType Directory -Path 'C:\Users\claude\Lvl3Quant\output\cnn_mamba_v3_4_2_fixedmtl' -Force\""
```

### Step 3 — scp v3.3 weights Jupiter → Razer

```bash
scp /home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_intra_ckpt.pt /home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_feature_stats.npz claude@razer:C:/Users/claude/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/
```

### Step 4 — Write inference adapter modules on Razer

Pattern: copy `live_trading_linux/cnn_mamba_v2_inference.py` shim (which re-exports from `live_trading/cnn_mamba_v2_inference.py`). Create:
- `C:\Users\claude\Lvl3Quant\live_trading_linux\v3_4_2_inference.py`
- `C:\Users\claude\Lvl3Quant\live_trading_linux\v3_3_inference.py`

Each adapter must expose a class with method `predict(features_window: np.ndarray) -> dict[str, float]` returning at minimum `{pred_log_ret_1s, pred_log_ret_5s, pred_log_ret_10s, pred_log_ret_30s, conf}`.

Model classes:
- v3.4.2: `CNNMambaV341BookResidual` from `scripts/v3_4_research/dispatch_v34_2_fixedmtl.py` (loaded by importlib.spec_from_file_location — see `v342_run_oot_inference.py` lines 60-65 for the pattern)
- v3.3: `CNNMambaV33UncertaintyWeighted` from `alpha_discovery/deep_models/train_cnn_mamba_v3_3.py`

**Risk**: v3.4.2 uses dual-trunk (book features + event features); the live MBO stream may not have book-tensor features built. Check `streaming_features_smart_v3.py` for what feature channels are available live. If book features missing, v3.4.2 inference can't run live without first building the book-tensor extraction pipeline on Razer — that adds 1+ day of work.

If book features are not live-available: **FALLBACK** = log v3.4.2 prediction = NaN with a `reason: book_features_missing` field, and document the gap in the Friday report. The harness still ships with 2 model channels (v2 + v3.3) instead of 3.

### Step 5 — Modify paper_trading_v2_1s_short_top05.py

Add at module top after v2 inference instantiation:
```python
from v3_4_2_inference import V342Inference  # may fail per step 4 risk
from v3_3_inference import V33Inference
```

In the per-event hot path, after computing v2 prediction, ALSO call:
```python
pred_v33 = v33_model.predict(features_window)
pred_v342 = v342_model.predict(features_window_with_book) if book_features_available else None
```

Add to the JSONL log emit:
```python
{
  "ts_ns": ts_ns,
  "v2_pred_1s": v2_pred["pred_log_ret_1s"],
  "v33_pred_1s": pred_v33["pred_log_ret_1s"],
  "v342_pred_1s": pred_v342["pred_log_ret_1s"] if pred_v342 else None,
  "passed_gate": bool,
  "fill_outcome": "filled|cancelled|sl|tp",
  ...
}
```

**Keep gate logic v2-only.** v3.3/v3.4.2 predictions are recorded, not traded. That preserves apples-to-apples with HC #418 backtest expectation while collecting multi-model live corpus per HC #443 R2.

### Step 6 — Relaunch on Razer

Kill PID 33424 cleanly:
```bash
ssh claude@razer "powershell -Command \"Stop-Process -Id 33424 -Force\""
```

Relaunch via `spawn_shadow_v2.py` (or its equivalent — read that file for the Win32_Process.Create pattern per HC #401). The new process should pick up the modified `paper_trading_v2_1s_short_top05.py`.

Verify within 5 min that new JSONL log file in `live_trading_linux/logs/` contains all 3 prediction fields per event.

### Step 7 — Update watchdog config

The watchdog's `processes` dict (in `live_stack_watchdog.py`) currently watches `legacy_paper_top5` (dead) and `shadow_v2_1s_short_top05` (the new shadow). Remove the legacy entry; rename shadow_v2 to `harness_v2_v33_v342` so naming matches the new role.

---

## Estimated effort

- Step 1-3: 15 min (mechanical)
- Step 4 (adapter modules): 60-90 min IF book features are live-available; +day if not
- Step 5 (paper trader mod): 30-45 min
- Step 6-7: 20 min
- **Total**: 2-3 hours if no book-feature blocker; 1+ day if blocked

## Acceptance criteria

1. JSONL log in `live_trading_linux/logs/` has 3 prediction fields per event for at least 1 RTH session.
2. Watchdog status JSON shows the harness process as alive with counters advancing.
3. No regression to v2 gate behavior (still top-0.5% short, ultra-tight, near-zero fills).
4. Friday closest-to-profit report includes a "live corpus" appendix showing first-N hours of multi-model predictions with realized MFE/MAE.

## What NOT to do (anti-patterns observed in prior sessions)

- ❌ Defer to next session because "no clear GPU work available" — the harness is the work
- ❌ Speculate on v3.4.3 architecture refinements — that path is concluded
- ❌ Launch more filter/confluence sweeps on existing NPZs — HC #448 R1 closes that
- ❌ Spend context writing yet another state-of-the-project doc instead of building the harness
