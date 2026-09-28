# Label Coverage Diagnosis — MFE/MAE Relabel Parquets (`data/relabel/`)

## TL;DR

The previous report noted "MFE/MAE relabel parquets cover only ~12 min/day (~6% of prediction steps)". Root cause identified and fixed.

**Root cause**: `scripts/relabel_mfe_mae_multi_horizon.py::process_one_date()` uses the validity gate

```python
invalid = (bid <= 0) | (ask <= 0) | (ask < bid)
mids_clean[invalid] = np.nan
```

This gate is INCORRECT for the source data format. The `bid_price_1` and `ask_price_1` columns inside `data/processed/mbo_book_features/<date>_book_features.npz` are NOT absolute prices in ticks — they are **relative offsets around an unknown session anchor** that goes negative for the majority of each session. Only the (rare) window where the anchor happens to land at zero passes the `bid > 0 & ask > 0` test.

Evidence (date 20260223):

| Column | min | max | description |
|--------|-----|-----|-------------|
| bid_price_1 | -322 | 27600 | relative-offset, NOT abs price |
| ask_price_1 | -320 |    70 | relative-offset, NOT abs price |
| spread = ask - bid | 1 | (>0) | valid for 99.9% of events |

Out of 12,399,120 events on 20260223:
- `bid > 0 & ask > 0`: 770,307 (**6.2%**) — what the script accepts
- `0 < spread < 20`: 12,390,374 (**99.9%**) — what the script SHOULD accept

The "valid" window 14:59:45–15:12:12 ET is just where the relative encoding's absolute values happen to be positive. It's not a data outage — it's a coding bug in the validity filter.

## Verification of "12 minutes / day" claim

`time_to_mfe_s` is non-NaN only at event indices in the contiguous range [2,281,598, 3,070,107], a ~12-min wall-clock window — exactly the 6.2% of events that pass the buggy filter. So the report's anomaly note was correct; the underlying issue is in the relabel script's filter, not in Razer's compute or the input data.

## Correct validity gate

```python
spread = ask - bid
invalid = (spread <= 0) | (spread >= 20.0) | ~np.isfinite(spread)
```

ES futures during RTH are 1-2 ticks wide ~99% of the time. Anything outside (spread <= 0 or spread >= 20 ticks) is either a halt, a tick-anomaly, or a session boundary — exactly the events we want to skip.

## Smoke-test results (`scripts/expand_mfe_mae_full_session.py`)

Tested on dates 20260223, 20260301, 20260401 — all four horizons (1s/5s/10s/30s). Total wall time 44s on 3 workers.

| date | events | MFE-valid OLD | MFE-valid NEW | pred-step valid OLD | pred-step valid NEW |
|------|--------|---------------|---------------|---------------------|---------------------|
| 20260223 | 12,399,120 | 770,302 | **12,390,305** (16x) | 3,081 / 49,585 (6.2%) | 49,560 / 49,585 (**99.9%**) |
| 20260301 | 382,960 | 0 | 369,166 | 0 / 1,520 (0.0%) | 1,475 / 1,520 (97.0%) |
| 20260401 | 16,771,094 | 16,764,122 | 16,757,223 (~same) | 67,051 / 67,073 (100%) | 67,030 / 67,073 (99.9%) |

Observations:
- 20260223 jumped from 6.2% to 99.9% — confirming the script-bug interpretation.
- 20260301 went 0% to 97% — that date had ZERO coverage in the old data (a Globex partial day; appears to have been skipped entirely by the original script).
- 20260401 was already at 100% — at some point the original pipeline appears to have been re-run on later dates with a working version, but the early Feb–Mar dates were left in the buggy state.

## Output

- `scripts/expand_mfe_mae_full_session.py` — corrected validity gate, reuses the same O(N) monotonic-deque windowed scan from the original (imported via sys.path).
- Output dir: `data/relabel_full/` (parallel to `data/relabel/` to keep the original sparse data untouched).
- Full 32-date OOT expansion queued in background after smoke test passed.

## Why this matters for tradability

The stream-continuation harness has been forced to score P&L from the model's own `target_log_ret_*` arrays inside the NPZ rather than from the empirical MFE/MAE. With full-session relabel parquets, the next-generation stream backtest can:

- Score P&L from `mfe_within_horizon` instead of point-target log returns (catches the actual best-case move during the hold).
- Validate the time-to-MFE empirical distribution to decide whether the model's `pred_pred_time_to_mfe_secs` head is calibrated.
- Compute realistic FIFO-exit P&L (HC #464 R2b) using full-day mid trajectory.
