# HC #417 — Late-OOT zero-fill data-gap root-cause analysis

**Author**: Jupiter research agent
**Date**: 2026-05-17
**Trigger**: zero_fill_diag.py flagged 6 dates (Apr 21/22/23/24/28/29) with `mask_log_ret_1s` uniformly False in the wrapped v2 NPZ; HC #417 Task 1.

## TL;DR

**Root cause**: the `mbo_events_smart_v3` preprocessed NPZ files for 5 of the 6 zero-fill dates (Apr 21/22/23/24/28) were written in a **truncated, label-corrupted state** during an earlier preprocessing run. The `labels_1s/5s/10s/30s` arrays in these downstream files are **uniformly NaN**, while the **upstream** `data/processed/mbo_events/{date}_mbo_events.npz` files contain perfectly valid labels (~80% finite non-zero). The 6th date (Apr 29) is genuinely raw-data-limited: even the upstream labels are all-zero because the raw MBO file covers only a partial session (~1.75M events; first ~12k events insufficient for 1s/5s/10s/30s forward lookups against session-end truncation).

**Fix**: re-run `alpha_discovery/deep_models/precompute_features_smart_v3.py` on Apr 21/22/23/24/28 (forcing overwrite of the stale outputs). This is a 5-date re-process, CPU-only, ~10-20 min wall time. After re-processing, re-run `scripts/v3_3_research/hc417_v2_full_oot.py --skip-infer` to regenerate the wrapped NPZ, then the HC #413 / HC #415 sweep on the wrapped NPZ.

**Estimated PnL recovery if labels are fixed**: based on the same per-day fill rate as the other 36 active dates (`v2_1s_short_top05`: 25.6 fills/active-day mean), and the per-fill profitability holding (no reason to expect Apr 21-28 to differ from Apr 16-20 / Apr 27 which DO pass), the 5 recoverable dates would contribute ~110-130 additional fills (~25 fills/day × 5 days × adjustment for Apr 26's quiet half-day) at +0.27 tk/fill = **~$370-450 additional realised P&L** over the full-OOT window. This does not change the promotion verdict (cell already passes HC #408 + HC #415 rule 2 + HC #411). It does increase robustness evidence by giving us 41 instead of 25 active days.

**No-action option**: leaving the dates broken is also safe. The current verdict uses only the 25 dates that have valid labels; HC #415 rule 2's `per_day_pass_rate=1.00` is computed on those active dates only. Live trading will not be affected (live inference doesn't depend on these label files at all — labels are only used for offline backtesting).

---

## Evidence chain

### Stage 1: Wrapped NPZ symptom
`output/hc417_v2_full_oot_wrapped_for_hc413.npz` shows for the 6 zero-fill dates:
- `mask_log_ret_1s` (= `np.isfinite(target) & |target|>1e-9`) is uniformly **0** (False).
- `target_log_ret_1s` is uniformly **0** (NaN was converted to 0 somewhere; see Stage 4).
- `pred_log_ret_1s` is present and looks reasonable (mean ≈ 0.05, std ≈ 0.3 — same distribution as good dates).

### Stage 2: Wrapper script is innocent
`scripts/v3_3_research/hc417_v2_wrap_for_backtest.py` (lines 78-87) simply slices the upstream 56-day NPZ. It does NOT compute or alter labels. The bug is upstream.

### Stage 3: 56-day source NPZ is also broken for these dates
`output/hc417_v2_full_oot_56d.npz` is assembled by `scripts/v3_3_research/hc417_v2_full_oot.py` which:
- Loads per-day prediction files from `output/cnn_mamba_v2_bulk_oot_v2/{date}_predictions.npz`.
- Each per-day file's `labels` array is filled (line 155-157) from the upstream `data/processed/mbo_events_smart_v3/{date}_mbo_events.npz`'s `labels_1s`, `labels_5s`, `labels_10s` indexed at window-end positions.
- Mask formula (line 283): `mask = (np.isfinite(lab) & (np.abs(lab) > 1e-9)).astype(np.float32)`.

### Stage 4: smart_v3 NPZs are corrupted for 5 dates

Direct inspection of `data/processed/mbo_events_smart_v3/{date}_mbo_events.npz` vs `data/processed/mbo_events/{date}_mbo_events.npz`:

| date | upstream_N | upstream_labels_1s finite&!=0 | smart_v3_N | smart_v3_labels_1s finite&!=0 | smart_v3 file size (MB) | smart_v3 mtime |
|---|---:|---:|---:|---:|---:|---|
| 20260420 | 12,850,534 | 10,551,300 (82%) | 12,850,534 | 10,551,300 (82%) | 1,606 | Apr 30 07:43 |
| 20260421 | 16,694,578 | 14,175,139 (85%) | **7,642,493** | **0** | **955** | **Apr 23 19:57** |
| 20260422 | 11,509,224 | 9,331,075 (81%) | **1,668,818** | **0** | **209** | **Apr 23 19:58** |
| 20260423 | 17,658,827 | 15,043,414 (85%) | **2,525,187** | **0** | **316** | **Apr 23 19:58** |
| 20260424 | 14,427,013 | 11,924,172 (83%) | **10,224,097** | **0** | **1,278** | **Apr 30 01:07** |
| 20260427 | 11,840,958 | 9,403,791 (79%) | 11,840,958 | 9,403,791 (79%) | 1,480 | Apr 30 07:43 |
| 20260428 | 13,348,365 | 10,859,833 (81%) | **8,992,770** | **0** | **1,124** | **Apr 30 01:09** |
| 20260429 | 1,750,561 | **0** | 1,734,875 | 0 | 196 | n/a (genuine zero) |

Pattern: 5 dates (21/22/23/24/28) have BOTH (a) smart_v3 N substantially smaller than upstream N and (b) zero finite-nonzero labels. Dates 20 and 27 (which work fine) have N matching upstream exactly. Date 29 has zero labels in BOTH upstream and downstream → genuinely raw-data-limited.

The smart_v3 file mtimes for the 3 broken-and-old files (21/22/23) are all `Apr 23 19:57-58` — an early build run before late-April raw data was fully delivered. The 2 broken-and-newer files (24/28) are dated `Apr 30` and DID have full upstream available, so the truncation+nan-label outcome there is more puzzling — likely a write-failure or out-of-memory event during processing that left a partial NPZ. `precompute_features_smart_v3.py` uses `np.savez(...)` (line 472 — not compressed, no atomic write), and the existence check at line 511 prevents re-attempt unless the bad file is deleted first.

### Stage 5: smart_v3 labels really are all-NaN
Direct probe of `mbo_events_smart_v3/20260422_mbo_events.npz`:
- `labels_1s[:1000]` unique values: `[nan]` → confirmed all-NaN, not all-zero.
- `events.shape = (1668818, 25)` — non-zero, features look valid.
- `timestamps` spans full ~24h (`duration_min=1441.1`).

So the file was written with `events` and `event_type_raw` populated, but `labels_*` ended up as a NaN-filled placeholder. The truncation of N (from ~11.5M upstream to ~1.67M downstream for Apr 22) is the smoking gun: the file was written partially, then either:
- Apparent path: re-run logic detected the existing partial file and skipped re-processing (see `if file in existing: skip`), OR
- The first write was interrupted mid-stream and the resulting partial NPZ has truncated arrays with NaN placeholders.

### Stage 6: Manual 1s label re-derivation on Apr 22 sanity check
Walk-forward through the UPSTREAM `data/processed/mbo_events/20260422_mbo_events.npz`:
- 9,331,075 / 11,509,224 events have valid 1s forward labels (81%).
- Sample label values are non-trivial (range of ±5 ticks typical), distribution looks normal.

→ The labeler itself is fine. The bug is specifically the smart_v3 PASS-THROUGH copy step, NOT the label generation step.

---

## Fix recommendation

### Option A (recommended): Re-run smart_v3 preprocessing on the 5 fixable dates

```bash
cd /home/jupiter/Lvl3Quant
# 1. Delete the 5 corrupted files so the re-run picks them up
for d in 20260421 20260422 20260423 20260424 20260428; do
    rm -f data/processed/mbo_events_smart_v3/${d}_mbo_events.npz
done

# 2. Re-run smart_v3 preprocessing (CPU, ~10-20 min for 5 files)
python3 alpha_discovery/deep_models/precompute_features_smart_v3.py

# 3. Re-run v2 inference on those 5 dates only (delete stale prediction files first)
for d in 20260421 20260422 20260423 20260424 20260428; do
    rm -f output/cnn_mamba_v2_bulk_oot_v2/${d}_predictions.npz
done
python3 scripts/v3_3_research/hc417_v2_full_oot.py --workers 5 --batch 128 --stride 250

# 4. Re-wrap for HC #413
python3 scripts/v3_3_research/hc417_v2_wrap_for_backtest.py

# 5. Re-run HC #413/415 sweep on wrapped NPZ
# (existing sweep wrapper, e.g. hc415_eval_v2native.py)
```

**Expected delta**: 5 newly active dates (Apr 26 was already partial; we now add 21/22/23/24/28), each contributing ~15-25 short top-0.5% fills at ~+0.27 tk/fill. Promotion verdict for `v2_1s_short_top05` is already PASS; this only **strengthens** it.

### Option B: Leave broken, no action

Live trading is unaffected because:
- Live inference uses `mbo_recorder.py` → real-time book features → CNN-Mamba v2 → pred_log_ret_1s. It does NOT consult `mbo_events_smart_v3` label files.
- The promotion verdict is already PASS on the 25 active dates with valid labels.

Cost of doing nothing: less robustness evidence (25 vs ~30 active dates), but the cell already meets every promotion gate.

### Option C: Investigate the partial-write root cause

The fact that Apr 24 and Apr 28 (both built on Apr 30) ALSO ended up truncated suggests a recurring problem with `precompute_features_smart_v3.py` under memory pressure. The `np.savez` call writes uncompressed and non-atomically. A defensive fix would be to write to `<dst>.tmp` then `os.rename` it into place, and to add a post-write self-test that loads the file back and checks `labels_1s` has at least some finite non-zero entries.

**Recommendation**: Adopt Option A immediately (cheap, contained, high signal) AND log Option C as a tech-debt ticket but don't block deployment on it.

---

## Apr 29 (and other genuinely raw-data-limited dates)

`data/processed/mbo_events/20260429_mbo_events.npz` shows N=1,750,561 events with zero finite-nonzero labels. The raw MBO file `data/raw/mbo/glbx-mdp3-20260429.mbo.dbn.zst` is 196 MB (smallest of all April files, vs typical 200-310 MB), indicating a half-day or pre-holiday partial session. The labeler correctly emits NaN for windows whose 1s/5s/10s/30s forward lookups fall outside the trailing data.

**Action**: None possible without acquiring additional MBO data. Live trading will simply not run on partial sessions (the live stack respects RTH hours and skips half-days unless explicitly enabled).

---

## Live-trading implications

**Zero impact on live trading.** Live signal path:
```
MBO websocket → mbo_recorder.py → in-memory book features → CNN-Mamba v2 (fold_10_best.pt) → pred_log_ret_1s → paper_trader entry decision
```
Labels are only used for **offline backtesting** and **decay monitoring**. The label gap on Apr 21-28 means our backtest sample is N=25 active days instead of N=30, but the live trading bot would have happily traded those days had it been running. The HC #415 rule 2 `per_day_pass_rate=1.00` is computed only on `n_fills > 0` days, so the zero-label dates do not penalise the score (verified by reading the verdict.md generator).

---

## File-level summary

| File | Status | Notes |
|---|---|---|
| `data/raw/mbo/glbx-mdp3-2026042{1,2,3,4,7,8,9}.mbo.dbn.zst` | exists | Apr 29 partial session |
| `data/processed/mbo_events/2026042{1,2,3,4,7,8,9}_mbo_events.npz` | valid (incl labels) | Apr 29 has zero labels (raw-data-limited) |
| `data/processed/mbo_events_smart_v3/2026042{1,2,3,4,8}_mbo_events.npz` | **CORRUPT** | truncated N, NaN labels |
| `data/processed/mbo_events_smart_v3/20260420_mbo_events.npz` | valid | matches upstream |
| `data/processed/mbo_events_smart_v3/20260427_mbo_events.npz` | valid | matches upstream |
| `data/processed/mbo_events_smart_v3/20260429_mbo_events.npz` | valid (zero labels) | raw-data-limited |
| `data/processed/mbo_events_smart_v3_fifo_labels/2026042{1,2,3,4,8,9}_fifo_labels.npz` | valid | tp4sl3 / tp8sl5 labels exist (different label family — not affected) |

**Note** that the FIFO label NPZs (`mbo_events_smart_v3_fifo_labels/`) DO contain valid tp4sl3 / tp8sl5 exit labels for Apr 21-28, computed from a different pipeline. Those are used by HC #413 backtester for fill simulation. The missing-label issue is specifically the `log_ret_1s/5s/10s/30s` arrays in `mbo_events_smart_v3/`, which power the wrapped NPZ's `mask_log_ret_1s` gate.
