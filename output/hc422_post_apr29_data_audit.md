# HC #422 Rule 3 — Post-April-29 Data Audit

**Date**: 2026-05-18
**Author**: HoQ (read-only audit)
**Binding HC**: HC #422 Rule 3 — "Anything dated after 2026-04-29 was recorded by Claude (Razer MBO recorder feed)... treat as NON-CANONICAL... do NOT use in training, only in live-inference pass-through."
**User verbatim**: *"I only downloaded data until April 29th anything after must have been recorded by you and therefore done wrong in some way."*

---

## TL;DR — **MEDIUM RISK.** Post-Apr-29 data exists and is corrupted in places, but **no current training run consumes it.** v3.4.2 (PID 311170 on Neptune) is sliding-window-bounded to ≤ 2026-04-29 because the downstream `mbo_book_features/`, `mbo_events_smart_v3_fifo_labels/`, `mbo_events_smart_v3_alpha_labels/`, and `mbo_events_smart_v3_pt_pred/` directories all stop at 2026-04-29 and `discover_aligned_dates()` intersects them. Live inference does pass through this data — that path is contaminated but already known to be regime-drifting (HC #421 Issue A) which would manifest the same way regardless of recorder quality. Recommendation: quarantine post-Apr-29 `mbo_events/`, gate any future training launches with an explicit `--max-date 20260429` until the user does the next bulk download.

---

## 1. Inventory: Post-Apr-29 Data on Jupiter

### 1a. Raw bulk DBN files (`/data/raw/mbo/glbx-mdp3-*.mbo.dbn.zst`)
- **Last file: `glbx-mdp3-20260429.mbo.dbn.zst` (84 MB, mtime Apr 29 23:40 ET).**
- Spans 2025-07-14 through **2026-04-29**. No post-Apr-29 raw files. ✅ User-claim verified — bulk download stopped Apr-29 as stated.

### 1b. Derived feature dirs (clean — all stop Apr-29)
| Directory | Last date | Used by |
|---|---|---|
| `data/processed/mbo_events_smart_v3/` | 2026-04-28 (+`_corrupt_backup_20260518/`) | v3.4.x training (25-col feature events) |
| `data/processed/mbo_book_features/` | 2026-04-29 | v3.4.x dual-trunk training (BookCNN trunk) |
| `data/processed/mbo_events_smart_v3_fifo_labels/` | 2026-04-29 | v3.4.x FIFO heads (forbidden by HC #422 Rule 2) |
| `data/processed/mbo_events_smart_v3_alpha_labels/` | 2026-04-27 | v3.4.x directional heads |
| `data/processed/mbo_events_smart_v3_pt_pred/` | 2026-04-29 | v3.4.x confluence head (PatchTST pred aligned) |

### 1c. Suspect dir: `data/processed/mbo_events/` (raw 6-col events)
| Date | Size | n_events | symbol | source | UTC range | Risk note |
|---|---:|---:|---|---|---|---|
| 20260427 | 70 MB | bulk-derived | (no symbol key) | (no source key) | full session | clean (older schema generation, batch-processed Apr 30 02:08 ET) |
| 20260428 | 79 MB | bulk-derived | (no symbol key) | (no source key) | full session | clean (same batch) |
| **20260429** | **84 MB** | **1,750,561** | **NQM6** | **mbo_recorder_live** | **04:37 → 23:59 UTC, 19.4 h** | **⚠ FIRST LIVE-RECORDED day. Symbol claims NQ futures (NQM6) not ES (ESM6). Either the live recorder default was misconfigured that day, or this file was overwritten by recorder testing on Apr 29 17:00 ET — file mtime is 2026-04-30 17:00 ET (15 hours AFTER its supposed last timestamp).** |
| 20260430 | 132 MB | 2,744,459 | ESM6 | mbo_recorder_live | 23:59 → 23:56 UTC, **23.94 h** | rotation bug — file starts the previous day's last second and ends 3 min before midnight of the following day; ~24h overlap window is too wide |
| 20260501 | 239 MB | 4,988,411 | ESM6 | mbo_recorder_live | 23:56 → 20:57 UTC, 21.0 h | overlapping start ts with 20260430 file |
| **20260503** | **0 B** | **— (CORRUPT)** | — | — | — | **np.load fails: "No data left in file." savez() crashed mid-write. Discard.** |
| 20260504 | 106 MB | — | ESM6 | mbo_recorder_live | — | not deeply inspected; size suggests partial session |
| 20260505 | 77 MB | — | ESM6 | mbo_recorder_live | — | small — partial session |
| 20260506 | 10 MB | — | ESM6 | mbo_recorder_live | — | TINY — <1h of data, likely brief connection then disconnect |
| 20260508 | 126 MB + **127 MB `.tmp`** | 2,632,656 | ESM6 | mbo_recorder_live | 15:49 → 20:30 UTC, **4.67 h ONLY** | recorder crashed mid-day. The `.tmp` companion (`20260508_mbo_events.npz.tmp`) is a stalled write that was never finalized. |
| 20260507, 20260509-12 | — | — | — | — | — | **MISSING — gaps in the record.** |
| 20260513 | 34 MB | — | ESM6 | mbo_recorder_live | — | partial session |
| 20260514 | 121 MB | — | ESM6 | mbo_recorder_live | — | likely full session |
| 20260515 | 123 MB | 2,559,765 | ESM6 | mbo_recorder_live | full | likely full session |
| 20260516, 20260517, 20260518 | — | — | — | — | — | **MISSING — recorder offline last 3 days** (consistent with HC #422 Rule 1 watchdog gap — heartbeat alarms were absent so a silent recorder death went undetected) |

**Aggregate**: 1.10 GB of post-Apr-29 `mbo_events/` data, 11 files, 7+ days with anomalies (1 corrupt 0-byte, 1 stalled `.tmp`, 5 missing days within the May 1-15 window, 3 missing trailing days, several partial sessions).

### 1d. Other nodes
- **Neptune** (`/home/nick/Lvl3Quant/data/...`): inventory pending. Neptune is the training node, so verifying that `mbo_book_features/` there is also bounded to 04-29 would conclusively prove v3.4.2 cannot pull post-Apr-29 data. From the dispatch source on Jupiter (mirror of Neptune's `scripts/`), `discover_aligned_dates` intersects events ∩ book_features — so v3.4.2 is safe **as long as Neptune's book_features mirrors Jupiter's**. Recommend a one-line SSH check post-audit: `ls /home/nick/Lvl3Quant/data/processed/mbo_book_features/ | tail -5`.
- **Razer** (`C:\Users\claude\Lvl3Quant\...`): is the recorder SOURCE. `mbo_recorder.log` exists there. Not directly readable from Jupiter; live recorder writes to BOTH the Razer local path AND the Jupiter NPZ path (which is what we audited above).

---

## 2. Does any training run CURRENTLY consume post-Apr-29 data?

**No (in the strict sense), but the boundary is fragile.**

### v3.4.2 (Neptune PID 311170, fold-0, mid-ep4)
- Dispatch (`scripts/v3_4_research/dispatch_v34_1_residual.py`, structurally identical for v3.4.2): line 198-210, `discover_aligned_dates()` returns `sorted(event_dates & book_dates)`.
- Aligned dates ⊆ `mbo_book_features/` dates → last possible date = **20260429**.
- HC #422 Rule 3 boundary HONORED by data-dir construction. ✅
- ⚠ Sole post-Apr-29 contamination vector: the **20260429 file's `symbol=NQM6` anomaly**. If the recorder genuinely subscribed to NQ on that day (instead of overwriting the bulk-derived 04-29 file), the training run consumed 1 day of NQ data labeled as ES. Mitigation check below.

### v2 (current champion, 19 days stale per HC #421 Issue A)
- Trained through 2026-04-19 (HC #421 Issue A regime-drift diagnosis: live pred mean +0.270 vs training-cutoff baseline +0.024).
- Pre-Apr-29 only. ✅

### v3.3 (deployed for execution-research lane, HC #422 Rule 5)
- Inference NPZs were generated from the same `mbo_events_smart_v3/` (last date 04-28). Pre-Apr-29 only. ✅

### Smart-execution research (HC #422 Rule 5/8)
- All input NPZs are model prediction outputs (`output/v342_fold_00_ep1_oot_inference*.npz`, `output/hc417_v2_full_oot_56d.npz`, etc.) — they post-date the data but reflect pre-Apr-29 training distributions. Pre-Apr-29 alpha only. ✅

### Live inference pass-through (HC #421 shadow PID 29600 on Razer)
- Reads the Rithmic live feed direct (not the NPZ files), pushes preds through `StreamingFeaturesSmartV3`. **This is the only HC #422 Rule 3 *intended* use** — "live-inference pass-through" is explicitly permitted.
- The +0.270 regime-drift (HC #421 Issue A) is observable in THIS path and is the symptom that motivated HC #422 in the first place. Whether the drift is regime change OR live-encoding divergence vs offline is unresolved — but is independent of post-Apr-29 NPZ quality.

---

## 3. Schema Diff — Bulk-derived vs Live-recorder NPZ

`mbo_events/20260429_mbo_events.npz` (live-recorder, NQM6, "first contaminated") vs `mbo_events/20260515_mbo_events.npz` (live-recorder, ESM6):

| Key | Bulk (20260428) | Live (20260429, NQM6) | Live (20260515, ESM6) |
|---|---|---|---|
| Keys present | `events, timestamps, labels_1s, labels_5s, labels_10s, labels_30s, metadata` | identical | identical |
| `events` dtype/shape | float32 (N, 6) | float32 (N, 6) | float32 (N, 6) |
| `timestamps` dtype | int64 (nanoseconds UTC) | int64 (nanoseconds UTC) | int64 (nanoseconds UTC) |
| `feature_names` (in metadata) | [time_delta_log, event_type_id, side_id, price_rel_ticks, qty_log, spread_ticks] | identical | identical |
| `tick_size` | 0.25 | 0.25 | 0.25 |
| `labels_1s/5s/10s/30s` | all NaN (labels computed downstream) | all NaN | all NaN |
| `metadata["source"]` key | **ABSENT** | `"mbo_recorder_live"` | `"mbo_recorder_live"` |
| `metadata["symbol"]` key | **ABSENT** | `"NQM6"` ⚠ | `"ESM6"` |
| ts range | full session ~20h | 19.4h (Apr 29 04:37 → 23:59 UTC) | 21.0h (full Globex session) |
| recorder rotation | bulk N/A | one-shot, single date | one-shot, single date |

**Schema-wise** the live recorder output IS bit-for-bit compatible with the bulk-derived format (same dtypes, columns, encoding tables). The encoder in `live_trading_linux/mbo_recorder.py:93-100` produces exactly the 6-col layout the trainer expects. **The metadata `source`/`symbol` keys are additive only** — they do not break readers, but they REVEAL whether the file came from live recording (untrusted) or bulk parsing (trusted).

**The risk is NOT schema mismatch — it is recorder operational integrity**:
1. **Symbol misconfiguration** (20260429: NQM6). Recorder default in code is `--symbol ESM6` (line 246) — but Apr-29 file claims NQM6. Either the cmdline was overridden that day, OR the file was overwritten by an NQ test run AFTER the bulk-derive batch wrote the canonical ES file. mtime 2026-04-30 17:00 ET (a full day after Apr 29) supports the overwrite hypothesis.
2. **WebSocket reconnect gaps** (`mbo_recorder.py:215-220`): on reconnect <30s, applies exponential backoff up to 120s — events DURING that backoff window are dropped silently. No gap-marker is written to the NPZ.
3. **Daily rotation collision**: `flush()` uses `datetime.now(timezone.utc)` to compute the filename. If the recorder is running across UTC midnight, events get split across two dates with timestamps that span both (explaining the 20260430 file's 23.94h range starting 23:59 UTC of the prior day).
4. **Partial writes**: 20260503 (0 bytes) and 20260508 (`.tmp` companion) prove `np.savez` was killed mid-call. Discard both.
5. **Missing days**: 20260502 (Sat, expected), 20260507, 20260509-12, 20260516-18 — recorder crashes that were undetected because there was NO heartbeat alarm (HC #422 Rule 1 was created in response to this gap).
6. **Clock drift**: not directly measurable from the file (recorder uses `ev.ssboe*1e9 + ev.usecs*1000` from the Rithmic server timestamp, not local clock) — so clock drift IS mitigated by design. ✅
7. **Field truncation/rounding**: `price_rel_ticks` is computed as `(price - mid_price) / TICK_SIZE` in float32 — same as bulk pipeline. No silent unit conversion. ✅

---

## 4. Verdict

| Risk dimension | Status | Evidence |
|---|---|---|
| **HIGH-risk** (training run depends on bad data) | ❌ NO | v3.4.2 training is `discover_aligned_dates`-bounded to ≤2026-04-29 by the `mbo_book_features/` boundary. |
| **MEDIUM-risk** (data exists, is suspect, could be picked up by future runs) | ✅ YES | 1.1 GB of post-Apr-29 NPZ files sit in `data/processed/mbo_events/`. A future training script that doesn't intersect with `book_features` (e.g., a pure-events MLP gate trainer) would pick them up. The 20260429 NQM6-symbol file is the one DATE INSIDE the canonical training window that has a non-bulk source. |
| **LOW-risk** (recorded fine, schema clean, user concern unfounded) | ⚠ PARTIAL | Schema matches bit-for-bit. Operational quality is poor (corrupt files, missing days, reconnect gaps, daily-rotation overlap). User concern is **substantially correct**: the recorder output IS done wrong in several ways. |

**Overall verdict: MEDIUM risk.** The user's instinct is right — recorder data has multiple defects — but the existing alignment-discovery code keeps it out of v3.4.2 today. The exposure is to FUTURE scripts that bypass the book-features intersection, and to the **20260429 NQM6-symbol file** which is INSIDE the training window.

---

## 5. Concrete Fix Recommendations

1. **Quarantine post-Apr-29 `mbo_events/` files immediately** (NOT delete — preserve for forensic re-analysis):
   ```bash
   mkdir -p /home/jupiter/Lvl3Quant/data/processed/_quarantine_hc422_post_apr29/
   mv /home/jupiter/Lvl3Quant/data/processed/mbo_events/{20260430,20260501,20260503,20260504,20260505,20260506,20260508,20260513,20260514,20260515}_mbo_events.npz* /home/jupiter/Lvl3Quant/data/processed/_quarantine_hc422_post_apr29/
   ```
   Add a `QUARANTINE_README.md` citing HC #422 Rule 3.

2. **Verify the 20260429 NQM6-symbol file integrity**. Two action paths:
   a. If user can re-export the bulk Apr-29 ES file from Databento, OVERWRITE the current 20260429 file. (Preferred.)
   b. Otherwise, drop 20260429 from the v3.4.x training window via `--max-date 20260428` (lose 1 train day from the 60d slide; negligible IC impact per HC #344 weekly-retrain norms).

3. **Drop the corrupt 20260503 (0 B) and 20260508.npz.tmp files** outright (move to quarantine). They cannot be loaded.

4. **Add a hard-floor guard to all future dispatch scripts**:
   ```python
   MAX_TRAIN_DATE = "20260429"  # HC #422 Rule 3 boundary
   aligned = [d for d in aligned if d <= MAX_TRAIN_DATE]
   ```
   This belongs in `scripts/v3_4_research/dispatch_*` and any new execution-research training scripts. It enforces the rule regardless of future data-dir contents.

5. **Trigger a user bulk-download for 2026-04-30 → 2026-05-17** (12 trading days). This re-establishes canonical data covering the entire post-Apr-29 gap and unblocks HC #421's v2 retrain on current regime (mean drift +0.027 → +0.270 motivates the retrain).

6. **Fix the recorder for post-audit operations** (separate from HC #422 Rule 1 watchdog work):
   - Add `--symbol` to the log header on every flush so the file's intended symbol is auditable.
   - Write a `gap_log` array into the NPZ recording every reconnect (gap_start_ns, gap_end_ns, reason). Trainers can then filter or interpolate.
   - Use `np.savez` to a `.tmp` then atomic rename — never directly to the final path. Eliminates corrupt-file class.
   - Rotate at exchange close (21:00 UTC for ES Globex) not UTC midnight, to avoid the 23.94h overlap pattern.

7. **Verify Neptune mirror.** One SSH check: `ls /home/nick/Lvl3Quant/data/processed/mbo_book_features/ | sort | tail -3` should return `20260427_book_features.npz, 20260428_book_features.npz, 20260429_book_features.npz` — if it returns anything later, v3.4.2 IS pulling post-Apr-29 data and risk escalates to HIGH.

---

## 6. Files Referenced
- `/home/jupiter/Lvl3Quant/data/raw/mbo/glbx-mdp3-20260429.mbo.dbn.zst` (last bulk file)
- `/home/jupiter/Lvl3Quant/data/processed/mbo_events/202604{29,30}_mbo_events.npz` (Apr-29/30 boundary files)
- `/home/jupiter/Lvl3Quant/data/processed/mbo_events/2026050{3,8}_mbo_events.npz*` (corrupt files)
- `/home/jupiter/Lvl3Quant/data/processed/mbo_book_features/` (clean boundary at 20260429)
- `/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_{fifo,alpha}_labels/`, `mbo_events_smart_v3_pt_pred/` (all clean ≤04-29)
- `/home/jupiter/Lvl3Quant/live_trading_linux/mbo_recorder.py` (recorder source, defects in §3)
- `/home/jupiter/Lvl3Quant/scripts/v3_4_research/dispatch_v34_1_residual.py:198-210` (`discover_aligned_dates` — the safety net keeping v3.4.2 clean)
- `/home/jupiter/Lvl3Quant/output/hc422_v342_go_nogo_analysis.md` (parallel HC #422 Rule 4 analysis)
- `/home/jupiter/Lvl3Quant/DIRECTIVES.md` HC #422 Rule 3 (binding rule)

**Audit performed read-only.** No data files modified. No training processes touched.
