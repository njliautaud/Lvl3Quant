# Spec — 2026-05-06 Post-Close Rebuild

**Goal:** Apply every fix mandated by HCs #223–#231 in one coordinated rebuild. After-hours window. Redeploy Razer before next session open. Restart Neptune Split DQN with corrected rewards.

**Non-negotiable principle:** Nothing is "fixed" without verification. No claim of done unless the verification step passes.

---

## A. Razer paper_trader fixes

### A.1 Long-only directional bias [HC #223, #231(D)] — BLOCKING

**Evidence:** Today's signal log shows 296 signals, 296 LONG, 0 SHORT. `pred_1s` distribution: min=+0.192, max=+2.312, mean=+0.948, n_negative=0/296. Model never produced a negative output all day.

**Diagnosis** (in progress via subagent): three hypotheses — feature normalization broken on Razer, inference post-processing bug, or model checkpoint genuinely long-biased. RCA report due before patches.

**Fix path** depends on diagnosis outcome:
- If **normalization bug**: rebuild Razer feature pipeline to use training-time scaler (saved to disk during training); validate by piping today's MBO through Jupiter and checking pred dist symmetry.
- If **post-processing bug**: patch the offending code in `paper_trading_mamba_v2_patched.py` or upstream inference module.
- If **model bias**: select alternate fold checkpoint (compare folds 0-11 prediction balance on validation), or retrain with class-balance loss term — but retraining is multi-day, NOT in tonight's scope.

**Verification:** After fix, run on yesterday's tape (2026-04-29 MBO) — pred_1s should show roughly balanced distribution (n_neg ≈ n_pos within ±20%) AND signals should split L/S roughly proportionally.

### A.2 Max hold time data-driven [HC #226, #231(C)] — HIGH PRIORITY

**Current:** `paper_trading_mamba_v2_patched.py:791` — `--max-hold-minutes default=15.0` → 900 seconds. Used at lines 548, 602.

**Data:** From `/home/jupiter/Lvl3Quant/output/mfe_mae_horizon_analysis/mfe_mae_horizon_results.json`:
- Signal horizons predicted: 1s, 5s, 10s
- MFE growth (overall direction-balanced, all signals):
  - @ 5s: mean 1.69 ticks, p90 5.0
  - @ 10s: mean 2.63 ticks, p90 7.0
  - @ 30s: mean 4.45 ticks, p90 11.0
  - @ 60s: mean 8.27 ticks, p90 22.5

MFE keeps growing because price walks; the question is when SIGNAL-DRIVEN edge stops dominating. From signal_decay_analysis 2026-05-01: edge persists ~30s.

**Decision:** `max_hold_seconds = 60` (down from 900). Justification:
- 6× the 10s signal horizon — generous buffer
- p90 of 10s-MFE is 7 ticks; by 60s most signal-driven move is captured
- Hard backstop only — primary exits should be MFE-revert, signal-decay, trailing-stop

**Patch:** Change line 791 default from `15.0` (minutes) to `1.0` (= 60s). Keep `--max-hold-minutes` argument name for backward-compat.

**Verification:** EOD trade analytics (HC #225) show `time_stop` exit reason < 20% of total exits.

### A.3 Primary exits — signal decay + MFE peak detection [HC #226]

**Current:** Exits driven by max_hold_time (modal), risk-stop, trailing stop, opposite signal.

**Required additional exits:**
- `signal_decay_exit`: if `events_since_entry > signal_horizon_events × 1.5` AND `current_signal_alignment < 0.3 × entry_signal_strength`, exit.
- `mfe_peak_revert_exit`: track per-trade `rolling_max_unrealized_ticks`. If `current_unrealized < rolling_max_unrealized − give_back_ticks` (e.g. 2 ticks), exit. (essentially a trailing stop calibrated on MFE, not entry.)

**Implementation:** Add to position-management loop in `paper_trading_mamba_v2_patched.py` near lines 540-610.

**Verification:** Exit reason breakdown shows `mfe_peak_revert` and `signal_decay` as significant fractions; `time_stop` < 20%.

### A.4 HC #225 — trade analytics breakdown [REQUIRED EOD output]

EOD report (also surfaced in heartbeat) must include:
1. Total trades, win rate
2. Avg hold time WINNERS vs LOSERS
3. Avg winner $ + avg winner ticks
4. Avg loser $ + avg loser ticks
5. Max winner / max loser
6. Exit reason breakdown WITH P&L by reason (not just count)
7. Order type breakdown (market vs limit)
8. Long count / short count + per-side WR + per-side avg P&L

**Implementation:** Either inside paper_trader's existing trade journal OR a separate analyzer that reads `logs/mamba_v2_signals_*.jsonl` + `logs/paper_trades_*.jsonl`. Cleaner = analyzer script run at EOD by cron.

**Verification:** Run analyzer on 2026-05-06 logs, output matches the expected fields.

### A.5 HC #224 — latency tracking [REQUIRED]

Per-stage latency logging:
1. MD event received → feature added to buffer
2. Feature buffer → CNN-Mamba inference
3. Inference → confluence/ensemble
4. Signal → order placement
5. Order → ack

Log p50/p95/p99 in heartbeat + EOD summary. Threshold: end-to-end p95 > 100ms is a problem.

**Implementation:** Add `time.perf_counter_ns()` instrumentation at each stage boundary. Aggregate in a circular buffer, dump percentiles every heartbeat tick.

### A.6 Order type + commission logging [HC #227]

Every order placement logs:
```json
{"order_id": "...", "order_type": "market"|"limit"|"IOC", "side": "B"|"S",
 "qty": int, "limit_price": float|null, "queue_position_estimate": int|null,
 "placed_ts_ns": int, "fill_ts_ns": int|null, "expired_ts_ns": int|null}
```

Every fill deducts commission ($2.35 = 0.188 ticks/side). No spread crossing cost — fill price is what it is. Real measured slippage = `expected_mid_at_placement − actual_fill_price` for non-passive.

**Verification:** Reconcile internal P&L against Rithmic-reported P&L; should match within ±5% (only difference is timing of marks).

### A.7 FIFO queue position + adverse selection + fill latency [HC #228, #229B, #231(E)]

- Queue position: derive from L3 book snapshot at order placement time. Phase 1 (tonight): rough estimate = `total_qty_at_level` from BBO event stream (we don't have full L3 yet, see C below). Phase 2 (later): exact from full L3 book reconstruction.
- Fill latency: order_placed_ts → fill_ts, p50/p95/p99 by order type.
- Adverse selection: 5s post-fill midprice direction sign for passive fills. Report adverse_selection_rate live.

---

## B. Neptune Split DQN fixes [HC #221, #230, #211, #219]

### B.1 Reward redesign per `REWARD_DESIGN.md`

See companion doc: `alpha_discovery/execution/REWARD_DESIGN.md`. Three heads, each with proper reward + bounds + activity costs. Counterfactual cancel rewards. MFE-capture exit primary.

### B.2 Patches to fifo_rl_env.py

- Remove `MARKET_ORDER_COST_TICKS`, `LIMIT_ORDER_COST_TICKS` constants
- Raise `ALPHA_GATE_THRESHOLD` from 0.05 to 0.50 (or tier-table-driven)
- Rewrite `_force_close_position` reward with MFE-capture
- Add `price_improvement_bonus` to entry action handler
- Add `per_entry_activity_cost = 0.10`
- Add `PER_STEP_OPEN_COST = 0.001`
- Hook counterfactual buffer for cancel rewards (post-episode pass)

### B.3 Patches to train_split_dqn.py

- Reward clipping `[-5, +5]` before replay buffer storage
- Reduce `n_step` 50 → 20
- Verify Huber loss (HC #211)
- Per-head Q-loss logging + per-epoch greedy eval

### B.4 Kill + relaunch sequence

1. Wait for current eval pass to complete (provides baseline metric for "before reward fix")
2. SIGTERM (graceful checkpoint) the running PID 449707 on Neptune
3. Verify last checkpoint `fold1_ep9` saved cleanly + metadata flushed
4. Apply patches (B.2, B.3)
5. Pre-launch validator: predicted RSS < 24GB (HC #200, #211)
6. Launch from **CLEAN STATE** (NOT warm-start from polluted checkpoint) — fold 0 ep 0
7. First-epoch gate: trade count ≤ 1/10 of pre-fix value (target: 5-20k trades, not 200k+)
8. Q-loss stable (no explosion past 10 over first 1000 updates)

**Verification:** First epoch shows expected trade count + stable Q-loss + balanced ENTRY action distribution (no-op ≥30%, entry ≤30%, etc).

---

## C. MBO durability + book state persistence [HC #229]

### C.1 Phase 1 tonight: single-connection fan-out recorder

**Problem:** Rithmic license = 1 connection. Paper trader currently uses it; old `mbo_recorder.py` was a 2nd connection conflict.

**Fix:** The paper trader's existing Rithmic socket already receives the full MBO event stream. Hook a recorder INSIDE the paper_trader process that writes every event to disk in append-only mode, fsync'd per second.

**Output:** `C:\Users\claude\Lvl3Quant\data\raw_mbo_live\YYYYMMDD\events_HHMM.jsonl` (per-session-window file). Fsync per 1s. Roll over at session boundary.

**Backup:** Cron on Razer rsyncs to Jupiter every 5 min. EOD verifier (4:15pm ET cron) checks today's file exists, non-empty, has plausible event count, has been backed up.

**Verification:** Deliberate kill of paper_trader mid-session → confirm zero data loss in the recorded file (last fsync ≤ 1s before kill).

### C.2 Phase 2 (deferred — 2-3 day project): Full L3 book reconstruction + snapshotting

This is too large for tonight. Scope:
- Maintain in-memory full L3 book by replaying MBO from session open
- Snapshot every 30s + on every order placement
- On resume: load snapshot + replay deltas
- Hook into fill sim for FIFO queue position

**Document only tonight; implement in a follow-up session.** Spec doc `ORDER_BOOK_STATE.md` to be written separately.

### C.3 MBO backfill missing dates

Awaiting user OK on Databento spend. Missing dates:
- 2026-04-30 (Thu)
- 2026-05-01 (Fri)
- 2026-05-04 (Mon)
- 2026-05-05 (Tue)
- 2026-05-06 (Wed, partial)

---

## D. Spread crossing cost — DELETE everywhere [HC #231(A)]

**Audit list** (30+ files matched on `MARKET_ORDER_COST_TICKS|1\.376|spread_cost|crossing_cost`):

Active code (must fix):
- `alpha_discovery/execution/fifo_rl_env.py` — constants + comments
- `alpha_discovery/execution/train_fifo_rl_sac.py`
- `alpha_discovery/execution/train_fifo_rl.py`
- `alpha_discovery/execution/monte_carlo_execution.py`
- `alpha_discovery/execution/live_rl_inference.py`
- `alpha_discovery/execution/rl_execution_agent.py`
- `live_trading/rl_execution_agent.py`
- `live_trading/trade_journal.py`
- `live_trading/test_rl_pipeline.py`
- `configs/live_paper_trading.yaml`

Output JSONs (historical, leave alone — but flag in any future regen):
- `output/blended_cost_execution_analysis.json`
- `output/fifo_time_exit_backtest_results.json`
- `output/mfe_mae_horizon_analysis/mfe_mae_horizon_results.json`
- `output/lower_threshold_analysis/lower_threshold_results.json`
- `output/signal_quality_analysis/signal_quality_report.json`
- `output/trade_profitability_xgb/report_market.json`
- `output/fill_prob_xgb/*.json`
- `output/vol_conditioned_exec/*.csv`

Documentation:
- `CLAUDE.md` — cost constants section needs updating per HC #231(A)

**Verification:** `grep -rn "MARKET_ORDER_COST_TICKS\|1\.376\|spread_cost\|crossing_cost" alpha_discovery/ live_trading/ configs/` returns zero hits in active source files (output JSONs are historical, OK to keep).

---

## E. Validation gate before redeploy

Before redeploying Razer, ALL of the following must pass:

1. ✅ Long-only RCA complete + fix verified by feature/inference comparison test
2. ✅ Paper trader patched: max_hold=60s, primary exits MFE-revert + signal_decay, HC #225 analytics implemented, HC #224 latency tracked, HC #227 order/commission logging
3. ✅ MBO durability recorder running inside paper_trader with deliberate-kill test passing
4. ✅ Dry-run on yesterday's tape (2026-04-29 MBO):
   - Trade count reasonable (single-digit to low-tens, not hundreds)
   - Per-side balance: long_count and short_count both > 0 (target ratio 30/70 to 70/30)
   - Exit reasons: time_stop < 20%
   - P&L distribution sane (no implausible single trades)
5. ✅ Heartbeat shows new metrics (latency p50/p95/p99, fill rate, adverse_sel_rate)
6. ✅ Status report posted to user

---

## F. Status reporting

**Pre-bed report tonight (~10pm ET deadline):**
- What was fixed and verified
- What was deferred and why (with target date)
- Current state of Neptune relaunch
- Current state of Razer redeploy (or block reason)

**Morning briefing tomorrow:**
- Cluster status
- Overnight Neptune training progress
- Razer overnight (Globex) trade summary if redeployed

---

## G. Out of scope for tonight (deferred work)

- Full L3 book reconstruction + snapshotting (HC #229B Phase 2) — 2-3 day project
- MBO date backfill — awaiting user OK on Databento spend
- Razer 4-model deployment (PatchTST + razer_ppo + sac_v7) per HC #220 — separate task, after tonight's blockers fixed
- Counterfactual cancel reward implementation in fifo_rl_env.py — implementing minimal version tonight, full version in follow-up
- HC #198 dashboard cleanup — non-blocking
