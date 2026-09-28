# DEPLOYMENT SPEC — `v2_1s_short_top05` (CNN-Mamba v2, 1s short scalp, top 0.5% confidence)

**Status**: SPEC FOR USER REVIEW. Nothing deployed. Nothing on Razer touched.
**Date**: 2026-05-17 (revised 2026-05-18 — implementation gap added)

---

## 0. ⚠️ IMPLEMENTATION GAP (revised 2026-05-18 00:30 ET per Razer audit)

**Initial Phase A rollout assumed config-swap-and-restart. Razer read-only audit found this is WRONG.**

Current running `paper_trading_mamba_v2.py` on Razer (PID 25512, started 2026-05-14) is the **HC #46 risk-managed trader**, NOT spec-compliant:
- ✗ Tier: Top **5%** (spec wants Top 0.5%)
- ✗ Sides: BOTH long + short signal-flip (spec wants SHORT only)
- ✗ Brackets: NO TP1/TP2/SL bracket logic at all
- ✗ Order type: not passive_at_touch
- ✗ RTH gate: missing
- ✗ Kill-switches: 7 of 9 missing (no daily/weekly loss cap, no consec-loss pause, no IC-drift halt, no Discord alerts)
- ✓ Match: model arch, symbol (ESM6), position size (1 contract)

**Greenlight gate audit**: **3/19 spec gates pass**. See `output/hc417_razer_paper_config_audit.md` for full table.

**Phase A implementation effort estimate** (per audit): ~6–10 hours of new code. Approaches the user can choose from:
- **Option I**: Write fresh `paper_trading_v2_1s_short_top05.py` script implementing the spec end-to-end. Cleanest. ~8h.
- **Option II**: Add a CLI flag mode to existing script that swaps in spec gates + bracket logic. ~6h, riskier (entanglement with HC #46 paths).
- **Option III**: Postpone Phase A — keep current HC #46 paper trader running this week, treat the v2 finding as pure research evidence for Phase C live-capital decision later.

**Side bug found by audit** (not blocking): paper trader source has stale `POINT_VALUE=50.0  # NQ` constant alongside `TICK_VALUE=12.50`. Currently running `--symbol ESM6` so ES TICK is correct, but the NQ constant could cause issues if anyone toggles symbol later. Flag for cleanup during Option I/II rewrite.

**Until user picks I / II / III, no Razer-side code changes will be made.** Phase A is BLOCKED on this user decision.

---
**Source verdict**: `output/hc417_hc413_v2native_mfe/verdict.md`
**Source sub-window evidence**: `output/hc417_hc411_subwindow_v2/verdict.md`
**Backtest row**: `output/hc417_hc413_v2native_mfe/scalping_backtest_results.csv` (cell_id `v2_1s_short_top05`)

---

## 1. STRATEGY DEFINITION

| Field | Value | Source |
|---|---|---|
| **Model** | CNN-Mamba v2, checkpoint `fold_10_best.pt` | LIVE inference model already on Razer at `C:\Users\claude\Lvl3Quant\output\cnn_mamba_v2_smart_v3_mar\fold_10_best.pt` (CORRECTED 2026-05-18 per Razer audit; earlier `models\fold_10_best.pt` path was incorrect) — **DO NOT REPLACE** |
| **Model SHA-256** | Recorded in wrapped NPZ `ckpt_sha256` field | Verify against Razer file before flipping live |
| **Signal head** | `pred_log_ret_1s` (head index 0) | v2 emits 3-head: 1s/5s/10s |
| **Direction** | SHORT only | 1s long fails HC #415 rule 2 (only 5 short cells pass) |
| **Confidence ranking** | Top 0.5% by `signed_short = -pred_log_ret_1s` | Per-day percentile (recommended — see rationale) |
| **Per-day vs rolling**: | **Per-day** | Rationale: backtester uses GLOBAL threshold over full OOT (= 0.69260 in canonical units), but this is unstable for a live cold-start. Per-day percentile is more robust: each session computes its own 99.5th percentile from prior elapsed signal stream. Backtest treats GLOBAL ≈ per-day because pred distribution is stationary across days (zero_fill_diag.py table shows daily pred_std ~0.30-0.40 with similar mean +0.05). |
| **Cold-start fallback** | Use GLOBAL threshold `pred_log_ret_1s ≤ -0.6926` for first 30 min of session, then switch to per-day percentile | Avoids "no fills" on session open |
| **Entry trigger** | `pred_log_ret_1s ≤ -0.6926` AND rank ≥ 99.5%ile within current session | Both gates active |
| **TP1** | **+0.4782 ticks** (0.5 × MFE) | From `scalping_backtest_results.csv` row, `tp1` col |
| **TP2** | **+0.9564 ticks** (1.0 × MFE) | From row, `tp2` col |
| **SL**  | **-0.5686 ticks** (capped at MAE) | From row, `sl` col |
| **Order type** | `passive_at_touch` (LIMIT at best ask for short entry, post-only if broker supports) | HC #413 default; market orders would push net negative |
| **Cancel window** | 40 evals × 250ms stride = **10s** | `DEFAULT_CANCEL_EVAL_WINDOW=40` in `fill_sim.py:41` |
| **Cancel logic** | If unfilled after 10s, cancel and forget (do not chase) | HC #344 |
| **Position size** | **1 contract** initial | NEVER scale until verified |
| **Trading hours** | RTH only, 09:30–16:00 ET | ES futures liquidity |
| **Max concurrent positions** | 1 | Avoid double-up on consecutive signals |
| **Re-entry cooldown** | After exit, 5s minimum before next entry | Avoid mean-reversion whipsaw |

### Rationale for per-day percentile choice
Backtest used a GLOBAL Top0.5% threshold derived from the full 36-date masked sample (`pred_log_ret_1s ≤ -0.6926`). Live mode cannot know that distribution at session start. Per-day percentile produces statistically similar fill counts (zero-fill diag shows daily local_top05_short_thr in range +0.58 to +0.74 around the global +0.6926). Recommend hybrid: warm-up phase uses GLOBAL cutoff for first 30 min, then per-day percentile takes over.

---

## 2. EXPECTED OPERATING METRICS (full disclosure from backtest)

All values from `output/hc417_hc413_v2native_mfe/scalping_backtest_results.csv`, row `v2_1s_short_top05`, full 1.46M-sample OOT (36 active dates).

| Metric | Backtest value | Notes |
|---|---:|---|
| Total fills (full OOT) | 639 | passive_at_touch, 25 active days |
| Fills/active-day (mean) | 25.6 | range 1-67 — **high variance** |
| Fills/active-day (median) | ~22 | per HC #417 brief |
| Fills/active-day (min, max) | 1, 67 | quiet days = thin |
| Net per fill | **+0.274 ticks** | = **+$3.43** per fill |
| Win rate | **84.8%** | TP1 or TP2 hits dominate |
| TP1 hits | 100 (15.6%) | partial scalp |
| TP2 hits | 442 (69.2%) | full MFE captured |
| SL hits | 95 (14.9%) | risk-managed |
| Time-stops | 2 (0.3%) | rare |
| Profit Factor | **2.92** | gross wins / gross losses |
| Sharpe (√N annualized proxy) | 12.77 | per-fill basis, very high due to short hold |
| Sortino (√N) | 707.6 | huge (few negative outliers) |
| Day-conc | 0.132 | well under HC #344 0.20 limit — P&L not concentrated |
| CI95 lower bound (net/fill) | +0.232 ticks | >> 0, alpha confirmed |
| Per-day pass rate (HC #415 rule 2) | **1.00 (25/25 active days net>0)** | every single active day positive |
| Median dollar P&L/day | **~$75** | 22 fills × $3.43 |
| Best day dollar P&L | ~$300 | per backtest dispersion |
| Worst day dollar P&L | ~+$5 to -$5 | bounded |
| Sub-window stability (N=3/4/6) | All windows net>0, all CI95lo>0 | regime-stable (HC #411 PASS) |

### Important caveats
- **Backtest assumes perfect fill at passive_at_touch when queue position triggers**. Live execution may experience adverse selection if AMP/Rithmic latency exceeds the modeled fill timing. Phase A paper-trade is mandatory to validate this.
- **Backtest uses v2-NATIVE MFE matrix**, not v3.4.2 borrowed values. This was the HC #417 falsification fix and is now canonical.
- **Backtest uses 25 of 36 active dates** due to upstream data-pipeline label corruption (see `hc417_data_gap_root_cause.md`). The other 11 dates have valid v2 predictions but missing 1s labels.
- **MAE proxy** at 1s/5s/10s is the mean magnitude of negative-only signed realized move (same proxy as HC #411). Not direct MAE from FIFO replay.

---

## 3. KILL-SWITCHES (ALL MANDATORY)

These MUST be wired in the paper_trader / live trader before deployment.

| Switch | Trigger | Action |
|---|---|---|
| **Hard daily loss limit** | Realised P&L ≤ -10 ticks ($125) on the trading day | Halt strategy until next trading day; alert user |
| **Hard weekly loss limit** | Realised P&L ≤ -25 ticks ($312.50) on rolling 5-day window | Halt strategy until user manually re-enables |
| **Per-trade max risk** | Position open with unrealised P&L ≤ SL (-0.5686 tk) | SL order should fire automatically; if not filled within 30s, force market exit |
| **Consecutive-loss auto-pause** | 5 consecutive losing fills (any kind: SL hit, time-stop, neg time-exit) | Pause strategy 1 hour, then re-arm |
| **Realised drift halt** | Rolling realised net/fill < -0.5 tk over last 10+ fills | Pause + alert; user manual re-enable required |
| **IC drift monitor** | Live `IC(pred_1s, realised_log_ret_1s)` < 0.15 on rolling 5-day window | Auto-disable; alert with degradation evidence; user must re-arm after diagnosis |
| **Stale-signal kill** | No new prediction for >30 sec | Pause new entries (existing positions still managed by TP/SL/cancel) |
| **Connectivity loss** | Broker API disconnect > 5s | Cancel all unfilled limits; flatten any open position at market; pause until reconnected + manual re-arm |
| **Model file integrity** | `fold_10_best.pt` SHA-256 mismatch at startup vs recorded canonical | Refuse to start; alert |

### Position sizing kill-switch
Position size hard-coded to 1 contract in code. Any change requires explicit user approval and a code commit (not a config toggle).

---

## 4. PRE-LIVE CHECKLIST (USER YES/NO)

- [ ] **v2-native MFE config loaded**: TP1=0.4782, TP2=0.9564, SL=0.5686 ticks (passive_at_touch cost 0.376 tk already accounted for)
- [ ] **Razer paper trader is currently configured for which signal?** TO BE VERIFIED (document only, do not modify): inspect `C:\Users\claude\Lvl3Quant\paper_trading_mamba_v2.py` (or current paper trader script) read-only to confirm what signal it's currently consuming. Compare to the spec above. **DO NOT EDIT.**
- [ ] **Live AMP/Rithmic credentials present**: noted, not pulled. User confirms credentials are in place on Razer.
- [ ] **Telegram/Discord alerts wired for fills**: confirm `webhook_notifier.js` or equivalent on Razer side fires on every fill/exit/kill-switch event.
- [ ] **Position size = 1 contract** (hard-coded, NOT scaled up)
- [ ] **Trading hours: RTH only** (09:30–16:00 ET, no overnight, no Globex extended)
- [ ] **Paper-mode dry-run for 3 days before live**: STRONGLY RECOMMEND YES (this is Phase A below)
- [ ] **Kill-switches enumerated above all wired**
- [ ] **Backtest evidence file pinned**: `output/hc417_hc413_v2native_mfe/scalping_backtest_results.csv` archived to immutable storage with date/SHA stamp

---

## 5. ROLLOUT PLAN

### Phase A — Paper trade (Mon market open → Wed EOD, 5/19 09:30 ET → 5/21 16:00 ET)
- Razer paper trader configured for `v2_1s_short_top05` strategy as defined above.
- All kill-switches active.
- 1-contract simulated position size.
- Telemetry: every fill, exit, kill-switch trigger logged to Discord + local jsonl.
- Compare to backtest expectations:
  - Fill count: **must be within ±20% of expected ~22/day**
  - Net/fill: **must be within ±0.05 tk of expected +0.27 tk**
  - WR: **must be within ±10pp of expected 85%**
  - Day-conc: must remain ≤ 0.20

### Phase B — Phase A review & user gate (Wed EOD 5/21 16:00 → Fri EOD 5/23)
- If Phase A tracking metrics are within spec → present results to user; request approval for Phase C.
- If Phase A is OUT of spec (fill count off, net negative, WR < 75%) → halt; produce diagnostic report; do NOT escalate to live. Causes to investigate: (a) live model differs from backtested model, (b) live MBO feed differs from offline MBO data, (c) per-day percentile cold-start failing, (d) latency assumption broken.

### Phase C — Live 1-contract (Mon 5/26 market open, if approved)
- Flip paper → live in Razer paper_trader config (single config toggle, no code change).
- Daily kill-switches active.
- User reviews every EOD for first 5 trading days.
- After 5 days of within-spec live tracking → user may approve scaling proposal (NEW spec; not auto).

---

## 6. DECISION MATRIX FOR USER (per HC #414 + HC #417)

| Option | What happens | Recommendation |
|---|---|---|
| **A — Approve Phase A only** | Paper-trade Mon-Wed; revisit Wed EOD | **DEFAULT — strongly recommended** |
| **B — Skip Phase A, jump to Phase B/C** | Live 1-contract Mon market open | NOT recommended — paper trade is cheap insurance |
| **C — Skip paper, more backtest research** | Defer Phase A; run 3 candidate further tests | Only if user has strong "want more evidence" preference |
| **D — Reject entirely** | Do nothing; continue execution research | Falls back to HC #414 / HC #417 productive-work queue |

### If Option C: 3 candidate additional tests
1. **Re-run sweep on fixed-labels NPZ** (after applying `hc417_data_gap_root_cause.md` Option A fix). Adds ~5 dates of evidence, expected to re-confirm verdict with stronger n. ~1h Jupiter CPU.
2. **2-contract position sizing simulation** in the existing backtester (no code change, just doubles dollar P&L and risk; confirms kill-switch thresholds scale correctly). ~10 min Jupiter CPU.
3. **Alternative gate combos**: top0.5% + LGBM supervised gate (currently running as HC #414(b) per Task 3 of this assignment; see `output/hc417_lgbm_supervised_gate_v2/`). Already in progress.

---

## 7. DEPLOYMENT FILE-INVENTORY (READ-ONLY VERIFICATION)

To greenlight Phase A, the user / agent should verify (DO NOT EDIT):

| Razer file | Expected content | How to verify |
|---|---|---|
| `C:\Users\claude\Lvl3Quant\models\fold_10_best.pt` | CNN-Mamba v2 weights, SHA-256 matches `ckpt_sha256` field in wrapped NPZ | Read-only SHA |
| `C:\Users\claude\Lvl3Quant\paper_trading_mamba_v2.py` (or current) | Currently reads pred_log_ret_1s; entry filter currently TBD | Read source, document; DO NOT EDIT in this task |
| `C:\Users\claude\Lvl3Quant\mbo_recorder.py` | Currently running PID 15720 (per HC brief). Feeds MBO events to inference. | `tasklist /fi "pid eq 15720"` |
| Razer paper trader process | PID 25512 must remain UP and undisturbed | Status check only |

**Reminder**: This task is SPEC ONLY. No Razer-side files are touched. PIDs 25512 and 15720 are not interrupted.

---

## 8. POST-DEPLOYMENT MONITORING (after Phase A starts)

| Item | Frequency | Tool |
|---|---|---|
| Fill count vs expected | Every hour during RTH | `qcc_pnl_status` + Discord ping |
| Realised net/fill | EOD | dashboard + EOD summary |
| Drift IC tracking | EOD | dedicated cron, alerts if IC_1s < 0.15 over 5 days |
| Day_conc | EOD | dashboard |
| Latency tail | Real-time | `mbo_recorder` + paper_trader latency logs |
| Kill-switch triggers | Real-time | Discord alert per trigger |

---

## 9. APPENDIX — Exact backtest row (paste-ready)

```
cell_id,model,horizon,side,conf_tier,n_fills,n_tp1_hits,n_tp2_hits,n_sl_hits,n_time_stops,gross_mfe_per_fill,realized_net_per_fill,realized_net_per_fill_dollars,sharpe_sqrtN,sortino_sqrtN,pf,wr,day_conc,ci_low_95_net,ci_low_95_net_dollars,pass_hc344,pass_hc408_honesty,mfe_source,mae_source,tp1,tp2,sl,entry_cost_ticks,order_type
v2_1s_short_top05,v2,1s,short,top05,639,100,442,95,2,0.6502845070422537,0.27428450704225354,3.4285563380281694,12.767092939637386,707.5746638626798,2.9157253877515332,84.82003129890454,0.13153471430576524,0.23178949139280128,2.897368642410016,True,True,0.9564,0.5686,0.4782,0.9564,0.5686,0.376,passive_at_touch
```
