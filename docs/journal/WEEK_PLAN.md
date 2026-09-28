# WEEK PLAN — HC #413/#415/#417 — DELIVER TRADABLE BY FRI 5/22 (FULL-OOT EDITION)

**Last updated**: 2026-05-17 ~23:55 ET, HC #417 falsification complete (both caveats resolved)

## ⭐ MONDAY (5/18) MORNING — USER-DECISION FLOW (added 2026-05-18 00:30 ET)

**`v2_1s_short_top05` is promotion-grade. Deployment spec ready at `output/hc417_v2_1s_short_top05_DEPLOYMENT_SPEC.md`.**

User decision required Monday AM (one of A/B/C/D):
- **A — Approve Phase A (paper trade Mon-Wed, 1 contract, all kill-switches)** ← RECOMMENDED DEFAULT
- B — Skip paper, go live 1-contract Mon AM
- C — Defer paper; run 3 more research tests (fixed-labels resweep, 2-ctr sim, LGBM gate sweep [LGBM already done — negligible])
- D — Reject; back to HC #417 productive-work queue

### Phase A — Mon 5/19 09:30 ET → Wed 5/21 16:00 ET (PAPER)
- Strategy: `v2_1s_short_top05` (CNN-Mamba v2 fold_10, pred_log_ret_1s, top 0.5% short)
- Order: passive_at_touch, TP1=+0.4782, TP2=+0.9564, SL=-0.5686 tk, 10s cancel window
- Position size: 1 contract; max 1 concurrent; 5s re-entry cooldown
- Hours: RTH only (09:30-16:00 ET)
- Kill-switches active (daily -10tk, weekly -25tk, 5 consec losses 1h pause, IC drift <0.15)
- Tracking gates: fill count within ±20% of expected ~22/day; net/fill within ±0.05 tk of +0.27 tk; WR ≥ 75%; day_conc ≤ 0.20
- NOTHING ON RAZER MODIFIED THIS WEEKEND. Spec ready for Mon AM wire-up.

### Phase B — Wed 5/21 EOD → Fri 5/23 EOD (REVIEW)
- If Phase A within spec → present results + scaling proposal for user approval
- If out of spec → halt, produce diagnostic, do NOT escalate

### Phase C — Mon 5/26 (LIVE 1-contract, IF user approves)
- Flip paper → live (single config toggle)
- Daily user review for 5 trading days
- After 5 days within spec → user may approve scaling proposal (separate spec)

### Tech-debt (non-blocking)
- `data/processed/mbo_events_smart_v3/` corruption on 5 dates documented in `output/hc417_data_gap_root_cause.md`. Optional re-process opportunity (~20 min CPU). Live trading unaffected.

---

## ⭐ MONDAY (5/18) PRIORITY UPDATE — RAZER WIRE-UP MOVES EARLIER
HC #417 v2 falsification (Phase 1+2+3+4) confirms the 5-cell finding is robust:
- v2-NATIVE MFE matrix yields STRONGER results (5 cells still pass HC #408 + HC #415; net/fill improved on survivors; best cell v2_1s_short_top05 jumps from +0.207 to +0.274 tk/fill, CI95lo +0.232, pdpr=1.00)
- 4 of 5 cells survive HC #411 sub-window stability at N=3, N=4, AND N=6
- Late-OOT "zero-fill" dates are missing 1s target labels (data pipeline gap), NOT signal decay
- **Recommendation**: bring Razer wire-up forward to Monday AM. Live-paper-trade v2_1s_short_top05 first (passive_at_touch, TP1=0.48tk, TP2=0.96tk, SL=0.57tk, gate pred_1s<=-0.6926). v2_1s_short_top1 as second config. Defer 5s/10s cells until 1s is verified live.
- See SESSION_STATE.md top entry + `output/hc417_hc413_v2native_mfe/`, `output/hc417_hc411_subwindow_v2/`, `output/hc417_zero_fill_diagnosis.md`.
**Active HCs**: #413 (production deadline Fri), #414 (Razer pivot contingency), #415 (multi-gate + all-OOT + FIFO + persistence), #416 (productive+honest+v3.4.2 is the bet), **#417 (full-OOT window Feb-end→Apr-29 + ALL nodes productive + full autonomy)**
**Single overriding deliverable**: by Fri 5/22 EOD, either Razer is paper-trading a winning config with positive-net + per_day_pass_rate≥80% on the **full ~40-day Feb→Apr OOT window** under FIFO canonical replay, OR explicit verdict "no live-worthy alpha after full-OOT multi-gate exhaustive search" → HC #414 triggers.

---

## ⭐ HC #417 GAME-CHANGER DISCOVERY (2026-05-17 23:10 ET)

**`data/precomputed_obs/` already has all 56 trading days of feature tensors for Feb 23 → Apr 29 2026.** Every model with weights on disk can produce a full-OOT NPZ in batch-inference mode — no retraining needed for the baseline eval.

**Implication**: HC #411's "0 regime-stable cells" verdict was on 16-day data. The HC #415 "0 cells pass" verdict was on 16-day data. ALL prior negative verdicts have ~3× more data available to either confirm or overturn them. This is the single biggest leverage point of the week.

**HC #417 work queue** (consume top-down):
1. **(Razer, weekend GPU)** Build batch wrapper around `live_trading_linux/cnn_mamba_v2_inference.py` + run on 56-day OOT → `output/hc417_v2_full_oot_56d.npz`. Generates the v2 baseline we never had.
2. **(Jupiter CPU, parallel)** Replicate same pattern for v3.3 weights (`output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_intra_ckpt.pt` from train_cnn_mamba_v3_3.py model class) → `output/hc417_v33_full_oot_56d.npz`. Slower (CPU) but parallel-safe.
3. **(Saturn, when SSH revived)** HC #415 multi-gate exhaustive combo (3-gate, 4-gate) on existing v3.4.2 16d NPZ as warm-up; then on 56d NPZs when they land.
4. **(Jupiter)** Generate `target_fifo_tp4sl3_net` + `target_fifo_tp8sl5_net` labels for the 56-day window if not already there. These are derivable from precomputed_obs MBO replay.
5. **(Neptune, after v3.4.2 60d ep-1 done ~22:00 ET 5/18)** Run a SEPARATE inference pass on the 56-day OOT with the ep-1 weights → `output/hc417_v342_full_oot_56d.npz`.
6. Re-run HC #413 + HC #411 + HC #415 sweeps on all three 56-day NPZs. Compare under SAME window.
7. Multi-gate sweep on COMBINED 3-model ensemble (agreement gating).

---

## RECOVERY PROCEDURE (read first on any session reset)
1. Read this file (WEEK_PLAN.md) — tells you the current milestone.
2. Read SESSION_STATE.md (newest entry on top) — tells you what is RUNNING.
3. Read DIRECTIVES.md HC #415, #414, #413 (top of file) — binding rules.
4. Read RUN_HISTORY.md last 5 entries — what was already tried.
5. Run `mcp__qcc__qcc_node_status live_check=true` to verify reality.
6. Check `mcp__discord__read_channel general 30` for latest user messages.
7. Re-arm 6 crons (see "Cron Reference" at bottom of this file).
8. THEN dispatch the next pending task in this plan.

---

## METRIC HIERARCHY (HC #415 rule 6 — use THIS, not legacy IC-first)
- **Primary**: per_day_pass_rate (HC #415 rule 2), Sortino, realized_net_tk_per_fill (HC #405 fill-price)
- **Secondary**: Sharpe, PF, WR, day_conc, adv_sel_30s
- **Diagnostic**: IC, IC@conf-tier, MFE/MAE (RESEARCH ONLY, not promotion evidence)
- **Never primary**: raw P&L, mid-based alpha, single-day numbers

## GATING POLICY (HC #415 rule 1 — multi-output, not just IC)
A promoted cell MUST be a TUPLE of model outputs, e.g.:
- `direction_1s == direction_5s == direction_10s` (cross-horizon agreement)
- `sigma_head < sigma_threshold` (model is confident)
- `predicted_MFE > 1.5 × TP_threshold` (room to move)
- `P(reversal_within_hold) < 0.3` (trend persists)
- `vol_30s ∈ [vol_low, vol_high]` bucket where backtest passes
- `book_imbalance_sign == predicted_side`
- `confidence ∈ Top0.5%` AND `|prediction| > absolute_floor`

## ACCEPTANCE GATE (HC #415 rule 2 — all-OOT-stability)
A cell promotes to deploy IFF ALL of:
- per_day_pass_rate ≥ 80% (days with net_tk > 0)
- per_day_pass_rate_strict ≥ 70% (days with net_tk > 0.376)
- max_single_day_pnl_share ≤ 40%
- min_n_days_with_fills ≥ 10 (if OOT spans ≥15 days)
- HC #408 honesty: n ≥ 50, day_conc ≤ 0.20, CI_low_95 > 0
- HC #344 day_conc gate
- Full FIFO canonical replay (HC #74/#377/#397B) used

---

## DAILY MILESTONES

### Mon 2026-05-18
- [ ] **08:00 ET** — Read v3.4.2 60d training status. Should be ~10h into fold-0 ep-1 (~halfway). PID 311170. If dead → relaunch from `fold_00_intra_ckpt.pt`.
- [ ] **AM Jupiter CPU**: HC #415 multi-output gating sweep on v3.3 15d NPZ (already on disk at `output/cnn_mamba_v3_3_uncertainty_weighted/`). Output → `output/hc415_multi_gate_sweep_v33/`.
  - Enumerate gates: cross-horizon-direction, sigma, MFE-prob, reversal, vol-bucket, book-imbalance, top-rank+magnitude.
  - For each (cell × gate-combo), run FIFO canonical replay (HC #74/#377/#397B).
  - Compute: per_day_pass_rate, per_day_pass_rate_strict, max_single_day_pnl_share, Sortino, net_tk/fill, adv_sel_30s, day_conc.
  - Filter to ALL HC #415 rule-2 passers. Rank by Sortino.
- [ ] **AM Jupiter CPU**: Razer paper-trader log-visibility audit (HC #410 P4). Verify ledger fields, Discord alert hook, kill-switch hook before market close.
- [ ] **~22:00 ET** — v3.4.2 60d fold-0 ep-1 OOT NPZ lands on Neptune. SCP to Jupiter. Run sanity gate (HC #413 rule 4). Run HC #415 multi-gate sweep on it. Compare to v3.3.
- [ ] **EOD**: Decision matrix per HC #414. If v3.4.2 ≥ 1 cell passes HC #415 → Wed Razer wire-up of v3.4.2. If only v3.3 passes → Wed Razer wire-up of v3.3. If neither → HC #414 TRIGGERS, Razer pivots to training.

### Tue 2026-05-19
- [ ] **AM**: Backtest winner with TP/SL scalping on 15+ day OOT. Tune entry threshold to maximize Sortino subject to HC #415 rule-2 gate.
- [ ] **PM**: Wire winner to Razer (inference engine + TP/SL executor + ledger + Discord + kill-switch). Smoke test E2E.
- [ ] **Decision deadline 12:00 ET**: if v3.4.2 60d still hasn't produced a HC #415-passing cell AND v3.3 hasn't either → HC #414 trigger, Razer pivots to training a regime-classifier ensemble + LGBM scalping baseline.

### Wed 2026-05-20
- [ ] **09:30 ET market open**: First paper-trading day with winner config. Monitor every fill.
- [ ] Verify realized fills match backtest distribution (fill prices, queue position, adv-sel).
- [ ] EOD: compute day's net, contribution to OOT cumulative, gate evaluation.

### Thu 2026-05-21
- [ ] Second paper-trading day. Verify P&L distribution.
- [ ] If kill-switch trips → diagnose + decide whether to continue Fri.

### Fri 2026-05-22
- [ ] If Wed+Thu paper P&L > 0 AND no kill-switch AND realized matches backtest → **GO LIVE 1 contract**.
- [ ] Else: ship "paper-only — alpha possible but not live-worthy because [reason]" verdict.
- [ ] EOD verdict post to #general.

---

## ACTIVE EXPERIMENTS (kept in sync with SESSION_STATE.md)

### Neptune
- **v3.4.2 60d fold-0 ep-1**: PID 311170, MLflow `e5f0f79b...`, started 2026-05-17 21:35 ET, ETA fold-0 complete ~22:00 ET 5/18.
- Source: `/tmp/v342_resume_launcher.py` resuming from `fold_00_intra_ckpt.pt`.
- HC #0 compliant (60d sliding). HC #386 patches active. BS=8, workers=1, bf16, AMP.
- DO NOT TOUCH unless dead.

### Jupiter
- **HC #415 multi-gate sweep on v3.3 15d** (to be dispatched this session).
- Output → `output/hc415_multi_gate_sweep_v33/`.

### Razer
- LIVE host placeholder. Paper-trader PID 25512 carry-over (HC #413 P4 to audit Mon AM).
- HC #414 dormant pending Mon EOD decision.

---

## CRON REFERENCE (re-arm on session reset; current IDs in SESSION_STATE.md)
- `37 * * * *` — MAMBA_PULSE (35-min health)
- `23 */2 * * *` — DEEP_CHECK (2h cluster)
- `23 8 * * *` — MORNING_BRIEF (8:23 ET daily)
- `41 15 * * 1-5` — EOD_SUMMARY (3:41 ET weekdays)
- `7 9 * * *` — USAGE_AM (9:07)
- `7 15 * * *` — USAGE_PM (15:07)

## KEY FILES
- `/home/jupiter/Lvl3Quant/output/hc411_regime_agnostic_20260517_215211/mfe_at_confidence_matrix.csv` — TP/SL inputs
- `/home/jupiter/Lvl3Quant/scripts/hc413_scalping_backtester/` — canonical FIFO scalping backtester
- `/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz` — v3.3 5d OOT
- `/home/jupiter/Lvl3Quant/data/v3_3/fold_00_predictions.npz` — v3.3 15d concat NPZ
- `/home/jupiter/Lvl3Quant/output/v342_fold_00_ep1_oot_inference_extended.npz` — v3.4.2 16d
- Pending: `/home/jupiter/Lvl3Quant/output/v342_fold_00_ep1_oot_inference_60d.npz` — v3.4.2 60d (lands ~22:00 ET 5/18)
- Pending: `/home/jupiter/Lvl3Quant/output/hc415_multi_gate_sweep_v33/` — first HC #415 deliverable

## ABSOLUTE NO-NOs (HC violations)
- No expanding window (HC #0 — sliding 60d only)
- No midpoint-based execution promotion (HC #74 — FIFO market replay only)
- No 2.0-tick or 1.24-tick or 1.0-tick spread cost (HC #405 — 0.376 commission only on fill-price replay)
- No "+1 tick" added to realized fill-price P&L (HC #405)
- No training on Razer until HC #414 triggers
- No re-launching v3.4.2 60d cold (use `fold_00_intra_ckpt.pt` resume)
- No SIGKILL on SSH "FAILED" without `ps -p` verification (HC #406)
- No promoting a cell on aggregate P&L when 1-2 days carry > 60% (HC #415 rule 2)
- No reporting IC as primary (HC #415 rule 6 — Sortino/per_day_pass_rate is primary)
- No "awaiting your approval" / "standing by" on routine engineering (HC #393)
