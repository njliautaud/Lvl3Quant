# TASK QUEUE — Persistent across restarts
# Updated: 2026-06-16 ~23:32 ET
#
# === WORK LOOP PROTOCOL ===
# 1. On EVERY restart/recovery: READ this file FIRST
# 2. On EVERY user message: ADD new tasks to IN PROGRESS or QUEUED
# 3. Work top task in IN PROGRESS to completion
# 4. When done: move to COMPLETED with timestamp, pick up next QUEUED item
# 5. On crash/reset: resume the top IN PROGRESS task immediately
# 6. NEVER ask user what to do — read this file, it tells you
#
# Priority: IN PROGRESS (resume) > QUEUED (next up) > BACKLOG (when idle)

## IN PROGRESS — Currently Working

- [x] HC #638: Deep 30-min model COMPLETE — NN doesn't beat LightGBM. Champion = v4 Strategy B (LightGBM 30-min)
- [x] Feature ablation COMPLETE — lean model (43 feat) beats full (65): Sharpe 3.23 vs 2.70
- [ ] Monitor lh-30min-paper engine for live signal quality (first live signals expected next RTH session)
- [x] Validate lean model OOT — DONE, lean wins 3/3 splits. Sharpe 3.90 on 37d holdout
- [x] Update lh-30min-paper engine to use lean feature set (43 features) — DONE, restarted
- [x] Confidence sweep → 5% optimal (Sharpe 4.03 vs 2.35 at 15%). Paper engine updated.

## QUEUED — Next Up (work in order)

## BACKLOG — When Idle

- [ ] WhatsApp + Instagram fully linked (HC #625)

## COMPLETED

- [2026-06-17 12:00] Feature ablation DONE — lean model (43 features) is new candidate champion. Sharpe 3.23, Sortino 5.99, WR 57.2%. Regime gap 0.46 passes but close to limit.
- [2026-06-17 09:44] HC #637 Phase 1 COMPLETE — Longer-horizon directional model v1 built, validated, paper engine running. LightGBM on MBO hourly features: IC 0.40-0.55 across 1h-EOD horizons, regime-agnostic (2% gap at 2h), paper engine backtest Sharpe 3.24 / OOT Sharpe 2.46. PM2: lh-paper-engine.
- [2026-06-16 22:50] Neptune DQN v5 relaunched — PYTHONPATH fix applied, workers loading buffers, GPU will engage shortly. MLflow exp: split_dqn_v5_v342_r2
- [2026-06-16 22:48] Infra audit done — 18 PM2 processes (1.2GB / 46GB RAM = 16%), not resource-constrained. Work loop is the real fix for context resets.
- [2026-06-16 22:42] Autonomous work loop built — HC #634 in DIRECTIVES, TASK_QUEUE.md protocol established
