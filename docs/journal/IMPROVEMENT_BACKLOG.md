# IMPROVEMENT BACKLOG — Continuous Gap-Scan Log (HC #590)

**Created 2026-06-09 ~19:10 ET** per HC #590. Append-only. Every deep check + every morning briefing scans for new gaps and closes existing ones.

Format:
```
## YYYY-MM-DD HH:MM ET — [LANE] — [STATUS] — short title
- Gap: what is broken / missing / stale
- Impact: why it matters (risk-adjusted, deploy-gate, infra reliability, etc.)
- Fix: action taken or planned (owner: self / sub-agent N / cron)
- Closed: timestamp + verification
```

Status tags: `OPEN` | `IN-PROGRESS` | `CLOSED` | `WONTFIX` (with reason)

Lanes: `A1-ETF` | `A2-K6` | `A3-WHEEL` | `INFRA` | `REPORTING` | `LIVE` | `DATA` | `MODEL`

---

## 2026-06-09 19:10 ET — INFRA — OPEN — Backlog file did not exist
- Gap: continuous-improvement loop had no persistent backlog; gaps found ad-hoc were lost across restarts
- Impact: HC #590 R5 violated by default; no audit trail of what got better
- Fix: created this file; will append every deep check
- Closed: 2026-06-09 19:10 ET — file in place

## 2026-06-09 19:10 ET — INFRA — CLOSED — Initial gap-scan inventory compiled
- Gap: full inventory of open gaps across all 3 lanes + live + infra had never been compiled
- Impact: cannot prioritize per HC #590 R6 without inventory
- Fix: dispatched read-only sub-agent; produced 15 prioritized gaps across lanes A1/A2/A3 + infra + reporting
- Closed: 2026-06-09 19:15 ET — inventory recorded below

---

# INITIAL GAP INVENTORY (2026-06-09 19:15 ET)

## P1 — Silent breakage risk (paper engines could die unnoticed)

### P1-1 [A1 ETF] STALENESS — CLOSED
- Gap: no watchdog if etf_rotation cron silently fails for >24h; state.json mtime never checked
- Fix: scripts/etf_rotation_staleness_check.sh + cron 0 11 * * 2-6
- Closed: 2026-07-05 ~08:15 ET — script existed, cron added

### P1-2 [A2 K=6] WEEKLY-MISS — CLOSED
- Gap: weekly Monday cron — if it fails one Monday, no alarm fires for a week
- Fix: scripts/megacap_weekly_check.sh + cron 0 10 * * 2 (Tuesday morning verification)
- Closed: 2026-07-05 ~08:15 ET — script existed, cron added

### P1-3 [A3 WHEEL] DAEMON-HEALTH — CLOSED
- Gap: pm2 wheel-paper-engine has autorestart but no external liveness check; live_stack_watchdog only covers Razer
- Fix: scripts/wheel_liveness_check.sh + cron */15 * * * 1-5
- Closed: 2026-07-05 ~08:15 ET — script existed, cron added

### P1-4 [ALL] NAV-SNAPSHOT — CLOSED
- Gap: no daily NAV time series for any lane (state files overwrite in place); can't compute Sharpe/DD later
- Fix: scripts/nav_snapshot.py + cron 30 16 * * 1-5 + one-time backfill of today's row
- Updated 2026-07-05: added wheel_v4, wheel_balanced, wheel_diversified, lh_2h lanes. Fixed state file paths. Cron entry added. First snapshot captured (etf_rotation, megacap_k6, wheel, wheel_diversified).
- Closed: 2026-07-05 ~06:38 ET

### P1-5 [A3 WHEEL] IV-FEED — IN-PROGRESS
- Gap: yfinance IV often returns absurd pre-market values; floored silently with no alert
- Fix: minimal edit to wheel_paper_engine.py to count floor invocations + scripts/wheel_iv_feed_check.sh
- Status: build in flight

## P2 — Deploy-gate + model + reporting

### P2-1 [DEPLOY-GATE] AUTOMATION — OPEN
- Gap: HC #428 R1/R2 gates run as one-shot scripts; validation_pack.json has "deploy": null waiting for a programmatic verdict
- Fix proposal: pass_deploy_gates.py reads validation_pack.json → emits PASS/FAIL; post-backtest hook
- Status: queued (effort M)

### P2-2 [A2 K=6] META-CLASSIFIER — CLOSED (NEGATIVE)
- Gap: HC #589 R2 deliverable; script existed but had never run end-to-end (sector ETF load bug + previous sub-agent died after feature build).
- Action 2026-06-09: applied min-fix (load sector ETFs directly from price cache when _load_prices returns nothing for them — zero architecture change); ran full 23-fold WF; logged parent + 23 fold runs to MLflow experiment k6_meta_classifier_v1.
- Result: gate at P<0.45 cuts Sharpe 2.34 -> 1.43, CAGR 105% -> 39%, MaxDD -19.6% -> -35.2%, Calmar 5.36 -> 1.11. Every threshold 0.30-0.55 underperforms ungated baseline. Regime gap NOT closed (1.75 -> 1.87). HC #428 R1 FAIL in both.
- Verdict: meta-gate as specified does NOT add edge — it suppresses returns symmetrically while K=6 has asymmetric green/red regime behavior the gate cannot exploit.
- Findings: research/findings/k6_meta_classifier_v1.md (full diagnosis + 5 candidate next iterations).
- Closed: 2026-06-09 22:25 ET. Live K=6 paper state UNTOUCHED.

### P2-3 [A1 ETF] REGIME-FEATURE-GAP — CLOSED
- Gap: ETF rotation picker uses pure price momentum only; no VIX, yield curve, DXY, sector dispersion
- Fix: ETF Rotation v2 built with yield curve RoC, fed funds, graded regime gate. Sharpe 1.90, gap 0.04 (PASSES R1). ETF v3 (quality-weighted) Sharpe 2.39, gap 0.19. Both paper trading.
- Closed: 2026-07-10 ~15:30 ET — v2 and v3 deployed, paper A/B/C test running

### P2-4 [A3 WHEEL] IV-AWARE-PUT-PICKER — OPEN
- Gap: pick_short_put ranks by delta/dte only; iv_rank_floor was a documented gate, currently disabled
- Fix proposal: re-enable iv_rank_floor=0.25 + vol-rich sizing tilt when IV > median
- Status: queued (effort M)

### P2-5 [REPORTING] PROMPTS-INCOMPLETE — CLOSED
- Gap: EOD/morning/deep-check prompts didn't require Sharpe/Sortino/PF/WR (HC #69), % returns (HC #580), or regime split (HC #428 R1)
- Fix: amended all 3 prompt files to require the trio + HC #590 gaps-closed line
- Closed: 2026-06-09 19:20 ET

### P2-6 [ALL] CROSS-LANE DUPLICATION — OPEN
- Gap: 3 independent PaperState dataclasses, 3 cost models, 3 _log_trade impls; no shared base
- Fix proposal: extract live_trading_linux/paper_core.py; refactor engines to inherit
- Status: queued (effort M) — defer until P1 watchdogs land

## P3+ — Lower priority

### P3-1 [A3 WHEEL] NAV-DROP-ALERT — CLOSED
- Gap: no intraday NAV-drop alert; equity logged every 5 min but not compared to start-of-day
- Fix: added check_nav_drop() to wheel_v4_paper.py. Tracks SOD NAV in sod_nav.json, alerts once/day if drop ≥2.5%. Fires Discord webhook + log warning. Wired into run_cycle after NAV computation.
- Closed: 2026-07-05 ~07:38 ET

### P3-2 [A1 ETF] CASH-NEGATIVE — CLOSED
- Gap: cash_usd permitted to go negative (currently -$1441, implicit margin debt) without alert
- Fix: added cash-negative check after rebalance in etf_rotation_paper_engine.py. Logs margin debt always; fires Discord webhook if margin >10% of NAV.
- Closed: 2026-07-05 ~07:42 ET

### P4-1 [A2 K=6] SIZING — OPEN
- Gap: K=6 uses 1/K equal-weight only; no vol-target like A1
- Fix proposal: port A1's vol-target block
- Status: queued (effort S)

### P4-2 [INFRA] EVENT_TRIGGER GPU-IDLE LEAK PAST GAME-FILTER — OPEN (found 2026-06-09 ~21:35 ET)
- Gap: Persistent monitor's game-process filter (Wine/Steam/Deadlock) catches alert messages but the legacy EVENT_TRIGGER [NEPTUNE_GPU_IDLE] dispatch path still fires when Nick games on Neptune. False-positive on restart #32536 — GPU at 48% util was Wine/Steam, 5 min old.
- Fix proposal: extend filter to EVENT_TRIGGER emitter (not just send_to_discord wrapper); also sanity-check util threshold — true idle is <5%, not <50%.
- Status: queued (effort S, P3 priority — wakes Claude on user's downtime, burns tokens)

### P4-2-UPDATE 2026-06-09 ~21:38 ET
- Bidirectional leak confirmed: same Nick-gaming session also fired NEPTUNE_GPU_BUSY trigger (util=57%, Deadlock 4GB GPU mem). Both idle→busy and busy→idle transitions leak past the filter — single fix covers both.
- Better filter design: gate on nvidia-smi --query-compute-apps result; if dominant process matches steam/deadlock/proton/wine/explorer.exe, suppress the EVENT_TRIGGER regardless of util %.

### P3-1 [INFRA] MONITORING CRONS NOT DURABLE — OPEN (found 2026-06-09 ~21:42 ET)
- Gap: All 6 monitoring crons are session-only (in-memory). Every restart wipes them; recovery procedure recreates them by hand. Saw 3 restart-cycle cron-wipes in 15 min during an EVENT_TRIGGER storm tonight.
- Fix proposal: re-issue all 6 with durable:true so they persist to .claude/scheduled_tasks.json. Then update the /recovery skill to dedupe instead of blindly recreating (otherwise duplicates accumulate on every restart).
- Status: queued (effort S, P3 — monitoring goes dark in the gap between restart and next /recovery)

## 2026-06-09 22:42 ET — INFRA — CLOSED — Discord lint enforcement (HC #433 / #393)
- Gap: HC #433 (no garbage data) and HC #393 (no awaiting-approval) were enforced by self-discipline only. Repeated violations across sessions.
- Impact: user has explicitly called this out multiple times; wastes user time and trust
- Fix: built PreToolUse hook on mcp__discord__send_to_discord that scans every outbound message for banned patterns (paths, PIDs, hex hashes, reset counters, bare HC refs, wait-for-approval phrases) and injects a strong system reminder. Message still goes out (user always hears from us) but next message is corrected.
- Closed: 2026-06-09 22:42 ET — smoke tests pass on both banned and clean drafts. Hook wired into ~/.claude/settings.json. Active next session (and likely this one — settings watcher hot-reloads).

## 2026-06-09 22:43 ET — PROCESS — CLOSED — Single-brain orchestration (no specialist subagents)
- Gap: only one subagent (neptune-operator) existed. Every cluster check, deploy-gate validation, gap scan, backtest analysis was done by main Opus brain — slow and expensive.
- Impact: blocked the autonomy posture HC #393 demands (everything serialized through one expensive model)
- Fix: defined 5 specialist subagents in ~/.claude/agents/ with appropriate models:
  - discord-critic (Haiku) — rewrites drafts pre-send when active critique needed
  - cluster-watchdog (Haiku) — fast health snapshots for monitoring loops
  - deploy-gate-checker (Sonnet) — runs HC #428 R1/R2 against any config or MLflow run
  - gap-scanner (Sonnet) — runs HC #590 R2 5-category scan every cycle
  - quant-researcher (Sonnet) — full backtest analysis with regime split + deploy gates
- Closed: 2026-06-09 22:43 ET — 5 agents live and addressable via Task tool

## 2026-06-09 22:44 ET — PROCESS — CLOSED — Memory recall was pull-only (had to remember to call it)
- Gap: memory MCP existed but I had to manually invoke recall/check_pending. Often forgotten under load → prior decisions re-litigated, contradicted, or lost.
- Impact: HC #81 startup-reading-order rule got patchy enforcement
- Fix: built UserPromptSubmit hook that extracts keywords from every user message and greps DIRECTIVES.md / SESSION_STATE.md / RUN_HISTORY.md / IMPROVEMENT_BACKLOG.md for relevant prior context, then injects matching snippets as a system reminder. Silent on trivial messages (under 8 chars or 0 hits). Loud on substantive ones.
- Closed: 2026-06-09 22:44 ET — smoke test pulls ~5500 chars of relevant context for a regime-symmetry question, silent on "hi there"

## 2026-06-09 22:45 ET — INFRA — OPEN — P1: model routing not yet implemented
- Gap: I still use Opus for trivial lookups (file finds, log greps). Cheap Haiku subagent route exists but is not used by default.
- Impact: slower responses, higher cost
- Fix planned: lean on cluster-watchdog (Haiku) for status snapshots, discord-critic (Haiku) for message polish. Add Haiku-backed "file-finder" subagent for grep/glob tasks.
- ETA: this week

## 2026-06-11 00:2x ET — GAP FOUND+FIXED same cycle (HC #590 R3): stale OS-crontab inject set was post-HC#600 token-burn source
- OS crontab still fired old prompts (mamba 3h, deep_check 6h, usage 2x/day) with stale HC #565/#566 lane text — woke fresh sessions twice tonight with obsolete orders.
- FIX: consolidated to 3 injects (combined_pulse_4h.txt at 37 past 1,5,9,13,17,21 + morning 8:23 + EOD 15:41); pulse prompt reads DIRECTIVES first so it can't go stale. 5 bash watchdogs untouched. Backup in /tmp.
- Session-level crons RETIRED (were dying every restart + double-firing vs OS injects). SessionStart hook text rewritten in both settings files (JSON-validated): verify crontab, don't create session crons. Closes the P3-1 durability gap structurally.

## 2026-06-23 ~09:00 ET — MODEL — CLOSED — Mid-trade thesis validation (no in-trade intelligence)
- Gap: champion strategy "trades a screenshot" — enter, set fixed SL/TP, wait. No mid-trade intelligence. User explicitly flagged this as unacceptable.
- Impact: 77% of losing trades were actually right on direction but got stopped out on entry noise. No ability to detect losers early or confirm winners.
- Fix: built tick-level LightGBM classifier reading order book at T+5/10/15/30s. AUC 0.60-0.73. Best config: T+10s cut → Sharpe 13.08 (2.4× baseline), regime gap 0.33 PASS.
- Closed: 2026-06-23 09:00 ET — midtrade_thesis_v1 complete on Neptune, MLflow exp 16.

## 2026-06-23 ~12:45 ET — MODEL — CLOSED — Time/day filter not validated on correct dataset
- Gap: previous session claimed baseline FAILS regime gate (gap 0.54) — used wrong 223-trade dataset instead of 210-trade champion.
- Impact: false negative led to unnecessary filter complexity. Baseline actually passes with gap 0.12.
- Fix: re-ran on correct dataset. Mon-Wed filter → Sharpe 9.16, gap 0.024. Morning+Mon-Wed → Sharpe 9.08, WR 31.4%.
- Closed: 2026-06-23 12:45 ET — timeday_filter_analysis.py on Jupiter.

## 2026-06-23 ~13:00 ET — MODEL — IN-PROGRESS — Combined mid-trade + time/day filter stacking
- Gap: two independent improvements (time filter + mid-trade classifier) not yet tested together.
- Impact: potential Sharpe 15+ if improvements stack multiplicatively.
- Fix: dispatched midtrade_combined_v1 to Neptune. Running now.
- ETA: today
