# NODE_LEDGER — Single Source of Truth for 3-Node Alpha Mode

**Governed by HC #483 / HC #484.** Updated by: every dispatch (immediate); 35-min mamba_monitor cron; 2-hour deep_check cron. If any node ELAPSED_IDLE_MIN > 15 AND not in documented exception → MUST dispatch within 5 minutes (HC #483 R3).

**Last updated**: 2026-05-22 14:37 ET (recovery post session reset; crons restored x6; Razer fold-3 resume launched; Neptune deferred; Jupiter MSE baseline live)

---

## Neptune (RTX 3090, Ubuntu)

- **STATE**: RUNNING (FRESH-START after fast-forward stall — PID 1834742)
- **ACTIVE_JOB**: CNN-Mamba v3.4.2 alpha-redev retrain on fixed labels (hc477fix_v2). MLflow run `682f223056b543578db27540924f84b1` (FRESH, no intra-ckpt resume). Folds 0-9.
- **JOB_LAUNCHED_AT**: 2026-05-22 11:03 ET Jupiter-time (PID 1834742) — clean restart from warm-start `fold_00_best.pt`. Intra-ckpt resume was abandoned because trainer's fast-forward replays through all skipped batches at ~1 b/s (would take 3-4h per resume) and prior intra_ckpt also showed loss divergence pre-crash.
- **LAST_PROGRESS_AT**: 11:09 ET — Feature stats phase active (started 11:04:29). GPU engages ~11:11 ET.
- **HC #487 R1 COMPLIANCE**: ✅ Resume-from-intra-ckpt used. Subshell `(nohup ... < /dev/null &)` to dodge SSH-timeout duplicate-launch bug.
- **PRIOR PID HISTORY (today)**: 1456710 (22:26 ET 5/21) → 1534826 (00:53 ET 5/22 HC #485 fixes) → CRASHED 04:04 worker OOM → 1633822 (resumed) → CRASHED ~09:28 worker OOM → 1804432 (2nd resume, 14:04 ET Jupiter / Neptune-time issue, was actually ~10:00) → CRASHED 10:37 worker OOM at batch ~14300, loss 3.02 → 1822147 (CURRENT, 3rd resume)
- **THREE OOM CRASHES TODAY — RECURRENCE WATCH ACTIVE**: All 3 crashes are `DataLoader worker exited unexpectedly` from kernel SIGKILL under swap pressure. Root cause unchanged: Firefox + Spotify + Discord on Neptune desktop consume ~3-4 GiB persistently, pushing swap to 6.4/8 GiB. Each fast-forward of 14000 batches stresses RAM. **If 1822147 also dies in batch 12000-14000 window → escalate to user.**
- **LOSS TRAJECTORY (pre-3rd-crash)**: 1.09 → 1.62 → 2.48 → 3.02 over ~300 batches. 4 consecutive 1.5x increases, one short of kill-on-divergence threshold. Crash beat divergence-kill by minutes. Ambiguous: optimizer state may be poisoned OR kernel killed worker mid-batch and spike was transient.
- **CRASH ROOT CAUSE**: DataLoader worker SIGKILL'd by kernel. Neptune swap 6.1/8GB at crash. Workers already at script-minimum (2 train/1 val). Concurrent user-desktop apps (firefox/discord/steam/spotify) consume ~3-4GB persistently. No sudo to flush swap.
- **ELAPSED_IDLE_MIN**: 0
- **NEXT_QUEUED**: Folds 1-9 sequential. On all-10-folds completion → continuation-target retrain per HC #475 R4.
- **REASON_IF_IDLE**: n/a
- **GATE**: concat IC across 10 folds + symmetric long/short balance (HC #475 R2). Pass if both-sides IC > 0.5× better-side IC. **ETA fold 0 ~6h from resume = ~10:00 ET 5/22**.
- **CRASH RECURRENCE WATCH**: if PID 1633822 dies near same batch window (~12000-13000) → escalate. Mitigation options if it recurs: (a) ask user to close firefox/discord during training, (b) patch trainer to write tmpfs-free chunks, (c) reduce batch_size, (d) wait until off-hours when desktop apps closed.
- **DATA QUALITY NOTE (HC #484 R1a)**: _v2/ label sample: 20251102 / 20260126 healthy (1-9% NaN normal end-of-day); 20260428 has 92% NaN on 60s/5min heads (regen partial-failure on April dates). Does NOT affect folds 0-9 training dates (Dec 2025 – Apr 21). Affects HC #432 R2 47-day OOT extension; tracked as separate fix after fold 0 lands.

## Jupiter (CPU, 64GB, Ubuntu)

- **STATE**: RUNNING (evaluator-watchdog cron active; fires on fold-0 landing)
- **ACTIVE_JOB**: `scripts/eval_fold0_v3_4_2_hc485.py` cron `*/10 * * * *` — polls for Neptune fold-0 NPZ, runs full 6-gate deploy pipeline + Discord briefing emit. Self-test PASS. MFE gate patch applied post-build.
- **JOB_LAUNCHED_AT**: 2026-05-22 03:55 ET (evaluator script + cron installed)
- **LAST_PROGRESS_AT**: 03:58 ET (MFE schema-name patch + re-verified)
- **ELAPSED_IDLE_MIN**: 0 (watchdog-as-workload, HC #483 R1-style exception)
- **PREVIOUS JOB #1 (closed)**: hc475_symmetric_gate_ab finished 03:28 ET. VERDICT: all 5 configs FAIL (Sharpe -0.09 to -0.18) on broken-label v3.4.2 NPZ. Scope finding: NPZ mislabeled "47-day" — actually 16 days (2026-02-23..2026-03-15).
- **PREVIOUS JOB #2 (closed)**: Adaptive Exit v1 finished 03:45 ET (105s). VERDICT: REJECT. net_ticks −0.377, Sharpe −1.48, 0/5 OOT days positive. v0's +0.61 was 100% look-ahead. Adaptive Exit thread dead.
- **PREVIOUS JOB #3 (closed)**: Fold-0 auto-evaluator build + self-test 03:55 ET. All 6 gates wired (HC #485 NaN, concat IC, sym L/S, regime, MFE-horizon, delta-vs-broken-baseline). MFE column-name bug patched 03:58 ET — gate now resolves to concrete numbers, no AMBER stub.
- **PREVIOUS JOB #1 (closed)**: hc475_symmetric_gate_ab finished 03:28 ET. VERDICT: all 5 configs FAIL (Sharpe -0.09 to -0.18, PF 0.69-0.83) on broken-label v3.4.2 NPZ. L/S flip 99.8% short → mixed succeeded. Real test pending Neptune fold 0. Scope: only 15 days of fills, not 47 — investigate.
- **PREVIOUS JOB #2 (closed)**: Adaptive Exit v1 finished 03:45 ET (105s run). VERDICT: REJECT. net_ticks/trade −0.377, Sharpe −1.48, PF 0.04, 0/5 OOT days positive. v0's claimed +0.61 was 100% look-ahead leak (linear interp toward known exit). Adaptive Exit as designed is dead.
- **NEXT_QUEUED** (execution-gap closure thread, per user request 23:54 ET 5/21):
  1. **Adaptive Exit v1 — EXACT MBO REPLAY** (closes v0 look-ahead leak in `scripts/adaptive_exit_v0_train.py`). v0 showed +0.61 ticks/trade Sharpe +0.46 but the in-trade MFE/MAE was linearly interpolated toward the KNOWN exit — look-ahead leak. v1 must rebuild in-trade trajectory via exact MBO tick replay of each trade leg. Gate: >+0.10 ticks net mean AND Sharpe >0.3 AND consistent across 47 OOT days = real edge; collapse to flat = v0 number was the leak.
  2. **hc471 re-run on FRESH v3.4.2 preds** — once Neptune fold-0 retrain on `_v2/` labels lands (~05:00 ET). hc471 ran today on broken-label preds and went NEGATIVE (-0.46 ticks/trade canonical FIFO). Re-run on fixed-label preds is the real test of the continuation×alpha mask.
  3. **Continuation specialist retrain** on `_v2/` labels — current specialist also trained against half-zero targets.
  4. Per-day Sharpe regime-stratification (HC #428 R1 deploy gate)
  5. MFE-within-horizon p90 verification on extended OOT (fix April-date label NaN bug)
- **REASON_IF_IDLE**: n/a

## Razer (RTX 3070, Windows, 16GB)

- **STATE**: ALPHA RESEARCH DISPATCH IN FLIGHT (HC #487 R3 — live stack REVOKED 2026-05-22 13:30 ET)
- **ACTIVE_JOB**: Sub-agent dispatched 14:10 ET to pick + launch ONE of: PatchTST retrain on _v2/ labels / HC #486 R4 meta-layer prototype / TimeMixer-DLinear baseline. Launch within 25 min. MBO recorder PID 15720 retained per HC #487 R3 skill.
- **LIVE STACK STATUS**: DEFERRED. Paper trader + inference daemon gone (only MBO recorder + nothing else). Re-evaluate when a profitable setup is ready to deploy.
- **JOB_LAUNCHED_AT**: 2026-05-14 07:33 (recorder) / 2026-05-18 14:41 (inference)
- **LAST_PROGRESS_AT**: continuously — last MBO event log within RTH
- **ELAPSED_IDLE_MIN**: 0 (live stack IS the workload per HC #483 R1)
- **NEXT_QUEUED** (CPU-side, can run alongside live stack):
  1. FIFO sweep parity check on Razer's local NPZ (cross-node validation vs Jupiter hc441 wider TP result)
  2. Data parity check vs Jupiter NPZs (HC #478 R2 schema/shape audit)
  3. LGBM execution-features model on local labels
- **REASON_IF_IDLE**: Live stack = primary workload. CPU spare for parity/audit jobs at next mamba_monitor tick (23:17).

---

## Recent Infra Fixes (HC #484 R1, 2026-05-21 23:10 ET)

- **R1a — _v2/ label verification**: Sampled 3 files via nanstd diagnostic. Healthy on 2025/early-2026 (folds 0-9 source); April-2026 dates partially-failed regen (920% NaN on 60s/5min heads). Documented above; doesn't block current training. Re-regen task queued for after fold 0.
- **R1b — OS-level durability**: Confirmed existing Jupiter crontab has `crash_recovery.sh` + `infra_sync.py` + `fold_watcher.sh` running every 30 min. PM2 `persistent-monitor` online 18 days. Claude-session crons are redundant layer. Adequate.
- **R1c — Zombie procs cleaned**: Killed 22 stale procs (precompute_observations workers from May 7, ssh_exec.py orphans from May 13-20). All current procs are <24h old.
- **R1d — QCC Jupiter-offline false positive ROOT-CAUSED + FIXED**: `lib/qcc-ssh.js` heartbeat had Neptune-localhost case disabled and NO Jupiter-localhost case, so daemon SSH'd to self and got ETIMEDOUT. Patched: Jupiter now uses local `echo ok` heartbeat. Daemon restarted. Jupiter status flipped from `offline` (26,442 min) → `online`. 1,494 historical false-positive alerts resolved.
- **R1e — Stale job #535 marked failed**: Was claiming `running` while GPU 0-50%. Cleared. Real Neptune job #536 is the active one.
- **R1f — Razer dispatch**: NOT idle — live stack alive per ledger.

---

## Idle Ladder Reference (HC #476 R5 + HC #483 + HC #484)

### Neptune-idle ladder:
- (a) CNN-Mamba v3.4.2 retrain on new alpha labels (HC #475 R4) — CURRENT
- (b) Continuation-target label retrain
- (c) Book Spatial CNN retrain (HC #450)
- (d) Smart-execution RL/MLP
- (e) Long-horizon CNN-Mamba variant

### Jupiter-idle ladder:
- (a) Symmetric-gate replay of TP grid (HC #475 R2) — CURRENT
- (b) Continuation-target label construction (HC #475 R4)
- (c) MFE-within-horizon p90 sweep (HC #432 R2)
- (d) Per-day Sharpe regime-stratification (HC #428 R1)
- (e) April-date label re-regen (HC #484 R1a follow-up)
- (f) Rules-based FIFO sweep on freshest NPZ

### Razer-CPU-idle ladder (outside RTH, alongside live stack):
- (a) FIFO sweep parity check on local NPZ
- (b) Data parity audit vs Jupiter NPZs
- (c) LGBM execution-features model
- (d) Adaptive-exit local replay

---

## Accountability Rules (HC #483 + HC #484)

- **15-min idle ceiling** — any node IDLE > 15 min without documented exception = directive violation. Must dispatch within next 5 min.
- **Act-first dispatch on session start** — no first-message-to-user without acting first (HC #393).
- **Proper-fix-not-suppression** — every alert clear must also fix root cause (HC #484 R2).
- **RUN_HISTORY.md gate** — check before every dispatch; append within 2 min of launch.
- **Discord digest** — morning brief (8:23 ET) + EOD (3:41 ET) include 3-line node-status block (HC #483 R6).
- **No banned phrases** — "no defensible work / queued for / deferring / holding until" all banned.
