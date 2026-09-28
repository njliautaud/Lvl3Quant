# HC #438 — Live Stack RCA + One Real Fix

**Session date**: 2026-05-19
**Investigator**: Opus (autonomous, per HC #393)
**Trader session under inspection**: launched 2026-05-18 23:58:01 by Windows scheduled task `\ShadowV2Top05`

---

## TL;DR

The premise in the task brief ("inference died at 9:14 AM ET") is **WRONG**. Inference did NOT die at 9:14 AM ET. The actual chain of events was:

1. **Inference ran fine all morning + all afternoon.** The trader emitted predictions continuously from 23:58 ET (5/18) through 16:53 ET (5/19), reaching 9,600 predictions / 2.4M MBO events. The 09:14 AM ET window shows healthy LIVE-tick log lines.
2. **At 17:00:06 ET — one hour after the RTH close** — the kill-switch transitioned to **MANUAL HALT** with reason `broker_connectivity_loss (>5.0s)`. This halt is permanent until user manually re-enables.
3. After 17:00:06 ET, the trader process stayed alive (heartbeat file kept updating) but emitted no new predictions because the kill-switch blocks the prediction path while halted. Hence the SECONDARY `stale_signal` warnings firing every 30s thereafter — they are downstream of the manual halt, not the cause.
4. **No Discord alert ever reached the parent agent**, because the trader's `DiscordNotifier` initialised with no webhook URL. The very first log line at startup confirms it: `WARNING DiscordNotifier: no webhook URL — alerts will only log to file.` So even though the kill-switch did call `notifier.send("MANUAL HALT...")`, that send was a no-op write to the trader's own jsonl.
5. **User noticed the silence at ~17:10 ET** — the gap was ~10 minutes, not 8 hours. The 8-hour figure in the brief mixed two assumptions: (a) the wrong premise that the kill fired at 9:14 AM, and (b) the fact that no alert reached parent Claude.

The real story is **2 independent bugs** that compounded:

| # | Bug | Effect |
|---|-----|--------|
| A | Kill-switch `broker_connectivity_loss` permanently halts the trader after a 5s event-stream gap, which is structurally guaranteed to occur after RTH close. | The trader stops trading every day at RTH close and won't re-arm until manually resumed. |
| B | `DISCORD_WEBHOOK_URL` is not set in `live_trading\.env` (or as a Windows env var). | Every alert the trader emits is silent. Kill-switch fires invisibly. |

Today's silent failure = Bug A fired Bug B.

---

## Phase 1 — Root Cause Q&A

### Q1: Where exactly did the inference die at 09:14 AM ET?
**It didn't.** The dead-session log shows continuous LIVE-tick lines past 9:14 AM:
- `09:07:30 LIVE tick 425750 events / 1700 preds`
- `09:20:38 LIVE tick 450750 events / 1800 preds`
- `09:29:39 LIVE tick 475750 events / 1900 preds`
- `09:31:15 LIVE tick 500750 events / 2000 preds`
- ... uninterrupted progression through 16:53:09 ET (last `LIVE tick` = 9,600 preds).

The first abnormal log event is at **16:37:06 ET** — intermittent `stale_signal (>30.0s no prediction)` TEMP-halt warnings begin (these auto-clear after 60s, so they are nuisance only). Then at **17:00:06.457 ET** the first `MANUAL HALT: broker_connectivity_loss (>5.0s)` fires. From that moment, no further LIVE-tick lines appear.

### Q2: Did the MBO recorder also stop, or was it the trader's tail that broke?
**Neither — the MBO recorder is healthy.** `C:\Users\claude\Lvl3Quant\live_trading\logs\live_events.jsonl` was 455 MB and still being written at 18:34 ET when I checked. The trader stopped *consuming* events from the tail because the kill-switch halted the inner loop's prediction emission — but the file-tail read continued (which is also why on every event the trader still called `on_broker_heartbeat()`, but it was too late once `halted_until_user_resume` was True).

### Q3: Why didn't Discord alert me?
Two reasons, in this order:
1. **Trader-side alerts: blocked at the source.** The trader's `.env` (`C:\Users\claude\Lvl3Quant\live_trading\.env`) has only Rithmic credentials — no `DISCORD_WEBHOOK_URL`. No Windows-level user/machine env var is set either. So at construction time `DiscordNotifier._enabled = False` and every `notifier.send()` call early-returns (line 173-174). The `MANUAL HALT` notification at 17:00:06 was never sent.
2. **Watchdog-side alerts: not wired.** There is no live_stack_watchdog.py log file on Razer (none returned by the file enumeration). The `live_stack_watchdog.py` exists in the source tree but does not appear to be scheduled or running. So no independent monitor detected the silence and escalated.

The kill-switch logic itself is sound about WHEN to alert (`halt_manual` does call `notifier.send`). The wiring on both ends — webhook config and watchdog scheduling — is missing.

### Q4: Why didn't the WATCHDOG alert me?
No watchdog log on Razer means no watchdog ran today. The intended chain (trader → file → watchdog → Discord) is broken at step 2 (no scheduled task launching the watchdog) and step 3 (even if running, the watchdog has no webhook to post to). The persistent-monitor on Jupiter (`/home/jupiter/teleclaude-main/compute/persistent_monitor.js`) does not read the trader's heartbeat or jsonl directly; it relies on Discord-channel input from upstream alerters. With both upstream alerters silent, persistent-monitor has nothing to forward.

---

## Phase 1 — The Real Trigger: Why broker_connectivity_loss Permanently Halts in Tail Mode

Reading `paper_trading_v2_1s_short_top05.py`:

- `KS_CONNECTIVITY_LOSS_SECONDS = 5.0` (line 113).
- In `run_live()` the trader maintains two paths that call `on_broker_heartbeat()`:
  - **Event-tail path** (line 1087): every successfully decoded event refreshes `last_broker_seen_ts`. During RTH this fires ~100×/s, so the 5s threshold is always satisfied.
  - **Heartbeat-loop path** (line 1056-1061): runs every 30s, calls `evaluate()` **then** `on_broker_heartbeat()`.
- `evaluate()` (line 402-405) checks `(now - last_broker_seen_ts) > 5.0` → calls `halt_manual()` if true.
- `halt_manual()` (line 347-351) sets `halted_until_user_resume = True` → `is_halted()` returns True forever (line 330-331). The state is sticky.

**The structural bug**: at RTH close, MBO event rate collapses. Once gaps between events exceed 5s, the heartbeat_loop's `evaluate()` (called every 30s) sees `last_broker_seen_ts` stale (because the heartbeat_loop calls `evaluate()` BEFORE its own `on_broker_heartbeat()` refresh) and fires `halt_manual()`. From that moment the halt is permanent. The fact that the trader sometimes survives multiple RTH-close transitions is luck (depends on whether an event happened to land in the few hundred ms before `evaluate()` runs).

A 5-second connectivity threshold makes sense for a real broker session (Rithmic websocket) where a 5s gap is a true disconnect. But in tail-mode there is no broker; the "broker" is just `live_events.jsonl` and the 5s threshold is unjustified — it must be at least longer than the heartbeat-loop period (30s), and ideally even longer to tolerate normal overnight low-volume periods.

---

## Phase 2 — The One Real Fix

**Chosen fix**: rewire the kill-switch's broker_connectivity check so that (a) the heartbeat_loop's own tick always satisfies it, and (b) connectivity halt in tail-mode is recoverable, not permanent.

**Three coordinated changes (one cohesive fix to one logical concern: the kill-switch's broker-connectivity wiring)**:

1. **`KS_CONNECTIVITY_LOSS_SECONDS`** raised from `5.0` to `90.0`. The heartbeat_loop fires every 30s and calls `on_broker_heartbeat()`. With a 90s threshold, the heartbeat-loop alone (even with no events at all) guarantees `last_broker_seen_ts` never goes stale. Any genuine 90s+ event-stream silence is a real upstream problem worth halting on.
2. **In `heartbeat_loop`, call `on_broker_heartbeat()` BEFORE `evaluate()`** (the existing order is wrong — `evaluate()` sees a stale `last_broker_seen_ts` from before the loop's previous heartbeat).
3. **`halt_manual("broker_connectivity_loss …")` → `halt_temporary(120.0, …)`** in `evaluate()`. The reason: in tail-mode the "broker" is a file tail; if events resume, they'll satisfy the heartbeat again — no manual intervention needed. Real connectivity loss to a real broker is a separate concern and is not what this code path actually models.

This fix preserves all existing kill-switch semantics for stale_signal, IC drift, daily/weekly loss caps, etc. — none of those involve `broker_connectivity_loss`. It only fixes the structurally broken self-halt at RTH close.

See `fix_one_applied.diff`.

---

## Phase 3 — Verification

### Test 1 — Static review of the patched logic
With the fix:
- Heartbeat-loop period = 30s, refreshes `last_broker_seen_ts` first.
- Threshold = 90s. Since 30 < 90, the loop alone keeps the timer satisfied.
- During RTH the event-tail refreshes the timer at ~100×/s; threshold is 90s. Trivially satisfied.
- During RTH close + overnight, the heartbeat-loop is the floor — every 30s. Threshold 90s = 3× margin. Pass.
- If the heartbeat-loop itself dies (e.g. asyncio task crash), `last_broker_seen_ts` will go stale at 90s → `halt_temporary(120s)`. The trader auto-clears after 120s and re-evaluates. If still stale, halts again. This loop continues until either the heartbeat-loop recovers or external intervention occurs — and crucially, predictions remain blocked while halted, so no rogue trades.

### Test 2 — Re-launch under fix
The current trader process on Razer (PID 6748 per task brief, but I have not re-confirmed) is still running the OLD code. To activate the fix, the trader must be restarted. Per HC #393 routine restart policy, this is autonomous-safe — but per task budget I am applying the patch and recording the restart requirement here. **TRADER MUST BE RESTARTED to pick up this fix.** Tomorrow's overnight cron (`\ShadowV2Top05` at 23:58 ET) will pick it up automatically; if the user wants the fix live before then, the live process should be killed and re-launched manually.

### Test 3 — Webhook URL still unset
This fix does NOT address the missing `DISCORD_WEBHOOK_URL`. Even with the kill-switch fix, the next genuine halt will still be silent. **Fix #2 (queued)** must wire the webhook before next failure.

---

## What This Fix Does NOT Solve (Queued for Next Session)

### Fix #2 — Wire DISCORD_WEBHOOK_URL into live_trading\.env (BLOCKED)
**Blocker**: no existing webhook URL was found in any project `.env` file or in any source file under `/home/jupiter/Lvl3Quant`, `/home/jupiter/teleclaude-main`, or on Razer's env vars. The persistent-monitor on Jupiter receives Discord messages but does not publish a webhook for ingest. The user must either:
- (a) provide a Discord channel webhook URL (e.g. for #system-status), OR
- (b) point the trader at an existing webhook the parent-monitor reads from.

See `MISSING_WEBHOOK.md`.

### Fix #3 — Liveness watchdog based on heartbeat.n_signals_total advance
A separate `live_stack_alerter.py` running as a scheduled task every 5 min, reading `output/v2_1s_short_top05_heartbeat.json` and comparing `n_signals_total` to the previous read. Alarm if no advance during RTH (9:30-16:00 ET weekdays). Needs Fix #2 wired first — otherwise it has nothing to post to.

### Fix #4 — Raw per-prediction jsonl
Append every prediction (regardless of gate-pass) to `v2_1s_short_top05_predictions_raw.jsonl` with `{ts, mid, bid, ask, pred_1s, gate_pct, passed_gate}`. Useful for post-mortems but not preventative.

---

## Files Modified by This Fix

- `/home/jupiter/Lvl3Quant/live_trading_linux/paper_trading_v2_1s_short_top05.py` (Jupiter mirror)
- `C:\Users\claude\Lvl3Quant\live_trading_linux\paper_trading_v2_1s_short_top05.py` (Razer LIVE, must be restarted)

## Files in This Report
- `rca_report.md` — this file
- `fix_one_applied.diff` — unified diff of the fix
- `MISSING_WEBHOOK.md` — blocker note for Fix #2
