# HC #424 — Live Performance Audit, 2026-05-18

**Author:** Claude Opus 4 (autonomous, HC #393 + HC #424)
**Generated:** 2026-05-18 ~18:15 ET (post RTH close)
**Scope:** Razer live host, both paper traders, full RTH session 09:30–16:00 ET.

---

## Bottom line

**P&L today: $0.00 (gross + net). 0 fills on either paper trader. WR / Sortino / Sharpe undefined (n=0).**

The HC #424 task brief asserted that PID 29600 was dead and that PID 25512 had won +$45.30 today. **Both assertions are wrong.** The audit corrects them below.

---

## Razer process state at 18:08 ET (post-close)

| PID | Name | Path | Start | CPU(s) | Status |
|-----|------|------|-------|--------|--------|
| 15720 | `python.exe mbo_recorder.py` | `C:\Users\claude\Lvl3Quant\mbo_recorder.py --symbol ESM6 --exchange CME --flush-minutes 5` | 5/14 07:33 ET | 2753 | ALIVE, recording (488 MB live_events.jsonl, mtime 18:07 ET) |
| 25512 | `python.exe paper_trading_mamba_v2.py` | `live_trading\paper_trading_mamba_v2.py --min-tier Top5%` | 5/14 09:30 ET | 417 | ALIVE BUT IDLE — counters frozen since 5/14 00:04 ET |
| 29600 | `pythonw.exe paper_trading_v2_1s_short_top05.py --shadow` | `live_trading_linux\paper_trading_v2_1s_short_top05.py` | 5/18 08:06 ET | 5801 | ALIVE, ingesting (n_signals_total=8710) but GATE-STARVED |
| 32980 | `pythonw.exe live_stack_watchdog.py` | `live_trading_linux\live_stack_watchdog.py` | 5/18 14:41 ET | 15 | ALIVE, alarming correctly but **Discord webhook not configured** |

**Correction to HC #424 brief:** PID 29600 was reported dead. It is alive (verified via `Get-Process -Id 29600` and `Get-WmiObject Win32_Process | Where-Object ProcessId -eq 29600`). The HC brief was looking at `Get-Process python` which does not include `pythonw.exe` — that's why PID 29600 didn't appear in the brief's snapshot. The shadow trader has been running continuously since 08:06 ET; CPU 5801s in ~10 hours = ~13% sustained utilization (consistent with ingesting 8710 prediction events over the session).

---

## Per-trader breakdown

### Legacy PID 25512 — `paper_trading_mamba_v2.py --min-tier Top5%`

**Today's contribution: 0 fills, $0 P&L.**

The "+$45.30 single trade" widely cited in HC #422 ("first confirmed profitable live paper trade") and HC #424 brief actually occurred on **2026-05-13** at approximately 09:14 ET, by a PREVIOUS process that wrote into the same log file `paper_mamba_v2_20260513_0618.log`. PID 25512 started 5/14 09:30 ET and inherited that log via append; its first own STATUS line (5/14 00:04 ET — actually a few minutes before its real PID start because the log's clock is the recorder's, not the trader's) already shows `events=729739 preds=1458 signals=1 trades=1 P&L=$45.30` and those numbers have NEVER moved since.

Evidence:
- 5/13 06:24 ET first STATUS line: `events=11954 preds=22 signals=0 trades=0`
- 5/13 09:14 ET: `events=647764 preds=1294 signals=1 1 trades WR=100% P&L=$45.30 risk_exits=1`
- 5/14 00:04 ET (and every 5-min STATUS line since, through 18:04 ET 5/18): `events=729739 preds=1458 signals=1 1 trades WR=100% P&L=$45.30 risk_exits=1` — **identical, character-for-character.**
- CPU 417s over 4 days alive ≈ 0.1% utilization. The legacy is heartbeating its log (5-min STATUS write keeps file mtime fresh and fools the watchdog's JSONL-mtime check) but not consuming the live event stream.

This is a textbook **counter-content freeze** failure: the process is alive, the log writes, but the values logged are static. The HC #422 Rule 1 watchdog catches counter-value-staleness in the shadow's status JSON (because the watchdog parses the JSON), but for legacy there is no status JSON, only the log line — so the watchdog only checks log mtime, not log content. **Blindspot.**

Why is PID 25512 not ingesting? Likely candidates:
- Lost MBO event-bus subscription at some moment (the recorder restarted? a pipe broke?).
- Failed silently to re-attach after a transient error.
- Without a stack trace we cannot pin it; restart will be diagnostic.

Per HC #424 R4 + HC #422 R6, legacy must NOT be killed without user OK (it's the "preserved +$45 winner"). However, the preserved winner has actually been non-functional for 4 days. **Recommendation (escalation, not action):** kill and relaunch PID 25512 in tomorrow's pre-open after confirming with user, with logging that captures attach/subscribe state.

### Shadow PID 29600 — `paper_trading_v2_1s_short_top05.py --shadow`

**Today's contribution: 0 fills, $0 P&L. 8710 predictions made. 0 passed the gate.**

JSONL summary (`v2_1s_short_top05_paper_20260518_080606.jsonl`, 252 lines, all `event=discord_alert`):
- 12:06 UTC (08:06 ET): "engine loaded. shadow=True symbol=ESM6"
- 12:06 UTC: "LIVE START — symbol=ESM6 shadow=True"
- 20:23 UTC (16:23 ET) → present: alternating `TEMP HALT 60s: stale_signal (>30.0s no prediction)` and `MANUAL HALT (USER ACTION NEEDED): broker_connectivity_loss (>5.0s)`.

The kill-switch latched into broker_connectivity_loss MANUAL HALT around 17:47 ET. Per the kill-switch implementation, MANUAL HALT requires user re-arm — it does NOT auto-clear. While in MANUAL HALT, the trader sits idle even though the model is still producing predictions.

But the more important finding is from the watchdog status snapshot: `n_signals_total` IS advancing (latest 8710 at 18:09 ET, was advancing at every 16-min watchdog cycle), and `n_signals_passed_gate=0` since the watchdog's first observation at 18:41 UTC (14:41 ET — exactly when the watchdog started). **So the trader has been generating predictions all day but ZERO predictions have passed the entry gate.** This is consistent with HC #423 §3's hypothesized +0.21 pred mean shift due to encoder mismatch: every prediction is pushed positive (long bias), but the trader's `--shadow` config only enters SHORT at the top-0.5% confidence tail, so a positive bias makes the SHORT tail unreachable.

### MBO recorder PID 15720

Healthy. 488 MB live_events.jsonl, fresh mtime, ESM6 + CME, 5-min flush. No findings.

### Watchdog PID 32980

Started 14:41 ET. Has been correctly firing alarms every 16 min for shadow's GATE-STARVED + JSONL-STALE conditions. **Critical gap: `webhook_set: false` in the startup event — the watchdog cannot find the Discord webhook URL because neither `$env:DISCORD_WEBHOOK_LIVE_STACK` nor a `.env` file with that key is set on Razer.** All "alarm" events are written to `logs/watchdog/watchdog_events.jsonl` but nobody is reading that file in near-real-time. The user has no idea the watchdog has been screaming for 4 hours.

---

## Why the silent failure went undetected

1. **No Discord webhook on the watchdog** → all alarms ring into the void on Razer's local disk.
2. **Watchdog blind to legacy counter-content freeze** → legacy registry entry has empty `counters: []` and no status-file path, so only log mtime is checked. The legacy's 5-min STATUS heartbeat keeps mtime fresh, satisfying the watchdog while the actual counter values never change.
3. **Kill-switch MANUAL HALT requires user re-arm but user is not notified** → trader self-disables and stays disabled.
4. **HC #424 brief snapshot used `Get-Process python` which excludes `pythonw.exe`** → shadow looked dead in the audit but was actually alive.

---

## Recommendations (for tomorrow's pre-open)

| # | Action | Owner | Priority |
|---|--------|-------|----------|
| 1 | Configure Discord webhook on Razer: write `live_trading_linux\.env` with `DISCORD_WEBHOOK_LIVE_STACK=<url>`, restart watchdog | Claude | P0 |
| 2 | Patch watchdog to PARSE legacy STATUS log lines (regex on `events=<n> preds=<n>`) and alarm when the parsed values don't change | Claude | P0 |
| 3 | Investigate why legacy PID 25512 lost ingestion; restart with subscribe-attempt logging | Claude (with user OK) | P1 |
| 4 | Apply HC #423 §3 encoder fix to shadow: port `paper_trading_mamba_v2.py` lines 727–751 encoder path; verify pred mean shifts back to ~0 | Claude | P0 |
| 5 | Clear shadow's MANUAL HALT state file pre-open and add auto-re-arm logic with operator override (HC #422 R1 + R6) | Claude | P1 |

---

## Answer to user's question "how was live performance today?"

**Net P&L: $0. Zero trades fired. Both paper traders were online but neither traded:**

- The legacy "preserved +$45 winner" has been internally frozen since 5/14 and is logging stale stats from 5/13. Today it contributed nothing.
- The new shadow v2_1s_short_top05 ran all day, made 8710 predictions, but the gate let zero through (consistent with the +0.21 pred-mean encoder bug from HC #423). It then latched into broker-connectivity-loss MANUAL HALT around 17:47 ET and has been alarming-but-silent for hours because Razer has no Discord webhook configured.

Tomorrow's pre-open priorities are: (1) wire the Discord webhook, (2) patch the watchdog's legacy blindspot, (3) apply the HC #423 §3 encoder fix, (4) restart legacy with subscribe diagnostics.
