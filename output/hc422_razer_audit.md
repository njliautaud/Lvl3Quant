# HC #422 Rule 7 — Razer Live-Trading Stack Robustness Audit

**Date**: 2026-05-18 (PM ET) | **Scope**: All Razer-side processes feeding the live paper trading loop. | **Trigger**: PID 29600 shadow `v2_1s_short_top05` fired zero signals for 5.5h without detection. User caught it first. HC #422 Rule 1 violation. **Read-only audit per HC #420 authorization.**

---

## 1. Severity Table — Failure Modes vs. Coverage

Legend: **C** = current detection method, **G** = gap, **R** = recommended fix, **E** = effort (S = <1h, M = 1-4h, L = 0.5-2 days, XL = >2 days).

| Sev | # | Failure mode | C (detection today) | G (gap) | R (fix) | E |
|---|---|---|---|---|---|---|
| **CRIT** | F1 | Shadow `passed_gate=0` for full RTH session (regime drift, HC #421 Issue A) | Watchdog Rule-1 alarm now wired; before today: nothing | Was the 5.5h silent failure. Alert wired but only AFTER deploy of watchdog; no fallback if Discord 4xx/5xx | Already covered by `live_stack_watchdog.py` + Jupiter-side alerter polling status JSON. Must verify deployed | S (verify) |
| **CRIT** | F2 | MBO recorder dies → all downstream paper traders go silent but `psutil.is_alive()` for the paper traders still True | JSONL mtime check on `live_events.jsonl` (in watchdog registry) | Watchdog only knows about file mtime — if recorder restarts cleanly but loses Rithmic session, file may still get a stale rotation write. No `last_rithmic_tick_ts` heartbeat from recorder | Augment `mbo_recorder.py` to emit `logs/mbo_recorder_status.json` every 5s with `{events_counter, last_tick_ts, rithmic_connected}`. Register in watchdog. | M |
| **CRIT** | F3 | Paper traders not writing the heartbeat JSON the watchdog expects | Watchdog registry lists `output/v2_1s_short_top05_heartbeat.json` + `logs/paper_status.json` (legacy) | `paper_trading_v2_1s_short_top05.py` DOES write `HEARTBEAT_PATH` (verified line 128) every 60s via `maybe_write_heartbeat`. **Legacy `paper_trading_mamba_v2.py` does NOT write a JSON status** — watchdog cannot fire STALL alarms on legacy, only DEAD + JSONL-mtime | Add a periodic status-writer thread to legacy trader: `{events_counter, preds_counter, fills_today, model_sha, last_event_ts}` every 10s | M |
| **CRIT** | F4 | Watchdog process itself dies (unhandled exception in psutil iter, perm error on a JSONL glob, etc.) | None on Razer; Jupiter alerter detects `watchdog_status.json` stale > 3 min RTH | If Jupiter dies, watchdog death is invisible. Currently: alerter is a one-shot script run via cron — no PM2/systemd guarantee of "Jupiter alive" | Wrap watchdog in a `while True: try/except` outer loop with restart-self via Win32_Process.Create on fatal. Add PM2 keep-alive on Jupiter alerter. | M |
| **CRIT** | F5 | Discord webhook URL rotated / forbidden / silently 4xx → all alarms go to /dev/null | `post_discord()` logs failure to local JSONL; Jupiter alerter falls back to `alerter_pending.jsonl` | No second-channel egress (no Telegram fallback, no QCC alert hook). If Jupiter dies + Discord broken, user sees nothing | Add Telegram bot fallback in `razer_watchdog_alerter.py`. Also `qcc_alert_send` MCP call on every CRIT-severity alarm | M |
| **HIGH** | F6 | Rithmic auth expires (paper account session token) → no events, recorder process alive | JSONL mtime stall → watchdog fires after 10 min RTH | 10 min is too generous for a critical feed; ES at RTH has ~dozens of quotes/sec — 30s silent is already broken | Reduce JSONL stall threshold to **90s RTH** for MBO recorder specifically (per-process override in registry). Add explicit `last_rithmic_tick_ts` to status file | S |
| **HIGH** | F7 | GPU OOM on RTX 3070 8GB (legacy 769MB + shadow 87MB + future N traders) | None | Watchdog doesn't poll VRAM. Inference returns silent garbage on torch OOM in some code paths | Add `nvidia-smi --query-gpu=memory.used,utilization.gpu` poll to watchdog. Alarm `memory.used > 7000 MiB` OR `utilization.gpu == 0` for >2 cycles during RTH | S |
| **HIGH** | F8 | Tailscale tunnel down → Jupiter alerter sees `ssh_fail`, watchdog Discord posts still work IF Razer has internet | Jupiter alerter logs `ssh_fail`; eventually fires "ALERTER CANNOT REACH RAZER" | If BOTH Tailscale AND Razer-side Discord broken (e.g., Razer offline entirely), nothing pages. Also: Jupiter has no clock-skew check vs Razer | Add a Razer-side direct Discord ping every 5 min (independent of process state). Jupiter cron also pings Tailscale subnet health | M |
| **HIGH** | F9 | Reboot of Razer (Windows Update) — what auto-resumes? | None — all current processes were manually launched via Win32_Process.Create after the last reboot | **Nothing auto-resumes on reboot today.** No Task Scheduler entry; no `RunOnce`. Manual relaunch required for: MBO recorder, legacy paper, shadow, watchdog | Register each as a **Task Scheduler "At startup, run as user, highest privileges"** task pointing at the same `launch_*.bat` files. Test by rebooting Razer once in off-hours. | L |
| **HIGH** | F10 | Disk full on `C:\` (paper JSONLs grow unbounded — one trader generates ~50-200 MB/RTH session) | None | No log rotation; no disk-space alarm | One-line `shutil.disk_usage()` poll in watchdog (alarm < 5 GB). Add `logrotate`-equivalent .bat run weekly via Task Scheduler | S |
| **MED** | F11 | Clock skew Razer↔Jupiter (suspected 1h DST mismatch per recent pulses) | None | Watchdog uses local `America/New_York` via `zoneinfo`. If Razer system clock is wrong, "RTH" decision is wrong → STALL thresholds applied at wrong times | Jupiter alerter compares `status.ts` to Jupiter `datetime.now(UTC)` — alarm if delta > 30s. Force `w32tm /resync` weekly | S |
| **MED** | F12 | Win32_Process.Create wrapper vs child orphan (wrapper PID 3936 alive, child python dead) | Watchdog matches by cmdline substring — substring is in BOTH; could match wrapper and miss dead child | False-OK risk | In watchdog, when matching, require `cmdline contains '.py'` AND process `name in {python.exe, pythonw.exe}` — not cmd.exe wrapper | S |
| **MED** | F13 | Model `.pt` file replaced underfoot → silent prediction-distribution shift on next restart | SHA verified once at load time (in code); not exported anywhere | If silent file swap happens between restarts, no record | Add `model_sha` to heartbeat JSON. Watchdog alarms on change between cycles | S |
| **MED** | F14 | Multiple paper traders sharing single Rithmic session — if session drops, ALL go silent at once but recorder may still write last cached events | Cross-process stale-correlation not checked | Watchdog evaluates each process independently; missing the recorder-is-source rule | Add a "blast radius" rule: if MBO recorder is stale/dead, suppress dependent paper-trader STALL alarms (they're symptoms, not root causes), but escalate recorder alarm to CRITICAL | S |
| **MED** | F15 | SSH host-key rotation (Razer reinstall, fresh Windows imaging) breaks Jupiter alerter SSH | `BatchMode=yes` causes silent fail; logged as `ssh_fail` | Repeated `ssh_fail` will fire `fetch_failed` alarm — covered, but until first alarm runs the gap is invisible | Jupiter cron alarms on N consecutive `ssh_fail` events | S |
| **LOW** | F16 | JSONL bloat in `live_trading_linux/logs/` (paper traders, MBO recorder) | None | One full day of MBO events JSONL can hit GB-scale | Daily gzip-and-archive scheduled task. Not urgent unless disk fills | S |
| **LOW** | F17 | Win32_Process.Create dispatch fails silently if WMI service is degraded | None — `Create` returns a `ProcessId` but if WMI is hung it may hang the launch script | Rare on Windows 11 | If `launch_watchdog.bat` reports `ReturnValue != 0`, alarm. Manual check today | S |

---

## 2. Heartbeat Coverage Map (post-deploy of `live_stack_watchdog.py`)

| Link | Heartbeat present? |
|---|---|
| MBO recorder PID alive | ✅ (cmdline match) |
| MBO recorder publishing events | ✅ JSONL mtime, but threshold 10 min RTH = too lax (F6) |
| MBO recorder Rithmic-connected | ❌ (F2) — no `last_rithmic_tick_ts` exported |
| Legacy paper trader PID alive | ✅ |
| Legacy paper trader gating signals | ❌ (F3) — no heartbeat JSON writer |
| Shadow trader PID alive | ✅ |
| Shadow trader `passed_gate` cumulative | ✅ — registry tracks `n_signals_passed_gate`; GATE-STARVED alarm wired |
| Watchdog process itself | ⚠️ Partial — Jupiter alerter polls, but no Razer-side self-restart on crash (F4) |
| Discord delivery | ⚠️ Fallback to `alerter_pending.jsonl` only; no Telegram (F5) |
| GPU/VRAM | ❌ (F7) |
| Disk space | ❌ (F10) |
| Clock skew | ❌ (F11) |
| Reboot survival | ❌ (F9) |

---

## 3. Quick Wins (ship in < 1h, close worst gaps)

1. **(F6) Add per-process JSONL stall threshold override** in `PROCESS_REGISTRY`: MBO recorder gets 90s RTH instead of 10 min. ES quote frequency is dozens/sec; 90s silent = broken.
2. **(F7) Add GPU/VRAM poll** to `evaluate_process` cycle — one `subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'])` per cycle. Alarm `memory.used > 7000 MiB` or `utilization.gpu == 0` for >2 RTH cycles.
3. **(F10) Add disk-space check** — `shutil.disk_usage('C:/')` per cycle; alarm < 5 GB free.
4. **(F11) Add clock-skew check in Jupiter alerter** — compare `status.ts` to `datetime.now(UTC)`; alarm if delta > 30s.
5. **(F14) Add "blast-radius suppression"** — when MBO recorder is in STALL/DEAD, mark dependent paper-trader stalls as "downstream of recorder" so user gets ONE alert, not three.

---

## 4. Strategic Fixes (multi-day work)

### 4.A — Auto-restart layer (`relauncher.py`, Razer-side)
Watchdog is alert-only. A separate `relauncher.py` reads `watchdog_status.json` every 30s and acts:
- **DEAD** → relaunch via `Invoke-CimMethod Win32_Process Create` using a `process_registry.json` of canonical launch commands. Max 3 attempts / 30 min, then halt + page.
- **STALL** (counter frozen) → graceful kill → force kill → relaunch.
- **GATE-STARVED** (passed_gate=0 RTH) → **DO NOT restart**. Halt + page. Regime drift won't be fixed by restart (HC #421 Issue A).
- **Model SHA changed** → halt + page; require human verify.
- Every relaunch logged to `logs/watchdog/relaunch_events.jsonl`.

### 4.B — Reboot survival (F9)
Register each live process as a Windows Task Scheduler "At startup, run as user `claude`, highest privileges" task pointing at `launch_*.bat`. Same launch path under all conditions (Win32_Process.Create stays the supported pattern; Task Scheduler just kicks `launch_watchdog.bat` etc. at boot). Test with one off-hours reboot.

### 4.C — Multi-config publish/subscribe bus (HC #422 Rule 6)
Replace fan-out JSONL with a shared in-memory ring buffer (mmap or ZeroMQ PUB). MBO recorder = sole publisher; N paper traders = subscribers. Watchdog registers all subscribers from a config file (`logs/watchdog/process_registry.json`) so new traders auto-enroll without watchdog redeploy. Today's 3 fixed registry entries become dynamic.

### 4.D — Two-channel alerting (F5, F8)
Telegram fallback in `razer_watchdog_alerter.py` AND `live_stack_watchdog.py` (Razer-side); QCC alert hook on every CRIT. Direct test of both channels via a daily synthetic alarm at 09:25 ET pre-open.

### 4.E — Recorder + legacy paper heartbeat JSON contracts (F2, F3)
Augment `mbo_recorder.py` and `paper_trading_mamba_v2.py` to emit periodic JSON status files matching the watchdog contract. Until this lands, watchdog has only DEAD+JSONL-mtime visibility on those two processes — counter freeze (the legacy 11:39 ET `events_counter=729,739` symptom) cannot fire its specific alarm.

---

## 5. Bottom Line

After today's `live_stack_watchdog.py` deployment **plus** the 5 Quick Wins (~1h work), Razer goes from **0% active heartbeat coverage** to **~70%**. The remaining 30% — auto-recovery, reboot survival, multi-channel alerting, multi-config bus — is the Strategic Fixes block above and matches HC #422 Rules 1, 6, and 7 falsification gates.

**Single highest-blast-radius node**: MBO recorder. Fix F2 (recorder status JSON + Rithmic tick heartbeat) is the single most valuable change because EVERY paper trader silently dies the moment the recorder is broken, and today the watchdog cannot distinguish "no market" from "feed broken" on the recorder.

**Most-likely-to-bite-again class**: regime drift (F1). The 5.5h silent failure was 50% a watchdog gap and 50% a distribution-drift gap. The watchdog will catch it next time within 10 min; the underlying drift is an alpha/execution research problem, not an infra problem (separate doc per HC #421).

---

## 6. References

- DIRECTIVES.md HC #422 Rules 1, 6, 7 (this audit's mandate)
- DIRECTIVES.md HC #421 Issue A (regime drift root cause)
- DIRECTIVES.md HC #401 (Win32_Process.Create — only known SSH-disconnect-survival pattern)
- `live_trading_linux/live_stack_watchdog.py` (the watchdog)
- `ops/razer_watchdog_alerter.py` (Jupiter-side companion)
- `live_trading_linux/paper_trading_v2_1s_short_top05.py:128` (HEARTBEAT_PATH writer — works)
- `live_trading_linux/paper_trading_mamba_v2.py` (no JSON heartbeat writer — gap F3)
- `live_trading_linux/mbo_recorder.py` (no JSON heartbeat writer — gap F2)
- `logs/alerter/razer_watchdog_latest.json` (live snapshot proving watchdog is currently running, cycle 4)
