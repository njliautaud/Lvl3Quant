# HC #424 — Razer Multi-Config Paper Audit

**Author:** Claude Opus 4 (HC #393 + HC #424)
**Generated:** 2026-05-18 ~18:25 ET
**Companion doc:** `output/hc424_live_perf_5_18.md` (live perf audit for today)

---

## 1. Every PID running on Razer (post-close 5/18)

| PID | Process | Purpose | Source file | Logs / JSONL | Dependencies | Heartbeat hook | Restart policy (current) |
|-----|---------|---------|-------------|--------------|--------------|----------------|--------------------------|
| 15720 | `python.exe mbo_recorder.py --symbol ESM6 --exchange CME --flush-minutes 5` | Records every MBO event from Rithmic to a single append-only JSONL — the master event bus the paper traders subscribe to. | `C:\Users\claude\Lvl3Quant\mbo_recorder.py` | `C:\Users\claude\Lvl3Quant\logs\mbo_recorder.log` + `C:\Users\claude\Lvl3Quant\live_trading\logs\live_events.jsonl` | Rithmic WebSocket | Watchdog checks log mtime (passes — actively logging) | Manual; no auto-restart |
| 25512 | `python.exe paper_trading_mamba_v2.py --symbol ESM6 --exchange CME --weights ...fold_10_best.pt --stats ...fold_09_feature_stats.npz --device cuda --min-tier Top5%` | Legacy CNN-Mamba v2 paper trader, Top5% confidence gate. Was the "+$45 winner" baseline. | `C:\Users\claude\Lvl3Quant\live_trading\paper_trading_mamba_v2.py` | `C:\Users\claude\Lvl3Quant\live_trading\logs\paper_mamba_v2_20260513_0618.log` | MBO recorder (subscribe), CNN-Mamba weights | Watchdog only checks log mtime (FAILING — see §3) | Manual |
| 29600 | `pythonw.exe paper_trading_v2_1s_short_top05.py --shadow --symbol ESM6 --device cuda --sha256 300e338d3c...` | Shadow paper trader: 1s-horizon short-only, top-0.5% gate. Per HC #421/422 intended cutover candidate (still running in `--shadow` mode pending Rithmic broker wiring per HC #421 Step G1). | `C:\Users\claude\Lvl3Quant\live_trading_linux\paper_trading_v2_1s_short_top05.py` | `C:\Users\claude\Lvl3Quant\live_trading_linux\logs\v2_1s_short_top05_paper_20260518_080606.{log,jsonl}` + `C:\Users\claude\Lvl3Quant\output\v2_1s_short_top05_heartbeat.json` | MBO recorder, CNN-Mamba weights, encoder | Watchdog parses heartbeat JSON for `n_signals_total` + `n_signals_passed_gate` (WORKING — fired GATE-STARVED 13 times) | Manual |
| 32980 | `pythonw.exe live_stack_watchdog.py` | Active heartbeat watchdog (HC #422 R1). Polls process registry every 60s. | `C:\Users\claude\Lvl3Quant\live_trading_linux\live_stack_watchdog.py` | `C:\Users\claude\Lvl3Quant\logs\watchdog\watchdog_events.jsonl` + `...\watchdog_status.json` | psutil | Self-writes status file every cycle (cycle=195 currently) | Manual; no self-restart |

---

## 2. Why shadow PID 29600 was reported "dead" in the HC #424 brief (it was not)

The brief based its claim on `Get-Process python` output, which filters by process **name `python`**. PID 29600 runs as `pythonw.exe` (windowless Python), so it does not match `Get-Process python` — but it does match `Get-Process pythonw` or a broader `Get-WmiObject Win32_Process | Where Name -match 'python'`. Cross-verification via `Get-Process -Id 29600` returns the alive process (StartTime 5/18 08:06 ET, CPU 5800s, ThreadCount 27).

**Action item for future audits:** Replace `Get-Process python` with `Get-WmiObject Win32_Process | Where Name -in 'python.exe','pythonw.exe'` (or filter by CommandLine containing `.py`).

---

## 3. Why PID 29600 generated zero fills today (the REAL failure mode)

Not death — **GATE STARVATION + KILL-SWITCH LATCH**.

- The shadow's heartbeat JSON shows `n_signals_total=8710` advancing all day (model IS running, encoder IS encoding, MBO recorder IS publishing).
- But `n_signals_passed_gate=0` for the entire ≥4 hours the watchdog has been observing (and presumably for the entire 8-hour session).
- Around 17:47 ET, the kill-switch latched into `broker_connectivity_loss (>5.0s) MANUAL HALT` and has been re-asserting it every 30s ever since. MANUAL HALT requires user re-arm and does not auto-clear.

**Root cause hypothesis (HC #423 §3 encoder mismatch):** The shadow's `StreamingFeaturesSmartV3.encode()` invocation is missing the `delta_ns / 1_000_000` normalization step the legacy trader applies at `paper_trading_mamba_v2.py` lines 727-751. The result is a +0.21 pred-mean shift in the live encoder vs training (documented in HC #421 Issue A as "regime drift" but never directly tested against training-day feat_vec first moments — HC #423 §3 mandates that test). With predictions biased positive, the short-only top-0.5% tail is unreachable → gate starves → zero fills.

**Why the brief said "PID 29600 died at 16:48 ET":** likely a confusion between (a) the watchdog correctly flagging `JSONL-STALE: ...not written for 411min` because the JSONL hasn't grown since pre-RTH (only kill-switch alerts written, log lines are sparse compared to the 8710 preds in the heartbeat JSON), and (b) actual process death. The watchdog status file shows `pid: 29600` consistently across all 195 cycles; the process never died.

---

## 4. The watchdog blindspots (HC #422 R1 violation root causes)

### Blindspot A: Discord webhook unconfigured
The watchdog code searches for `DISCORD_WEBHOOK_LIVE_STACK` or `DISCORD_WEBHOOK` in `live_trading_linux/.env`. That file does not exist on Razer. The only `.env` on Razer is `live_trading/.env` which has Rithmic creds only. Result: every watchdog alarm has been silently logged to local disk; the user has never been notified.

**Fix (P0 for 5/19 pre-open):** create `C:\Users\claude\Lvl3Quant\live_trading_linux\.env` with a `DISCORD_WEBHOOK_LIVE_STACK=<url>` line. (User must provide the URL.) Then restart PID 32980 via Win32_Process.Create.

### Blindspot B: legacy counter-content freeze undetected
Legacy registry entry has `counters: []` and `status_file: None`. So the watchdog only checks (a) process alive, (b) JSONL `paper_mamba_v2_*.log` mtime. The legacy heartbeats the log every 5min with a STATUS line, keeping mtime fresh. But the STATUS line content `events=729739 preds=1458` has been identical character-for-character since 5/14 00:04 ET.

**Fix (P0 for 5/19 pre-open):** patch `live_stack_watchdog.py` `PROCESS_REGISTRY` entry for `legacy_paper_top5`:
- Add a `log_status_regex: r"events=(\d+) preds=(\d+)"` field
- In `evaluate_process`, when `log_status_regex` is set, tail the JSONL/log, find the most recent match, extract groups; track those as virtual counters with the same stall logic as the shadow's status-JSON counters.
- A 10-min freeze of `events=` during RTH → alarm.

### Blindspot C: kill-switch MANUAL HALT not user-notified
Shadow's `DiscordNotifier` ALSO has no webhook URL (its env var `DISCORD_WEBHOOK_URL` is unset; the launch did not pass `--webhook`). Every `discord_alert` event is logged to the JSONL only. Combined with the watchdog's missing webhook, the kill-switch state is invisible.

**Fix (P0):** include `DISCORD_WEBHOOK_URL=<url>` in `live_trading_linux\.env` so both shadow and watchdog can pick it up.

### Blindspot D: no auto-restart for ANY watched process
The watchdog only alarms; it does not invoke Win32_Process.Create on a dead PID. HC #424 R4 requires auto-restart within 60s. This is missing.

**Fix (P1):** add `auto_restart: True` + `restart_cmd: [...]` per registry entry; on DEAD alarm, invoke restart via subprocess with `CREATE_NEW_PROCESS_GROUP|DETACHED_PROCESS` and log the new PID.

### Blindspot E: no self-watchdog
The watchdog's own crash is detected only by external means. SESSION_STATE.md notes a Jupiter-side 3-min cron `bad131f4`/`51da6f13` should read `watchdog_status.json` mtime — but the user did not confirm this cron is actively running.

**Fix (P1):** verify the Jupiter-side cron exists; if not, add it.

---

## 5. Crash modes already known + recovery state

| Mode | Detection today | Auto-recovery | Status |
|------|-----------------|---------------|--------|
| Shadow zero-fill / gate-starved | Watchdog `GATE-STARVED` alarm (working, but silent) | None | Open |
| Legacy counter-content freeze | Not detected (Blindspot B) | None | Open |
| MBO recorder log rotation | Glob picks newest, OK | None | OK |
| GPU OOM | psutil PID death + watchdog DEAD alarm (silent) | None | Open |
| Kill-switch MANUAL HALT | Logged to JSONL (silent) | None — by design requires user | Open |
| SSH disconnect kills child process | HC #401 Win32_Process.Create | Active | OK |

---

## 6. Dependency graph

```
                  Rithmic WebSocket
                         |
                         v
              [PID 15720 mbo_recorder]
                  |             |
            live_events.jsonl   memory pub
                  |             |
   .--------------+-------------+----------.
   |              |             |          |
   v              v             v          v
[PID 25512    [PID 29600   [future       (watchdog
 legacy        shadow       v2_1s_long    subscribes
 top5%]        top0.5%      / other        to none —
                short]      configs]       polls each)
   |              |
   log only       heartbeat.json + JSONL
   (no JSON       (status file polled by
   status)        watchdog)
```

**Link without heartbeat alarm today:** Legacy → log (Blindspot B). All other links covered (with the caveat that alarms are silent due to Blindspot A/C).

---

## 7. Plan to make EVERY link auto-recover within 60s

Per HC #424 R4 the watchdog must restart any dead watched PID within 60s. The launch pattern is HC #401 Win32_Process.Create. Concretely, add to `live_stack_watchdog.py`:

```python
RESTART_COMMANDS = {
    "mbo_recorder": [
        "C:\\Python311\\python.exe",
        "C:\\Users\\claude\\Lvl3Quant\\mbo_recorder.py",
        "--symbol", "ESM6", "--exchange", "CME", "--flush-minutes", "5",
    ],
    "legacy_paper_top5": [
        "C:\\Python311\\python.exe",
        "C:\\Users\\claude\\Lvl3Quant\\live_trading\\paper_trading_mamba_v2.py",
        "--symbol", "ESM6", "--exchange", "CME",
        "--weights", "C:\\Users\\claude\\Lvl3Quant\\output\\cnn_mamba_v2_smart_v3_mar\\fold_10_best.pt",
        "--stats",   "C:\\Users\\claude\\Lvl3Quant\\output\\cnn_mamba_v2_smart_v3_mar\\fold_09_feature_stats.npz",
        "--device", "cuda", "--min-tier", "Top5%",
    ],
    "shadow_v2_1s_short_top05": [
        "C:\\Python311\\pythonw.exe",
        "C:\\Users\\claude\\Lvl3Quant\\live_trading_linux\\paper_trading_v2_1s_short_top05.py",
        "--shadow", "--symbol", "ESM6", "--device", "cuda",
        "--sha256", "300e338d3c16137fc587b10cce92204e8fe0a486fc3c7aaf53856fdd8c21281e",
        # TODO: --webhook from env DISCORD_WEBHOOK_URL once available
    ],
}

def restart_process(name: str) -> bool:
    cmd = RESTART_COMMANDS.get(name)
    if not cmd:
        return False
    # Use Win32_Process.Create equivalent (HC #401): subprocess.Popen with
    # creationflags=DETACHED_PROCESS|CREATE_NEW_PROCESS_GROUP so the child
    # survives the watchdog exiting.
    import subprocess
    flags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    p = subprocess.Popen(cmd, creationflags=flags, close_fds=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         cwd=str(LVL3_ROOT))
    log_event({"type": "auto_restart", "process": name, "new_pid": p.pid, "cmd": cmd})
    post_discord(f":arrows_counterclockwise: **AUTO-RESTART** {name} -> PID {p.pid}")
    return True
```

Trigger inside `should_fire_discord` block on DEAD-alarm: call `restart_process(name)`, wait one poll cycle, re-check; if still DEAD, escalate to user with @mention.

---

## 8. Adding a 3rd paper config — HC #422 R6 publish-subscribe pattern

**Current state:** NOT implemented. Each paper trader does its OWN ingestion of MBO events via subscribing to the Rithmic ws stream itself (or re-reading the recorder's JSONL — needs verification). Result: every new config = a new full inference engine + ws subscription = N×GPU memory + N×duplicated work.

**Proposed minimal scaffold:**

1. **Single inference daemon** — a new process `live_trading_linux\inference_daemon.py` that:
   - Subscribes to MBO recorder output (memory queue or file tail of `live_events.jsonl`).
   - Runs CNN-Mamba v2 forward pass once per event window.
   - Publishes the full multi-head prediction tuple (`pred_1s, pred_5s, pred_10s, pred_30s, mfe, mae, vol, confidence`) to a local IPC channel — easiest: Redis pub/sub on localhost (Razer already has Redis from previous experiments per CLAUDE.md) OR a memory-mapped ring buffer at `C:\Users\claude\Lvl3Quant\live_trading_linux\state\predictions.mmap`.

2. **N thin paper-trader subscribers** — each config (legacy-top5, shadow-1s-short-top05, future long-only-top5, etc.) becomes a SLIM process that:
   - Subscribes to the prediction channel only (no model loading).
   - Applies its own gate + position sizing + kill-switch logic.
   - Writes its own JSONL + heartbeat JSON.

3. **Watchdog covers all subscribers** by reading `watchdog_config.json`:
   ```json
   {
     "inference_daemon": {
       "cmdline_match": ["inference_daemon.py"],
       "heartbeat_file": "output/inference_daemon_heartbeat.json",
       "counters": ["n_events_in", "n_preds_out"],
       "restart_cmd": ["python.exe", "live_trading_linux/inference_daemon.py"]
     },
     "trader_legacy_top5": { ... },
     "trader_shadow_1s_short_top05": { ... },
     "trader_long_only_top5": { ... }
   }
   ```

4. **Time estimate to build:** 6-10 hours (1 inference daemon + 2 subscriber refactors + watchdog config refactor). Best done on a non-trading day or pre-market.

**Recommendation:** defer to user. The existing duplicated architecture works for ≤3 traders; the pub-sub refactor is worth it only if we plan ≥4 configs side by side, OR if GPU memory becomes the constraint (currently CNN-Mamba v2 fits ~1 GB so 4 copies = ~4 GB on a 8 GB 3070 — borderline but functional).

---

## 9. Bottom line on HC #424 R4 compliance status (5/18 18:30 ET)

| Requirement | Status |
|-------------|--------|
| ≥2 paper traders running side-by-side | YES (legacy 25512 + shadow 29600) |
| HC #423 §3 encoder fix applied to shadow | NO — diagnosed but not patched |
| Heartbeat watchdog covers every PID | PARTIAL — covers all but legacy is mtime-only (Blindspot B) |
| Watchdog alarms reach user | NO — webhook unconfigured (Blindspot A) |
| Auto-restart on PID death within 60s | NO (Blindspot D) |
| Live perf audit for today | YES — `output/hc424_live_perf_5_18.md` |

**Closest to clean for 5/19 RTH open:** wire the Discord webhook (10 min, needs URL from user) + patch the legacy counter-content regex into the watchdog (1 hour) + apply HC #423 §3 encoder fix (2-3 hours, code review + targeted port). Auto-restart and pub-sub refactor are P1, not blocking tomorrow's open.
