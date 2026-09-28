# Razer LIVE Engine Cutover Playbook

**Target window:** Wednesday 2026-04-29, 8:00 AM ET (90 minutes before equity-market cash open, while NQ overnight session is still quiet).

**Authority:** DIRECTIVES.md HC #32 (locked 2026-04-28 19:39 ET). Razer becomes the LIVE engine: MBO recorder + paper engine + live inference (CNN Mamba v2). Jupiter is freed for fillsim/research.

**Constraint:** HC #6a — only one Rithmic AMP session at a time. Jupiter must release the session BEFORE Razer claims it.

---

## 0. Pre-flight (Tuesday evening / Wed 7:30 AM)

These items are *prerequisites* and were NOT completed during the 2026-04-28 evening provisioning. Do them BEFORE step 1 below or the cutover will fail.

- [ ] **Copy Rithmic protobuf bundle to Razer.** Source: `/home/jupiter/teleclaude-main/downloads/rithmic_api/0.89.0.0/samples/samples.py/` (contains `base_pb2.py`, `request_login_pb2.py`, etc., plus `rithmic_ssl_cert_auth_params`). Destination on Razer: pick a stable path, e.g. `C:\Users\claude\rithmic_pb\`. Then set the system env var: `setx RITHMIC_PB_DIR "C:\Users\claude\rithmic_pb" /M`. Verify with `python -c "import os, sys; sys.path.insert(0, os.environ['RITHMIC_PB_DIR']); import base_pb2; print('OK')"`.
- [ ] **Patch paper_engine.py log path.** Line 63 of `C:\Users\claude\Lvl3Quant\live_trading\paper_engine.py` hard-codes `_LOG_DIR = Path("/home/jupiter/Lvl3Quant/live_trading_linux/logs")`. On Razer this becomes `C:\home\jupiter\...` which is wrong. Override via env var or edit the file: change to `_LOG_DIR = Path(os.environ.get("PAPER_ENGINE_LOG_DIR", r"C:\Users\claude\Lvl3Quant\live_trading\logs"))`.
- [ ] **Verify shim package.** `C:\Users\claude\Lvl3Quant\live_trading_linux\` should contain `__init__.py` plus shim modules (`lgbm_inference.py`, `rithmic_client.py`, `streaming_features.py`, `streaming_features_smart_v3.py`, `cnn_mamba_v2_inference.py`, `execution_strategies.py`, `signal_engine.py`, `trading_cards.py`). Each shim re-exports `from live_trading.<mod> import *`. Created during 2026-04-28 provisioning; should still be in place.
- [ ] **Smoke-test paper_engine import.** `cd C:\Users\claude\Lvl3Quant\live_trading & set PYTHONPATH=C:\Users\claude\Lvl3Quant & python -c "import paper_engine; print('OK')"`. Expect "OK" with no traceback. PM2 ecosystem already sets PYTHONPATH for the runtime case.
- [ ] **Discord webhook URL.** `discord_watchdog.py` reads `DISCORD_WEBHOOK_URL` env var. Set it on Razer (system-wide): `setx DISCORD_WEBHOOK_URL "<url>" /M`. Without it, the watchdog logs alerts to file but skips Discord posts.
- [ ] **Verify PM2 daemon survives SSH session exit.** Repeated observations on 2026-04-28 showed the PM2 daemon being killed when each `qcc_ssh_exec` SSH session closed. Fix candidates: (a) ensure pm2-windows-startup service is running, (b) start PM2 from a Scheduled Task that runs as SYSTEM, (c) launch via `psexec -d` to detach. Confirm `pm2 list` from a *new* SSH session still shows the running services. This is the highest-priority pre-flight risk.

---

## 1. Verify Razer PM2 healthy (08:00 ET)

```powershell
# From a fresh SSH session as user `claude`
C:\Users\claude\AppData\Roaming\npm\pm2.cmd list
```

Expect: empty list (or only paper-engine if you started it for a soak test). No mbo-recorder, no live-inference yet — those start AFTER Jupiter releases Rithmic.

If PM2 daemon is missing, restart:
```powershell
C:\Users\claude\AppData\Roaming\npm\pm2.cmd resurrect
```

---

## 2. Stop Jupiter's recorder (08:02 ET)

On Jupiter, identify and stop the active recorder + paper engine. Service names may differ — check first:

```bash
pm2 list                 # find live recorder/engine names
pm2 stop mbo-recorder paper-engine    # adjust names as needed
pm2 logs mbo-recorder --lines 5 --nostream    # confirm clean shutdown
```

Note the last MBO event timestamp from the log so we can verify continuity on Razer.

---

## 3. Wait 30 seconds for Rithmic session to release

Rithmic's AMP gateway has a brief grace period before a new session can authenticate from a different host. 30s is conservative.

```powershell
Start-Sleep -Seconds 30
```

---

## 4. Start all three Razer services (08:03 ET)

```powershell
cd C:\Users\claude\Lvl3Quant\live_trading
C:\Users\claude\AppData\Roaming\npm\pm2.cmd start ecosystem.config.js
C:\Users\claude\AppData\Roaming\npm\pm2.cmd list
```

Expect: `mbo-recorder`, `paper-engine`, `live-inference` all `online` with PIDs.

Tail err logs for the first 60 seconds:
```powershell
type C:\Users\claude\Lvl3Quant\live_trading\logs\mbo-recorder_err.log
type C:\Users\claude\Lvl3Quant\live_trading\logs\paper-engine_err.log
type C:\Users\claude\Lvl3Quant\live_trading\logs\live-inference_err.log
```

A clean startup shows no Python tracebacks. Rithmic auth success should appear in mbo-recorder's out log.

---

## 5. Verify market-open tick rate (09:35 ET)

Compare events/min in `C:\Users\claude\Lvl3Quant\data\processed\mbo_events\` to last week's same time bucket from Jupiter's archive. Acceptable range: within +/-20% of the historical baseline.

```powershell
# Count events written in the last 5 minutes
Get-ChildItem C:\Users\claude\Lvl3Quant\data\processed\mbo_events\*.jsonl |
  Where-Object { $_.LastWriteTime -gt (Get-Date).AddMinutes(-5) } |
  Get-Content | Measure-Object -Line
```

If event rate is far below baseline: check Rithmic auth, network, GPU-driver issues on Razer.

---

## 6. Verify v2 inference is emitting signals (by 09:40 ET)

```powershell
type C:\Users\claude\Lvl3Quant\live_trading\logs\live-inference_out.log
```

Look for messages indicating model loaded (fold_10_best.pt), feature stats loaded (fold_09_feature_stats.npz), and signal generation cadence (one inference per 500-event stride at window=1000).

---

## 7. Verify paper trades are landing (by 10:00 ET)

```powershell
Get-ChildItem C:\Users\claude\Lvl3Quant\live_trading\logs\mamba_v2_signals_NQM6_*.jsonl |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
```

Expect a JSONL with at least one signal per Top1% prediction. If empty after 25 min of trading: check `paper-engine_err.log` for signal-routing errors.

---

## 8. Snapshot PM2 state for boot persistence (10:05 ET)

```powershell
C:\Users\claude\AppData\Roaming\npm\pm2.cmd save
```

This writes `~/.pm2/dump.pm2`. Combined with `pm2-startup install` (already done 2026-04-28), services will resurrect on Razer reboot.

---

## 9. Start Discord watchdog (10:10 ET)

The watchdog is NOT under PM2 (so it survives PM2 daemon crashes). Start as a detached Scheduled Task or Windows Service. Quick option:

```powershell
schtasks /Create /TN "PM2Watchdog" /TR "C:\Python311\python.exe C:\Users\claude\Lvl3Quant\live_trading\discord_watchdog.py" /SC ONSTART /RU SYSTEM /F
schtasks /Run /TN "PM2Watchdog"
```

Verify it logged its startup line:
```powershell
type C:\Users\claude\Lvl3Quant\live_trading\logs\watchdog.log
```

---

## 10. Discord post (10:15 ET)

Send to `#system-status`:

> "Live engine cutover complete: Razer = MBO recorder + paper engine + CNN Mamba v2 inference. Jupiter freed for fillsim/research. PM2 boot-persistent (pm2 save snapshotted). Discord watchdog active. v2 paper signals flowing — first trade landed at <HH:MM>."

Then update `MEMORY.md` compute section with the new layout.

---

## Rollback (if anything fails between steps 4-7)

```powershell
# On Razer
C:\Users\claude\AppData\Roaming\npm\pm2.cmd stop all
C:\Users\claude\AppData\Roaming\npm\pm2.cmd delete all
```

Then on Jupiter:
```bash
pm2 start ecosystem.config.js   # or whatever the previous command was
```

Wait 30s for Rithmic to release/reclaim. Verify Jupiter is recording again. Diagnose Razer offline. Reschedule cutover.

---

## Artifacts referenced in this playbook

- `C:\Users\claude\Lvl3Quant\live_trading\ecosystem.config.js` — PM2 service config (3 services, sets PYTHONPATH)
- `C:\Users\claude\Lvl3Quant\live_trading\discord_watchdog.py` — out-of-band PM2 health watcher
- `C:\Users\claude\Lvl3Quant\live_trading_linux\` — Razer-side shim package mapping `live_trading_linux.X` → `live_trading.X`
- `C:\Users\claude\Lvl3Quant\output\cnn_mamba_v2_smart_v3_mar\fold_10_best.pt` — v2 inference weights
- `C:\Users\claude\Lvl3Quant\output\cnn_mamba_v2_smart_v3_mar\fold_09_feature_stats.npz` — feature normalization stats
