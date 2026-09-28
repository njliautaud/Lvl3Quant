# HC #438 -- How To Activate Fix #1

## Status
- Patched file is staged on Razer at:
  `C:\Users\claude\Lvl3Quant\live_trading_linux\paper_trading_v2_1s_short_top05.py.hc438patched`
- SHA-256: `4c0684e428ad49163fdba0c5a27211ec3325302eb50182e5fff68be21202b41f`
- Python syntax check: PASS
- The currently-running trader (whichever PID owns `\ShadowV2Top05`) is still on OLD code in MANUAL HALT. It will not trade until restarted.

## Two Activation Paths

### Path A: Wait for tonight's scheduled relaunch (LOW RISK)
1. At any time before 23:58 ET, swap the file:
   ```powershell
   ssh claude@razer
   powershell
   $live = 'C:\Users\claude\Lvl3Quant\live_trading_linux\paper_trading_v2_1s_short_top05.py'
   Copy-Item $live "$live.pre_hc438_backup"
   Move-Item -Force "$live.hc438patched" $live
   ```
2. Kill the existing halted trader process so the 23:58 ET scheduled task starts a fresh process tomorrow. (The halted process is harmless but consuming RAM.)
3. Tomorrow's `\ShadowV2Top05` task picks up the patched file automatically.

### Path B: Restart NOW for tonight's overnight (MEDIUM RISK)
Same file swap as above, then:
```powershell
Get-Process pythonw | Where-Object {$_.CommandLine -like '*v2_1s_short_top05*'} | Stop-Process -Force
schtasks /Run /TN \ShadowV2Top05
```
Verify within 60 seconds:
- `Get-Process pythonw` shows a new PID
- Trader log shows fresh `LIVE START` line
- `output\v2_1s_short_top05_heartbeat.json` mtime advances within 30s

## Verification After Restart
1. Check trader log: should see `LIVE START symbol=ESM6 shadow=True ...` within 30s.
2. Wait 5 minutes. Confirm `LIVE tick` lines are appearing (event counter advancing).
3. Confirm no `MANUAL HALT: broker_connectivity_loss` lines appear during the first overnight low-event window. Previous behaviour: would fire within 1-2 hours of low-event night. Expected new behaviour: never fires unless the recorder itself is dead for >90s.

## Rollback
If something breaks, restore the backup:
```powershell
$live = 'C:\Users\claude\Lvl3Quant\live_trading_linux\paper_trading_v2_1s_short_top05.py'
Move-Item -Force "$live.pre_hc438_backup" $live
```
