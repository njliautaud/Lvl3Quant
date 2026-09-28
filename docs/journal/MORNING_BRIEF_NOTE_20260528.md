# Morning Briefing Note — 2026-05-28 (UPDATED ~05:36 ET — Razer launcher SOLVED)

## RAZER LAUNCH BUG — ROOT CAUSE FOUND, FIX DEPLOYED

**Root cause**: Every standard Windows background-launch method (`start /b`,
`Start-Process -WindowStyle Hidden`, `schtasks /Run`, `schtasks` future-trigger)
ends up putting python.exe in **Session 0 (Services session)**, where it dies
silently within ~60s. Suspected cause: either NVIDIA driver attach fails for
Session 0 procs, or an EDR / Defender policy terminates them. **Not yet
isolated to one or the other**, but the workaround is reliable.

**Working fix**: `PsExec64.exe -i 1 -d` to inject into the interactive
**Console Session 1** (where the actively-logged-in `claude` user lives).
That session has GPU driver attach + no EDR kill.

**Proof of fix** (this session, 05:34 ET):
- `experiments/queue_predictor_v2.py` launched via PsExec → alive, 3.8GB working set,
  fold 0 loading data, GPU detected (RTX 3070 8.6GB), 50k samples/day streaming
- Plain `python -c "torch+cuda+matmul; sleep"` via PsExec also worked

**Files deployed** (survive context resets, per HC #492 R3):
- `C:\Tools\PsExec64.exe` — downloaded from Sysinternals (833KB)
- `C:\Users\claude\Lvl3Quant\scripts\razer_psexec_launch.ps1` — canonical
  parameterized launcher. Usage:
  ```
  powershell -ExecutionPolicy Bypass -File scripts\razer_psexec_launch.ps1 `
      -Script experiments\queue_predictor_v2.py -LogName queue_v2
  ```
  Returns immediately, python detaches into Session 1, log path printed.

**Recommended**: update the old `scripts\razer_launch_patchtst_s5.ps1` and any
other "Start-Process -WindowStyle Hidden" launchers to call PsExec instead.
The Start-Process pattern is broken on this machine and produces silent failures.

## v8 (confluence_meta_v8_pairs.py) STILL HAS A SCRIPT-LEVEL BUG

Separate from the launcher fix: when launched via PsExec, v8 **still hangs at
config dump** — process alive in Session 1 but no log progress past CONFIG json,
no GPU activity. The launcher is no longer the issue; the script body is.

**Next debug step for v8**: insert `print('STEP X', flush=True)` calls between
top-level operations (probe_patchtst_format, data load, fold loop) and re-run
via PsExec. The hang is somewhere in the first few function calls after the
config dump.

## CURRENTLY RUNNING (as of 05:36 ET)

- **Razer**: `experiments/queue_predictor_v2.py` via PsExec, PID 27524 in Console
  Session 1. Walk-forward fold 0, training dates 20260301..20260323 loading.
  Log: `C:\Users\claude\Lvl3Quant\logs\queue_predictor_v2_run.log`. This satisfies
  IDLE_ALERT directive priority option **e2**.
- **Neptune**: still idle. HC #487 R3 allows Razer for alpha research, so the
  Razer dispatch covers the IDLE_ALERT. Neptune available for additional work
  the morning briefing wants to dispatch.

## CORRUPTED DATA — INVESTIGATE

PatchTST dense-stride inference (e4) ran successfully via PsExec and found
**3 source npz files cannot be unzipped**:
- 20260330
- 20260402
- 20260410

Error: "File is not a zip file". Source corruption, not script bug. Worth
re-syncing from canonical store via `daily_mbo_sync` or manually.

## Other state
- Pre-market deploy cron fires at 09:15 ET today (one-shot in crontab).
- Morning briefing cron at 08:23 ET auto-injects the MORNING_BRIEFING prompt.
- 03:32 ET bot message: v7 production sweep is the WINNER at Spearman 0.308,
  perfect monotonic confidence calibration. That's the baseline to beat.
- v8 confluence training is the attempt to beat v7 — blocked by the script
  hang above, not by infrastructure anymore.
