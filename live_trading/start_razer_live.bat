@echo off
REM =========================================================================
REM start_razer_live.bat — Master launcher for Razer live trading services
REM
REM Starts the watchdog process which manages:
REM   - mbo_recorder.py (MBO market data recorder)
REM   - paper_trading_mamba_v2.py (CNN-Mamba v2 live inference)
REM
REM Usage:
REM   start_razer_live.bat          — normal launch (watchdog in background)
REM   start_razer_live.bat --fg     — foreground mode (for debugging)
REM
REM To auto-start on reboot, create a scheduled task:
REM   schtasks /create /tn "RazerLiveTrading" /tr "C:\Users\claude\Lvl3Quant\live_trading\start_razer_live.bat" /sc onstart /ru claude /rl HIGHEST
REM =========================================================================

REM --- Set environment variables ---
set PYTHONPATH=C:\Users\claude\Lvl3Quant
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
set PYTHONUNBUFFERED=1

REM --- Working directory ---
cd /d C:\Users\claude\Lvl3Quant\live_trading

REM --- Create logs directory if needed ---
if not exist logs mkdir logs

REM --- Check if watchdog is already running ---
tasklist /FI "IMAGENAME eq python.exe" /FO CSV /NH 2>NUL | findstr /I "razer_watchdog" >NUL 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [%date% %time%] Watchdog already running, skipping launch.
    echo [%date% %time%] Watchdog already running, skipping launch. >> logs\launcher.log
    goto :EOF
)

REM --- Log launch ---
echo [%date% %time%] Starting Razer live trading watchdog >> logs\launcher.log

REM --- Check for foreground flag ---
if "%1"=="--fg" (
    echo Running watchdog in foreground mode...
    C:\Python311\python.exe razer_watchdog.py
    goto :EOF
)

REM --- Launch watchdog in background (no window) ---
echo Starting watchdog in background...
start /B /MIN "" C:\Python311\python.exe razer_watchdog.py >> logs\watchdog_console.log 2>&1

REM --- Verify it started ---
timeout /t 2 /nobreak >NUL
tasklist /FI "IMAGENAME eq python.exe" /FO CSV /NH 2>NUL | findstr /I "python" >NUL 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [%date% %time%] Watchdog launched successfully >> logs\launcher.log
    echo Watchdog launched successfully.
) else (
    echo [%date% %time%] WARNING: Watchdog may have failed to start >> logs\launcher.log
    echo WARNING: Watchdog may have failed to start. Check logs\watchdog.log
)
