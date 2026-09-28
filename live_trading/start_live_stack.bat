@echo off
REM ╔══════════════════════════════════════════════════════════════════╗
REM ║  RAZER LIVE STACK — One-Click Launcher                          ║
REM ║  Starts: MBO Recorder + CNN-Mamba v2 Inference + Paper Trader   ║
REM ║  Auto-restarts on crash. Health monitoring. Discord alerts.     ║
REM ╚══════════════════════════════════════════════════════════════════╝

cd /d C:\Users\claude\Lvl3Quant\live_trading

REM Set environment
set PYTHONPATH=C:\Users\claude\Lvl3Quant
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
set PYTHONUNBUFFERED=1

REM Validate first
echo [%date% %time%] Running startup validation...
C:\Python311\python.exe live_stack.py --dry-run
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Validation failed! Fix issues above before launching.
    pause
    exit /b 1
)

echo.
echo [%date% %time%] Validation passed. Launching live stack...
echo.

REM Launch the unified stack
C:\Python311\python.exe live_stack.py --symbol ESM6 --exchange CME --device cuda --min-tier "Top1%%" --window 1000 --stride 500

echo.
echo [%date% %time%] Live stack exited.
pause
