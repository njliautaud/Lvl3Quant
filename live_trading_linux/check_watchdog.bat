@echo off
REM Verify watchdog is alive and producing output
echo === Process list (python.exe) ===
tasklist /FI "IMAGENAME eq python.exe" /FO TABLE
echo.
echo === Watchdog stdout log (last 15 lines) ===
if exist "C:\Users\claude\Lvl3Quant\logs\watchdog\watchdog_stdout.log" (
    powershell -ExecutionPolicy Bypass -Command "Get-Content 'C:\Users\claude\Lvl3Quant\logs\watchdog\watchdog_stdout.log' -Tail 15"
) else (
    echo NO LOG FILE YET
)
echo.
echo === Watchdog status file ===
if exist "C:\Users\claude\Lvl3Quant\logs\watchdog\watchdog_status.json" (
    type "C:\Users\claude\Lvl3Quant\logs\watchdog\watchdog_status.json"
) else (
    echo NO STATUS FILE YET
)
