@echo off
REM HC #422 Rule 1 — launch live_stack_watchdog.py via Win32_Process.Create
REM so it survives SSH disconnect (HC #401 launch pattern).
REM Logs to C:\Users\claude\Lvl3Quant\logs\watchdog\watchdog_stdout.log

if not exist "C:\Users\claude\Lvl3Quant\logs\watchdog" mkdir "C:\Users\claude\Lvl3Quant\logs\watchdog"

powershell -ExecutionPolicy Bypass -Command "$cmd = 'cmd.exe /c C:\Python311\python.exe -u C:\Users\claude\Lvl3Quant\live_trading_linux\live_stack_watchdog.py >> C:\Users\claude\Lvl3Quant\logs\watchdog\watchdog_stdout.log 2>&1'; $proc = ([WMICLASS]\"Win32_Process\").Create($cmd); Write-Host ('PID=' + $proc.ProcessId + ' ReturnValue=' + $proc.ReturnValue)"
