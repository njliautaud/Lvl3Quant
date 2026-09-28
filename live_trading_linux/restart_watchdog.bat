@echo off
REM Kill any running watchdog (matches by cmdline substring) and relaunch
echo === Killing existing watchdog instances ===
powershell -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process -Filter 'Name=\"python.exe\"' | Where-Object { $_.CommandLine -like '*live_stack_watchdog.py*' } | ForEach-Object { Write-Host ('Killing PID ' + $_.ProcessId); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
timeout /t 2 /nobreak > nul
echo.
echo === Launching watchdog v2 via Win32_Process.Create ===
powershell -ExecutionPolicy Bypass -Command "$cmd = 'cmd.exe /c C:\Python311\python.exe -u C:\Users\claude\Lvl3Quant\live_trading_linux\live_stack_watchdog.py >> C:\Users\claude\Lvl3Quant\logs\watchdog\watchdog_stdout.log 2>&1'; $proc = ([WMICLASS]\"Win32_Process\").Create($cmd); Write-Host ('NEW_PID=' + $proc.ProcessId + ' ReturnValue=' + $proc.ReturnValue)"
timeout /t 5 /nobreak > nul
echo.
echo === Watchdog tail after restart ===
powershell -ExecutionPolicy Bypass -Command "Get-Content 'C:\Users\claude\Lvl3Quant\logs\watchdog\watchdog_stdout.log' -Tail 10"
echo.
echo === Status JSON ===
type "C:\Users\claude\Lvl3Quant\logs\watchdog\watchdog_status.json"
