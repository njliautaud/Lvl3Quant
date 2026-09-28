@echo off
echo === All python.exe processes with full cmdline ===
powershell -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process -Filter 'Name=\"python.exe\"' | Select-Object ProcessId, CommandLine | Format-List"
echo.
echo === Get-Process for specific PIDs ===
powershell -ExecutionPolicy Bypass -Command "foreach ($pid in @(1436, 15720, 25512, 16204, 29600)) { $p = Get-Process -Id $pid -ErrorAction SilentlyContinue; if ($p) { Write-Host ('PID ' + $pid + ' alive: ' + $p.ProcessName) } else { Write-Host ('PID ' + $pid + ' NOT alive') } }"
echo.
echo === Is there anything matching v2_1s_short_top05 anywhere? ===
powershell -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*v2_1s_short_top05*' } | Select-Object ProcessId, Name, CommandLine | Format-List"
