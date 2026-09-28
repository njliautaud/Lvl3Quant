@echo off
echo === .env files containing DISCORD ===
powershell -ExecutionPolicy Bypass -Command "Get-ChildItem -Path 'C:\Users\claude\Lvl3Quant' -Recurse -Filter '.env*' -ErrorAction SilentlyContinue | ForEach-Object { Write-Host ('--- ' + $_.FullName + ' ---'); Get-Content $_.FullName | Where-Object { $_ -like '*DISCORD*' -or $_ -like '*WEBHOOK*' } }"
echo.
echo === Shadow JSONL files (search) ===
powershell -ExecutionPolicy Bypass -Command "Get-ChildItem -Path 'C:\Users\claude\Lvl3Quant' -Recurse -Filter '*v2_1s_short_top05*' -ErrorAction SilentlyContinue | Select-Object FullName, Length, LastWriteTime | Format-List"
echo.
echo === Legacy paper status / JSONL files ===
powershell -ExecutionPolicy Bypass -Command "Get-ChildItem -Path 'C:\Users\claude\Lvl3Quant\logs' -Recurse -Filter 'paper_trading*' -ErrorAction SilentlyContinue | Select-Object FullName, Length, LastWriteTime | Format-List"
echo.
echo === Watchdog log + status (post-cycle-2) ===
if exist "C:\Users\claude\Lvl3Quant\logs\watchdog\watchdog_stdout.log" (
    powershell -ExecutionPolicy Bypass -Command "Get-Content 'C:\Users\claude\Lvl3Quant\logs\watchdog\watchdog_stdout.log' -Tail 8"
)
