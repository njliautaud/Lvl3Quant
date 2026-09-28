@echo off
echo === Shadow heartbeat JSON ===
type "C:\Users\claude\Lvl3Quant\output\v2_1s_short_top05_heartbeat.json"
echo.
echo === Shadow active JSONL (last lines) ===
powershell -ExecutionPolicy Bypass -Command "$f = Get-ChildItem 'C:\Users\claude\Lvl3Quant\live_trading_linux\logs\v2_1s_short_top05_paper_*.jsonl' | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($f) { Write-Host ('Active: ' + $f.FullName + ' size=' + $f.Length + ' mtime=' + $f.LastWriteTime); Get-Content $f.FullName -Tail 5 }"
echo.
echo === Shadow active LOG (last 20 lines) ===
powershell -ExecutionPolicy Bypass -Command "$f = Get-ChildItem 'C:\Users\claude\Lvl3Quant\live_trading_linux\logs\v2_1s_short_top05_paper_*.log' | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($f) { Write-Host ('Active: ' + $f.FullName + ' size=' + $f.Length + ' mtime=' + $f.LastWriteTime); Get-Content $f.FullName -Tail 20 }"
echo.
echo === Search for ANY legacy paper trader output files ===
powershell -ExecutionPolicy Bypass -Command "Get-ChildItem -Path 'C:\Users\claude\Lvl3Quant' -Recurse -Filter '*paper*' -ErrorAction SilentlyContinue | Where-Object { $_.Name -notlike '*v2_1s_short*' -and $_.Name -notlike '*.py*' -and $_.LastWriteTime -gt (Get-Date).AddDays(-3) } | Sort-Object LastWriteTime -Descending | Select-Object -First 15 FullName, Length, LastWriteTime | Format-List"
echo.
echo === Searching env files for any webhook URL ===
powershell -ExecutionPolicy Bypass -Command "Get-ChildItem -Path 'C:\Users\claude\Lvl3Quant' -Recurse -Include '.env','*.env' -ErrorAction SilentlyContinue | ForEach-Object { Write-Host ('--- ' + $_.FullName + ' ---'); Get-Content $_.FullName -ErrorAction SilentlyContinue }"
