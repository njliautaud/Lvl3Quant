# Razer Live Suite - Scheduled Task Setup
# Sets up persistent processes that survive reboot + SSH disconnect
# Environment variables are embedded in wrapper batch files

$ErrorActionPreference = "Stop"
$LT = "C:\Users\claude\Lvl3Quant\live_trading"
$LOGS = "$LT\logs"

# Ensure logs dir exists
if (-not (Test-Path $LOGS)) { New-Item -ItemType Directory -Path $LOGS -Force }

# Create wrapper batch files with env vars baked in
$recorderBat = @"
@echo off
set PYTHONPATH=C:\Users\claude\Lvl3Quant
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
set PYTHONUNBUFFERED=1
cd /d C:\Users\claude\Lvl3Quant\live_trading
C:\Python311\python.exe mbo_recorder.py --symbol ESM6 --exchange CME
"@
Set-Content -Path "$LT\svc_recorder.bat" -Value $recorderBat -NoNewline

$inferenceBat = @"
@echo off
set PYTHONPATH=C:\Users\claude\Lvl3Quant
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
set PYTHONUNBUFFERED=1
cd /d C:\Users\claude\Lvl3Quant\live_trading
C:\Python311\python.exe paper_trading_mamba_v2.py --symbol ESM6 --exchange CME --device cuda --min-tier Top1%% --window 1000 --stride 500
"@
Set-Content -Path "$LT\svc_inference.bat" -Value $inferenceBat -NoNewline

# Register scheduled tasks - AtStartup trigger + manual start capability
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)

$actionRec = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$LT\svc_recorder.bat`" > `"$LOGS\svc_recorder.log`" 2>&1"
$actionInf = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$LT\svc_inference.bat`" > `"$LOGS\svc_inference.log`" 2>&1"

Register-ScheduledTask -TaskName "Lvl3_MBO_Recorder" -Action $actionRec -Trigger $trigger -Settings $settings -User "claude" -Force | Out-Null
Register-ScheduledTask -TaskName "Lvl3_Live_Inference" -Action $actionInf -Trigger $trigger -Settings $settings -User "claude" -Force | Out-Null

Write-Output "Scheduled tasks created:"
Get-ScheduledTask -TaskName "Lvl3_*" | Format-Table TaskName, State -AutoSize

# Start recorder immediately
Write-Output "Starting MBO Recorder..."
Start-ScheduledTask -TaskName "Lvl3_MBO_Recorder"
Start-Sleep -Seconds 10

# Start inference (staggered to avoid Rithmic session conflicts)
Write-Output "Starting Live Inference..."
Start-ScheduledTask -TaskName "Lvl3_Live_Inference"
Start-Sleep -Seconds 5

# Verify
Write-Output "`nProcess check:"
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, WorkingSet64, StartTime | Format-Table -AutoSize
Get-ScheduledTask -TaskName "Lvl3_*" | Format-Table TaskName, State -AutoSize
Write-Output "DONE"
