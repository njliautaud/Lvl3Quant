# Register Razer live suite scheduled tasks
# Tasks run the bat files directly (which handle their own logging)
$ErrorActionPreference = "Stop"
$LT = "C:\Users\claude\Lvl3Quant\live_trading"

$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)

# Recorder task - runs svc_recorder.bat (which has env vars + output redirect baked in)
$actionRec = New-ScheduledTaskAction -Execute "$LT\svc_recorder.bat"
Register-ScheduledTask -TaskName "Lvl3_MBO_Recorder" -Action $actionRec -Trigger $trigger -Settings $settings -User "claude" -Force | Out-Null

# Inference task - runs svc_inference.bat (uses --follow-events, no Rithmic connection)
$actionInf = New-ScheduledTaskAction -Execute "$LT\svc_inference.bat"
Register-ScheduledTask -TaskName "Lvl3_Live_Inference" -Action $actionInf -Trigger $trigger -Settings $settings -User "claude" -Force | Out-Null

Write-Output "Tasks registered:"
Get-ScheduledTask -TaskName "Lvl3_*" | Format-Table TaskName, State -AutoSize

# Start recorder
Start-ScheduledTask -TaskName "Lvl3_MBO_Recorder"
Start-Sleep -Seconds 15

# Check
$procs = Get-Process python -ErrorAction SilentlyContinue
if ($procs) {
    Write-Output "Python processes running:"
    $procs | Format-Table Id, WorkingSet64, StartTime -AutoSize
} else {
    Write-Output "WARNING: No Python processes found - check logs"
}

# Start inference (staggered)
Start-ScheduledTask -TaskName "Lvl3_Live_Inference"
Start-Sleep -Seconds 10

# Final check
$procs2 = Get-Process python -ErrorAction SilentlyContinue
Write-Output "Final process check:"
if ($procs2) {
    $procs2 | Format-Table Id, WorkingSet64, StartTime -AutoSize
} else {
    Write-Output "WARNING: No Python processes"
}
Write-Output "DONE"
