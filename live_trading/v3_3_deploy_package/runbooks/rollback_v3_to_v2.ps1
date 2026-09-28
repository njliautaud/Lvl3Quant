# rollback_v3_to_v2.ps1 — HC #368 instant revert path
#
# Runs on Razer. Restores most-recent v2 snapshot.
# Idempotent. Safe to run anytime.
#
# Usage:
#   .\rollback_v3_to_v2.ps1
#   .\rollback_v3_to_v2.ps1 -SnapshotPath "C:\Users\claude\Lvl3Quant\rollback_snapshots\v2_20260514_153021"

param(
    [string]$SnapshotPath = ""
)

$ErrorActionPreference = "Stop"
$LvlRoot = "C:\Users\claude\Lvl3Quant"
$ActiveConfigPath = "$LvlRoot\live_trading\framework_config.json"

if (-not $SnapshotPath) {
    $SnapshotPath = (Get-ChildItem "$LvlRoot\rollback_snapshots\v2_*" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
}

if (-not $SnapshotPath -or -not (Test-Path $SnapshotPath)) {
    Write-Host "NO SNAPSHOT FOUND. Falling back to default v2 config."
    & "$LvlRoot\live_trading\launch_v2_stack.ps1"
    exit 0
}

Write-Host "[rollback] using snapshot: $SnapshotPath"

# Kill all running python (v3.3 daemon + paper trader)
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 3

# Restore v2 framework_config.json
if (Test-Path "$SnapshotPath\framework_config.json") {
    Copy-Item -Path "$SnapshotPath\framework_config.json" -Destination $ActiveConfigPath -Force
    Write-Host "[rollback] config restored"
}

# Relaunch v2 stack
& "$LvlRoot\live_trading\launch_v2_stack.ps1"

# Signal Discord via flag file
@{
    ts = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
    message = "✅ Rollback to v2 complete — snapshot: $SnapshotPath"
    source = "rollback_v3_to_v2.ps1"
} | ConvertTo-Json | Out-File -FilePath "$LvlRoot\deploy_status.json" -Encoding utf8 -Force

Write-Host "ROLLBACK COMPLETE"
