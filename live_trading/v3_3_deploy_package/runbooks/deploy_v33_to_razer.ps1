# deploy_v33_to_razer.ps1 — HC #368 one-button deploy with auto-rollback
#
# RUN ON RAZER (Windows PowerShell).
# Authorized by user explicit "deploy v3.3 now" — NOT autonomous (HC #366(f) hard-cost exception).
#
# Steps:
#   1. Snapshot current v2 state for rollback.
#   2. Verify v3.3 ckpt + config + daemon files present.
#   3. Stop v2 stack.
#   4. Start v3.3 daemon with active config.
#   5. Verify first prediction within 60s.
#   6. Send Discord "v3.3 LIVE" confirmation.
#   7. Auto-rollback to v2 if ANY step fails.
#
# Usage:
#   .\deploy_v33_to_razer.ps1 -ConfigName v33_short_top1pct_passive
#   .\deploy_v33_to_razer.ps1 -ConfigName v33_short_top01pct_5s_exit -DryRun

param(
    [Parameter(Mandatory=$true)]
    [string]$ConfigName,
    [switch]$DryRun = $false,
    [int]$FirstPredictionTimeoutSec = 60
)

$ErrorActionPreference = "Stop"
$LvlRoot = "C:\Users\claude\Lvl3Quant"
$PackageRoot = "$LvlRoot\live_trading\v3_3_deploy_package"
$SnapshotDir = "$LvlRoot\rollback_snapshots\v2_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
$ActiveConfigPath = "$LvlRoot\live_trading\framework_config.json"
$ActiveConfigBackup = "$ActiveConfigPath.v2_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
$CandidateConfigPath = "$PackageRoot\configs\candidate_configs\$ConfigName.json"

function Send-Discord([string]$msg) {
    Write-Host "[discord] $msg"
    # Discord MCP is in the Claude main session, not Razer. PowerShell here writes a flag file
    # that the Claude session reads and posts. (deploy_status.json watched by cron.)
    @{
        ts = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
        message = $msg
        source = "deploy_v33_to_razer.ps1"
    } | ConvertTo-Json | Out-File -FilePath "$LvlRoot\deploy_status.json" -Encoding utf8 -Force
}

function Rollback-ToV2 {
    param([string]$Reason)
    Send-Discord "🔴 v3.3 DEPLOY FAILED ($Reason) — auto-rollback to v2 in progress"

    # Kill any v3.3 daemon
    Get-Process python -ErrorAction SilentlyContinue | Where-Object {
        $_.MainWindowTitle -match "v33" -or $_.Path -match "v33"
    } | Stop-Process -Force -ErrorAction SilentlyContinue

    # Restore v2 config
    if (Test-Path $ActiveConfigBackup) {
        Copy-Item -Path $ActiveConfigBackup -Destination $ActiveConfigPath -Force
    }

    # Relaunch v2 stack
    & "$LvlRoot\live_trading\launch_v2_stack.ps1"

    Send-Discord "✅ v2 stack restored — investigate v3.3 failure before retry"
    exit 1
}

# ============================================================
# STEP 1: Snapshot current v2 state
# ============================================================
Write-Host "[step 1] Snapshot v2 state to $SnapshotDir"
if (-not $DryRun) {
    New-Item -ItemType Directory -Path $SnapshotDir -Force | Out-Null
    Copy-Item -Path $ActiveConfigPath -Destination "$SnapshotDir\framework_config.json" -ErrorAction SilentlyContinue
    Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, ProcessName, StartTime, Path |
        ConvertTo-Json | Out-File "$SnapshotDir\running_processes.json"
    Copy-Item -Path $ActiveConfigPath -Destination $ActiveConfigBackup -Force
}

# ============================================================
# STEP 2: Verify v3.3 artifacts exist
# ============================================================
Write-Host "[step 2] Verify v3.3 ckpt + config + daemon files"
$ckpt = "$LvlRoot\output\cnn_mamba_v3_3_uncertainty_weighted\fold_00_intra_ckpt.pt"
$daemon = "$PackageRoot\inference\v33_inference_daemon.py"
$nets = "$PackageRoot\safety_nets\safety_nets.py"

$missing = @()
if (-not (Test-Path $ckpt)) { $missing += $ckpt }
if (-not (Test-Path $CandidateConfigPath)) { $missing += $CandidateConfigPath }
if (-not (Test-Path $daemon)) { $missing += $daemon }
if (-not (Test-Path $nets)) { $missing += $nets }

if ($missing.Count -gt 0) {
    Write-Host "MISSING: $($missing -join ', ')"
    if (-not $DryRun) { Rollback-ToV2 "missing artifacts" }
    exit 2
}

# ============================================================
# STEP 3: Safety-nets self-test
# ============================================================
Write-Host "[step 3] Run safety-nets self-test"
if (-not $DryRun) {
    $selfTest = python "$nets" --self-test 2>&1
    Write-Host $selfTest
    if ($LASTEXITCODE -ne 0) { Rollback-ToV2 "safety-nets self-test failed" }
}

# ============================================================
# STEP 4: Stop v2 stack
# ============================================================
Write-Host "[step 4] Stop v2 stack"
if (-not $DryRun) {
    Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 3
    if (Get-Process python -ErrorAction SilentlyContinue) {
        Rollback-ToV2 "could not stop v2 python procs"
    }
}

# ============================================================
# STEP 5: Switch active config to v3.3 candidate
# ============================================================
Write-Host "[step 5] Switch active config → $ConfigName"
if (-not $DryRun) {
    Copy-Item -Path $CandidateConfigPath -Destination $ActiveConfigPath -Force
}

# ============================================================
# STEP 6: Start v3.3 daemon
# ============================================================
Write-Host "[step 6] Start v3.3 inference daemon + paper trader"
if (-not $DryRun) {
    $logDir = "$LvlRoot\logs\v3_3_live"
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"

    # Inference daemon
    Start-Process -FilePath "python" `
        -ArgumentList "$daemon --config $ActiveConfigPath --output-jsonl $logDir\v33_predictions_$ts.jsonl" `
        -WindowStyle Hidden `
        -RedirectStandardOutput "$logDir\v33_inference_$ts.log" `
        -RedirectStandardError "$logDir\v33_inference_${ts}_err.log"

    # Paper trader (existing one with v3.3 config)
    Start-Process -FilePath "python" `
        -ArgumentList "$LvlRoot\live_trading\paper_trader.py --config $ActiveConfigPath --pred-jsonl $logDir\v33_predictions_$ts.jsonl" `
        -WindowStyle Hidden `
        -RedirectStandardOutput "$logDir\v33_paper_$ts.log" `
        -RedirectStandardError "$logDir\v33_paper_${ts}_err.log"
}

# ============================================================
# STEP 7: Verify first prediction within timeout
# ============================================================
Write-Host "[step 7] Verify first prediction lands within $FirstPredictionTimeoutSec sec"
if (-not $DryRun) {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $predFound = $false
    $predFile = (Get-ChildItem "$LvlRoot\logs\v3_3_live\v33_predictions_*.jsonl" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName

    while ($sw.Elapsed.TotalSeconds -lt $FirstPredictionTimeoutSec) {
        if ($predFile -and (Test-Path $predFile)) {
            $lines = Get-Content $predFile -ErrorAction SilentlyContinue
            if ($lines -and $lines.Count -gt 0) { $predFound = $true; break }
        }
        Start-Sleep -Seconds 2
    }

    if (-not $predFound) {
        Rollback-ToV2 "no prediction within $FirstPredictionTimeoutSec sec"
    }
}

# ============================================================
# STEP 8: Confirm to Discord
# ============================================================
Send-Discord "✅ v3.3 LIVE — config=$ConfigName, time=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "DEPLOY COMPLETE"
