# launch_razer_paper_cli.ps1
# PowerShell launcher for the Razer paper trader CLI wrapper (HC #351).
# Designed to be invoked via WMI Win32_Process Create from Jupiter (HC #308).
#
# Usage (locally on Razer for debugging):
#   powershell -ExecutionPolicy Bypass -File launch_razer_paper_cli.ps1 `
#       -ConfigPath C:\Users\claude\Lvl3Quant\live_trading\razer\example_config.json
#
# Usage (remote dispatch from Jupiter):
#   Invoke-WmiMethod -ComputerName razer -Class Win32_Process -Name Create `
#     -ArgumentList @('powershell -ExecutionPolicy Bypass -File C:\Users\claude\Lvl3Quant\live_trading\razer\launch_razer_paper_cli.ps1 -ConfigPath C:\Users\claude\Lvl3Quant\live_trading\razer\example_config.json')

param(
    [Parameter(Mandatory=$true)]
    [string]$ConfigPath,

    [string]$PythonExe = "python",

    [string]$WrapperPath = "C:\Users\claude\Lvl3Quant\live_trading\razer\paper_trader_cli.py",

    [string]$LogDir = "C:\Users\claude\Lvl3Quant\live_trading\logs",

    [switch]$DryRun,

    # Extra params passed straight through to paper_trader_cli.py (e.g. "--cnn-threshold 0.55")
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

# Required env vars for the trading daemon (mirrors razer_watchdog.py SERVICE_ENV).
$env:PYTHONPATH = "C:\Users\claude\Lvl3Quant"
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"

if (-not (Test-Path $ConfigPath)) {
    Write-Error "Config not found: $ConfigPath"
    exit 1
}

if (-not (Test-Path $WrapperPath)) {
    Write-Error "Wrapper not found: $WrapperPath"
    exit 1
}

# Ensure log dir exists
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$stdoutLog = Join-Path $LogDir "paper_trader_cli_${ts}.out.log"
$stderrLog = Join-Path $LogDir "paper_trader_cli_${ts}.err.log"

$argsList = @($WrapperPath, "--config", $ConfigPath)
if ($DryRun) { $argsList += "--dry-run" }
if ($ExtraArgs) { $argsList += $ExtraArgs }

Write-Host "Launching: $PythonExe $($argsList -join ' ')"
Write-Host "stdout -> $stdoutLog"
Write-Host "stderr -> $stderrLog"

# Use Start-Process so WMI returns immediately; tee to logs.
$proc = Start-Process -FilePath $PythonExe `
    -ArgumentList $argsList `
    -NoNewWindow `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

Write-Host "Started PID $($proc.Id)"
exit 0
