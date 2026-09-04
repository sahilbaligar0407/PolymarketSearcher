<#
.SYNOPSIS
  Install MarketLab as a background service on Windows.

.DESCRIPTION
  The PRD specifies a systemd --user service. This machine is Windows, where systemd does
  not exist, so this script provides the documented equivalent: a Scheduled Task that runs
  at logon, restarts on failure, and writes to the same log directory.

  Scheduled Tasks are the right primitive here rather than a true Windows Service because
  MarketLab is a user-scoped research process: it needs the user's environment, it reads
  .env from the repo, and it should stop when the user logs out rather than running as
  SYSTEM with broader privileges than it needs.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\install_service.ps1
  powershell -ExecutionPolicy Bypass -File scripts\install_service.ps1 -Uninstall
#>
[CmdletBinding()]
param(
    [switch]$Uninstall,
    [string]$TaskName = "MarketLab",
    [decimal]$Bankroll = 50
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

if ($Uninstall) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Stop-ScheduledTask   -TaskName $TaskName -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed scheduled task '$TaskName'."
    } else {
        Write-Host "No scheduled task named '$TaskName' found."
    }
    return
}

# Resolve uv the same way bootstrap.sh does - it is often not on PATH.
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) {
    foreach ($c in @("C:\Python314\Scripts\uv.exe", "C:\Python313\Scripts\uv.exe",
                     "$env:USERPROFILE\.local\bin\uv.exe")) {
        if (Test-Path $c) { $uv = $c; break }
    }
}
if (-not $uv) { throw "uv not found. Install it with: python -m pip install uv" }

$logDir = Join-Path $RepoRoot "data\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# `marketlab run` is the foreground daemon; the task supervises it.
$action = New-ScheduledTaskAction `
    -Execute $uv `
    -Argument "run marketlab run --bankroll-per-strategy $Bankroll" `
    -WorkingDirectory $RepoRoot

$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

# Restart on failure, never stop for being "long running", and do not stop on battery -
# this is a continuously running research process by design.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -DontStopOnIdleEnd `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal -Force | Out-Null

Write-Host "Installed scheduled task '$TaskName'."
Write-Host ""
Write-Host "  Start:   Start-ScheduledTask -TaskName $TaskName"
Write-Host "  Stop:    Stop-ScheduledTask  -TaskName $TaskName"
Write-Host "  Status:  Get-ScheduledTask   -TaskName $TaskName | Get-ScheduledTaskInfo"
Write-Host "  Logs:    Get-Content -Wait '$logDir\marketlab.jsonl'"
Write-Host "  Remove:  powershell -File scripts\install_service.ps1 -Uninstall"
Write-Host ""
Write-Host "Live trading is unaffected by this: it stays hard-disabled unless every gate in"
Write-Host "docs\live_safety.md is satisfied."
