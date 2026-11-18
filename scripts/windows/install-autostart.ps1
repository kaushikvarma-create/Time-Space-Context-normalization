# Keep the WSL distro (and so cron + the 15-minute capture tick) running after a Windows restart.
#
# Registers a per-user scheduled task that, at logon, starts a hidden `sleep infinity` inside the
# distro. That process holds WSL open, so systemd keeps cron running. Re-running this script
# replaces the task. Remove it with:  Unregister-ScheduledTask -TaskName SBI-WSL-Keepalive
#
# Usage (from WSL):
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(wslpath -w scripts/windows/install-autostart.ps1)" -Distro "$WSL_DISTRO_NAME"
param(
    [Parameter(Mandatory = $true)][string]$Distro,
    [string]$TaskName = "SBI-WSL-Keepalive"
)
$ErrorActionPreference = "Stop"

$known = (wsl.exe -l -q) -replace "`0", "" | ForEach-Object { $_.Trim() } | Where-Object { $_ }
if ($known -notcontains $Distro) {
    throw "WSL distro '$Distro' not found (have: $($known -join ', '))"
}

# conhost --headless runs wsl.exe without a console window (Windows 11).
$action = New-ScheduledTaskAction -Execute "$env:WINDIR\System32\conhost.exe" `
    -Argument "--headless $env:WINDIR\System32\wsl.exe -d $Distro --exec /bin/sleep infinity"
$trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Description "Keeps WSL '$Distro' running so the SBI capture cron keeps ticking." -Force | Out-Null

Start-ScheduledTask -TaskName $TaskName
Start-Sleep -Seconds 3
$info = Get-ScheduledTask -TaskName $TaskName
Write-Output "task '$TaskName' registered; state: $($info.State)"
