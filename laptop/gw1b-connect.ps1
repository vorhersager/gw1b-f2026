# One command from Windows (PowerShell) to get a GPU JupyterLab on Pegasus + the ssh tunnel:
#
#     .\gw1b-connect.ps1 <netid> [gw1b jupyter options]      e.g.  .\gw1b-connect.ps1 jsmith --gpus 1 --time 3:00:00
#
# Requires the built-in OpenSSH client (Windows 10+) and VPN off campus. You will be asked for your
# GW password / 2FA twice (once to start the job, once for the tunnel). Ctrl-C closes the tunnel.
param(
    [Parameter(Mandatory = $true, Position = 0)] [string] $NetId,
    [Parameter(ValueFromRemainingArguments = $true)] [string[]] $JupyterArgs
)
$GW1B_HOME_REMOTE = if ($env:GW1B_HOME_REMOTE) { $env:GW1B_HOME_REMOTE } else { "/scratch/gw1b-class/group/gw1b-f2026" }   # <- instructor: set once
$LoginHost = if ($env:GW1B_LOGIN_HOST) { $env:GW1B_LOGIN_HOST } else { "pegasus.arc.gwu.edu" }
$LocalPort = if ($env:GW1B_LOCAL_PORT) { $env:GW1B_LOCAL_PORT } else { "8888" }

$remote = "bash -lc 'source $GW1B_HOME_REMOTE/activate.sh >/dev/null 2>&1; gw1b jupyter --local-port $LocalPort $($JupyterArgs -join ' ')'"
Write-Host "[gw1b] connecting to $LoginHost as $NetId ..."
$out = & ssh -tt "$NetId@$LoginHost" $remote 2>&1 | Out-String
Write-Host $out
$m = [regex]::Match($out, 'ssh -N -L (\S+) -L (\S+) (\S+)')
if (-not $m.Success) { Write-Error "[gw1b] could not find the tunnel command in the output above"; exit 1 }
$fwd1 = $m.Groups[1].Value; $fwd2 = $m.Groups[2].Value; $target = $m.Groups[3].Value
$url = [regex]::Match($out, 'http://localhost:\d+/lab\?token=[a-f0-9]+').Value
Write-Host ""
Write-Host "[gw1b] opening tunnel: ssh -N -L $fwd1 -L $fwd2 $target"
Write-Host "[gw1b] JupyterLab: $url"
Write-Host "[gw1b] Colab: Connect -> Connect to a local runtime -> $($url -replace '/lab\?', '/?')"
Write-Host "[gw1b] (Ctrl-C closes the tunnel; the job keeps running)"
Start-Job -ScriptBlock { param($u) Start-Sleep 4; Start-Process $u } -ArgumentList $url | Out-Null
& ssh -N -o ServerAliveInterval=60 -L $fwd1 -L $fwd2 $target
