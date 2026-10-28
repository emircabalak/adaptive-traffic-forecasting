# Starts (or continues) the LearnedBlend pretraining as a detached, console-less process.
# Safe to run any time: it does nothing if it is already running, and it resumes from the last
# saved chunk in results\blend_theta.json.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\start_pretrain_blend.ps1
$root = Split-Path $PSScriptRoot -Parent
$py = "C:\Users\emirc\venvs\adaptive\Scripts\pythonw.exe"
$running = Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
    Where-Object { $_.CommandLine -match "pretrain_blend\.py" }
if ($running) { "Pretraining is already running (pid $($running.ProcessId -join ', '))"; exit }
$cmd = "`"$py`" -u scripts\pretrain_blend.py --log results\pretrain_blend.log"
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmd; CurrentDirectory = $root}
"Started pretraining, pid $($r.ProcessId). Follow it in results\pretrain_blend.log"
