# Starts (or continues) the test-set experiments as a detached, console-less process.
# Safe to run any time: it does nothing if they are already running. Finished (variant, seed,
# dur) runs are skipped, so after a shutdown just run this again.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\start_experiments.ps1
$root = Split-Path $PSScriptRoot -Parent
$py = "C:\Users\emirc\venvs\adaptive\Scripts\pythonw.exe"
$running = Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
    Where-Object { $_.CommandLine -match "run_experiments\.py" }
if ($running) { "Experiments are already running (pid $($running.ProcessId -join ', '))"; exit }
$cmd = "`"$py`" -u scripts\run_experiments.py --log results\experiments.log"
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmd; CurrentDirectory = $root}
"Started experiments, pid $($r.ProcessId). Follow it in results\experiments.log"
