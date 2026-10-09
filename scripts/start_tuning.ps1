# Starts (or continues) the tuning queue as a detached, console-less process.
# Safe to run any time: it does nothing if the queue is already running. Finished configs and
# seeds are skipped, so after a shutdown just run this again.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\start_tuning.ps1
$root = Split-Path $PSScriptRoot -Parent
$py = "C:\Users\emirc\venvs\adaptive\Scripts\pythonw.exe"
$running = Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
    Where-Object { $_.CommandLine -match "tune\.py.*--queue" }
if ($running) { "Tuning queue is already running (pid $($running.ProcessId -join ', '))"; exit }
$cmd = "`"$py`" -u scripts\tune.py --queue results\tune_queue.jsonl --log results\tune.log"
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmd; CurrentDirectory = $root}
"Started tuning queue, pid $($r.ProcessId). Follow it in results\tune.log"
