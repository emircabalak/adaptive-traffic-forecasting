# Starts (or continues) baseline training as a detached, console-less process.
# Safe to run any time: it does nothing if training is running or already finished.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\start_training.ps1
$root = Split-Path $PSScriptRoot -Parent
$py = "C:\Users\emirc\venvs\adaptive\Scripts\pythonw.exe"

if (Get-ChildItem "$root\checkpoints\metr_exp1_best_*.pth" -ErrorAction SilentlyContinue) {
    "Training already finished: $((Get-ChildItem "$root\checkpoints\metr_exp1_best_*.pth").Name)"; exit
}
$running = Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
    Where-Object { $_.CommandLine -match "train_(baseline|resumable)\.py" }
if ($running) { "Training is already running (pid $($running.ProcessId -join ', '))"; exit }

$cmd = "`"$py`" -u scripts\train_resumable.py --log results\baseline_train.log"
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmd; CurrentDirectory = $root}
"Started training, pid $($r.ProcessId). Follow it in results\baseline_train.log"
