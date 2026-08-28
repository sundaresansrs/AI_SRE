# Attempt 6 Monitoring Checkpoint
$checkTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Output "=== CHECKPOINT: $checkTime ==="
Write-Output ""

# 1. Process alive?
$proc = Get-CimInstance Win32_Process -Filter "ProcessId = 18636" -ErrorAction SilentlyContinue
$worker = Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq 18636 -and $_.Name -like "python*" }
if ($proc -and $worker) {
    Write-Output "✅ Merge process (Parent PID 18636 / Worker PID $($worker.ProcessId)) ALIVE"
} elseif ($proc) {
    Write-Output "✅ Parent process (PID 18636) ALIVE"
} else {
    Write-Output "❌ CRITICAL: Process DEAD — check logs"
}

# 2. Latest log lines (last 15 lines to see current phase)
$logPath = "C:\AI-SRE\data\processed\gaia_merge_full.log"
if (Test-Path $logPath) {
    Write-Output ""
    Write-Output "=== Last 15 log lines (current phase) ==="
    Get-Content $logPath -Tail 15
} else {
    Write-Output "⚠️ Log file not found"
}

# 3. Progress counts (how many chunks/files written?)
if (Test-Path $logPath) {
    $biz = (Select-String -Path $logPath -Pattern "business parquet write complete" -ErrorAction SilentlyContinue).Count
    $met = (Select-String -Path $logPath -Pattern "metric parquet write complete" -ErrorAction SilentlyContinue).Count
    $tra = (Select-String -Path $logPath -Pattern "trace parquet write complete" -ErrorAction SilentlyContinue).Count
    Write-Output ""
    Write-Output "=== Chunks/Files Written ==="
    Write-Output "Business: $biz | Metric: $met | Trace: $tra"
}

# 4. Output file status
$outPath = "C:\AI-SRE\data\processed\gaia_unified.parquet"
if (Test-Path $outPath) {
    $size = (Get-Item $outPath).Length
    Write-Output ""
    Write-Output "✅ FINAL OUTPUT EXISTS: $([math]::Round($size/1GB, 2)) GB"
} else {
    Write-Output ""
    Write-Output "⏳ Output file not yet created (still merging)"
    Get-ChildItem "C:\AI-SRE\data\processed" -Filter "gaia_unified_*.parquet" -ErrorAction SilentlyContinue | Select-Object Name, Length, LastWriteTime
}

# 5. Disk space
$d = Get-PSDrive C
$freeGB = [math]::Round($d.Free/1GB, 2)
$usedGB = [math]::Round($d.Used/1GB, 2)
Write-Output ""
Write-Output "=== Disk Space ==="
Write-Output "Free: $freeGB GB | Used: $usedGB GB"