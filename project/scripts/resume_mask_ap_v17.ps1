$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
foreach ($stage in @('P4D17R1','P4S17','P4A17','P4E17')) {
    & 'D:\python\python.exe' -B scripts/pipeline.py run --stage $stage --resume
    if ($LASTEXITCODE -ne 0) { throw "阶段 $stage 未完成；保留已有记录，检查日志后用同一入口恢复。" }
}
