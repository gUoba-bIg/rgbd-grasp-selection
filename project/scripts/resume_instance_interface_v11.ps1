$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
foreach ($stage in @('P3I11', 'P3M11', 'P3E11')) {
    & 'D:\python\python.exe' -B scripts/pipeline.py run --stage $stage --resume
    if ($LASTEXITCODE -ne 0) { throw "第十一轮阶段 $stage 未完成。请保留记录并检查对应阶段日志。" }
}
