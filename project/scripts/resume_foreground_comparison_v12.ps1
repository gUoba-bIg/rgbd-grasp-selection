$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
foreach ($stage in @('P3G12', 'P3S12', 'P3E12')) {
    & 'D:\python\python.exe' -B scripts/pipeline.py run --stage $stage --resume
    if ($LASTEXITCODE -ne 0) { throw "第十二轮阶段 $stage 未完成。请保留已有输出并检查日志，不要强制覆盖。" }
}
