$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pipelinePython = 'D:\python\python.exe'
foreach ($stage in @('P2D6', 'P2V6', 'P2E6')) {
    & $pipelinePython -B scripts/pipeline.py run --stage $stage --resume
    if ($LASTEXITCODE -ne 0) { throw "阶段 $stage 未完成；已提交数据与帧缓存保留。" }
}
