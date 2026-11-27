$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pipelinePython = 'D:\python\python.exe'
foreach ($stage in @('P2D8', 'P2R8', 'P2T8', 'P2E8')) {
    & $pipelinePython -B scripts/pipeline.py run --stage $stage --resume
    if ($LASTEXITCODE -ne 0) { throw "阶段 $stage 中止；已完成文件、帧缓存和训练检查点均保留。" }
}
