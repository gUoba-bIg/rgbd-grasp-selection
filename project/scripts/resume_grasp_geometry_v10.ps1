$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
foreach ($stage in @('P3D10', 'P3A10', 'P3G10')) {
    & 'D:\python\python.exe' -B scripts/pipeline.py run --stage $stage --resume
    if ($LASTEXITCODE -ne 0) {
        throw "第十轮阶段 $stage 中止；已提交文件和候选均保留。请检查对应日志，勿删除结果或从头训练。"
    }
}
