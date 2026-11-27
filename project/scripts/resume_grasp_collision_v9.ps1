$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
& 'D:\python\python.exe' -B scripts/pipeline.py run --stage P3C9 --resume
if ($LASTEXITCODE -ne 0) { throw '第九轮中止；已提交候选块及旧结果均保留，请检查日志后按同一入口恢复。' }
