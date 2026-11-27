$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
foreach ($stageId in @('P4R16', 'P4T16', 'P4V16', 'P4E16')) {
    & 'D:\python\python.exe' -B scripts/pipeline.py run --stage $stageId --resume
    if ($LASTEXITCODE -ne 0) {
        throw "Stage $stageId stopped; saved checkpoints and diagnostics were retained."
    }
}
