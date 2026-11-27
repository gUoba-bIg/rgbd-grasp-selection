$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
foreach ($stageId in @('P3F15', 'P3L15', 'P3E15R1')) {
    & 'D:\python\python.exe' -B scripts/pipeline.py run --stage $stageId --resume
    if ($LASTEXITCODE -ne 0) {
        throw "Stage $stageId stopped; all committed results and failure evidence are retained."
    }
}
