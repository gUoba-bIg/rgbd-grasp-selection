$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONDONTWRITEBYTECODE = '1'
foreach ($stageName in @('P4R18', 'P4C18', 'P4T18')) {
    & 'D:\python\python.exe' -B 'scripts/pipeline.py' run --stage $stageName --resume
    if ($LASTEXITCODE -ne 0) {
        throw "Stage $stageName stopped; retain checkpoints and inspect its logs."
    }
}
