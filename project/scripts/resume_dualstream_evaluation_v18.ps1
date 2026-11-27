$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONDONTWRITEBYTECODE = '1'
foreach ($stageName in @('P4S18', 'P4A18', 'P4E18')) {
    & 'D:\python\python.exe' -B 'scripts/pipeline.py' run --stage $stageName --resume
    if ($LASTEXITCODE -ne 0) {
        throw "Stage $stageName stopped; keep sealed frames and inspect its logs."
    }
}
