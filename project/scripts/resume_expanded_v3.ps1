$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    & 'D:\python\python.exe' -B 'scripts/pipeline.py' run --through P2E3 --resume
    if ($LASTEXITCODE -ne 0) {
        throw "Stage execution stopped with exit code $LASTEXITCODE. Checkpoints and logs are preserved."
    }
}
finally {
    Pop-Location
}
