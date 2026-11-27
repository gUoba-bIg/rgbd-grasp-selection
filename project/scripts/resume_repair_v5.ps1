$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    foreach ($stage in @('P2R5','P2T5','P2E5')) {
        & 'D:\python\python.exe' -B 'scripts/pipeline.py' run --stage $stage --resume
        if ($LASTEXITCODE -ne 0) {
            throw "Stage $stage stopped with exit code $LASTEXITCODE. Checkpoints and logs are preserved."
        }
    }
}
finally {
    Pop-Location
}
