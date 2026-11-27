$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    & 'D:\python\python.exe' -B 'scripts/pipeline.py' run --stage P2A4 --resume
    if ($LASTEXITCODE -ne 0) {
        throw "Diagnosis stopped with exit code $LASTEXITCODE. Committed frames and logs are preserved."
    }
}
finally {
    Pop-Location
}
