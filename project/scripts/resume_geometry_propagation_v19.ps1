$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $ProjectRoot
try {
    foreach ($Stage in @('P6R19', 'P6E19')) {
        & 'D:\python\python.exe' -B scripts/pipeline.py run --stage $Stage --resume
        if ($LASTEXITCODE -ne 0) { throw "Stage $Stage did not complete; retain all evidence and inspect logs." }
    }
    & '.\runtime\envs\grasp_eval\Scripts\python.exe' -B scripts/verify_geometry_propagation_readonly_v19.py
    if ($LASTEXITCODE -ne 0) { throw 'Synthetic geometry read-only verification failed.' }
}
finally {
    Pop-Location
}
