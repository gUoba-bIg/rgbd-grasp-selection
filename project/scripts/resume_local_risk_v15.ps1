$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    foreach ($stage in @('P3F15', 'P3L15', 'P3E15')) {
        & 'D:\python\python.exe' -B 'scripts/pipeline.py' run --stage $stage --resume
        if ($LASTEXITCODE -ne 0) { throw "Stage $stage stopped; saved results are retained." }
    }
} finally {
    Pop-Location
}
