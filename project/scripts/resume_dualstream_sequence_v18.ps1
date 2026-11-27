$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $ProjectRoot
try {
    & 'D:\python\python.exe' -B scripts/pipeline.py run --stage P4T18 --resume
    if ($LASTEXITCODE -ne 0) { throw 'Paired training did not complete; inspect retained logs.' }
    & 'D:\python\python.exe' -B scripts/freeze_dualstream_evaluation_v18.py
    if ($LASTEXITCODE -ne 0) { throw 'Evaluation binding failed; no evaluation was started.' }
    foreach ($Stage in @('P4S18', 'P4A18', 'P4E18')) {
        & 'D:\python\python.exe' -B scripts/pipeline.py run --stage $Stage --resume
        if ($LASTEXITCODE -ne 0) { throw "Stage $Stage did not complete; inspect retained logs." }
    }
    & '.\runtime\envs\grasp_eval\Scripts\python.exe' -B scripts/verify_dualstream_evaluation_readonly_v18.py
    if ($LASTEXITCODE -ne 0) { throw 'Final read-only verification failed; inspect retained logs.' }
}
finally {
    Pop-Location
}
