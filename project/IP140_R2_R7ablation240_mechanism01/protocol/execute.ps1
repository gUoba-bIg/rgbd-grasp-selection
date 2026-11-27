$ErrorActionPreference = 'Stop'
$versionDir = Split-Path -Parent $PSScriptRoot
$researchRoot = Split-Path -Parent $versionDir
Set-Location -LiteralPath $researchRoot
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$python = Join-Path $researchRoot 'runtime/envs/grasp_eval/Scripts/python.exe'
& $python -B (Join-Path $versionDir 'code/dispatch_ablation.py')
$workerStatus = $LASTEXITCODE
if ($workerStatus -ne 0) {exit $workerStatus}
& $python -B (Join-Path $versionDir 'reporting/finish.py')
exit $LASTEXITCODE
