$ErrorActionPreference = 'Stop'
$projectPath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$runnerPath = Join-Path $PSScriptRoot 'pipeline.py'
$runnerPython = 'D:\python\python.exe'
if (-not (Test-Path -LiteralPath $runnerPython -PathType Leaf)) {
    throw "The recorded stage-runner interpreter is missing: $runnerPython"
}
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
Push-Location -LiteralPath $projectPath
try {
    & $runnerPython -B $runnerPath run --through P2L --resume
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $runnerPython -B $runnerPath run --through P3S --resume
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
