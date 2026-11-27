param(
    [ValidateSet('status', 'run')]
    [string]$Action = 'status',
    [string]$Stage,
    [string]$Through,
    [switch]$Resume,
    [switch]$NewRun,
    [switch]$Json,
    [string]$Python = 'D:\python\python.exe'
)
$ErrorActionPreference = 'Stop'
$projectPath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable was not found: $Python"
}
if ($Resume -and $NewRun) { throw 'Resume and NewRun cannot be combined.' }
if ($Stage -and $Through) { throw 'Choose Stage or Through, not both.' }
$env:PROJECT_ROOT = $projectPath
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPYCACHEPREFIX = Join-Path $projectPath 'runtime\cache\pycache'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$env:TEMP = Join-Path $projectPath 'runtime\tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
$pipelineArgs = @((Join-Path $PSScriptRoot 'pipeline.py'), $Action)
if ($Action -eq 'run') {
    if ($Stage) { $pipelineArgs += @('--stage', $Stage) }
    elseif ($Through) { $pipelineArgs += @('--through', $Through) }
    else { throw 'Run requires Stage or Through.' }
    if ($Resume) { $pipelineArgs += '--resume' }
    if ($NewRun) { $pipelineArgs += '--new-run' }
} elseif ($Json) {
    $pipelineArgs += '--json'
}
& $Python @pipelineArgs
exit $LASTEXITCODE
