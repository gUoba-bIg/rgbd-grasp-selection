[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$Target)
$ErrorActionPreference='Stop'
$t=[IO.Path]::GetFullPath($Target)
& (Join-Path $t 'bin/python.exe') -I -B (Join-Path $PSScriptRoot 'adapters/summarize.py') $t
if ($LASTEXITCODE -ne 0){throw 'Unable to summarize the supplied cold output.'}
