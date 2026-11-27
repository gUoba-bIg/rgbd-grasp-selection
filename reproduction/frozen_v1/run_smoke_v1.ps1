[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$Target)
$ErrorActionPreference='Stop'
$t=[IO.Path]::GetFullPath($Target)
if(!(Test-Path -LiteralPath (Join-Path $t 'setup_complete.json'))){throw 'Run setup_v1.ps1 in an unused directory first.'}
& (Join-Path $t 'bin/python.exe') -I -B (Join-Path $PSScriptRoot 'adapters/driver.py') $t
if ($LASTEXITCODE -ne 0){throw 'Cold reproduction stopped. Keep the complete cold_receipts directory.'}
