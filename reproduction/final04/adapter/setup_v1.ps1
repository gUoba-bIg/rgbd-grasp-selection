[CmdletBinding()]
param([string]$Target,[string]$Archives)
$ErrorActionPreference='Stop'
$kit=$PSScriptRoot
if (!$Target) { $Target=Join-Path (Split-Path -Parent $kit) 'cold_v1' }
if (!$Archives) { $Archives=Join-Path (Split-Path -Parent $kit) 'v1_archive' }
$Target=[IO.Path]::GetFullPath($Target)
if (Test-Path -LiteralPath (Join-Path $Target 'bin')) { throw 'Select an unused setup directory.' }
if ((Test-Path -LiteralPath $Target) -and @(Get-ChildItem -Force -LiteralPath $Target).Count -gt 0) { throw 'Cold setup needs an empty target.' }
[void][IO.Directory]::CreateDirectory($Target)
$runtime=Join-Path $Target 'bin'
$lock=Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $kit 'locks/python312.receipt.json') | ConvertFrom-Json
$zip=Join-Path $Archives $lock.name
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $zip).Hash.ToLowerInvariant() -ne $lock.sha256) {throw 'CPython installer SHA256 mismatch'}
Add-Type -AssemblyName System.IO.Compression.FileSystem
[IO.Compression.ZipFile]::ExtractToDirectory($zip,$runtime)
$env:PYTHONDONTWRITEBYTECODE='1';$env:PYTHONNOUSERSITE='1';$env:PYTHONUTF8='1'
& (Join-Path $runtime 'python.exe') -I -B (Join-Path $kit 'adapters/setup.py') $Target $Archives
if ($LASTEXITCODE -ne 0) { throw "Cold setup failed ($LASTEXITCODE)." }
