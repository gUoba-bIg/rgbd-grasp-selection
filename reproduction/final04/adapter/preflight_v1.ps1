[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$Target)
$ErrorActionPreference='Stop'
$t=[IO.Path]::GetFullPath($Target)
$e0=$env:PATH
try {
 $env:PATH=("$t\bin","$t\bin\DLLs","$env:SystemRoot\System32","$env:SystemRoot" -join ';')
 $env:PYTHONUTF8='1'
 & (Join-Path $t 'bin/python.exe') -I -B (Join-Path $PSScriptRoot 'adapters/register_host.py') $t
 if($LASTEXITCODE -ne 0){throw 'Unable to register exact Windows x64 AMSI server.'}
 foreach($role in @('vision','rank','physics')){
  $file=Join-Path $t "setup_logs/preflight_$role.json"
  if(Test-Path -LiteralPath $file){throw "Preflight result exists: $file"}
  $data=& (Join-Path $t "env/$role/Scripts/python.exe") -I -B (Join-Path $PSScriptRoot 'adapters/preflight.py') $t $role
  if($LASTEXITCODE -ne 0){throw "Preflight failed in $role"}
  $line=@($data | Where-Object {$_ -match '^\{"role"'});if($line.Count -ne 1){throw "Missing preflight return ($role)"}
  [IO.File]::WriteAllText($file,$line[0],[Text.UTF8Encoding]::new($false))
 }
} finally {$env:PATH=$e0}
