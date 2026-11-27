[CmdletBinding()]
param([string]$Archives,[switch]$Deep)
$ErrorActionPreference='Stop'
if (!$Archives){$Archives=Join-Path (Split-Path -Parent $PSScriptRoot) 'v1_archive'}
$receipts=@(Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'locks') -File -Filter '*.receipt.json')
foreach($r in $receipts){
 $m=Get-Content -LiteralPath $r.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
 foreach($pair in @(@((Join-Path $Archives $m.name),$m.sha256),@((Join-Path $PSScriptRoot $m.member_manifest),$m.member_manifest_sha256))){
  if((Get-FileHash -Algorithm SHA256 -LiteralPath $pair[0]).Hash.ToLowerInvariant() -ne $pair[1]){throw "Hash mismatch: $($pair[0])"}
 }
 if($Deep){Add-Type -AssemblyName System.IO.Compression.FileSystem; $z=[IO.Compression.ZipFile]::OpenRead((Join-Path $Archives $m.name));try{if($z.Entries.Count -ne $m.members){throw $m.name}}finally{$z.Dispose()}}
}
$ctx=Join-Path $PSScriptRoot 'metadata/baseline_files.sha256'
if(Test-Path -LiteralPath $ctx){
 foreach($s in Get-Content -LiteralPath $ctx -Encoding utf8){$a=$s -split '  ',2;if((Get-FileHash -LiteralPath (Join-Path $PSScriptRoot $a[1]) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $a[0]){throw "Changed baseline: $($a[1])"}}
}
"Verified $($receipts.Count) payloads and their locked inventories."
