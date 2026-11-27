[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$Component)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$item=Get-Item -LiteralPath $Component -Force
$sig=Get-AuthenticodeSignature -LiteralPath $item.FullName
$nm=if ($sig.SignerCertificate) {$sig.SignerCertificate.GetNameInfo([System.Security.Cryptography.X509Certificates.X509NameType]::SimpleName,$false)} else {''}
[ordered]@{actual=$item.FullName;bytes=[Int64]$item.Length;version=[string]$item.VersionInfo.FileVersion;signature_status=[string]$sig.Status;signer_simple=$nm;signer_subject=$sig.SignerCertificate.Subject;thumbprint=$sig.SignerCertificate.Thumbprint;sha256=(Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}|ConvertTo-Json -Depth 5 -Compress
