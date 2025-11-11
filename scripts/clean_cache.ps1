Param(
  [switch]$Yes
)

$cache = Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')) 'data\cache'
if (-not (Test-Path $cache)) {
  Write-Output "No cache directory found at $cache"
  exit 0
}

if (-not $Yes) {
  $resp = Read-Host "Delete cache directory $cache ? [y/N]"
  if ($resp -ne 'y' -and $resp -ne 'Y') { Write-Output 'Aborted'; exit 1 }
}

Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $cache
Write-Output "Deleted cache at $cache"

