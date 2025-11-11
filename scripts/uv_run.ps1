Param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Args
)

# Pinned uv run using the repo's uv venv interpreter
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Resolve-Path (Join-Path $here "..")
$py = Join-Path $repo "captionqa/Scripts/python.exe"

if (-not (Test-Path $py)) {
  Write-Error "Pinned interpreter not found at $py. Run 'uv venv captionqa' first." -ErrorAction Stop
}

& uv run -p $py @Args
exit $LASTEXITCODE

