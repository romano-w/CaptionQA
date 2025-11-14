Param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Args
)

# Pinned uv run using the repo's uv venv interpreter
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Resolve-Path (Join-Path $here "..")
$py = Join-Path $repo "captionqa/Scripts/python.exe"

# Default Hugging Face cache dir on D: drive unless user overrides HF_HOME
if (-not $env:HF_HOME) {
  $hfHome = "D:\HFCache\huggingface"
  if (-not (Test-Path $hfHome)) {
    New-Item -ItemType Directory -Path $hfHome -Force | Out-Null
  }
  $env:HF_HOME = $hfHome
}

if (-not (Test-Path $py)) {
  Write-Error "Pinned interpreter not found at $py. Run 'uv venv captionqa' first." -ErrorAction Stop
}

& uv run --no-sync --no-project -p $py @Args
exit $LASTEXITCODE
