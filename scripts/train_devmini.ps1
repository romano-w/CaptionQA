Param(
  [ValidateSet('avqa','captioning')]
  [string]$Task = 'avqa',
  [string]$DatasetRoot,
  [int]$Epochs = 1,
  [int]$BatchSize = 2,
  [double]$LR = 0.001,
  [string]$Device,
  [switch]$FP16,
  [string]$Output,
  [string]$Tokenizer,
  [string]$Pairs,
  [switch]$Eval
)

$ErrorActionPreference = 'Stop'
$repo = Resolve-Path (Join-Path $PSScriptRoot '..')
$uv = Join-Path $PSScriptRoot 'uv_run.ps1'

if (-not $Device) {
  $Device = (Get-Command nvidia-smi -ErrorAction SilentlyContinue) ? 'cuda' : 'cpu'
}

if ($Task -eq 'avqa') {
  if (-not $DatasetRoot) {
    if ($env:CAPTIONQA_DATASETS) {
      $candidates = @(
        Join-Path $env:CAPTIONQA_DATASETS 'avqa\AVQA'),
        (Join-Path $env:CAPTIONQA_DATASETS 'AVQA')
      foreach ($c in $candidates) { if (Test-Path $c) { $DatasetRoot = $c; break } }
    }
    if (-not $DatasetRoot) { throw "DatasetRoot not provided and CAPTIONQA_DATASETS not set" }
  }
  if (-not $Tokenizer) { $Tokenizer = (Join-Path $repo 'checkpoints/avqa_tokenizer.json') }
  if (-not $Output) { $Output = (Join-Path $repo 'checkpoints/avqa_tiny.pt') }

  $args = @('python','-m','captionqa.qa.train', $DatasetRoot, '--epochs', $Epochs, '--batch-size', $BatchSize, '--lr', $LR, '--device', $Device, '--tokenizer', $Tokenizer, '--output', $Output)
  if ($FP16) { $args += '--fp16' }
  & $uv @args

  if ($Eval) {
    $py = @"
from pathlib import Path
from captionqa.qa.eval import load_avqa_subset, run_fine_tuned
from captionqa.qa.summary import summarize_results
from captionqa.qa.tokenizer import SimpleWordTokenizer
ds = load_avqa_subset(Path(r'$DatasetRoot'), split='val', subset_size=8)
tok = SimpleWordTokenizer.load(Path(r'$Tokenizer'))
res = run_fine_tuned(Path(r'$Output'), ds, tok, device='$Device', max_length=8)
import json; print(json.dumps({'summary': summarize_results(res)}, indent=2))
"@
    & $uv python -c $py
  }
}
elseif ($Task -eq 'captioning') {
  if (-not $Pairs) {
    $Pairs = (Join-Path $repo 'data/dev-mini/captioning/pairs.json')
    if (-not (Test-Path $Pairs)) {
      $pairsDir = Split-Path $Pairs -Parent
      New-Item -ItemType Directory -Force -Path $pairsDir | Out-Null
      $dummy = (Join-Path $repo 'data/dev-mini/samples/dummy.mp4')
      $content = @(
        @{ video = $dummy; caption = 'a black scene with a tone' }
      ) | ConvertTo-Json -Depth 4
      Set-Content -Path $Pairs -Value $content -Encoding UTF8
    }
  }
  $args = @('python','-m','captionqa.captioning.train_captioning', $Pairs, '--epochs', $Epochs, '--lr', $LR, '--device', $Device)
  & $uv @args
}
else {
  throw "Unknown task: $Task"
}

