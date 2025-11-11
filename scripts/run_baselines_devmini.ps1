Param(
  [string]$CaptionManifest = 'data/dev-mini/captioning/samples.jsonl',
  [string]$CaptionRefs = 'data/dev-mini/captioning/refs.jsonl',
  [string]$CaptionOut = 'data/eval/captioning/360x_devmini',
  [string]$QAManifest = 'data/dev-mini/qa/manifest.jsonl',
  [string]$QARefs = 'data/dev-mini/qa/refs.jsonl',
  [string]$QAOut = 'data/eval/qa/devmini',
  [ValidateSet('fusion','qwen_vl')]
  [string]$Engine = 'qwen_vl'
)

$uv = Join-Path $PSScriptRoot 'uv_run.ps1'

Write-Output "[Captioning] Running baseline ($Engine) on $CaptionManifest"
& $uv python -m captionqa.captioning.baseline --manifest $CaptionManifest --engine $Engine --output-dir $CaptionOut --refs $CaptionRefs

if (-not (Test-Path $QAManifest)) {
  # Create a simple QA manifest using the dev-mini dummy clip
  $qaDir = Split-Path $QAManifest -Parent
  New-Item -ItemType Directory -Force -Path $qaDir | Out-Null
  $dummy = 'data/dev-mini/samples/dummy.mp4'
  $row = @{ id = 'qa1'; video = $dummy; question = 'What color is the scene?' } | ConvertTo-Json -Compress
  Set-Content -Path $QAManifest -Value $row -Encoding UTF8
}

Write-Output "[QA] Running Qwen-VL VQA baseline on $QAManifest"
& $uv python -m captionqa.qa.baseline_vqa --manifest $QAManifest --refs $QARefs --output-dir $QAOut

