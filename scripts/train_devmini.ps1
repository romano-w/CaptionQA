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
  if (-not $Output) { $Output = (Join-Path $repo 'checkpoints/caption_fusion.pt') }
  $args = @('python','-m','captionqa.captioning.train_captioning', $Pairs, '--epochs', $Epochs, '--lr', $LR, '--device', $Device, '--output', $Output)
  & $uv @args

  if ($Eval) {
    $predsPath = (Join-Path $repo 'data/eval/captioning/preds_devmini.jsonl')
    $refsPath = (Join-Path $repo 'data/eval/captioning/refs_devmini.jsonl')
    $py = @"
import json, sys
from pathlib import Path
import torch
from captionqa.captioning.cli import load_config
from captionqa.captioning.config import CaptioningConfig, build_pipeline
from captionqa.captioning.pipeline import generate_captions

pairs = json.loads(Path(r'$Pairs').read_text())
preds_path = Path(r'$predsPath'); preds_path.parent.mkdir(parents=True, exist_ok=True)
refs_path = Path(r'$refsPath'); refs_path.parent.mkdir(parents=True, exist_ok=True)

# Build default pipeline and try to load fine-tuned parts
cfg = CaptioningConfig.from_defaults()
sampler, venc, aenc, decoder, fusion = build_pipeline(cfg)
ckpt = Path(r'$Output')
if ckpt.exists():
    state = torch.load(ckpt, map_location='cpu')
    if 'fusion' in state and fusion.net is not None:
        fusion.net.load_state_dict(state['fusion'])
    if 'projector' in state:
        # ensure projector exists
        if decoder._cond_projector is None:
            dummy = torch.zeros(1, fusion.config.hidden_size)
            decoder.generate('warmup', conditioning=dummy)
        if decoder._cond_projector is not None:
            decoder._cond_projector.load_state_dict(state['projector'])

with preds_path.open('w', encoding='utf-8') as p, refs_path.open('w', encoding='utf-8') as r:
    for ex in pairs:
        vid = ex['video']
        cap = generate_captions(vid)
        p.write(json.dumps({'id': vid, 'prediction': cap}, ensure_ascii=False) + '\n')
        r.write(json.dumps({'id': vid, 'references': [ex['caption']]}, ensure_ascii=False) + '\n')

from captionqa.evaluation.run import main as eval_main
summary = eval_main(['--task','captioning','--preds',str(preds_path),'--refs',str(refs_path)])
import json as _j; print(_j.dumps(summary, indent=2))
"@
    & $uv python -c $py
  }
}
else {
  throw "Unknown task: $Task"
}
