# CaptionQA Living Roadmap
_Last updated: 2025-11-13_

This document is the single source of truth for what matters next in CaptionQA. Keep it concise, actionable, and up to date whenever goals shift or milestones finish.

---

## How To Use This Doc
- **Update after every milestone**: append new metrics, mark tasks complete, and add follow-ups so the context never goes stale.
- **Log decisions**: if priorities change, jot down the “why” here to avoid second-guessing later.
- **Link artifacts**: whenever you mention a run, note the command and artifact path (e.g., `data/eval/.../summary.json`).
- **Keep priorities explicit**: if a task drops in priority, move it rather than deleting so the history remains visible.

---

## Mission & Success Criteria
Deliver a reproducible 360° video captioning + QA baseline that:
1. Runs on dev-mini and 360x subsets end-to-end (captioning + QA) with logged metrics.
2. Documents manifests, commands, and troubleshooting steps so classmates/instructors can rerun it.
3. Provides a foundation for prompt/config tuning and future fine-tuning work.

Success looks like: `README` (or `docs/`) contains a baseline table with metrics, CI stays green, and we can spin up cloud or local runs on demand without manual babysitting.

---

## Current Progress Snapshot
- **Baseline infra**: Captioning + QA runners, manifest helper, evaluator CLI, and CI (Windows + Ubuntu) are in place.
- **Models**: Qwen2.5-VL 7B integrated for captioning and QA; temporal window support wired through manifests.
- **Evaluations**: Dev-mini captioning + QA artifacts exist (`data/eval/...devmini*`). 360x dev-mini caption run completed (100 clips) using TAL-derived references; metrics in `data/eval/captioning/360x_devmini/summary.json` are BLEU ≈ 0.0130, CIDEr ≈ 0.0050, SPICE ≈ 0.0536, but the predictions show the deterministic fallback prompt (Qwen frames failed to load).
- **Docs & tooling**: README now documents the `data/raw` junction workflow plus the TAL reference generator; scripts automate manifest + reference creation.
- **Open gaps**: Fix Qwen inference so frames load, run a QA baseline with real references, and surface metrics in README once they reflect true captions.

---

## Priority Stack
1. **Stabilize 360x Captioning Baseline (High)**  
   - Command: `./scripts/uv_run.ps1 python -m captionqa.captioning.baseline --manifest data/eval/captioning/360x_devmini/manifest.jsonl --engine qwen_vl --output-dir data/eval/captioning/360x_devmini --refs data/eval/captioning/360x_devmini/refs.jsonl`  
   - Deliverable: Updated predictions without fallback messaging + refreshed metrics logged here/README.  
   - Owner: Will (local or cloud run). Investigate Qwen frame loading (cache paths, VRAM) before next run.

2. **Build Real QA Baseline (High)**  
   - Generate a QA manifest tied to actual dataset references (360x or AVQA).  
   - Run `captionqa.qa.baseline_vqa` with Qwen-VL and collect accuracy/F1.  
   - Log metrics + manifest path here.

3. **Document Baselines (High)**  
   - Add “Baselines” section/table to README (or link to this doc) summarizing dev-mini + 360x results for fusion and Qwen-VL.  
   - Include manifest locations, engine configs, and evaluation references.

4. **Prompt & Config Iterations (Medium)**  
   - Experiment with Qwen-VL prompts, frame counts, decoding params.  
   - Use smaller manifests (`--limit 20`) for fast loops; graduate to full 100+ once improvements verified.

5. **Cloud Strategy & Automation (Medium)**  
   - Decide on Lambda Labs / RunPod / etc. for heavier runs.  
   - Capture setup notes, scripts, and data sync strategy so future sessions aren’t blocked by local hardware limits.

6. **Extended Roadmap Items (Low / Stretch)**  
   - Integrate official 360x annotations directly into the manifest helper (temporal spans, textual references).  
   - Add QA datasets beyond dev-mini (e.g., AVQA) with proper references.  
   - Explore fine-tuning or frozen-encoder training loops after baselines stabilize.

---

## Near-Term Roadmap
### 0-2 Days
- ✅ Improve baseline logging (2025-11-12).  
- ✅ Complete first Qwen-VL 360x evaluation with TAL references (2025-11-13).  
- 🔄 Debug the Qwen fallback issue (frames not loading / deterministic prompt in predictions).  
- 🔄 Draft README "Baselines" table once stable metrics exist.  
- 🔄 Outline QA manifest plan (dataset choice + schema).

### 1-2 Weeks
- ☐ Implement QA manifest generator + run captionqa.qa.baseline_vqa on a realistic subset.  
- ☐ Iterate on prompts/params using limited manifests; record results + parameter changes here.  
- ☐ Decide on (and document) remote GPU workflow for long runs.

### Stretch / Later
- ☐ Enhance manifest helper to ingest official annotations automatically.  
- ☐ Capture additional baseline engines (fusion + others) for comparison.  
- ☐ Explore fine-tuning or multi-engine ensembles once baselines are solid.
---

## Notes & Future Considerations
- **Performance**: Local hardware struggles with Qwen2.5-VL (5–6 min load time; 100-sample runs ~90 min). Plan to offload heavy runs to cloud to avoid blocking.  
- **Dataset access**: Keep Hugging Face token instructions current; note any new gated datasets here.  
- **Artifacts**: Store evaluation outputs under `data/eval/...` consistently; consider syncing key artifacts to cloud storage if multiple environments are used.  
- **Collab-ready**: When classmates need to reproduce runs, point them to this doc + README sections for commands, manifests, and troubleshooting.

---

## Update Checklist
1. **After every major run**: append metrics (BLEU/CIDEr/SPICE, Accuracy/F1) with paths + engine configs.  
2. **When priorities change**: edit the “Priority Stack” instead of creating ad-hoc to-do lists elsewhere.  
3. **Before meetings/demos**: skim “Current Progress” + “Roadmap” and ensure they reflect reality.  
4. **Quarterly or major milestone**: revisit Mission/Success Criteria to confirm they still match the project scope.  
5. **If unsure**: add a note under “Notes & Future Considerations” rather than letting context live only in chat threads.
