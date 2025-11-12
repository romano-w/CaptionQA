# CaptionQA Living Roadmap
_Last updated: 2025-11-12_

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
- **Evaluations**: Dev-mini captioning + QA artifacts exist (`data/eval/...devmini*`); 360x manifest prepared but long run still in progress (currently ~26% complete).
- **Docs & tooling**: README covers setup, huggingface auth, manifest helper usage, and troubleshooting; scripts automate dev-mini baselines.
- **Open gaps**: 360x + HF evaluation hasn’t finished; QA baseline uses a tiny single-example manifest; baseline metrics aren’t consolidated into docs yet.

---

## Priority Stack
1. **Finish 360x Captioning Baseline (High)**  
   - Command: `./scripts/uv_run.ps1 python -m captionqa.captioning.baseline --manifest data/eval/captioning/360x_devmini/manifest.jsonl --engine qwen_vl --output-dir data/eval/captioning/360x_devmini --dataset-name quchenyuan/360x_dataset_LR --split validation --id-column id --reference-column references`  
   - Deliverable: `summary.json` + note metrics here.  
   - Owner: Will (local or cloud run). _Update this entry once complete._

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
### 0–2 Days
- ✅ Improve baseline logging (done 2025-11-12).  
- 🔄 Finish Qwen-VL 360x evaluation, capture metrics, and note any hardware limitations encountered.  
- 🔄 Draft README “Baselines” table once new metrics land.  
- 🔄 Outline QA manifest plan (dataset choice + schema).

### 1–2 Weeks
- ☐ Implement QA manifest generator + run Qwen-VL baseline_vqa on a realistic subset.  
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

