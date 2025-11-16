# CaptionQA Living Roadmap
_Last updated: 2025-11-16_

This document is the single source of truth for what matters next in CaptionQA. Update it after every notable run so future contributors inherit accurate context.

---

## How To Use This Doc
- **Update after every milestone**: append new metrics, mark tasks complete, and add follow-ups so context never goes stale.
- **Log decisions**: jot down the “why” whenever priorities shift.
- **Link artifacts**: include run commands plus artifact paths (e.g., `data/eval/.../summary.json`).
- **Keep priorities explicit**: move items between priority buckets rather than deleting them.

---

## Mission & Success Criteria
Deliver a reproducible 360° video captioning + QA baseline that:
1. Runs end-to-end on dev-mini + 360x subsets (captioning + QA) with logged metrics.
2. Documents manifests, commands, and troubleshooting steps so classmates can rerun it.
3. Provides a foundation for prompt/config tuning and eventual fine-tuning.

Success looks like a README/docs site with baseline tables, CI staying green, and the ability to spin workloads up on cloud or local hardware without guesswork.

---

## Current Progress Snapshot
- **Baseline infra**: Captioning + QA runners, manifest helpers, evaluator CLI, and CI (Windows + Ubuntu) remain solid.
- **Models**: Qwen2.5‑VL‑7B runs end-to-end on devmini and Vast A10 (frame budgeting keeps VRAM usage sane).
- **Evaluations**: Caption dev-mini (100 clips) logged BLEU 0.0053 / CIDEr 0.0050 / SPICE 0.0536 @ `data/eval/captioning/360x_devmini/summary.json`. QA dev-mini (290 Qs) completed but accuracy is 0.0 because answers rarely match TAL strings exactly; see `data/eval/qa/360x_devmini/summary.json`.
- **Docs & tooling**: Vast setup script installs `hf_transfer`, skips existing datasets, renders a single HF download progress bar, and now points QA generation at the right TAL path. README summarizes the Vast workflow plus baseline metrics.
- **Open gaps**: Need friendlier QA scoring (normalization or keyword matching), deeper prompt/config sweeps, and a longer-form docs surface (GitHub Pages or MkDocs) for architecture details and troubleshooting.

---

## Priority Stack
1. **QA Scoring Improvements (High)**  
   - Inspect `data/eval/qa/360x_devmini/preds.jsonl` vs. refs to design normalization (case fold, keyword match, action mapping) so zero accuracy isn’t misleading.  
   - Update evaluator or add a lightweight post-processor and re-run the dev-mini QA baseline; document assumptions.

2. **Caption Prompt/Config Iterations (High)**  
   - Sweep prompt wording, frame counts, and temporal windows using reduced manifests (`--limit 20`), then re-run full dev-mini and refresh metrics in README/roadmap.  
   - Track configs near the metrics table for future comparisons.

3. **Docs Expansion (High)**  
   - Stand up a lightweight GitHub Pages / MkDocs site for architecture notes, dataset details, and troubleshooting beyond what fits in README.

4. **Cloud Workflow Hardening (Medium)**  
   - Re-verify `scripts/setup_vast_gpu.sh` on a fresh A10 with ≥100 GB disk or a persistent `/workspace/data` volume.  
   - Document recommended storage sizes + shared volume strategy to avoid “no space left” surprises.

5. **Extended Items (Stretch)**  
   - Integrate official 360x annotations directly into manifest builders.  
   - Add QA datasets beyond dev-mini (e.g., AVQA).  
   - Explore fine-tuning or fusion/ensemble experiments after baselines stabilize.

---

## Near-Term Roadmap
### 0-2 Days
- ✅ Vast setup script upgrades (hf_transfer install, skip existing datasets, aggregate progress bar, QA annotations flag) – 2025-11-16.  
- ✅ README trimmed + Vast workflow + baseline metrics documented – 2025-11-16.  
- ✅ Ran both caption + QA baselines on Vast dev-mini; metrics captured in `data/eval/.../summary.json` and README – 2025-11-16.  
- ⏳ Decide on docs approach (README vs. GitHub Pages) and set up a skeleton if using Pages.  
- ✅ Prototype QA normalization (string cleanup, keyword sets) and measure impact on dev-mini accuracy – 2025-11-16.  
- ✅ Added TAL-label prompt option + confusion matrix export for QA baseline – 2025-11-16.

### 1-2 Weeks
- ⏳ Iterate on prompts/configs using reduced manifests, then run the full 360x dev-mini set.  
- ⏳ Validate Vast workflow with persistent storage or larger disks; capture the recommendation in README/docs (include HF cache + data env var guidance).  
- ⏳ Stand up longer-form docs site once the approach is chosen.

### Stretch / Later
- ⏳ Enhance manifest helper to ingest official annotations automatically.  
- ⏳ Capture additional baseline engines (fusion + others) for comparison.  
- ⏳ Explore fine-tuning or multi-engine ensembles once baselines are solid.

---

## Notes & Future Considerations
- **Performance**: Local hardware struggles with Qwen2.5‑VL (5–6 min load; ~90 min per 100 clips). Keep heavy runs on cloud GPUs.  
- **Dataset access**: Maintain Hugging Face token instructions; document any new gated datasets.  
- **Artifacts**: Store evaluation outputs under `data/eval/...` consistently; consider syncing artifacts to shared storage across environments.  
- **Env vars**: Recommend setting `DATA_ROOT` (dataset volume) and `HF_HOME` (cache) explicitly on Vast so repeated runs don’t redownload assets.  
- **Collab-ready**: README now points directly here; once GitHub docs exist, funnel deep dives there.  
- **Downloads**: `hf_transfer` + aggregate progress bars dramatically clean up Vast logs—leave `HF_HUB_ENABLE_HF_TRANSFER=1` enabled whenever possible.

---

## Update Checklist
1. After every major run: append metrics with paths + configs.  
2. When priorities change: edit the stack here rather than keeping side lists.  
3. Before meetings/demos: skim “Progress” + “Roadmap” to ensure they reflect reality.  
4. Each milestone: confirm Mission/Success Criteria still match project scope.  
5. If unsure: add a note here instead of leaving context in private chats.
