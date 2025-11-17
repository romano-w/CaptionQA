# CaptionQA Living Roadmap
_Last updated: 2025-11-17_

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
- **Models**: Qwen2.5‑VL‑7B runs end-to-end on devmini across Vast A10 + A40 boxes with current VRAM budgeting.
- **Evaluations**: Caption dev-mini (100 clips) still logs BLEU 0.0053 / CIDEr 0.0050 / SPICE 0.0536 @ `data/eval/captioning/360x_devmini/summary.json`. QA dev-mini (290 Qs) now lands Accuracy/F1 = **0.1586** with normalization (`data/eval/qa/360x_devmini/summary.json`) and **0.1759** when forcing TAL labels (`data/eval/qa/360x_devmini_forceprompt/summary.json`); confusion exports live alongside each summary and no `<engine-unavailable>` answers remain.
- **Docs & tooling**: Vast setup script installs `hf_transfer`, skips existing datasets, renders a single HF download progress bar, and points QA generation at the right TAL path. README + this roadmap summarize the Vast workflow, commands, and latest metrics for hand-off-ready reproducibility.
- **Open gaps**: Finalize submission packaging (artifact bundles + docs polish), continue improving QA normalization/prompt heuristics (reduce “walking” bias), and stand up longer-form docs (GitHub Pages or MkDocs) for deeper architecture + troubleshooting notes.

---

## Priority Stack
1. **Delivery Package & Submission (High)**  
   - Bundle `data/eval/qa/360x_devmini{,_forceprompt}` (and captioning if rerun) plus confusion summaries for inclusion with the turn-in.  
   - Sweep README + docs for consistency, cite the 0.1586 / 0.1759 QA metrics, and capture any known issues/limitations so reviewers know what’s next.

2. **QA Scoring Improvements (High, post-delivery)**  
   - Mine `data/eval/qa/360x_devmini/preds.jsonl` for remaining raw phrases (dressing/phone/speaking) to extend normalization regexes.  
   - Iterate on the forced-label prompt/examples until confusion stops collapsing into “walking/standing.”

3. **Caption Prompt/Config Iterations (High)**  
   - Sweep caption prompts, frame counts, and temporal windows on reduced manifests (`--limit 20`), then refresh the full dev-mini metrics and document the config deltas.

4. **Docs Expansion (Medium)**  
   - Stand up a lightweight GitHub Pages / MkDocs site for deeper architecture notes, dataset details, and troubleshooting.

5. **Cloud Workflow Hardening (Medium)**  
   - Re-verify `scripts/setup_vast_gpu.sh` on a fresh A10/A40 with ≥100 GB disk or a persistent `/workspace/data` volume, then document the recommended storage strategy.

6. **Extended Items (Stretch)**  
   - Integrate official 360x annotations directly into manifest builders.  
   - Add QA datasets beyond dev-mini (e.g., AVQA).  
   - Explore fine-tuning or fusion/ensemble experiments once baselines + docs remain stable.

---

## Near-Term Roadmap
### Today – Submission Push (Nov 17)
- ✅ Restored 360x dev-mini manifests/refs and re-ran the normalized QA baseline (Acc/F1 0.1586) – `data/eval/qa/360x_devmini`.  
- ✅ Re-ran the forced-label QA baseline (Acc/F1 0.1759) – `data/eval/qa/360x_devmini_forceprompt`.  
- ⏳ Summarize key confusion takeaways + known issues in README/docs so reviewers know where accuracy still collapses (dressing/phone/speaking → walking).  
- ⏳ Archive/bundle `data/eval/qa/360x_devmini{,_forceprompt}` (and captioning outputs if refreshed) for final submission/backup.  
- ⏳ Final doc sweep (README + roadmap + submission notes) to ensure commands, metrics, and troubleshooting steps are reproducible.

### Post-Submission (This Week)
- ⏳ Decide on docs approach (README-only vs. GitHub Pages) and bootstrap the chosen surface.  
- ⏳ Iterate QA/caption prompts/configs on reduced manifests, then refresh the full dev-mini set once improvements are validated.  
- ⏳ Validate Vast workflow with persistent storage or larger disks; document the recommended HF cache/data volume strategy.

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
