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
- **Summary-augmented QA**: `baseline_vqa` now consumes caption outputs via `--summary-jsonl`. Full dev-mini runs (`data/eval/qa/360x_devmini_summary{,_forceprompt}`) scored 0.079 (normalized) and 0.155 (forced-label), collapsing most answers to “photographing.” Two mitigation attempts:
  - `captionqa.qa.summary_slices` trims captions to ≤2 sentences per QA span, but the 60-question debug run (`data/eval/qa/360x_devmini_summaryslice60`) only reached Accuracy/F1 **0.033**.
  - True per-window captions generated via `scripts/build_qa_caption_manifest.py` + `captionqa.captioning.baseline` live at `data/eval/captioning/360x_devmini_perwindow60`. Feeding those summaries back into QA (`data/eval/qa/360x_devmini_perwindow60`) nudged Accuracy/F1 to **0.05** on the same subset—still below the baseline but less catastrophic than clip-level summaries.
- **Docs & tooling**: Vast setup script installs `hf_transfer`, skips existing datasets, renders a single HF download progress bar, and points QA generation at the right TAL path. README + this roadmap summarize the Vast workflow, commands, and latest metrics for hand-off-ready reproducibility.
- **Open gaps**: Finalize submission packaging (artifact bundles + docs polish), continue improving QA normalization/prompt heuristics (reduce “walking” bias), and stand up longer-form docs (GitHub Pages or MkDocs) for deeper architecture + troubleshooting notes.

---

## Priority Stack
1. **Delivery Package & Submission (High)**  
   - Bundle `data/eval/qa/360x_devmini{,_forceprompt,_summary,_summary_forceprompt}` (and captioning if rerun) plus confusion summaries for inclusion with the turn-in.  
   - Sweep README + docs for consistency, cite the 0.1586 / 0.1759 QA metrics, note the new summary-aug flow, and capture any known issues/limitations so reviewers know what’s next.

2. **Ambitious QA Roadmap (High)**  
   - **Summary-banked QA v2**: Split captions into per-window snippets, store them alongside manifests (`scripts/build_qa_caption_manifest.py` + `captionqa.captioning.baseline`), and fine-tune prompts so QA can cite the relevant snippet without overwhelming the question. Target ≥0.20 accuracy on dev-mini with summary context.  
   - **Question-aware captioner**: Re-run captioning with higher temporal resolution + action-focused prompt to see if the working memory improves; log BLEU/CIDEr plus QA downstream impact.  
   - **Normalization automation**: `scripts/analyze_qa_mismatches.py` now joins preds/refs, exports CSVs, and prints naive regex hints per phrase; next iterate on auto-suggesting final TAL patterns.

3. **Model/Engineering Experiments (High)**  
   - Prototype a lightweight retriever that pulls similar TAL examples (few-shot in-context learning) before firing Qwen.  
   - Evaluate a second VLM (e.g., LLaVA-OneVision) on a 30-question slice to diversify errors and build an ensemble plan.

4. **Docs & Tooling Expansion (Medium)**  
   - Stand up a lightweight GitHub Pages / MkDocs site for deeper architecture notes, dataset details, and troubleshooting; scaffold lives under `docs/mkdocs.yml` with `docs/site/index.md`, `docs/site/captioning.md`, and `docs/site/qa.md`. Next steps: add experiment diary + theming plus a “current experiments” board.  
   - Add automation scripts (`scripts/run_eval_suite.sh`) that kick off caption + QA + summary runs and symlink the latest artifacts.

5. **Cloud Workflow Hardening (Medium)**  
   - Re-verify `scripts/setup_vast_gpu.sh` on a fresh A10/A40 with ≥100 GB disk or a persistent `/workspace/data` volume, then document the recommended storage strategy.

6. **Extended Items (Stretch)**  
   - Integrate official 360x annotations directly into manifest builders.  
   - Add QA datasets beyond dev-mini (e.g., AVQA).  
   - Explore fine-tuning or fusion/ensemble experiments once baselines + docs remain stable.

---

## Near-Term Roadmap
### Tonight – Ambition Block (Nov 17)
- ✅ Restored 360x dev-mini manifests/refs and re-ran the normalized QA baseline (Acc/F1 0.1586) – `data/eval/qa/360x_devmini`.  
- ✅ Re-ran the forced-label QA baseline (Acc/F1 0.1759) – `data/eval/qa/360x_devmini_forceprompt`.  
- ✅ Wired caption summaries into `baseline_vqa`; full runs stored under `data/eval/qa/360x_devmini_summary{,_forceprompt}` (currently worse than baseline).  
- ✅ Sliced caption summaries by QA windows (2-sentence cap) via `captionqa.qa.summary_slices` and ran a 60-question probe (`data/eval/qa/360x_devmini_summaryslice60`). Accuracy/F1 stayed at **0.033**, confirming that trimming alone doesn’t fix the “photographing” collapse; next attempt needs true per-window captioning or stricter prompt clauses.  
- ✅ Built the mismatch-mining helper at `scripts/analyze_qa_mismatches.py`; it joins refs/preds and lists the top raw phrases per TAL label to guide regex updates.  
- ✅ Drafted the MkDocs skeleton (`docs/mkdocs.yml`, `docs/site/index.md`, `docs/site/captioning.md`, `docs/site/qa.md`) so tomorrow’s polish can drop straight in.

### This Week (Nov 18–24)
- ✅ Decide on docs approach (README-only vs. GitHub Pages) and bootstrap the chosen surface. MkDocs now holds starter Captioning/QA pages; next add architecture + troubleshooting deep dives.  
- 🔜 Iterate QA/caption prompts/configs on reduced manifests, then refresh the full dev-mini set once improvements are validated.  
- 🔜 Validate Vast workflow with persistent storage or larger disks; document the recommended HF cache/data volume strategy.  
- 🔜 Prototype second VLM / retrieval experiment and record performance deltas.

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
