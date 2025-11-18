# CaptionQA Documentation Hub

Welcome to the staging area for the CaptionQA knowledge base. This site will eventually host the full architecture notes, troubleshooting guides, and experiment logs that no longer fit inside the README.

## Sections

- [Living Roadmap](../living_roadmap.md) &mdash; authoritative status + priority tracker.
- [Windows Troubleshooting](../windows_troubleshooting.md) &mdash; Vast/GPU setup reminders for the lab machines.
- Evaluation artifacts under `data/eval/...` are linked from the README until we promote them into dedicated pages.

## Build / Preview

```bash
# Install mkdocs if needed: python -m pip install mkdocs
mkdocs serve -f docs/mkdocs.yml
# or build static assets under ./public-docs
mkdocs build -f docs/mkdocs.yml
```

## Next Up

- Flesh out dedicated pages for captioning + QA pipelines (prompts, configs, troubleshooting).
- Embed run recipes + metric tables directly in the docs instead of copying from README.
- Add a changelog + experiment diary for summary-banked QA iterations.
