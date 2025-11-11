# Changelog

## 0.1.0-pre

Highlights
- CLI-driven panoramic captioning with projection, encoders, and decoder fallback.
- Dataset tooling: downloader (HF/GDrive/Git), 360SR organizer, evaluation CLI.
- AVQA model skeleton + zero-shot evaluation.

Recent sprint
- Day 0: uv env setup, deterministic captioning, dev-mini assets, docs.
- Day 1: GPU-batched encoders, .pt caching, CLI cache flags, tests.
- Day 2: Fusion head + soft-prompt multimodal decoding; metrics tightened; tests.
- Day 3: CLI smoke tests, evaluation CLI test; true equirectangular→perspective projection.
- Day 4: AVQA zero-shot now uses real features; confidence + temporal windows; tiny training scripts (AVQA + captioning), tokenizer, QA summary API.
- Day 5: Packaging polish (pins + extras), Windows troubleshooting docs, convenience scripts and configs.

Known issues
- Real LLM conditioning depends on installed transformers/tokenizers and GPU memory.
- Projection supports yaw/pitch/roll; heavy pitch sampling increases compute.

