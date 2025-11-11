Windows troubleshooting and tips
================================

Device and drivers
- Verify GPU visibility: run `nvidia-smi` (CUDA) or use the DirectML extra for non‑NVIDIA GPUs.
- Force CPU for any component by setting encoder/decoder `device` to `cpu` in config.

FFmpeg
- Ensure `ffmpeg` is on PATH. Check with `ffmpeg -version`.
- If audio extraction fails, the pipeline falls back to zeros; logs will mention the fallback.

Symlinks and Hugging Face cache
- Enable Windows Developer Mode or run PowerShell as Administrator to allow symlinks.
- To silence symlink warnings: set `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.

Long paths
- Enable long paths via Group Policy or registry. Alternatively, keep dataset roots shallow (e.g., `D:\CaptionQA\data`).

Antivirus file locks
- Organizer and downloader handle transient `PermissionError` with retries/copies, but consider excluding the dataset and cache directories from real‑time scanning.

uv usage
- Use the pinned wrapper: `./scripts/uv_run.ps1 ...`.
- For reproducible environments: `uv run --locked` (ensures `uv.lock` is respected).

Cache management
- Encoded features are stored under `data/cache/{visual,audio}` by default.
- Clear cache with the helper: `./scripts/clean_cache.ps1`.

