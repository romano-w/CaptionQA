"""Helpers for tidying a manually downloaded 360SR dataset drop.

This script is intended for the case where Google Drive splits the
`360SR-Challenge` folder download into several `drive-download-*`
directories that each contain fragment archives such as
`train_x4_150_199.zip`. Running the module consolidates those artifacts,
extracts them into the canonical directory structure, and moves the raw
archives into a dedicated `archives/` folder so the dataset tree is ready
for use by the CaptionQA pipelines.

Usage
-----

```bash
python -m captionqa.scripts.organize_360sr
# or, for a custom path
python -m captionqa.scripts.organize_360sr --base D:/CaptionQA/data/360sr/360SR-Challenge
```

The operation is idempotent: if an archive's contents are already
present, extraction is skipped. All archives (processed or skipped) are
relocated into `<base>/archives/` so that the main dataset directory only
contains ready-to-use files.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


DEFAULT_BASE = Path("datasets/360sr/360SR-Challenge")


def _move_temp_archives(base: Path) -> None:
    """Flatten any Google Drive `drive-download-*` directories.

    The temporary directories often hold duplicate zip archives (and
    occasionally spreadsheets). We move unique files into the base
    directory and delete the temporary folders once processed.
    """

    for temp_dir in sorted(base.glob("drive-download-*")):
        for item in temp_dir.rglob("*"):
            if not item.is_file():
                continue

            destination = base / item.name
            if destination.exists():
                item.unlink()
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            item.replace(destination)

        # Remove the now-empty temporary folder tree.
        shutil.rmtree(temp_dir, ignore_errors=True)


def _resolve_destination(odv_base: Path, zip_name: str) -> Path | None:
    lower = zip_name.lower()
    if lower.startswith("train_hr"):
        return odv_base / "train" / "HR"
    if lower.startswith("train_x4"):
        return odv_base / "train" / "x4"
    if lower.startswith("test_x4"):
        return odv_base / "test" / "x4"
    if lower.startswith("val_x4"):
        return odv_base / "val" / "x4"
    return None


def _already_extracted(zip_path: Path, destination: Path) -> bool:
    """Return True if the archive's top-level members already exist."""

    with zipfile.ZipFile(zip_path) as archive:
        members = {
            entry.split("/", 1)[0]
            for entry in archive.namelist()
            if entry.strip()
        }

    if not members:
        return False

    return all((destination / member).exists() for member in members)


def _extract_archives(base: Path) -> None:
    archive_dir = base / "archives"
    archive_dir.mkdir(exist_ok=True)

    odv_base = base / "Ntire2023-ODV360"
    odv_base.mkdir(parents=True, exist_ok=True)

    for zip_path in sorted(odv_base.glob("*.zip")):
        destination = _resolve_destination(odv_base, zip_path.stem)
        if destination is None:
            continue

        destination.mkdir(parents=True, exist_ok=True)

        if _already_extracted(zip_path, destination):
            print(f"[skip] {zip_path.name} already extracted -> {destination}")
        else:
            print(f"[extract] {zip_path.name} -> {destination}")
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(destination)

        archive_path = archive_dir / zip_path.name
        if archive_path.exists():
            zip_path.unlink()
            continue

        try:
            zip_path.replace(archive_path)
        except PermissionError:
            # On Windows an aggressive antivirus or indexing process can
            # momentarily hold a lock on the file. Fall back to a copy to
            # ensure the archive still lands in the expected location.
            shutil.copy2(zip_path, archive_path)
            zip_path.unlink(missing_ok=True)

    # Sweep any remaining archives left at the base level (e.g. files that
    # were moved out of drive-download directories) so that everything ends
    # up under `archives/`.
    for stray_zip in sorted(base.glob("*.zip")):
        archive_path = archive_dir / stray_zip.name
        if archive_path.exists():
            stray_zip.unlink()
            continue
        try:
            stray_zip.replace(archive_path)
        except PermissionError:
            shutil.copy2(stray_zip, archive_path)
            stray_zip.unlink(missing_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Organize 360SR dataset archives")
    parser.add_argument(
        "--base",
        type=Path,
        default=DEFAULT_BASE,
        help=(
            "Root directory of the 360SR data (default: datasets/360sr/360SR-Challenge "
            "relative to the repo root)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    base = args.base.expanduser().resolve()
    if not base.exists():
        raise SystemExit(f"Base directory {base} does not exist; nothing to organize.")

    print(f"Tidying 360SR dataset under: {base}")
    _move_temp_archives(base)
    _extract_archives(base)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())