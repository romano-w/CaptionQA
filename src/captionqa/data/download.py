"""Utilities for downloading datasets (package version).

Usage examples:

    python -m captionqa.data.download --list
    python -m captionqa.data.download 360x --output ./datasets
    python -m captionqa.data.download leader360v --output ./datasets --overwrite

This module is a direct migration of the original `data/download.py` so it can
be imported reliably from notebooks and scripts as part of the installed
`captionqa` package.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import requests
import gdown
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import (
    are_progress_bars_disabled,
    disable_progress_bars,
    enable_progress_bars,
)
from tqdm import tqdm


DatasetHandler = Callable[[Path, bool, bool], None]


_360X_RESOLUTION: str = "lr"


@dataclass(frozen=True)
class DownloadArtifact:
    url: str
    filename: Optional[str] = None
    extract: bool = False
    extract_subdir: Optional[str] = None
    checksum: Optional[str] = None
    checksum_algorithm: Optional[str] = None


@dataclass(frozen=True)
class DatasetTask:
    name: str
    description: str
    handler: DatasetHandler


def _resolve_checksum_parameters(
    checksum: str, checksum_algorithm: Optional[str]
) -> Tuple[str, str]:
    algo = (checksum_algorithm or "").lower() if checksum_algorithm else None
    value = checksum

    if ":" in checksum and not checksum_algorithm:
        algo_part, value = checksum.split(":", 1)
        algo = algo_part.strip().lower()
    elif checksum_algorithm is None:
        length_to_algo = {32: "md5", 64: "sha256"}
        algo = length_to_algo.get(len(checksum))

    if algo not in {"sha256", "md5"}:
        raise ValueError(
            "Checksum must specify an algorithm (sha256 or md5). "
            "Provide it explicitly via checksum_algorithm or use the 'algo:value' format."
        )

    normalized_value = value.strip().lower()
    return algo, normalized_value


def _calculate_checksum(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_checksum(
    path: Path, checksum: str, checksum_algorithm: Optional[str]
) -> None:
    algorithm, expected_value = _resolve_checksum_parameters(checksum, checksum_algorithm)
    actual_value = _calculate_checksum(path, algorithm)
    if actual_value != expected_value:
        try:
            path.unlink()
        except OSError:
            pass
        raise ValueError(
            "Checksum mismatch for '{path}'. Expected {algorithm} {expected} but got {actual}. "
            "The corrupted file has been removed; retry the download or use --overwrite.".format(
                path=path,
                algorithm=algorithm,
                expected=expected_value,
                actual=actual_value,
            )
        )


def _ensure_write_path(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination '{path}' already exists. Use --overwrite to replace it."
            )
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


def _download_http_file(url: str, destination: Path, overwrite: bool, dry_run: bool) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"File '{destination}' already exists. Use --overwrite to replace it."
        )
    if dry_run:
        print(f"[dry-run] Would download {url} -> {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    attempts = 3
    backoff = 2.0
    delay = 1.0
    last_error: Optional[Exception] = None
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            print(f"Downloading {url} -> {destination} (attempt {attempt}/{attempts})")
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()

                total = int(response.headers.get("content-length", 0)) or None
                with open(tmp_path, "wb") as handle, tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {destination.name}",
                ) as progress:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            handle.write(chunk)
                            progress.update(len(chunk))

            tmp_path.replace(destination)
            print(f"Download completed: {destination}")
            return
        except requests.RequestException as exc:
            last_error = exc
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            wait_time = delay * (backoff ** (attempt - 1))
            if attempt == attempts:
                break
            print(
                f"Download attempt {attempt} for {url} failed ({exc}). Retrying in {wait_time:.1f}s..."
            )
            time.sleep(wait_time)
        except Exception as exc:
            last_error = exc
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            break

    raise RuntimeError(
        f"Failed to download {url} after {attempts} attempts: {last_error}"
    ) from last_error


def _extract_archive(archive_path: Path, output_dir: Path, overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] Would extract {archive_path} -> {output_dir}")
        return

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive '{archive_path}' is missing")

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Extraction target '{output_dir}' already exists. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = archive_path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(output_dir)
    elif suffix in {".gz", ".bz2", ".xz"} and archive_path.name.endswith(".tar" + suffix):
        with tarfile.open(archive_path, "r:*") as archive:
            archive.extractall(output_dir)
    elif archive_path.suffixes[-2:] in ([".tar", ".gz"], [".tar", ".bz2"], [".tar", ".xz"]):
        with tarfile.open(archive_path, "r:*") as archive:
            archive.extractall(output_dir)
    elif archive_path.suffix == ".tar":
        with tarfile.open(archive_path, "r") as archive:
            archive.extractall(output_dir)
    else:
        raise ValueError(f"Unsupported archive format for '{archive_path}'")


def _download_artifacts(
    artifacts: Iterable[DownloadArtifact],
    dataset_dir: Path,
    overwrite: bool,
    dry_run: bool,
) -> None:
    for artifact in artifacts:
        filename = artifact.filename or Path(urlparse(artifact.url).path).name
        destination = dataset_dir / filename
        _download_http_file(artifact.url, destination, overwrite=overwrite, dry_run=dry_run)

        if not dry_run and artifact.checksum:
            _verify_checksum(
                destination,
                artifact.checksum,
                artifact.checksum_algorithm,
            )

        if artifact.extract:
            target_dir = (
                dataset_dir / artifact.extract_subdir
                if artifact.extract_subdir
                else dataset_dir
            )
            _extract_archive(destination, target_dir, overwrite=overwrite, dry_run=dry_run)


def _clone_repo(url: str, destination: Path, overwrite: bool, dry_run: bool) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"Repository destination '{destination}' already exists. Use --overwrite to replace it."
            )
        shutil.rmtree(destination)

    if dry_run:
        print(f"[dry-run] Would clone {url} -> {destination}")
        return

    subprocess.run([
        "git",
        "clone",
        "--depth",
        "1",
        url,
        str(destination),
    ], check=True)


def _download_huggingface_dataset(
    repo_id: str, destination: Path, overwrite: bool, dry_run: bool
) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"Dataset directory '{destination}' already exists. Use --overwrite to replace it."
            )
        shutil.rmtree(destination)

    if dry_run:
        print(f"[dry-run] Would download Hugging Face dataset {repo_id} -> {destination}")
        return

    destination.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    attempts = 3
    delay = 1.0
    backoff = 2.0
    last_error: Optional[Exception] = None

    def _iter_repo_files() -> Tuple[List[str], str]:
        info = api.dataset_info(repo_id)
        files = [s.rfilename for s in info.siblings if s.rfilename]
        return sorted(files), info.sha

    def _download_once() -> None:
        files, commit_sha = _iter_repo_files()
        total = len(files)
        if total == 0:
            print(f"No files found for dataset {repo_id}; nothing to download.")
            return
        print(f"Preparing to download {total} file(s) from {repo_id}...")

        progress_was_disabled = are_progress_bars_disabled()
        if not progress_was_disabled:
            disable_progress_bars()
        try:
            with tqdm(
                total=total,
                unit="file",
                desc=f"{repo_id} files",
            ) as progress:
                for repo_file in files:
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=repo_file,
                        repo_type="dataset",
                        revision=commit_sha,
                        local_dir=str(destination),
                        local_dir_use_symlinks=False,
                    )
                    progress.set_postfix_str(Path(repo_file).name)
                    progress.update(1)
        finally:
            if not progress_was_disabled:
                enable_progress_bars()

    for attempt in range(1, attempts + 1):
        try:
            destination.mkdir(parents=True, exist_ok=True)
            print(
                f"Downloading Hugging Face dataset {repo_id} -> {destination} "
                f"(attempt {attempt}/{attempts})"
            )
            _download_once()
            print(f"Download completed: {destination}")
            return
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            wait_time = delay * (backoff ** (attempt - 1))
            print(
                f"Download attempt {attempt} for {repo_id} failed ({exc}). "
                f"Retrying in {wait_time:.1f}s..."
            )
            if destination.exists():
                try:
                    shutil.rmtree(destination)
                    destination.mkdir(parents=True, exist_ok=True)
                except OSError:
                    pass
            time.sleep(wait_time)

    raise RuntimeError(
        f"Failed to download Hugging Face dataset {repo_id} after {attempts} attempts: {last_error}"
    ) from last_error


def _download_google_drive_file(
    file_id: str, destination: Path, overwrite: bool, dry_run: bool
) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"File '{destination}' already exists. Use --overwrite to replace it."
            )
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()

    if dry_run:
        print(
            f"[dry-run] Would download Google Drive file {file_id} -> {destination}"
        )
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    attempts = 3
    delay = 1.0
    backoff = 2.0
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            print(
                f"Downloading Google Drive file {file_id} -> {destination} "
                f"(attempt {attempt}/{attempts})"
            )
            gdown.download(id=file_id, output=str(destination), quiet=False, use_cookies=False)
            print(f"Download completed: {destination}")
            return
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            wait_time = delay * (backoff ** (attempt - 1))
            print(
                f"Download attempt {attempt} for Google Drive file {file_id} failed ({exc}). "
                f"Retrying in {wait_time:.1f}s..."
            )
            if destination.exists():
                try:
                    destination.unlink()
                except OSError:
                    pass
            time.sleep(wait_time)

    raise RuntimeError(
        f"Failed to download Google Drive file {file_id} after {attempts} attempts: {last_error}"
    ) from last_error


def _download_google_drive_folder(
    folder_id: str, destination: Path, overwrite: bool, dry_run: bool
) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"Directory '{destination}' already exists. Use --overwrite to replace it."
            )
        shutil.rmtree(destination)

    if dry_run:
        print(
            f"[dry-run] Would download Google Drive folder {folder_id} -> {destination}"
        )
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(
        id=folder_id,
        output=str(destination),
        quiet=False,
        use_cookies=False,
    )


def download_360x(dataset_dir: Path, overwrite: bool, dry_run: bool) -> None:
    resolution = _360X_RESOLUTION.lower()
    valid_resolutions = {"hr", "lr", "both"}
    if resolution not in valid_resolutions:
        raise ValueError(
            f"Unsupported 360x resolution '{resolution}'. Expected one of: {sorted(valid_resolutions)}"
        )

    requests_to_process = []
    if resolution in {"lr", "both"}:
        requests_to_process.append(
            ("quchenyuan/360x_dataset_LR", dataset_dir / "360x_dataset_LR")
        )
    if resolution in {"hr", "both"}:
        requests_to_process.append(
            ("quchenyuan/360x_dataset_HR", dataset_dir / "360x_dataset_HR")
        )

    print(f"Selected 360x resolution: {resolution}")
    for repo_id, destination in requests_to_process:
        _download_huggingface_dataset(
            repo_id=repo_id,
            destination=destination,
            overwrite=overwrite,
            dry_run=dry_run,
        )


def download_360dvd(dataset_dir: Path, overwrite: bool, dry_run: bool) -> None:
    archive_path = dataset_dir / "360DVD_dataset.zip"
    _download_google_drive_file(
        file_id="1W1eLmaP16GZOeisAR1q-y9JYP9gT1CRs",
        destination=archive_path,
        overwrite=overwrite,
        dry_run=dry_run,
    )

    _extract_archive(
        archive_path,
        dataset_dir,
        overwrite=overwrite,
        dry_run=dry_run,
    )


def download_leader360v(dataset_dir: Path, overwrite: bool, dry_run: bool) -> None:
    _download_huggingface_dataset(
        repo_id="Leader360V/Leader360V",
        destination=dataset_dir / "Leader360V",
        overwrite=overwrite,
        dry_run=dry_run,
    )


def download_360sr(dataset_dir: Path, overwrite: bool, dry_run: bool) -> None:
    destination = dataset_dir / "360SR-Challenge"
    _download_google_drive_folder(
        folder_id="1lDIxTahDXQ5w5x_UZySX2NOes_ZoNztN",
        destination=destination,
        overwrite=overwrite,
        dry_run=dry_run,
    )


def download_avqa(dataset_dir: Path, overwrite: bool, dry_run: bool) -> None:
    repo_destination = dataset_dir / "AVQA"
    _clone_repo(
        url="https://github.com/AlyssaYoung/AVQA.git",
        destination=repo_destination,
        overwrite=overwrite,
        dry_run=dry_run,
    )


DATASETS: Dict[str, DatasetTask] = {
    "360x": DatasetTask(
        name="360x",
        description="Panoramic video dataset with scene descriptions, action labels, and binaural audio.",
        handler=download_360x,
    ),
    "360dvd": DatasetTask(
        name="360dvd",
        description="Dense 360° video understanding dataset for video-language modeling.",
        handler=download_360dvd,
    ),
    "leader360v": DatasetTask(
        name="leader360v",
        description="Large-scale 360° dataset for object tracking and viewpoint-aware understanding.",
        handler=download_leader360v,
    ),
    "360sr": DatasetTask(
        name="360sr",
        description="Static panoramic scene classification dataset for spatial scene context models.",
        handler=download_360sr,
    ),
    "avqa": DatasetTask(
        name="avqa",
        description="Audio-visual question answering dataset repository with preprocessing utilities.",
        handler=download_avqa,
    ),
}


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CaptionQA datasets")
    parser.add_argument(
        "dataset",
        nargs="?",
        help="Dataset identifier (use --list to see options).",
        choices=sorted(DATASETS.keys()),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets"),
        help="Destination directory for downloaded datasets (default: ./datasets)",
    )
    parser.add_argument(
        "--360x-resolution",
        dest="resolution_360x",
        choices=("lr", "hr", "both"),
        default="lr",
        help=(
            "Select which 360x split to download: 'lr' (default), 'hr', or 'both'. "
            "Ignored for other datasets."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files or directories if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the operations that would be performed without downloading anything.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit.",
    )
    return parser.parse_args(argv)


def _list_datasets() -> None:
    print("Available datasets:")
    for key in sorted(DATASETS.keys()):
        task = DATASETS[key]
        print(f"  - {task.name}: {task.description}")


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    if args.list:
        _list_datasets()
        return 0

    if not args.dataset:
        print("Error: dataset identifier is required unless --list is used", file=sys.stderr)
        return 2

    dataset_task = DATASETS[args.dataset]
    dataset_root = args.output.resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)

    destination = dataset_root / dataset_task.name
    try:
        if dataset_task.name == "360x":
            global _360X_RESOLUTION
            _360X_RESOLUTION = args.resolution_360x.lower()
            if args.overwrite and destination.exists():
                shutil.rmtree(destination)
            destination.mkdir(parents=True, exist_ok=True)
        else:
            _ensure_write_path(destination, overwrite=args.overwrite)
            destination.mkdir(parents=True, exist_ok=True)
        dataset_task.handler(destination, args.overwrite, args.dry_run)
    except Exception as exc:
        print(f"Failed to download dataset '{dataset_task.name}': {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
