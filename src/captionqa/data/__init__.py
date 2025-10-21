"""CaptionQA dataset utilities.

Public API:
- DATASETS: mapping of dataset name -> DatasetTask
- main: CLI entry for dataset downloader (python -m captionqa.data.download)
"""

from .download import DATASETS, main  # noqa: F401
