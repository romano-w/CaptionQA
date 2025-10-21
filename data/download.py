"""Compatibility wrapper for the dataset downloader.

This module now forwards to the installable package module
`captionqa.data.download` so that notebooks and scripts can import the downloader
reliably regardless of the current working directory.

Prefer using:

    python -m captionqa.data.download ...

over the legacy path:

    python -m data.download ...
"""

from __future__ import annotations

import sys
import warnings

# Re-export public API for backward compatibility
from captionqa.data.download import DATASETS, main  # noqa: F401


def _warn_deprecated() -> None:
    warnings.warn(
        "'python -m data.download' is deprecated; use 'python -m captionqa.data.download' instead.",
        DeprecationWarning,
        stacklevel=2,
    )


if __name__ == "__main__":
    _warn_deprecated()
    raise SystemExit(main(sys.argv[1:]))
