"""Lightweight progress bar that degrades gracefully when stdout is not a TTY."""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass


@dataclass
class ProgressDisplay:
    """Terminal-friendly progress helper used across CLI baselines."""

    total: int

    def __post_init__(self) -> None:
        self.total = max(int(self.total), 1)
        self._is_tty = sys.stdout.isatty()
        term_width = shutil.get_terminal_size(fallback=(100, 20)).columns
        self._term_width = max(term_width, 60)
        digits = len(str(self.total))
        self._count_width = digits
        reserved = digits * 2 + 20  # brackets, slash, percent, etc.
        self._bar_width = max(10, min(30, self._term_width - reserved))

    def _trim_status(self, status: str) -> str:
        max_status = self._term_width - (self._bar_width + self._count_width * 2 + 25)
        if max_status <= 0 or len(status) <= max_status:
            return status
        if max_status <= 3:
            return status[:max_status]
        return status[: max_status - 3] + "..."

    def _render(self, completed: int, status: str, final: bool = False) -> None:
        status = self._trim_status(status)
        if not self._is_tty:
            prefix = "done" if final else "prog"
            print(f"[{completed}/{self.total}] {prefix}: {status}")
            return

        ratio = min(max(completed / self.total, 0.0), 1.0)
        filled = int(round(ratio * self._bar_width))
        bar = "#" * filled + "-" * (self._bar_width - filled)
        line = (
            f"[{completed:>{self._count_width}}/{self.total}] "
            f"|{bar}| {ratio * 100:5.1f}% {status}"
        )
        print("\r" + line.ljust(self._term_width), end="", flush=True)
        if final:
            print()

    def show_status(self, completed: int, status: str) -> None:
        self._render(completed, status, final=False)

    def finish_step(self, completed: int, status: str) -> None:
        final = completed >= self.total
        self._render(completed, status, final=final)
        if not self._is_tty and final:
            print()
