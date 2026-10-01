"""A fake disk for the retention tests and the wiring tests."""

from __future__ import annotations

from pathlib import Path

from seeingmon.store.retention import DiskUsage


class FakeDisk:
    """A partition of `total` bytes. The files under the root, and `other_used`, fill it."""

    def __init__(self, root: Path, total: int, other_used: int = 0) -> None:
        self.root = root
        self.total = total
        self.other_used = other_used

    def __call__(self, path: Path) -> DiskUsage:
        files = (p for p in self.root.rglob("*") if p.is_file())
        used = self.other_used + sum(p.stat().st_size for p in files)
        return DiskUsage(self.total, max(self.total - used, 0))
