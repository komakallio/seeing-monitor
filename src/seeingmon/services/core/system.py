"""Reads of the machine that `core` runs on: the load average and the memory in use.

The `health` record carries both. Linux exposes them in `/proc`, and other systems leave them
unknown (`None`), because the standard library has no portable call for them. The functions take
the text of a file, so a test needs no Linux, and `read_system_stats` reads the real files.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

KIB_PER_MIB = 1024.0


@dataclass(frozen=True, slots=True)
class SystemStats:
    """What the machine reports now. A value is `None` when the platform cannot tell."""

    load_1m: float | None
    memory_used_mb: float | None


def parse_loadavg(text: str) -> float | None:
    """The load average of the last minute, from the text of `/proc/loadavg`."""
    fields = text.split()
    if not fields:
        return None
    try:
        value = float(fields[0])
    except ValueError:
        return None
    return value if math.isfinite(value) and value >= 0 else None


def parse_meminfo(text: str) -> float | None:
    """The memory in use in megabytes: the total minus what is available, from `/proc/meminfo`.

    `MemAvailable` counts the cache that the kernel can free, so it measures the memory that
    programs hold. A system without that line (a kernel older than 3.14) reports `None`.
    """
    values: dict[str, float] = {}
    for line in text.splitlines():
        name, separator, rest = line.partition(":")
        if not separator or name not in ("MemTotal", "MemAvailable"):
            continue
        parts = rest.split()
        if not parts:
            continue
        try:
            values[name] = float(parts[0])  # the unit is kB
        except ValueError:
            continue
    if "MemTotal" not in values or "MemAvailable" not in values:
        return None
    used = values["MemTotal"] - values["MemAvailable"]
    return used / KIB_PER_MIB if used >= 0 else None


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="ascii", errors="replace")
    except OSError:
        return None


def read_system_stats(proc_dir: Path | str = "/proc") -> SystemStats:
    """Read the load and the memory from `proc_dir`. A file that is missing gives `None`."""
    base = Path(proc_dir)
    load_text = _read(base / "loadavg")
    memory_text = _read(base / "meminfo")
    return SystemStats(
        load_1m=None if load_text is None else parse_loadavg(load_text),
        memory_used_mb=None if memory_text is None else parse_meminfo(memory_text),
    )
