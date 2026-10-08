"""The memory of the recovery ladder: when it last asked for a reboot or a power cycle.

A reboot or a power cycle restarts the process that asked for it, so a timer in memory forgets the
request. Without a camera, the ladder then climbed to the reboot step again after every boot, and
the Pi rebooted about every quarter of an hour. The scheduler keeps the time of the last such
request in a small file, so that `[scheduler.ladder] destructive_interval_s` holds across boots.

The time is a UTC time, so the check needs a clock that the scheduler trusts. A Pi has no
real-time clock, and its clock is wrong until chrony synchronizes it. While the clock is not
synchronized, or when it runs behind the record, the scheduler counts the last request as just
made and asks for no reboot (see `Scheduler._remembered_destructive_age_s`).
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Protocol

_KEY = "last_destructive_utc_ns"


class LadderMemory(Protocol):
    """Where the scheduler keeps the time of its last reboot or power-cycle request."""

    def read(self) -> int | None:
        """The UTC time in nanoseconds of the last request, or `None` when there is none."""
        ...

    def write(self, utc_ns: int) -> None:
        """Keep the time of a request. The call returns after the record is on the disk."""
        ...


class FileLadderMemory:
    """A `LadderMemory` in a small JSON file. A write replaces the file in one step."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def read(self) -> int | None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            value = data[_KEY]
        except (OSError, ValueError, KeyError, TypeError):
            return None  # no file, or a file that is not ours: there is nothing to go by
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    def write(self, utc_ns: int) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(dir=self._path.parent, prefix=".ladder-", suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump({_KEY: int(utc_ns)}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self._path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(name)
            raise
