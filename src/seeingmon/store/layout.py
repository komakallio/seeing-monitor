"""The data directory: where each tier of files lives, and how to write a file safely.

Every lane that writes a file writes it through `DataLayout`, so that retention finds the file
and a crash never leaves half a file under its real name.

```
<data>/
  db/results.sqlite          the SQLite store (WAL and shared-memory files sit beside it)
  segments/YYYY/MM/DD/       per-frame metric segments (`seeingmon.store.segments`)
  survey/YYYY/MM/DD/         survey frames, for example FITS with Rice compression
  previews/YYYY/MM/DD/       JPEG previews
  bursts/<burst>/            one directory for each raw burst: the SER file, its JSON sidecar,
                             and an optional PINNED marker
```

**Atomic writes.** `write_atomic` writes to a hidden temporary file in the same directory, forces
it to disk, and renames it over the target. A reader sees the old file or the whole new file.
A crash leaves a hidden `.<name>.<pid>.<n>.tmp` file, which `clean_temp_files` removes later.

**Bursts.** A burst is a directory, so the SER file and its sidecar live and die together. Name
it with `new_burst`. `pin_burst` adds a `PINNED` marker file, and retention never deletes a
pinned burst or counts it against the burst quota.

**References.** A record that points at a file (`image_ref` of a survey frame) holds a path
relative to the data directory, with forward slashes. `relative` builds it, and `resolve`
turns it back into a path and refuses one that leaves the data directory.
"""

from __future__ import annotations

import contextlib
import itertools
import os
import re
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, BinaryIO

from pydantic import ConfigDict

from seeingmon.clock import NS_PER_S, utc_ns_to_datetime
from seeingmon.config import SectionModel

if TYPE_CHECKING:
    from seeingmon.config import Config

DB_DIRNAME = "db"
DB_FILENAME = "results.sqlite"
SEGMENTS_DIRNAME = "segments"
SURVEY_DIRNAME = "survey"
PREVIEWS_DIRNAME = "previews"
BURSTS_DIRNAME = "bursts"
PIN_MARKER = "PINNED"
TEMP_SUFFIX = ".tmp"

# Anything that `write_atomic` accepts as the contents of a file: bytes, or a callable that
# writes to the open file, which suits a large frame.
FileData = bytes | bytearray | memoryview | Callable[[BinaryIO], object]

_temp_counter = itertools.count()
_LABEL_UNSAFE = re.compile(r"[^a-z0-9_-]+")


class PathsConfig(SectionModel):
    """The `[paths]` section. It reads `data_dir` and ignores keys that other lanes add."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    data_dir: str


def fsync_directory(directory: Path) -> None:
    """Make a rename in `directory` durable. Windows cannot open a directory, so it skips this.

    The step is best effort. A file system that refuses to sync a directory has made the rename
    as durable as it can, so the function ignores that error and does not fail a write that
    already succeeded.
    """
    if os.name == "nt":
        return
    with contextlib.suppress(OSError):
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def is_temp_file(path: Path) -> bool:
    """Whether a file is a leftover of `write_atomic`: hidden, and with the `.tmp` suffix."""
    return path.name.startswith(".") and path.name.endswith(TEMP_SUFFIX)


def write_atomic(path: Path | str, data: FileData) -> Path:
    """Write a file so that a reader never sees a partial one. Return the path.

    The function creates the missing directories, writes to a hidden temporary file beside the
    target, forces it to disk, and renames it over the target. `data` is bytes, or a callable
    that takes the open binary file and writes to it. If the write fails, the function removes
    the temporary file, leaves the target as it was, and raises the error.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    while True:
        temp = target.with_name(f".{target.name}.{os.getpid()}.{next(_temp_counter)}{TEMP_SUFFIX}")
        try:
            descriptor = os.open(temp, flags, 0o644)
        except FileExistsError:
            continue  # a leftover from an earlier process with the same ID
        break
    try:
        with os.fdopen(descriptor, "wb") as handle:
            if callable(data):
                data(handle)
            else:
                handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            temp.unlink()
        raise
    fsync_directory(target.parent)
    return target


def _stamp(t_utc_ns: int, *, milliseconds: bool = False) -> str:
    text = utc_ns_to_datetime(t_utc_ns).strftime("%Y%m%dT%H%M%S")
    if milliseconds:
        return f"{text}.{(t_utc_ns // 1_000_000) % 1000:03d}Z"
    return f"{text}Z"


def _dated(base: Path, t_utc_ns: int) -> Path:
    moment = utc_ns_to_datetime(t_utc_ns)
    return base / f"{moment:%Y}" / f"{moment:%m}" / f"{moment:%d}"


def burst_name(t_utc_ns: int, label: str | None = None) -> str:
    """The name of a burst directory: the UTC start time and an optional label.

    The label keeps lowercase letters, digits, `_`, and `-`. The function replaces anything else
    with `-` and shortens the label to 40 characters.
    """
    name = _stamp(t_utc_ns)
    cleaned = "" if label is None else _LABEL_UNSAFE.sub("-", label.lower()).strip("-")[:40]
    return f"{name}-{cleaned}" if cleaned else name


class DataLayout:
    """The folders of the data directory, and the safe ways to put files in them."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)

    @classmethod
    def from_config(cls, config: Config) -> DataLayout:
        """Build the layout from `data_dir` in the `[paths]` section of the configuration."""
        return cls(config.section("paths", PathsConfig).data_dir)

    def __repr__(self) -> str:
        return f"DataLayout({str(self._root)!r})"

    @property
    def root(self) -> Path:
        return self._root

    @property
    def db_dir(self) -> Path:
        return self._root / DB_DIRNAME

    @property
    def db_path(self) -> Path:
        """The SQLite store. SQLite adds `-wal` and `-shm` files beside it."""
        return self.db_dir / DB_FILENAME

    @property
    def segments_dir(self) -> Path:
        return self._root / SEGMENTS_DIRNAME

    @property
    def survey_dir(self) -> Path:
        return self._root / SURVEY_DIRNAME

    @property
    def previews_dir(self) -> Path:
        return self._root / PREVIEWS_DIRNAME

    @property
    def bursts_dir(self) -> Path:
        return self._root / BURSTS_DIRNAME

    def create(self) -> None:
        """Create the data directory and the folder of every tier. Safe to repeat."""
        for directory in (
            self.db_dir,
            self.segments_dir,
            self.survey_dir,
            self.previews_dir,
            self.bursts_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    # --- names ---

    def survey_path(self, t_utc_ns: int, *, suffix: str = ".fits") -> Path:
        """The path of a survey frame: `survey/YYYY/MM/DD/<time>.fits`. No file is created."""
        return _dated(self.survey_dir, t_utc_ns) / f"{_stamp(t_utc_ns, milliseconds=True)}{suffix}"

    def preview_path(self, t_utc_ns: int, *, kind: str = "preview", suffix: str = ".jpg") -> Path:
        """The path of a preview: `previews/YYYY/MM/DD/<kind>-<time>.jpg`. No file is created."""
        name = f"{kind}-{_stamp(t_utc_ns, milliseconds=True)}{suffix}"
        return _dated(self.previews_dir, t_utc_ns) / name

    def burst_path(self, t_utc_ns: int, label: str | None = None) -> Path:
        """The directory of a burst, without creating it. See `burst_name`."""
        return self.bursts_dir / burst_name(t_utc_ns, label)

    def new_burst(self, t_utc_ns: int, label: str | None = None) -> Path:
        """Create the directory of a new burst and return it.

        Put the SER file and its JSON sidecar in it. If a burst of that name exists, the
        function adds `-2`, `-3`, and so on.
        """
        base = self.burst_path(t_utc_ns, label)
        base.parent.mkdir(parents=True, exist_ok=True)
        for number in itertools.count(1):
            path = base if number == 1 else base.with_name(f"{base.name}-{number}")
            try:
                path.mkdir()
            except FileExistsError:
                continue
            return path
        raise AssertionError("unreachable")  # itertools.count never ends

    # --- references ---

    def relative(self, path: Path | str) -> str:
        """The reference of a file: its path under the data directory, with forward slashes.

        Raises `ValueError` for a path outside the data directory.
        """
        candidate = Path(path)
        base = self._root.resolve()
        resolved = (candidate if candidate.is_absolute() else base / candidate).resolve()
        try:
            return resolved.relative_to(base).as_posix()
        except ValueError:
            raise ValueError("the path is outside the data directory") from None

    def resolve(self, ref: str) -> Path:
        """The path that a reference names. Raises `ValueError` if it leaves the data directory.

        A reference is relative, uses forward slashes, and has no `..` part. The function
        rejects an absolute path, a drive, and a path that a symbolic link carries outside.
        """
        parts = PurePosixPath(ref).parts
        if (
            not parts
            or PurePosixPath(ref).is_absolute()
            or ".." in parts
            or "\\" in ref
            or re.match(r"^[A-Za-z]:", ref)
        ):
            raise ValueError(f"{ref!r} is not a reference inside the data directory")
        base = self._root.resolve()
        resolved = base.joinpath(*parts).resolve()
        if not resolved.is_relative_to(base):
            raise ValueError(f"{ref!r} leaves the data directory")
        return resolved

    # --- writing ---

    def write_atomic(self, path: Path | str, data: FileData) -> Path:
        """Write a file under the data directory atomically. See the module-level function.

        Raises `ValueError` for a path outside the data directory.
        """
        self.relative(path)
        return write_atomic(path, data)

    def clean_temp_files(self, now_utc_ns: int, max_age_s: float) -> int:
        """Delete the temporary files of crashed writes that are older than `max_age_s`.

        `now_utc_ns` is the current time from your `Clock`, and file times are compared with it.
        Returns the number of files deleted.
        """
        limit = now_utc_ns - round(max_age_s * NS_PER_S)
        deleted = 0
        if not self._root.is_dir():
            return deleted
        for directory, _, names in os.walk(self._root):
            for name in names:
                path = Path(directory) / name
                if not is_temp_file(path):
                    continue
                try:
                    if path.stat().st_mtime_ns <= limit:
                        path.unlink()
                        deleted += 1
                except OSError:
                    continue
        return deleted

    # --- pins ---

    def _burst_directory(self, burst: Path | str) -> Path:
        candidate = Path(burst)
        path = candidate if candidate.is_absolute() else self.bursts_dir / candidate
        resolved = path.resolve()
        if resolved.parent != self.bursts_dir.resolve():
            raise ValueError("a burst is a directory directly under the bursts folder")
        if not resolved.is_dir():
            raise FileNotFoundError(f"there is no burst {candidate.name!r}")
        return resolved

    def pin_burst(self, burst: Path | str) -> Path:
        """Mark a burst so that retention keeps it. Return the marker file.

        `burst` is the directory of the burst or its name. Raises `FileNotFoundError` when the
        burst does not exist, and `ValueError` when the path is not a burst directory.
        """
        marker = self._burst_directory(burst) / PIN_MARKER
        write_atomic(marker, b"")
        return marker

    def unpin_burst(self, burst: Path | str) -> bool:
        """Remove the marker of a burst. Return whether the burst was pinned."""
        marker = self._burst_directory(burst) / PIN_MARKER
        try:
            marker.unlink()
        except FileNotFoundError:
            return False
        return True

    def is_pinned(self, burst: Path | str) -> bool:
        """Whether a burst carries the `PINNED` marker."""
        return (self._burst_directory(burst) / PIN_MARKER).is_file()
