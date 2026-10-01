"""Read-only access to the preview images and the FITS frames, for the REST API.

`core` writes the files through `DataLayout` (see `seeingmon.store.layout`):

    previews/YYYY/MM/DD/<kind>-<stamp>.jpg      the preview of an image
    survey/YYYY/MM/DD/<stamp>.fits              the FITS frame, for some images

The stamp is the UTC time of the image as `YYYYMMDDTHHMMSS.mmmZ`. The ID of an image is the name
of its preview without the extension, such as `preview-20261001T201500.123Z`. The FITS frame of an
image has the stamp of its ID, so the ID names both files.

**No path from a client.** The ID passes a strict pattern, and the code builds every path from the
parts that the pattern captured. A `/`, a `\\`, a `.`, and a `..` cannot pass it. The code then
resolves the path and refuses any file that does not sit exactly where the layout puts it, so a
symbolic link cannot lead out of the folder either. Nothing here writes: the module opens files for
reading only (the API streams them), and it lists folders.

**Limits.** The listing reads the newest `scan_days` day folders, and a page holds at most
`max_list_limit` images. A file that exceeds `max_preview_bytes` or `max_fits_bytes` counts as
absent in the listing, and a request for it gets an error.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from seeingmon.clock import NS_PER_S
from seeingmon.services.web.config import ImageSettings
from seeingmon.services.web.data import InvalidQueryError
from seeingmon.store.layout import DataLayout

IMAGE_ID = re.compile(
    r"(?P<kind>[a-z][a-z0-9_]{0,23})-"
    r"(?P<stamp>(?P<year>[0-9]{4})(?P<month>[0-9]{2})(?P<day>[0-9]{2})T(?P<time>[0-9]{6})"
    r"(?:\.(?P<millis>[0-9]{3}))?Z)"
)
_YEAR = re.compile(r"[0-9]{4}")
_TWO_DIGITS = re.compile(r"[0-9]{2}")
PREVIEW_SUFFIX = ".jpg"
FITS_SUFFIX = ".fits"


class ImageTooLargeError(Exception):
    """The file exceeds the size limit of the server."""


@dataclass(frozen=True, slots=True)
class ImageKey:
    """The parts of an image ID. Build it with `parse_image_id`."""

    kind: str
    stamp: str
    year: str
    month: str
    day: str
    t_utc_ns: int

    @property
    def id(self) -> str:
        return f"{self.kind}-{self.stamp}"

    @property
    def sort_key(self) -> tuple[str, str]:
        """Newer images sort higher. The stamp has a fixed width, so text order is time order."""
        return (self.stamp, self.kind)

    @property
    def date(self) -> str:
        return f"{self.year}{self.month}{self.day}"

    @classmethod
    def from_text(cls, text: str) -> ImageKey:
        """Parse an ID. Raises `ValueError` for text that is not an ID or names a missing time."""
        match = IMAGE_ID.fullmatch(text)
        if match is None:
            raise ValueError("not an image ID")
        time = match["time"]
        moment = datetime(
            int(match["year"]),
            int(match["month"]),
            int(match["day"]),
            int(time[0:2]),
            int(time[2:4]),
            int(time[4:6]),
            tzinfo=UTC,
        )
        millis = int(match["millis"] or 0)
        t_utc_ns = int(moment.timestamp()) * NS_PER_S + millis * 1_000_000
        return cls(
            match["kind"], match["stamp"], match["year"], match["month"], match["day"], t_utc_ns
        )


@dataclass(frozen=True, slots=True)
class ImageInfo:
    """One image: its ID, its time, the size of its preview, and the size of its FITS frame."""

    key: ImageKey
    size_bytes: int
    fits_bytes: int | None

    @property
    def id(self) -> str:
        return self.key.id

    @property
    def has_fits(self) -> bool:
        return self.fits_bytes is not None


def parse_image_id(text: str) -> ImageKey:
    """Parse an image ID. Raises `InvalidQueryError` for anything that is not one."""
    try:
        return ImageKey.from_text(text)
    except ValueError:
        raise InvalidQueryError("the image ID is not valid") from None


class ImageStore:
    """The images in the data directory, read-only."""

    def __init__(self, layout: DataLayout, settings: ImageSettings) -> None:
        self._layout = layout
        self._settings = settings

    @property
    def settings(self) -> ImageSettings:
        return self._settings

    # --- Paths -----------------------------------------------------------------------------

    def _file(self, base: Path, key: ImageKey, name: str) -> Path | None:
        """The file `name` in the day folder of `key` under `base`, or `None` when it is absent.

        The path must resolve to exactly the place that the layout names, or the file does not
        count: a symbolic link cannot lead somewhere else.
        """
        directory = base / key.year / key.month / key.day
        expected = directory.resolve() / name
        candidate = directory / name
        try:
            if candidate.is_symlink() or not candidate.is_file():
                return None
            if candidate.resolve() != expected:
                return None
        except OSError:
            return None
        return candidate

    def preview_path(self, key: ImageKey) -> Path | None:
        """The preview file of an image, or `None` when it does not exist."""
        return self._file(self._layout.previews_dir, key, f"{key.id}{PREVIEW_SUFFIX}")

    def fits_path(self, key: ImageKey) -> Path | None:
        """The FITS file of an image, or `None` when it does not exist."""
        return self._file(self._layout.survey_dir, key, f"{key.stamp}{FITS_SUFFIX}")

    def _info(self, key: ImageKey) -> ImageInfo | None:
        preview = self.preview_path(key)
        if preview is None:
            return None
        try:
            size = preview.stat().st_size
        except OSError:
            return None
        if size > self._settings.max_preview_bytes:
            return None
        fits = self.fits_path(key)
        fits_size: int | None = None
        if fits is not None:
            try:
                found = fits.stat().st_size
            except OSError:
                found = None
            if found is not None and found <= self._settings.max_fits_bytes:
                fits_size = found
        return ImageInfo(key, size, fits_size)

    # --- Reads -----------------------------------------------------------------------------

    def get(self, image_id: str) -> ImageInfo | None:
        """The image with this ID, or `None` when no such image exists."""
        return self._info(parse_image_id(image_id))

    def latest(self) -> ImageInfo | None:
        """The newest image, or `None` when there is none."""
        images, _ = self.recent(1)
        return images[0] if images else None

    def recent(self, limit: int, before: ImageKey | None = None) -> tuple[list[ImageInfo], bool]:
        """Up to `limit` images, newest first. Pass `before` to continue after an image.

        The second item says whether more images follow.
        """
        found: list[ImageInfo] = []
        for date, directory in self._day_folders():
            if before is not None and date > before.date:
                continue
            for key in self._keys_in(directory, date):
                if before is not None and key.sort_key >= before.sort_key:
                    continue
                info = self._info(key)
                if info is None:
                    continue
                if len(found) == limit:
                    return found, True
                found.append(info)
        return found, False

    def _day_folders(self) -> Iterator[tuple[str, Path]]:
        """The day folders of the previews, newest first, at most `scan_days` of them."""
        visited = 0
        for year in _names(self._layout.previews_dir, _YEAR):
            for month in _names(self._layout.previews_dir / year, _TWO_DIGITS):
                for day in _names(self._layout.previews_dir / year / month, _TWO_DIGITS):
                    if visited >= self._settings.scan_days:
                        return
                    visited += 1
                    yield f"{year}{month}{day}", self._layout.previews_dir / year / month / day

    @staticmethod
    def _keys_in(directory: Path, date: str) -> list[ImageKey]:
        keys: list[ImageKey] = []
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    name = entry.name
                    if not name.endswith(PREVIEW_SUFFIX):
                        continue
                    try:
                        key = ImageKey.from_text(name[: -len(PREVIEW_SUFFIX)])
                    except ValueError:
                        continue
                    if key.date == date:
                        keys.append(key)
        except OSError:
            return []
        keys.sort(key=lambda key: key.sort_key, reverse=True)
        return keys


def _names(directory: Path, pattern: re.Pattern[str]) -> list[str]:
    """The names of the sub-folders of `directory` that match `pattern`, newest first."""
    try:
        with os.scandir(directory) as entries:
            names = [
                entry.name
                for entry in entries
                if pattern.fullmatch(entry.name) and entry.is_dir(follow_symlinks=False)
            ]
    except OSError:
        return []
    return sorted(names, reverse=True)
