"""Retention: keep the data directory within its quotas, and stop raw capture when space runs out.

Call `RetentionManager.run_once` from an hourly task. A pass deletes the oldest files first, and
it never deletes a file that changed within the last `protect_recent_s` seconds, so the segment
that a writer holds open is safe.

**Policies.** Each tier has an age limit, and some have a size cap.

- Results (SQLite) stay forever. A record type that declares `retention_days` (the star lists,
  365 days) loses its rows after that time.
- Per-frame metrics (`segments/`) stay for `metrics_days` or up to `metrics_max_gb`, whichever
  limit comes first.
- Previews (`previews/`) stay for `previews_days`.
- Survey frames (`survey/`): every frame stays for `survey_full_days`. Then one frame for each
  night stays for `survey_thinned_days`, the frame nearest to the middle of the night.
- Raw bursts (`bursts/`) have a quota of `bursts_max_gb` for unpinned bursts. A pinned burst (a
  `PINNED` marker file) never counts against the quota and is never deleted.

A file's age is the time since its last change, from the file system. A burst is a directory, and
its age is the newest change inside it.

**Quota.** The data directory may use `quota_fraction` (25% by default) of the data partition.
The usage is every byte under the data directory, including the database and pinned bursts. When
the usage is over the quota, or the free space is below `min_free_gb` plus `resume_margin_gb`,
the pass shrinks the tiers in this order until enough space is free: per-frame metrics, previews,
survey frames, unpinned bursts. It never deletes results or pinned bursts. If that is not enough,
it writes an error event and goes on.

**Events.** Routine expiry by age writes no event. Every early deletion does: one
`retention.early_delete` event for each tier and reason in a pass, with the file count and the
size. Early means before the age limit, because a size cap, the quota, or low space forced the
deletion. The pass writes `retention.over_quota` when it cannot reach the quota, and
`retention.capture_stopped` and `retention.capture_resumed` when raw capture stops and resumes.

**Raw capture.** `capture_allowed()` returns `False` when the free space falls below `min_free_gb`,
and it returns `True` again when the free space rises `resume_margin_gb` above that limit. The
margin stops capture from flapping. The scheduler asks before it starts a burst.
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import stat
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from seeingmon.clock import NS_PER_S, Clock
from seeingmon.records.base import RECORD_TYPES
from seeingmon.store.config import GB, RetentionConfig
from seeingmon.store.db import Store, StoreError
from seeingmon.store.events import EventEmitter
from seeingmon.store.layout import PIN_MARKER, DataLayout, is_temp_file

logger = logging.getLogger(__name__)

DAY_NS = 86_400 * NS_PER_S

# The order in which pressure shrinks the tiers, and the name of each tier.
TIER_ORDER = ("metrics", "previews", "survey", "bursts")
_TIER_LABELS = {
    "metrics": "per-frame metric files",
    "previews": "preview images",
    "survey": "survey frames",
    "bursts": "unpinned raw bursts",
}

EXPIRED = "expired"
TIER_QUOTA = "tier_quota"
TOTAL_QUOTA = "total_quota"
LOW_SPACE = "low_space"
_REASON_TEXT = {
    TIER_QUOTA: "the tier exceeded its size cap",
    TOTAL_QUOTA: "the data directory exceeded its quota",
    LOW_SPACE: "the free space on the partition ran low",
}


@dataclass(frozen=True, slots=True)
class DiskUsage:
    """The size and the free space of the partition that holds the data directory, in bytes."""

    total_bytes: int
    free_bytes: int


# A function that takes the data directory and returns the usage of its partition. Tests pass a
# fake, and `system_disk_usage` asks the operating system.
DiskProbe = Callable[[Path], DiskUsage]


def system_disk_usage(path: Path) -> DiskUsage:
    """Ask the operating system for the usage of the partition that holds `path`."""
    usage = shutil.disk_usage(path)
    return DiskUsage(total_bytes=usage.total, free_bytes=usage.free)


@dataclass(frozen=True, slots=True)
class Deletion:
    """The files that one pass deleted from one tier for one reason.

    `reason` is `expired` (the age limit), `tier_quota` (the size cap of the tier),
    `total_quota` (the quota of the data directory), or `low_space`. `names` lists the burst
    names for the `bursts` tier and stays empty for the other tiers.
    """

    tier: str
    reason: str
    files: int
    size_bytes: int
    names: tuple[str, ...] = ()

    @property
    def early(self) -> bool:
        """Whether the deletion came before the age limit."""
        return self.reason != EXPIRED


@dataclass(frozen=True, slots=True)
class RetentionReport:
    """What one pass did and the state it left.

    `rows_expired` counts the rows that the pass deleted from each table record type with a
    retention. `shortfall_bytes` is how far the data directory still exceeds its quota, or how
    far the free space still falls short of its limit plus margin, after the pass deleted
    everything that retention may delete.
    """

    deletions: tuple[Deletion, ...]
    rows_expired: dict[str, int]
    temp_files_removed: int
    usage_bytes: int
    quota_bytes: int
    free_bytes: int
    shortfall_bytes: int
    capture_allowed: bool
    errors: int


@dataclass(frozen=True, slots=True)
class RetentionStatus:
    """The numbers for a health snapshot. `usage_bytes` comes from the last pass.

    `usage_bytes` and `quota_bytes` are `None` until the first pass runs. `low_space` is true when
    the free space is below `min_free_gb` right now.
    """

    total_bytes: int
    free_bytes: int
    usage_bytes: int | None
    quota_bytes: int | None
    capture_allowed: bool
    low_space: bool


@dataclass(frozen=True, slots=True)
class _Unit:
    """A file, or a whole burst directory, that retention deletes as one piece."""

    path: Path
    size: int
    mtime_ns: int
    pinned: bool = False
    is_dir: bool = False


@dataclass(slots=True)
class _Outcome:
    """What one deletion step did: the summary, the units it removed, and the failures."""

    deletion: Deletion
    done: list[_Unit]
    failed: int


def _order(unit: _Unit) -> tuple[int, str]:
    return (unit.mtime_ns, str(unit.path))


def _scan_files(root: Path) -> list[_Unit]:
    """List the regular files under a tier, except the temporary files of `write_atomic`."""
    units: list[_Unit] = []
    if not root.is_dir():
        return units
    for directory, _, names in os.walk(root):
        for name in names:
            path = Path(directory) / name
            if is_temp_file(path):
                continue
            try:
                info = path.lstat()
            except OSError:
                continue
            if stat.S_ISREG(info.st_mode):
                units.append(_Unit(path, info.st_size, info.st_mtime_ns))
    return units


def _scan_bursts(root: Path) -> list[_Unit]:
    """List the bursts: a directory is one unit, and a loose file directly in the folder too."""
    units: list[_Unit] = []
    if not root.is_dir():
        return units
    for entry in os.scandir(root):
        path = Path(entry.path)
        try:
            if entry.is_dir(follow_symlinks=False):
                size, newest, pinned = 0, 0, False
                for directory, _, names in os.walk(path):
                    for name in names:
                        info = (Path(directory) / name).lstat()
                        size += info.st_size
                        newest = max(newest, info.st_mtime_ns)
                        pinned = pinned or (name == PIN_MARKER and Path(directory) == path)
                if newest == 0:  # an empty directory ages by its own change time
                    newest = entry.stat(follow_symlinks=False).st_mtime_ns
                units.append(_Unit(path, size, newest, pinned, is_dir=True))
            elif entry.is_file(follow_symlinks=False):
                info = entry.stat(follow_symlinks=False)
                units.append(_Unit(path, info.st_size, info.st_mtime_ns))
        except OSError:
            continue
    return units


def _tree_size(root: Path) -> int:
    """The size of every file under `root`, in bytes."""
    total = 0
    if not root.is_dir():
        return total
    for directory, _, names in os.walk(root):
        for name in names:
            try:
                total += (Path(directory) / name).lstat().st_size
            except OSError:
                continue
    return total


def _gb(size_bytes: int) -> str:
    return f"{size_bytes / GB:.2f} GB"


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


class RetentionManager:
    """Keeps the data directory within its quotas. See the module documentation.

    Pass the layout, the retention configuration, a `Clock`, and an `EventEmitter` that writes
    the events. Pass the `Store` to let the pass expire old rows of the record types that declare
    a retention. `disk_usage` is the probe of the partition, and a test replaces it.
    """

    def __init__(
        self,
        layout: DataLayout,
        config: RetentionConfig,
        clock: Clock,
        events: EventEmitter,
        *,
        store: Store | None = None,
        disk_usage: DiskProbe = system_disk_usage,
    ) -> None:
        self._layout = layout
        self._config = config
        self._clock = clock
        self._events = events
        self._store = store
        self._probe = disk_usage
        self._lock = threading.Lock()
        self._capture_allowed = True
        self._over_quota = False
        self._last: RetentionReport | None = None
        self._boundary_ns = round(config.night_boundary_utc_hour * 3600 * NS_PER_S)

    # --- public ---

    def capture_allowed(self) -> bool:
        """Whether raw capture may run now. It reads the free space, so the answer is current.

        The function applies the stop and resume rules, and it writes the event when the state
        changes.
        """
        free = self._probe(self._layout.root).free_bytes
        self._update_capture(free)
        with self._lock:
            return self._capture_allowed

    def status(self) -> RetentionStatus:
        """The free space now, and the usage from the last pass, for a health snapshot."""
        disk = self._probe(self._layout.root)
        self._update_capture(disk.free_bytes)
        with self._lock:
            allowed = self._capture_allowed
            last = self._last
        return RetentionStatus(
            total_bytes=disk.total_bytes,
            free_bytes=disk.free_bytes,
            usage_bytes=None if last is None else last.usage_bytes,
            quota_bytes=None if last is None else last.quota_bytes,
            capture_allowed=allowed,
            low_space=disk.free_bytes < self._min_free_bytes,
        )

    def run_once(self) -> RetentionReport:
        """Run one pass: expire by age, enforce the size caps, enforce the quota, and report.

        The pass handles errors on single files and rows itself: it logs them, counts them in
        `errors`, and goes on.
        """
        now = self._clock.utc_ns()
        errors = 0
        deletions: list[Deletion] = []
        removed_temp = self._layout.clean_temp_files(now, self._config.temp_max_age_s)

        units = {
            "metrics": _scan_files(self._layout.segments_dir),
            "previews": _scan_files(self._layout.previews_dir),
            "survey": _scan_files(self._layout.survey_dir),
            "bursts": _scan_bursts(self._layout.bursts_dir),
        }

        def apply(tier: str, outcome: _Outcome) -> None:
            nonlocal errors
            errors += outcome.failed
            if outcome.deletion.files:
                deletions.append(outcome.deletion)
                gone = {unit.path for unit in outcome.done}
                units[tier] = [unit for unit in units[tier] if unit.path not in gone]

        # 1. Routine expiry by age. It writes no event.
        apply("metrics", self._expire("metrics", units["metrics"], self._config.metrics_days, now))
        apply(
            "previews", self._expire("previews", units["previews"], self._config.previews_days, now)
        )
        apply("survey", self._thin_survey(units["survey"], now))
        rows_expired, row_errors = self._expire_rows(now)
        errors += row_errors

        # 2. The size cap of each tier that has one.
        for tier, cap_gb in (
            ("metrics", self._config.metrics_max_gb),
            ("bursts", self._config.bursts_max_gb),
        ):
            counted = sum(unit.size for unit in units[tier] if not unit.pinned)
            excess = counted - round(cap_gb * GB)
            if excess > 0:
                apply(tier, self._shrink(tier, units[tier], excess, TIER_QUOTA, now))

        # 3. The quota of the data directory, and the free space.
        disk = self._probe(self._layout.root)
        quota = round(self._config.quota_fraction * disk.total_bytes)
        usage = _tree_size(self._layout.root)
        over_quota = usage - quota
        short_on_space = self._min_free_bytes + self._resume_margin_bytes - disk.free_bytes
        need = max(over_quota, short_on_space, 0)
        reason = TOTAL_QUOTA if over_quota >= short_on_space else LOW_SPACE
        for tier in TIER_ORDER:
            if need <= 0:
                break
            outcome = self._shrink(tier, units[tier], need, reason, now)
            apply(tier, outcome)
            need -= outcome.deletion.size_bytes
        self._prune_empty_directories(now)

        # 4. The events, and the state after the deletions.
        for deleted in deletions:
            if deleted.early:
                self._announce(deleted)
        disk = self._probe(self._layout.root)
        usage = _tree_size(self._layout.root)
        quota_excess = max(usage - quota, 0)
        shortfall = max(
            quota_excess, self._min_free_bytes + self._resume_margin_bytes - disk.free_bytes, 0
        )
        self._update_capture(disk.free_bytes)
        self._report_over_quota(quota_excess, usage, quota, units)
        with self._lock:
            report = RetentionReport(
                deletions=tuple(deletions),
                rows_expired=rows_expired,
                temp_files_removed=removed_temp,
                usage_bytes=usage,
                quota_bytes=quota,
                free_bytes=disk.free_bytes,
                shortfall_bytes=shortfall,
                capture_allowed=self._capture_allowed,
                errors=errors,
            )
            self._last = report
        return report

    # --- limits ---

    @property
    def _min_free_bytes(self) -> int:
        return round(self._config.min_free_gb * GB)

    @property
    def _resume_margin_bytes(self) -> int:
        return round(self._config.resume_margin_gb * GB)

    @property
    def _protect_ns(self) -> int:
        return round(self._config.protect_recent_s * NS_PER_S)

    # --- deletion ---

    def _delete(self, unit: _Unit) -> bool:
        try:
            if unit.is_dir:
                shutil.rmtree(unit.path)
            else:
                unit.path.unlink()
        except FileNotFoundError:
            return True  # something else removed it, which is the outcome that we wanted
        except OSError as exc:
            logger.warning("retention could not delete %s: %s", unit.path.name, exc)
            return False
        return True

    def _delete_all(self, tier: str, reason: str, victims: list[_Unit]) -> _Outcome:
        done: list[_Unit] = []
        failed = 0
        for unit in victims:
            if self._delete(unit):
                done.append(unit)
            else:
                failed += 1
        names = tuple(unit.path.name for unit in done) if tier == "bursts" else ()
        deletion = Deletion(tier, reason, len(done), sum(u.size for u in done), names)
        return _Outcome(deletion, done, failed)

    def _deletable(self, unit: _Unit, now: int) -> bool:
        return not unit.pinned and now - unit.mtime_ns >= self._protect_ns

    def _expire(self, tier: str, units: list[_Unit], days: float, now: int) -> _Outcome:
        limit = now - round(days * DAY_NS)
        victims = [
            unit
            for unit in sorted(units, key=_order)
            if unit.mtime_ns < limit and self._deletable(unit, now)
        ]
        return self._delete_all(tier, EXPIRED, victims)

    def _night(self, mtime_ns: int) -> int:
        """The index of the observing night that a time falls in."""
        return (mtime_ns - self._boundary_ns) // DAY_NS

    def _thin_survey(self, units: list[_Unit], now: int) -> _Outcome:
        """Delete survey frames past their age, and thin old nights to one frame each."""
        full_ns = round(self._config.survey_full_days * DAY_NS)
        keep_until = now - full_ns - round(self._config.survey_thinned_days * DAY_NS)
        victims = [unit for unit in units if unit.mtime_ns < keep_until]
        nights: dict[int, list[_Unit]] = {}
        for unit in units:
            if unit.mtime_ns >= keep_until:
                nights.setdefault(self._night(unit.mtime_ns), []).append(unit)
        for night, members in nights.items():
            night_start = self._boundary_ns + night * DAY_NS
            if night_start + DAY_NS > now - full_ns:
                continue  # part of this night is still inside the full-rate window
            middle = night_start + DAY_NS // 2
            keeper = min(members, key=lambda unit: (abs(unit.mtime_ns - middle), _order(unit)))
            victims += [unit for unit in members if unit is not keeper]
        victims = [unit for unit in sorted(victims, key=_order) if self._deletable(unit, now)]
        return self._delete_all("survey", EXPIRED, victims)

    def _shrink(
        self, tier: str, units: list[_Unit], need_bytes: int, reason: str, now: int
    ) -> _Outcome:
        """Delete the oldest deletable units of a tier until `need_bytes` are free."""
        victims: list[_Unit] = []
        freed = 0
        for unit in sorted(units, key=_order):
            if freed >= need_bytes:
                break
            if self._deletable(unit, now):
                victims.append(unit)
                freed += unit.size
        return self._delete_all(tier, reason, victims)

    def _prune_empty_directories(self, now: int) -> None:
        """Remove the date directories that deletions emptied. A new directory stays."""
        for root in (
            self._layout.segments_dir,
            self._layout.previews_dir,
            self._layout.survey_dir,
        ):
            if not root.is_dir():
                continue
            for directory, _, _ in os.walk(root, topdown=False):
                path = Path(directory)
                if path == root:
                    continue
                try:
                    if now - path.stat().st_mtime_ns >= self._protect_ns:
                        path.rmdir()  # fails, and is skipped, when the directory has files
                except OSError:
                    continue

    def _expire_rows(self, now: int) -> tuple[dict[str, int], int]:
        """Delete the old rows of the table record types that declare a retention."""
        rows: dict[str, int] = {}
        errors = 0
        if self._store is None:
            return rows, errors
        for cls in RECORD_TYPES.values():
            if cls.storage != "table" or cls.retention_days is None:
                continue
            try:
                count = self._store.expire(cls.record_type, now - cls.retention_days * DAY_NS)
            except (StoreError, sqlite3.Error) as exc:
                logger.warning("retention could not expire %s rows: %s", cls.record_type, exc)
                errors += 1
                continue
            if count:
                rows[cls.record_type] = count
        return rows, errors

    # --- state and events ---

    def _update_capture(self, free_bytes: int) -> None:
        stop = resume = False
        with self._lock:
            if self._capture_allowed and free_bytes < self._min_free_bytes:
                self._capture_allowed = False
                stop = True
            elif (
                not self._capture_allowed
                and free_bytes >= self._min_free_bytes + self._resume_margin_bytes
            ):
                self._capture_allowed = True
                resume = True
        detail = {"free_bytes": free_bytes, "min_free_bytes": self._min_free_bytes}
        if stop:
            self._events.emit(
                "warning",
                "retention.capture_stopped",
                f"The free space fell to {_gb(free_bytes)}, below the limit of "
                f"{_gb(self._min_free_bytes)}, so raw capture stops.",
                detail,
            )
        if resume:
            self._events.emit(
                "info",
                "retention.capture_resumed",
                f"The free space rose to {_gb(free_bytes)}, so raw capture resumes.",
                detail,
            )

    def _report_over_quota(
        self, excess: int, usage: int, quota: int, units: dict[str, list[_Unit]]
    ) -> None:
        """Write an error event when the pass cannot get under the quota, once for each episode."""
        with self._lock:
            entering = excess > 0 and not self._over_quota
            self._over_quota = excess > 0
        if not entering:
            return
        pinned = sum(unit.size for unit in units["bursts"] if unit.pinned)
        self._events.emit(
            "error",
            "retention.over_quota",
            f"The data directory uses {_gb(usage)} against a quota of {_gb(quota)}, and retention "
            "has nothing more to delete. Free space on the partition or raise the quota.",
            {
                "usage_bytes": usage,
                "quota_bytes": quota,
                "excess_bytes": excess,
                "pinned_burst_bytes": pinned,
            },
        )

    def _announce(self, deleted: Deletion) -> None:
        label = _TIER_LABELS[deleted.tier]
        detail: dict[str, object] = {
            "tier": deleted.tier,
            "reason": deleted.reason,
            "files": deleted.files,
            "size_bytes": deleted.size_bytes,
        }
        if deleted.names:
            detail["bursts"] = list(deleted.names)
        self._events.emit(
            "warning",
            "retention.early_delete",
            f"Deleted {_plural(deleted.files, 'file')} of {label} ({_gb(deleted.size_bytes)}) "
            f"before the age limit, because {_REASON_TEXT[deleted.reason]}.",
            detail,
        )
