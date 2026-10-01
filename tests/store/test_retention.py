"""Retention: age limits, size caps, the quota, the shrink order, and the capture gate."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import pytest

from seeingmon.clock import VirtualClock
from seeingmon.records.system import EventRecord
from seeingmon.store.config import GB, RetentionConfig
from seeingmon.store.db import Store
from seeingmon.store.events import EventEmitter
from seeingmon.store.layout import PIN_MARKER, DataLayout
from seeingmon.store.retention import DAY_NS, RetentionManager, system_disk_usage
from tests.store.builders import NS_PER_S, STATION, T0, make_health, make_star_list
from tests.store.disk import FakeDisk

NOW = T0 + 200 * DAY_NS  # 2026-07-20T00:00:00Z, the virtual time of the tests

Manager = Callable[..., RetentionManager]


def gb(size_bytes: int) -> float:
    """Express a small byte count in the gigabytes that the configuration uses."""
    return size_bytes / GB


def days(count: float) -> float:
    return count * 86_400


def make_file(path: Path, size: int, age_s: float, *, now: int = NOW) -> Path:
    """Create a file of `size` bytes that last changed `age_s` seconds before `now`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    stamp = now - round(age_s * NS_PER_S)
    os.utime(path, ns=(stamp, stamp))
    return path


def make_burst(
    layout: DataLayout, name: str, size: int, age_s: float, *, pinned: bool = False
) -> Path:
    """Create a burst directory with a SER file and a sidecar that add up to `size` bytes."""
    burst = layout.bursts_dir / name
    make_file(burst / "burst.ser", size - 10, age_s)
    make_file(burst / "burst.json", 10, age_s)
    if pinned:
        make_file(burst / PIN_MARKER, 0, age_s)
    return burst


def names(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def kinds(events: list[EventRecord]) -> list[str]:
    return [event.kind for event in events]


@pytest.fixture
def clock() -> VirtualClock:
    return VirtualClock(NOW)


@pytest.fixture
def layout(tmp_path: Path) -> DataLayout:
    data = DataLayout(tmp_path / "data")
    data.create()
    return data


@pytest.fixture
def events() -> list[EventRecord]:
    return []


@pytest.fixture
def emitter(clock: VirtualClock, events: list[EventRecord]) -> EventEmitter:
    return EventEmitter(
        events.append,
        clock,
        station_id=STATION,
        profile_id="profile-1",
        provenance={"software": "test"},
    )


@pytest.fixture
def make_manager(layout: DataLayout, clock: VirtualClock, emitter: EventEmitter) -> Manager:
    """Build a manager. By default the partition is huge and has no free-space limit, so a test
    sees only the rule that it sets up. Pass `disk`, `store`, or configuration overrides."""

    def build(
        disk: FakeDisk | None = None, store: Store | None = None, **overrides: Any
    ) -> RetentionManager:
        probe = disk if disk is not None else FakeDisk(layout.root, 10**15)
        settings = {"min_free_gb": 0.0, "resume_margin_gb": 0.0, **overrides}
        return RetentionManager(
            layout, RetentionConfig(**settings), clock, emitter, store=store, disk_usage=probe
        )

    return build


class TestAgeLimits:
    def test_metrics_and_previews_expire_after_seven_days_without_an_event(
        self, layout: DataLayout, make_manager: Manager, events: list[EventRecord]
    ) -> None:
        old_segment = make_file(layout.segments_dir / "2026/04/01/old.seg", 100, days(7.5))
        new_segment = make_file(layout.segments_dir / "2026/04/09/new.seg", 100, days(6.5))
        old_preview = make_file(layout.previews_dir / "2026/04/01/a.jpg", 50, days(8))
        new_preview = make_file(layout.previews_dir / "2026/04/09/b.jpg", 50, days(1))
        report = make_manager().run_once()
        assert not old_segment.exists()
        assert new_segment.exists()
        assert not old_preview.exists()
        assert new_preview.exists()
        assert sorted((d.tier, d.reason, d.files, d.size_bytes) for d in report.deletions) == [
            ("metrics", "expired", 1, 100),
            ("previews", "expired", 1, 50),
        ]
        assert not any(d.early for d in report.deletions)
        assert events == []  # routine expiry is not an early deletion

    def test_the_limits_follow_the_configuration(
        self, layout: DataLayout, make_manager: Manager
    ) -> None:
        segment = make_file(layout.segments_dir / "a.seg", 10, days(2.5))
        make_manager(metrics_days=2).run_once()
        assert not segment.exists()

    def test_a_file_that_last_changed_in_the_future_never_expires(
        self, layout: DataLayout, make_manager: Manager
    ) -> None:
        future = make_file(layout.segments_dir / "a.seg", 10, -days(400))  # after a clock jump
        make_manager().run_once()
        assert future.exists()

    def test_a_recent_file_survives_even_with_a_tiny_age_limit(
        self, layout: DataLayout, make_manager: Manager
    ) -> None:
        recent = make_file(layout.segments_dir / "open.seg.part", 10, 60)  # inside the 15 min guard
        make_manager(metrics_days=0.0001).run_once()  # an age limit of about 9 seconds
        assert recent.exists()

    def test_star_list_rows_expire_after_a_year_and_results_stay(
        self, layout: DataLayout, make_manager: Manager
    ) -> None:
        with Store.open(layout.db_path) as store:
            store.write_many(
                [
                    make_star_list(NOW - 400 * DAY_NS),
                    make_star_list(NOW - 300 * DAY_NS),
                    make_star_list(NOW - 10 * DAY_NS),
                ]
            )
            store.write(make_health(NOW - 400 * DAY_NS))  # results are kept forever
            report = make_manager(store=store).run_once()
            assert report.rows_expired == {"star_list": 1}
            assert store.count("star_list") == 2
            assert store.count("health") == 1

    def test_rows_stay_when_the_manager_has_no_store(
        self, layout: DataLayout, make_manager: Manager
    ) -> None:
        assert make_manager().run_once().rows_expired == {}

    def test_stale_temporary_files_are_removed(
        self, layout: DataLayout, make_manager: Manager
    ) -> None:
        stale = make_file(layout.previews_dir / ".live.jpg.1.0.tmp", 10, 7200)
        fresh = make_file(layout.previews_dir / ".live.jpg.1.1.tmp", 10, 60)
        report = make_manager().run_once()
        assert report.temp_files_removed == 1
        assert not stale.exists()
        assert fresh.exists()

    def test_a_directory_that_a_deletion_emptied_is_pruned(
        self, layout: DataLayout, emitter: EventEmitter
    ) -> None:
        # Directory times come from the real clock, so run this pass 30 days ahead of it.
        ahead = time.time_ns() + 30 * DAY_NS
        make_file(layout.segments_dir / "2026/04/01/old.seg", 10, days(20), now=ahead)
        make_file(layout.segments_dir / "2026/04/02/keep.seg", 10, days(1), now=ahead)
        manager = RetentionManager(
            layout,
            RetentionConfig(),
            VirtualClock(ahead),
            emitter,
            disk_usage=FakeDisk(layout.root, 10**15),
        )
        manager.run_once()
        assert not (layout.segments_dir / "2026" / "04" / "01").exists()
        assert (layout.segments_dir / "2026" / "04" / "02").is_dir()
        assert layout.segments_dir.is_dir()  # the root of the tier stays

    def test_a_directory_made_a_moment_ago_stays_for_a_writer_that_is_about_to_use_it(
        self, layout: DataLayout, emitter: EventEmitter
    ) -> None:
        fresh = layout.segments_dir / "2026" / "10" / "01"
        fresh.mkdir(parents=True)
        manager = RetentionManager(
            layout,
            RetentionConfig(),
            VirtualClock(time.time_ns()),
            emitter,
            disk_usage=FakeDisk(layout.root, 10**15),
        )
        manager.run_once()
        assert fresh.is_dir()


class TestSurveyFrames:
    """All frames for 7 days, then one a night for 60 days, then none.

    The night boundary is 12:00 UTC and NOW is 00:00 UTC. Night `n` ended `n - 0.5` days before
    NOW, started `n + 0.5` days before NOW, and has its middle (midnight) `n` days before NOW.
    """

    def night_frames(self, layout: DataLayout, night: int, hours: list[float]) -> list[Path]:
        """Create frames `hours` after the start of night `night`. Hour 12 is the middle."""
        return [
            make_file(
                layout.survey_dir / f"night{night}" / f"h{hour:04.1f}.fits",
                30,
                days(night + 0.5) - hour * 3600,
            )
            for hour in hours
        ]

    def test_a_night_older_than_seven_days_keeps_the_frame_nearest_its_middle(
        self, layout: DataLayout, make_manager: Manager
    ) -> None:
        early, middle, late = self.night_frames(layout, 10, [2, 12, 20])
        report = make_manager().run_once()
        assert not early.exists()
        assert middle.exists()
        assert not late.exists()
        assert [(d.tier, d.reason, d.files) for d in report.deletions] == [("survey", "expired", 2)]

    def test_a_night_inside_the_full_window_keeps_every_frame(
        self, layout: DataLayout, make_manager: Manager
    ) -> None:
        frames = self.night_frames(layout, 5, [2, 12, 20])
        make_manager().run_once()
        assert all(frame.exists() for frame in frames)

    def test_a_night_that_ends_inside_the_window_is_not_thinned_yet(
        self, layout: DataLayout, make_manager: Manager
    ) -> None:
        # Night 7 ends 6.5 days before NOW, so its late frames are inside the window. Its early
        # frame is already 7.4 days old, and it stays until the whole night leaves the window.
        early, late = self.night_frames(layout, 7, [2, 20])
        assert days(7) < days(7.5) - 2 * 3600
        make_manager().run_once()
        assert early.exists()
        assert late.exists()

    def test_a_thinned_night_is_stable_on_the_next_pass(
        self, layout: DataLayout, make_manager: Manager
    ) -> None:
        self.night_frames(layout, 10, [2, 12, 20])
        manager = make_manager()
        manager.run_once()
        survivors = names(layout.survey_dir)
        second = manager.run_once()
        assert names(layout.survey_dir) == survivors
        assert second.deletions == ()

    def test_every_frame_of_a_night_older_than_67_days_is_deleted(
        self, layout: DataLayout, make_manager: Manager
    ) -> None:
        old = self.night_frames(layout, 70, [2, 12, 20])
        (kept,) = self.night_frames(layout, 66, [12])  # thinned, and still inside the 60 days
        make_manager().run_once()
        assert not any(frame.exists() for frame in old)
        assert kept.exists()

    def test_the_thinned_period_follows_the_configuration(
        self, layout: DataLayout, make_manager: Manager
    ) -> None:
        (frame,) = self.night_frames(layout, 20, [12])
        make_manager(survey_thinned_days=5).run_once()  # 7 + 5 = 12 days
        assert not frame.exists()

    def test_the_night_boundary_follows_the_configuration(
        self, layout: DataLayout, make_manager: Manager
    ) -> None:
        # Two frames, 3 hours apart, on either side of midnight UTC, ten days ago.
        before = make_file(layout.survey_dir / "a.fits", 30, days(10) + 2 * 3600)
        after = make_file(layout.survey_dir / "b.fits", 30, days(10) - 3600)
        make_manager(night_boundary_utc_hour=0).run_once()  # midnight splits them into two nights
        assert before.exists()
        assert after.exists()
        make_manager(night_boundary_utc_hour=12).run_once()  # noon to noon puts them in one night
        assert (before.exists(), after.exists()) == (False, True)  # the one nearer to midnight


class TestTierCaps:
    def test_the_metrics_cap_deletes_the_oldest_files_first_and_writes_an_event(
        self, layout: DataLayout, make_manager: Manager, events: list[EventRecord]
    ) -> None:
        files = [
            make_file(layout.segments_dir / f"s{n}.seg", 1000, days(5) - n * 3600)
            for n in range(5)  # s0 is the oldest
        ]
        report = make_manager(metrics_max_gb=gb(2500)).run_once()
        # 5000 bytes against a cap of 2500: two files leave 3000, so a third must go.
        assert [f.exists() for f in files] == [False, False, False, True, True]
        (deletion,) = report.deletions
        assert (deletion.tier, deletion.reason, deletion.files, deletion.size_bytes) == (
            "metrics",
            "tier_quota",
            3,
            3000,
        )
        assert deletion.early
        (event,) = events
        assert (event.level, event.kind) == ("warning", "retention.early_delete")
        assert event.detail == {
            "tier": "metrics",
            "reason": "tier_quota",
            "files": 3,
            "size_bytes": 3000,
        }
        assert "before the age limit" in event.message

    def test_the_bursts_quota_counts_unpinned_bursts_and_spares_pinned_ones(
        self, layout: DataLayout, make_manager: Manager, events: list[EventRecord]
    ) -> None:
        pinned = make_burst(layout, "20260101T000000Z-keep", 1000, days(30), pinned=True)
        oldest = make_burst(layout, "20260102T000000Z", 60, days(20))
        middle = make_burst(layout, "20260103T000000Z", 60, days(10))
        newest = make_burst(layout, "20260104T000000Z", 60, days(5))
        report = make_manager(bursts_max_gb=gb(100)).run_once()
        assert pinned.is_dir()  # the pinned burst is exempt and does not count
        assert not oldest.exists()
        assert not middle.exists()
        assert newest.is_dir()
        (deletion,) = report.deletions
        assert deletion.names == ("20260102T000000Z", "20260103T000000Z")
        (event,) = events
        assert event.kind == "retention.early_delete"
        assert event.detail is not None
        assert event.detail["bursts"] == ["20260102T000000Z", "20260103T000000Z"]

    def test_a_burst_is_deleted_whole_with_its_sidecar(
        self, layout: DataLayout, make_manager: Manager
    ) -> None:
        burst = make_burst(layout, "20260101T000000Z", 500, days(3))
        make_manager(bursts_max_gb=0).run_once()
        assert not burst.exists()

    def test_a_burst_ages_by_its_newest_file(
        self, layout: DataLayout, make_manager: Manager
    ) -> None:
        # The SER file is old, but the sidecar changed a day ago, so burst `a` is the newer one.
        a = layout.bursts_dir / "a"
        make_file(a / "burst.ser", 100, days(30))
        make_file(a / "burst.json", 10, days(1))
        b = make_burst(layout, "b", 110, days(10))
        make_manager(bursts_max_gb=gb(150)).run_once()
        assert a.is_dir()
        assert not b.exists()

    def test_files_changed_within_the_guard_time_are_never_deleted(
        self, layout: DataLayout, make_manager: Manager
    ) -> None:
        recent = make_file(layout.segments_dir / "open.seg.part", 5000, 60)
        old = make_file(layout.segments_dir / "old.seg", 5000, 3600)
        make_manager(metrics_max_gb=gb(1000)).run_once()
        assert recent.exists()
        assert not old.exists()

    def test_a_file_that_cannot_be_deleted_is_counted_and_the_rest_go(
        self, layout: DataLayout, make_manager: Manager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stuck = make_file(layout.segments_dir / "a.seg", 1000, days(5))
        other = make_file(layout.segments_dir / "b.seg", 1000, days(4))
        real_unlink = Path.unlink

        def refuse(self: Path, missing_ok: bool = False) -> None:
            if self.name == "a.seg":
                raise PermissionError("in use")
            real_unlink(self, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", refuse)
        report = make_manager(metrics_max_gb=gb(500)).run_once()
        monkeypatch.undo()
        assert stuck.exists()
        assert not other.exists()
        assert report.errors == 1
        (deletion,) = report.deletions
        assert deletion.files == 1  # only what the pass really deleted


class TestQuota:
    def fill(self, layout: DataLayout, **sizes: int) -> dict[str, list[Path]]:
        """Create two units of half the size in each named tier. The older one comes first."""
        roots = {
            "metrics": layout.segments_dir,
            "previews": layout.previews_dir,
            "survey": layout.survey_dir,
        }
        made: dict[str, list[Path]] = {}
        for tier, size in sizes.items():
            ages = [days(3) - n * 3600 for n in (1, 2)]  # unit 1 is older than unit 2
            if tier == "bursts":
                made[tier] = [
                    make_burst(layout, f"2026010{n}T000000Z", size // 2, age)
                    for n, age in zip((1, 2), ages, strict=True)
                ]
            else:
                made[tier] = [
                    make_file(roots[tier] / f"{tier}{n}.dat", size // 2, age)
                    for n, age in zip((1, 2), ages, strict=True)
                ]
        return made

    def test_the_quota_is_a_quarter_of_the_partition(
        self, layout: DataLayout, make_manager: Manager
    ) -> None:
        self.fill(layout, metrics=100)
        report = make_manager(disk=FakeDisk(layout.root, 10_000)).run_once()
        assert report.quota_bytes == 2500
        assert report.usage_bytes == 100
        assert report.deletions == ()  # inside the quota

    def test_the_quota_fraction_follows_the_configuration(
        self, layout: DataLayout, make_manager: Manager
    ) -> None:
        report = make_manager(disk=FakeDisk(layout.root, 10_000), quota_fraction=0.5).run_once()
        assert report.quota_bytes == 5000

    def test_over_quota_shrinks_the_metrics_first(
        self, layout: DataLayout, make_manager: Manager, events: list[EventRecord]
    ) -> None:
        made = self.fill(
            layout, metrics=2000, previews=1000, survey=1000
        )  # 4000; the quota is 2500
        report = make_manager(disk=FakeDisk(layout.root, 10_000)).run_once()
        assert report.usage_bytes == 2000
        assert report.usage_bytes <= report.quota_bytes
        assert [p.exists() for p in made["metrics"]] == [False, False]  # 1,500 needed, 2,000 freed
        assert all(p.exists() for p in made["previews"] + made["survey"])
        (deletion,) = report.deletions
        assert (deletion.tier, deletion.reason, deletion.size_bytes) == (
            "metrics",
            "total_quota",
            2000,
        )
        assert kinds(events) == ["retention.early_delete"]

    @pytest.mark.parametrize(
        ("partition", "expected"),
        [
            (14_000, [("metrics", 1)]),  # quota 3,500 of 4,000 used: 500 over
            (10_000, [("metrics", 2), ("previews", 1)]),  # quota 2,500: 1,500 over
            (6_000, [("metrics", 2), ("previews", 2), ("survey", 1)]),  # quota 1,500: 2,500 over
            (
                2_000,  # quota 500: 3,500 over
                [("metrics", 2), ("previews", 2), ("survey", 2), ("bursts", 1)],
            ),
        ],
    )
    def test_the_tiers_shrink_in_the_order_metrics_previews_survey_bursts(
        self,
        layout: DataLayout,
        make_manager: Manager,
        events: list[EventRecord],
        partition: int,
        expected: list[tuple[str, int]],
    ) -> None:
        made = self.fill(layout, metrics=1000, previews=1000, survey=1000, bursts=1000)
        report = make_manager(disk=FakeDisk(layout.root, partition)).run_once()
        assert [(d.tier, d.files) for d in report.deletions] == expected
        assert all(d.reason == "total_quota" for d in report.deletions)
        assert report.usage_bytes <= report.quota_bytes
        survivors = {tier: [p.exists() for p in paths] for tier, paths in made.items()}
        for tier, files in expected:
            # The oldest units go first, so the survivors of a tier are a suffix of its units.
            assert survivors[tier] == [False] * files + [True] * (2 - files)
        for tier in set(made) - {tier for tier, _ in expected}:
            assert all(survivors[tier])  # a later tier is untouched
        assert [e.kind for e in events] == ["retention.early_delete"] * len(expected)

    def test_every_early_deletion_writes_one_event_for_its_tier_and_reason(
        self, layout: DataLayout, make_manager: Manager, events: list[EventRecord]
    ) -> None:
        self.fill(layout, metrics=1000, previews=1000, survey=1000)  # 3,000 used
        make_manager(disk=FakeDisk(layout.root, 4000)).run_once()  # quota 1,000: 2,000 must go
        details = [event.detail for event in events]
        assert [(d["tier"], d["reason"]) for d in details if d is not None] == [
            ("metrics", "total_quota"),
            ("previews", "total_quota"),
        ]
        assert [e.level for e in events] == ["warning", "warning"]

    def test_pinned_bursts_and_the_database_are_never_deleted_and_an_error_event_says_so(
        self, layout: DataLayout, make_manager: Manager, events: list[EventRecord]
    ) -> None:
        pinned = make_burst(layout, "20260101T000000Z-keep", 3000, days(40), pinned=True)
        database = make_file(layout.db_path, 1000, days(100))
        manager = make_manager(disk=FakeDisk(layout.root, 8000))  # quota 2,000, and 4,000 used
        report = manager.run_once()
        assert pinned.is_dir()
        assert database.exists()
        assert report.deletions == ()
        assert report.shortfall_bytes == 2000
        (event,) = events
        assert (event.level, event.kind) == ("error", "retention.over_quota")
        assert event.detail is not None
        assert event.detail["pinned_burst_bytes"] == 3000
        assert event.detail["excess_bytes"] == 2000
        manager.run_once()
        assert len(events) == 1  # the error appears when it starts, not on every pass

    def test_the_error_returns_when_a_new_episode_starts(
        self, layout: DataLayout, make_manager: Manager, events: list[EventRecord]
    ) -> None:
        pinned = make_burst(layout, "20260101T000000Z-keep", 3000, days(40), pinned=True)
        disk = FakeDisk(layout.root, 8000)
        manager = make_manager(disk=disk)
        manager.run_once()
        disk.total = 80_000  # the quota is 20,000 now, so the episode ends
        manager.run_once()
        disk.total = 8000
        manager.run_once()
        assert kinds(events) == ["retention.over_quota", "retention.over_quota"]
        assert pinned.is_dir()

    def test_low_free_space_frees_space_and_capture_resumes(
        self, layout: DataLayout, make_manager: Manager, events: list[EventRecord]
    ) -> None:
        made = self.fill(layout, metrics=2000)
        # 10,000 bytes: others use 7,500 and the metrics use 2,000, so 500 are free. The limit is
        # 1,000 and the margin 500, so the pass needs 1,000 more bytes.
        manager = make_manager(
            disk=FakeDisk(layout.root, 10_000, other_used=7_500),
            quota_fraction=1.0,
            min_free_gb=gb(1000),
            resume_margin_gb=gb(500),
        )
        assert manager.capture_allowed() is False
        report = manager.run_once()
        assert [p.exists() for p in made["metrics"]] == [False, True]
        (deletion,) = report.deletions
        assert (deletion.tier, deletion.reason, deletion.files) == ("metrics", "low_space", 1)
        assert report.free_bytes == 1500
        assert report.capture_allowed is True
        assert kinds(events) == [
            "retention.capture_stopped",
            "retention.early_delete",
            "retention.capture_resumed",
        ]

    def test_the_usage_counts_every_byte_under_the_data_directory(
        self, layout: DataLayout, make_manager: Manager
    ) -> None:
        make_file(layout.root / "other" / "notes.bin", 700, days(1))
        make_burst(layout, "20260101T000000Z", 300, days(1))
        report = make_manager(disk=FakeDisk(layout.root, 10**6)).run_once()
        assert report.usage_bytes == 1000


class TestCaptureGate:
    LIMITS: ClassVar[dict[str, float]] = {"min_free_gb": gb(1000), "resume_margin_gb": gb(500)}

    def test_capture_stops_below_the_limit_and_resumes_above_the_margin(
        self, layout: DataLayout, make_manager: Manager, events: list[EventRecord]
    ) -> None:
        disk = FakeDisk(layout.root, 10_000, other_used=8_000)  # 2,000 free
        manager = make_manager(disk=disk, **self.LIMITS)
        assert manager.capture_allowed() is True
        disk.other_used = 9_001  # 999 free: below the limit
        assert manager.capture_allowed() is False
        assert manager.capture_allowed() is False  # one event for one change
        disk.other_used = 8_600  # 1,400 free: above the limit, below the margin
        assert manager.capture_allowed() is False  # capture stays stopped
        disk.other_used = 8_500  # 1,500 free: the limit plus the margin
        assert manager.capture_allowed() is True
        assert manager.capture_allowed() is True
        assert kinds(events) == ["retention.capture_stopped", "retention.capture_resumed"]
        assert [e.level for e in events] == ["warning", "info"]

    def test_the_limit_itself_still_allows_capture(
        self, layout: DataLayout, make_manager: Manager
    ) -> None:
        disk = FakeDisk(layout.root, 10_000, other_used=9_000)  # exactly 1,000 free
        assert make_manager(disk=disk, **self.LIMITS).capture_allowed() is True
        disk.other_used = 9_001
        assert make_manager(disk=disk, **self.LIMITS).capture_allowed() is False

    def test_the_default_limit_is_one_gigabyte_with_a_margin_of_half_a_gigabyte(
        self, layout: DataLayout, make_manager: Manager
    ) -> None:
        total = 100 * GB
        disk = FakeDisk(layout.root, total, other_used=total - GB)  # exactly 1 GB free
        manager = make_manager(disk=disk, **{"min_free_gb": 1.0, "resume_margin_gb": 0.5})
        assert manager.capture_allowed() is True
        disk.other_used += 1
        assert manager.capture_allowed() is False
        disk.other_used = total - GB - GB // 2 + 1  # one byte short of 1.5 GB
        assert manager.capture_allowed() is False
        disk.other_used = total - GB - GB // 2  # exactly 1.5 GB
        assert manager.capture_allowed() is True

    def test_a_pass_applies_the_same_rule(
        self, layout: DataLayout, make_manager: Manager, events: list[EventRecord]
    ) -> None:
        disk = FakeDisk(layout.root, 10_000, other_used=9_500)  # 500 free
        manager = make_manager(disk=disk, quota_fraction=1.0, **self.LIMITS)
        assert manager.run_once().capture_allowed is False
        assert kinds(events) == ["retention.capture_stopped"]  # nothing to delete, no quota error
        disk.other_used = 0
        assert manager.run_once().capture_allowed is True
        assert kinds(events) == ["retention.capture_stopped", "retention.capture_resumed"]

    def test_the_event_says_how_much_space_is_free(
        self, layout: DataLayout, make_manager: Manager, events: list[EventRecord]
    ) -> None:
        disk = FakeDisk(layout.root, 10_000, other_used=9_700)
        make_manager(disk=disk, **self.LIMITS).capture_allowed()
        (event,) = events
        assert event.detail == {"free_bytes": 300, "min_free_bytes": 1000}
        assert "raw capture stops" in event.message

    def test_status_reports_the_free_space_and_the_usage_of_the_last_pass(
        self, layout: DataLayout, make_manager: Manager
    ) -> None:
        make_file(layout.previews_dir / "a.jpg", 400, days(1))
        disk = FakeDisk(layout.root, 10_000, other_used=1_000)
        manager = make_manager(disk=disk, **self.LIMITS)
        before = manager.status()
        assert (before.usage_bytes, before.quota_bytes) == (None, None)
        assert (before.total_bytes, before.free_bytes) == (10_000, 8_600)
        manager.run_once()
        after = manager.status()
        assert (after.usage_bytes, after.quota_bytes) == (400, 2500)
        assert after.capture_allowed is True
        assert after.low_space is False
        disk.other_used = 9_000
        low = manager.status()
        assert low.low_space is True
        assert low.capture_allowed is False


class TestRealDisk:
    def test_the_system_probe_reports_a_partition(self, tmp_path: Path) -> None:
        usage = system_disk_usage(tmp_path)
        assert usage.total_bytes > 0
        assert 0 <= usage.free_bytes <= usage.total_bytes

    def test_a_manager_with_the_system_probe_runs(
        self, layout: DataLayout, clock: VirtualClock, emitter: EventEmitter
    ) -> None:
        report = RetentionManager(layout, RetentionConfig(), clock, emitter).run_once()
        assert report.deletions == ()
        assert report.quota_bytes > 0
