"""`CoreApp` and the visibility summary: written when the split hour passes, and never at shutdown.

The rig starts at 22:00 UTC on 1 January. A split hour of 22.1 (22:06 UTC) ends the night six
minutes later, so a test needs no day of frames. A split hour in the dark is no advice for a
station (`[survey] night_split_utc_hour`), but the summary works the same at any hour.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from seeingmon.clock import NS_PER_S, iso_to_utc_ns
from seeingmon.records.visibility import VisibilitySummaryRecord
from seeingmon.store.db import StoreReader

from .rig import NIGHT, CoreRig, build_rig, read_all

SPLIT = "[survey]\nnight_split_utc_hour = 22.1\n"
SPLIT_NS = iso_to_utc_ns("2026-01-01T22:06:00Z")
assert NIGHT < SPLIT_NS


def summaries(rig: CoreRig) -> list[Any]:
    return [r for r in rig.records("visibility_summary") if isinstance(r, VisibilitySummaryRecord)]


def run_past_the_split(rig: CoreRig, seconds_after: float) -> None:
    while rig.clock.utc_ns() < SPLIT_NS + round(seconds_after * NS_PER_S):
        rig.run_for(10.0)


class TestTheSummaryOfCore:
    def test_core_writes_the_night_at_the_split_hour_without_a_star_summary(
        self, tmp_path: Path
    ) -> None:
        rig = build_rig(tmp_path, config_extra=SPLIT, polaris=True)
        try:
            assert rig.app.nightly is None  # the fake survey analyzer keeps no star summary
            rig.app.start()
            run_past_the_split(rig, 5.0)
            assert summaries(rig) == []  # the window that began before the split may be open
            run_past_the_split(rig, 45.0)  # one analysis window (10 s) and one task interval
            (summary,) = summaries(rig)
            assert summary.night == "2025-12-31"
            assert summary.t_utc_ns == SPLIT_NS - 24 * 3600 * NS_PER_S
            (visible,) = rig.events("polaris.visible")
            assert summary.first_visible_utc_ns == visible.t_utc_ns
            # Core started in the night and found Polaris at once, so it may have shown before.
            assert summary.first_censored is True
            assert summary.last_visible_utc_ns == SPLIT_NS  # still visible when the night ended
            assert summary.last_censored is True
            assert 0 < summary.visible_hours < 0.1
            windows = [w for w in rig.records("seeing_window") if w.t_utc_ns < SPLIT_NS]
            assert windows
            with_seeing = [w for w in windows if w.r0_cm is not None]  # type: ignore[attr-defined]
            seconds = sum(w.duration_s for w in with_seeing)  # type: ignore[attr-defined]
            assert summary.seeing_hours == pytest.approx(seconds / 3600.0, abs=1e-6)
            assert summary.first_visible_sun_deg is not None  # the rig has a site
            assert summary.first_visible_sun_deg < -40.0  # 22:00 UTC in January, 55 degrees north
        finally:
            rig.app.stop()

    def test_a_stop_before_the_split_hour_writes_no_summary(self, tmp_path: Path) -> None:
        rig = build_rig(tmp_path, config_extra=SPLIT, polaris=True)
        assert rig.app.storage is not None
        path = rig.app.storage.layout.db_path
        try:
            rig.app.start()
            rig.run_for(120.0)
            assert rig.events("polaris.visible")  # Polaris was visible at the stop
        finally:
            rig.app.stop()
        with StoreReader.open(path) as store:
            assert read_all(store, "visibility_summary") == []  # the night is not over
