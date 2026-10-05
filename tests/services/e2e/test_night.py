"""A simulated night through the whole of `core`, checked against the truth that the sim was given.

`build_night` (see `night.py`) runs the composition root on a virtual clock with the production
analyzers. The tests inject a Fried parameter, clouds, and a position of the Sun, and they read what
`core` stored: the windows, the survey records, the flags, and the events. They never wait for the
real time, and they assert on records and on virtual time only.

The short test runs on every push. The others run for 15 to 60 seconds of CPU, so they carry the
`slow` marker (run them with `--slow`).
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sep", reason="the survey path needs the survey extra")
pytest.importorskip("scipy", reason="the fast path needs the fast extra")

from seeingmon.clock import NS_PER_S, iso_to_utc_ns
from seeingmon.drivers.sim.detection import DetectionModel
from seeingmon.drivers.sim.params import SimParams
from seeingmon.records import RunRecord
from seeingmon.store.db import StoreReader

from ..core.rig import read_all
from .night import SIM_LATITUDE_DEG, Night, build_night, cloud

R0_TOLERANCE = 0.15  # one window of 6 s has a sampling error of 2 to 5%, and the bias is below 3%


def windows_with_r0(night: Night) -> list[Any]:
    return [w for w in night.records("seeing_window") if w.r0_cm is not None]


class TestAShortNight:
    """One cycle of the scheduler on a short window: it takes about ten seconds of CPU."""

    def test_the_chain_from_the_frames_to_the_records_works(self, tmp_path: Path) -> None:
        night = build_night(tmp_path, window_s=3.0)
        try:
            night.run_until(lambda: windows_with_r0(night) and night.records("sky_quality"))
            window = windows_with_r0(night)[0]
            assert window.r0_cm == pytest.approx(night.r0_cm, rel=0.3)  # three seconds are noisy
            assert window.readout_mode == "bin1"
            assert window.zenith_angle_deg == pytest.approx(90.0 - SIM_LATITUDE_DEG, abs=1.5)
            assert "degraded" not in window.flags

            frames = night.records("survey_frame")
            assert frames
            pointings = night.records("pointing")
            assert any(not p.flags for p in pointings)  # the long frame solved: the star is tracked
            sky = night.records("sky_quality")[0]
            assert sky.zero_point_mag is not None
            assert 19.0 < sky.zero_point_mag < 19.3
            assert sky.n_stars_used >= 12
            assert "dark_due" in sky.flags  # no dark library yet
            assert "moon" in sky.flags  # the Moon of 2 January is up in the evening, at the site

            (run,) = night.records("run")
            assert isinstance(run, RunRecord)
            assert run.versions["camera_driver"] == "sim"
            night.app.tick()
            health = night.records("health")[-1]
            assert health.dark_due is True
            assert health.components["scheduler"] == "ok"
            kinds = [e.kind for e in night.events()]
            assert kinds[:2] == ["core.started", "scheduler.start"]
            assert "scheduler.state_change" in kinds
        finally:
            night.app.stop()


@pytest.mark.slow
class TestTheFastPath:
    def test_the_windows_agree_with_the_injected_r0(self, tmp_path: Path) -> None:
        night = build_night(tmp_path, r0_cm=10.0)
        try:
            night.run_until(lambda: len(windows_with_r0(night)) >= 3)
            values = [w.r0_cm for w in windows_with_r0(night)]
            for value in values:
                assert value == pytest.approx(10.0, rel=R0_TOLERANCE)
            assert statistics.median(values) == pytest.approx(10.0, rel=0.07)
            for window in windows_with_r0(night):
                assert "degraded" not in window.flags
                assert "twilight" not in window.flags  # the Sun is 18 degrees below the horizon
                assert "cloud" not in window.flags
        finally:
            night.app.stop()

    def test_a_poorer_sky_reads_poorer(self, tmp_path: Path) -> None:
        night = build_night(tmp_path, r0_cm=6.0)
        try:
            night.run_until(lambda: len(windows_with_r0(night)) >= 2)
            values = [w.r0_cm for w in windows_with_r0(night)]
            assert statistics.median(values) == pytest.approx(6.0, rel=R0_TOLERANCE)
        finally:
            night.app.stop()


@pytest.mark.slow
class TestTheDaylightGate:
    def test_the_search_runs_by_day_and_measures_from_where_polaris_shows(
        self, tmp_path: Path
    ) -> None:
        """The Sun gates nothing. The simulator's sky at +6 degrees is far from saturating the
        brightness frame, so `auto` starts at once and the search looks for Polaris.

        The fast frames of these nights take 250 us, and at that exposure the detection estimate
        (`seeingmon.drivers.sim.detection`) puts the median-frame SNR of 10 at -0.54 degrees. The
        second detecting burst in a row starts measure, so `polaris.visible` comes there or a
        little later. 1 degree is the tolerance of the brief, for the bursts that run only in the
        fast slots of the cycle and the noise of the SNR of one burst.
        """
        start = "2026-01-01T14:30:00Z"
        night = build_night(tmp_path, start=start)
        try:
            night.run_for(30 * 60.0)  # the Sun is still above the horizon
            status = night.app.scheduler.status()
            assert status.state == "auto"
            assert status.search is not None
            assert status.search.mode == "search"
            assert status.counters.search_bursts > 10
            assert night.records("seeing_window") == []  # Polaris does not show in 250 us yet
            night.run_until(
                lambda: night.records("seeing_window"), limit_s=3 * 3600.0, slice_s=60.0
            )
            (change,) = night.events("scheduler.state_change")
            assert change.t_utc_ns - iso_to_utc_ns(start) < 60 * NS_PER_S  # the first frame
            (visible,) = night.events("polaris.visible")
            params = SimParams.from_profile(night.app.profile, night.app.profile.fast_mode.mode)
            estimate = DetectionModel.for_simulator(params, max_exposure_us=250).crossing_deg()
            assert estimate is not None
            assert visible.detail["sun_elevation_deg"] == pytest.approx(estimate, abs=1.0)
            first = night.records("seeing_window")[0]
            assert first.t_utc_ns >= visible.t_utc_ns - NS_PER_S
            assert "twilight" in first.flags  # the Sun is near the horizon, not 18 degrees down
        finally:
            night.app.stop()


@pytest.mark.slow
class TestClouds:
    def test_a_cloud_is_flagged_the_scheduler_responds_and_the_measurement_returns(
        self, tmp_path: Path
    ) -> None:
        night = build_night(tmp_path, clouds=[cloud(100.0, 150.0, transmission=0.02)])
        try:
            night.run_until(
                lambda: (
                    night.events("scheduler.cloud")
                    and len(night.events("scheduler.cloud")) >= 2
                    and len(windows_with_r0(night)) >= 6
                ),
                limit_s=900.0,
            )
            first, second = night.events("scheduler.cloud")[:2]
            assert first.detail["active"] is True
            assert second.detail["active"] is False
            cloudy = [r for r in night.records("sky_quality") if "cloud" in r.flags]
            assert cloudy
            assert all(r.cloud_fraction >= 0.5 for r in cloudy)
            assert all(r.zero_point_mag is None for r in cloudy)  # no star, no zero point
            clear = [r for r in night.records("sky_quality") if "cloud" not in r.flags]
            assert clear
            # The cloud hid the star: measure ended, and the stream searched until the cloud
            # passed, so no window closed while the cloud response held.
            (hidden,) = night.events("polaris.hidden")
            assert hidden.detail["reason"] == "star_missing"
            assert hidden.t_utc_ns < first.t_utc_ns
            assert not [w for w in night.records("seeing_window") if "cloud" in w.flags]
            lost = [w for w in night.records("seeing_window") if w.r0_cm is None]
            assert lost  # the window that the missing star ended
            assert all("partial" in w.flags for w in lost)
            visible = night.events("polaris.visible")
            assert len(visible) == 2
            assert visible[1].t_utc_ns > second.t_utc_ns  # the search found the star again
            after = [w for w in windows_with_r0(night) if w.t_utc_ns > second.t_utc_ns]
            assert after
            assert statistics.median([w.r0_cm for w in after]) == pytest.approx(
                10.0, rel=R0_TOLERANCE
            )
        finally:
            night.app.stop()


@pytest.mark.slow
class TestTheNightlySummary:
    def test_the_star_summary_closes_at_the_end_of_the_night_and_at_shutdown(
        self, tmp_path: Path
    ) -> None:
        # At 180 degrees east the split hour (12:00 UTC) is local midnight, in the middle of the
        # dark. The run starts three minutes before it and goes on for three after.
        night = build_night(
            tmp_path,
            start="2026-01-01T11:57:00Z",
            longitude_deg=180.0,
            config_extra="[survey.star_epoch]\nmin_frames = 2\n",
            sim_extra={"twilight": False},  # the sky of the sim follows its own site, not ours
        )
        path = night.app.storage.layout.db_path  # type: ignore[union-attr]
        try:
            split_ns = iso_to_utc_ns("2026-01-01T12:00:00Z")
            night.run_until(lambda: night.clock.utc_ns() > split_ns + 45 * NS_PER_S)
            summaries = night.records("star_epoch")
            assert len(summaries) == 1  # the first night closed without a frame of the next one
            first = summaries[0]
            assert first.night == "2025-12-31"
            assert first.t_utc_ns < split_ns  # it starts at the first survey frame of the night
            assert first.n_stars >= 20
            night.run_for(150.0)
        finally:
            night.app.stop("a test")
        with StoreReader.open(path) as store:
            nights = [r.night for r in read_all(store, "star_epoch")]  # type: ignore[attr-defined]
        assert nights == ["2025-12-31", "2026-01-01"]  # the open night was written at the stop
