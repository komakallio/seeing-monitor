"""A simulated night through the whole of `core`, checked against the truth that the sim was given.

`build_night` (see `night.py`) runs the composition root on a virtual clock with the production
analyzers. The tests inject a Fried parameter, clouds, and a position of the Sun, and they read what
`core` stored: the windows, the survey records, the flags, and the events. They never wait for the
real time, and they assert on records and on virtual time only.

The short test runs on every push. The others run for 15 to 60 seconds of CPU, so they carry the
`slow` marker (run them with `--slow`).
"""

from __future__ import annotations

import itertools
import re
import statistics
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sep", reason="the survey path needs the survey extra")
pytest.importorskip("scipy", reason="the fast path needs the fast extra")

from seeingmon.cli import main
from seeingmon.clock import NS_PER_S, iso_to_utc_ns, utc_ns_to_datetime
from seeingmon.drivers.sim.detection import DETECTION_SNR, DetectionModel
from seeingmon.drivers.sim.params import SimParams
from seeingmon.drivers.sim.sky import DAYLIGHT_SKY_MAG_ARCSEC2
from seeingmon.fastpath.config import FastPathConfig
from seeingmon.records import RunRecord
from seeingmon.scheduler.ephemeris import sun_elevation_deg
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

        The fast frames of these nights take 250 us. At that exposure, the detection estimate
        (`seeingmon.drivers.sim.detection`) keeps the matched SNR of the median frame above 10 at
        every Sun elevation (22 at +6 degrees), so it has no crossing, and the first two bursts
        find Polaris by day. With the SNR of the centroid aperture, the search waited for the
        Sun to reach -0.5 degrees.
        """
        start = "2026-01-01T14:30:00Z"
        night = build_night(tmp_path, start=start)
        try:
            night.run_until(lambda: night.records("seeing_window"), limit_s=30 * 60.0, slice_s=60.0)
            (change,) = night.events("scheduler.state_change")
            assert change.t_utc_ns - iso_to_utc_ns(start) < 60 * NS_PER_S  # the first frame
            status = night.app.scheduler.status()
            assert status.state == "auto"
            assert status.search is not None
            assert status.search.mode == "measure"
            (visible,) = night.events("polaris.visible")
            params = SimParams.from_profile(night.app.profile, night.app.profile.fast_mode.mode)
            model = DetectionModel.for_simulator(params, max_exposure_us=250)
            assert model.crossing_deg() is None
            sun = visible.detail["sun_elevation_deg"]
            assert sun > 5.0  # the first bursts, with the Sun near +6 degrees
            # The burst's median matched SNR against the median frame of the estimate. The run
            # gave 23.0 against 22.4, at +5.9 degrees, after 2 bursts.
            assert visible.detail["snr"] == pytest.approx(model.row(sun).snr_matched, rel=0.1)
            assert status.counters.search_bursts == 2
            first = night.records("seeing_window")[0]
            assert first.t_utc_ns >= visible.t_utc_ns - NS_PER_S
            assert "daylight" in first.flags  # the Sun is up
            assert "twilight" not in first.flags
        finally:
            night.app.stop()


@pytest.mark.slow
class TestPolarisAtDusk:
    def test_seeing_runs_where_the_detection_estimate_puts_polaris(self, tmp_path: Path) -> None:
        """A spring dusk with the fast stream of the camera: the full reference sensor, the real
        Polaris, at most 2 ms, and the adaptive exposure (`target_background_fraction` 0.3).

        The detection estimate (`seeingmon.drivers.sim.detection`) decides by the matched SNR of
        the median frame. It keeps that SNR at 41 or more in the model's full daylight, so it has
        no crossing of 10, and the search has no Sun limit. The run starts with the Sun at +11.5
        degrees, in the brightest sky of the model (its daylight sky holds from +10 degrees up),
        and seeing must run there: the first bursts find Polaris, and the first seeing window
        comes while the Sun is still above +10 degrees. An estimate with a crossing would need
        the run to start above it, and seeing to start within 1 degree of it: the tolerance of
        the brief, for the bursts that run once a cycle (one degree is 7 minutes here), the noise
        of the SNR of a burst, and the two detecting bursts in a row that measure needs.

        The centroids of these windows come from the wide aperture, whose noise is about four
        times the image motion in daylight. The noise model holds the sky, and it reads about 5%
        low there, so `r0` reads about 10% low, and the window carries `noisy`
        (`docs/research-notes.md`, "The seeing in a bright sky"). The first light of this test,
        before the sky term, read an `r0` of 3.4 cm against the injected 10 cm.
        """
        start = "2026-04-20T17:45:00Z"  # the Sun at +11.5 degrees on the synthetic site
        night = build_night(tmp_path, start=start, sensor="full", fast_exposure_us=2000)
        try:
            profile = night.app.profile
            mode = profile.fast_mode.mode
            fastpath = night.config.section("fastpath", FastPathConfig)
            airy = profile.airy_fwhm_px(mode)
            model = DetectionModel.for_simulator(
                SimParams.from_profile(profile, mode),
                filter_fwhms_px=tuple(w * airy for w in fastpath.matched_fwhm_airy_widths),
            )
            assert night.app.scheduler.config.fast.target_background_fraction == pytest.approx(
                model.target_background_fraction
            )
            start_sun = sun_elevation_deg(night.start_utc_ns, SIM_LATITUDE_DEG, 0.0)
            crossing = model.crossing_deg()
            if crossing is not None:  # pragma: no cover - the current estimate has none
                assert crossing < start_sun - 1.0  # the run must start above the crossing
            night.run_until(lambda: windows_with_r0(night), limit_s=40 * 60.0, slice_s=30.0)
            first = windows_with_r0(night)[0]
            sun = sun_elevation_deg(first.t_utc_ns, SIM_LATITUDE_DEG, 0.0)
            (visible,) = night.events("polaris.visible")
            visible_sun = visible.detail["sun_elevation_deg"]
            if crossing is None:
                # The brightest sky of the model: the daylight sky, which holds above +10 degrees.
                for elevation in (visible_sun, sun):
                    sky = model.row(elevation).sky_mag_arcsec2
                    assert sky == pytest.approx(DAYLIGHT_SKY_MAG_ARCSEC2), elevation
                assert model.row(sun).snr_matched >= DETECTION_SNR
            else:  # pragma: no cover - the current estimate has none
                assert sun == pytest.approx(crossing, abs=1.0)
                assert visible_sun == pytest.approx(crossing, abs=1.0)
            assert visible.detail["probe"] is False  # no limit, so no probe
            # The burst that confirmed Polaris: its median matched SNR sits near the median frame
            # of the estimate. The run gave 38.6 against 41.3, at +11.4 degrees, with an exposure
            # 2.4% short.
            assert visible.detail["snr"] == pytest.approx(
                model.row(visible_sun).snr_matched, rel=0.1
            )
            # The adaptive exposure: the background sits near its target. The run gave 1196 us
            # against 1226 us and 0.300.
            assert first.exposure_us < 2000
            # The loop counts the offset as sky, which makes the exposure 2.4% short
            # (`tests/scheduler/test_exposure.py`).
            assert first.exposure_us == pytest.approx(model.row(sun).exposure_us, rel=0.05)
            # The window counts the offset as background too, so the background sits on the target.
            assert first.background_fraction == pytest.approx(0.3, abs=0.01)
            # The window's star_snr is the SNR of the centroid aperture, a fifth of the matched
            # one. The run gave 8.2 against 8.3 for the median frame of the estimate.
            assert first.star_snr is not None
            assert first.star_snr == pytest.approx(model.row(sun).snr_centroid, rel=0.1)
            assert "daylight" in first.flags
            assert "twilight" not in first.flags
            # The aperture's noise in daylight: `noisy`, and r0 within the bias and the scatter
            # of the table (-11% and 7% for a window of 60 s). The run gave 9.2 cm for this
            # partial window, and 11.0 and 11.1 cm for the two full windows after it.
            assert "noisy" in first.flags
            assert first.r0_cm == pytest.approx(night.r0_cm, rel=0.3)
            # At most 2 ms, Polaris does not saturate in any window of the run.
            for window in night.records("seeing_window"):
                assert "saturated" not in window.flags
        finally:
            night.app.stop()


def solved_pointings(night: Night) -> list[Any]:
    """The pointing records of the survey frames that solved, in order."""
    return [p for p in night.records("pointing") if "unsolved" not in p.flags]


def first_full_solve(night: Night) -> Any | None:
    """The first solve with enough stars for no flag (12 matched, `[survey.pointing] few_stars`)."""
    return next((p for p in solved_pointings(night) if not p.flags), None)


def sun_at(t_utc_ns: int) -> float:
    return sun_elevation_deg(t_utc_ns, SIM_LATITUDE_DEG, 0.0)


@pytest.mark.slow
class TestTheSurveyAtDusk:
    """A winter dusk with the survey cycle of the camera (180 s) and the production pipeline.

    The run starts with the Sun 2.8 degrees down. The fast stream measures there, but a long survey
    frame of 1 s would hold about 14 times its saturation, so each survey step skips its long
    exposure and keeps its 1 ms frame, which the daylight gate reads. The first step measures the
    black level of the 1 ms frame with a frame of 32 us. Once the 1 ms frame shows less than about
    5 counts of the ADC of sky, the long frames start at 1 s and grow as the sky darkens.

    The runs gave (`docs/research-notes.md`, "The survey at dusk"): the first survey frame solves
    at -6.31 degrees (`few_stars`), and the first with 12 stars or more at -8.17 degrees (-8.55
    before the long frames in a bright sky took the full search). Long frames of a fixed 30 s first
    solve at -9.98 degrees. A degree is 3 survey steps here.
    """

    START = "2026-01-01T16:00:00Z"

    def test_the_first_survey_frame_solves_while_the_sky_is_bright(self, tmp_path: Path) -> None:
        night = build_night(tmp_path, start=self.START, cadence_s=180.0)
        try:
            night.run_until(
                lambda: first_full_solve(night) is not None, limit_s=2 * 3600.0, slice_s=60.0
            )
            status = night.app.scheduler.status()
            first = solved_pointings(night)[0]
            full = first_full_solve(night)
            assert full is not None
            # The tolerance is the step of the cycle, 0.37 degrees, and a second step for the noise
            # of the stars that a frame of 1 s detects.
            assert sun_at(first.t_utc_ns) == pytest.approx(-6.31, abs=0.8)
            assert sun_at(full.t_utc_ns) == pytest.approx(-8.17, abs=0.8)

            frames = night.records("survey_frame")
            longs = [f for f in frames if f.exposure_s >= 1.0]
            shorts = [f for f in frames if f.exposure_s < 1.0]
            # Every step took its 1 ms frame, and the bright steps skipped their long frame.
            assert status.counters.survey_long_skips >= 5
            assert len(shorts) == len(longs) + status.counters.survey_long_skips
            assert longs[0].exposure_s == 1.0
            assert sun_at(longs[0].t_utc_ns) < -5.0
            # One or two long frames saturate before the black level puts the long frames where
            # they serve (one at -5.6 degrees in the runs above), and none of them reports clouds.
            sky = night.records("sky_quality")
            saturated = [q for q in sky if "saturated_sky" in q.flags]
            assert 1 <= len(saturated) <= 2
            assert all(q.cloud_fraction is None and "cloud" not in q.flags for q in saturated)
            # The other long frames take the full search, which finds the sharp stars of a short
            # frame that the binned search loses, and they expect only the stars that it finds
            # with a chance of 0.9 or more, so the clear sky reads no clouds. The runs read 0 in
            # every such frame, and the binned search had read 0.43 to 0.80 in the frames of 1 to
            # 3 s. One missed star in the 4 to 13 expected is 0.08 to 0.25, so the tolerance of
            # 0.25 allows one, and the `cloud` flag (0.3) never sets.
            clear = [q for q in sky if "saturated_sky" not in q.flags]
            assert clear
            assert all(q.provenance.get("search") == "full" for q in clear)
            assert all("cloud" not in q.flags for q in clear)
            fractions = [q.cloud_fraction for q in clear if q.cloud_fraction is not None]
            assert fractions
            assert max(fractions) <= 0.25
            # The long exposure grows as the sky darkens, by at most 4 times a step.
            exposures = [f.exposure_s for f in longs]
            assert exposures == sorted(exposures)
            assert all(b / a <= 4.0 + 1e-9 for a, b in itertools.pairwise(exposures))
        finally:
            night.app.stop()

    def test_long_frames_of_a_fixed_30_s_solve_degrees_later(self, tmp_path: Path) -> None:
        """The comparison: `[survey.twilight] min_exposure_s` at 30 s turns the adaptation off.

        Each long frame saturates until the Sun is 9.6 degrees down, and the saturation guard keeps
        them from reporting clouds.
        """
        night = build_night(
            tmp_path,
            start="2026-01-01T16:20:00Z",  # the Sun 5.1 degrees down: nothing solves before -9
            cadence_s=180.0,
            config_extra="[survey.twilight]\nmin_exposure_s = 30.0\n",
        )
        try:
            night.run_until(lambda: solved_pointings(night), limit_s=2 * 3600.0, slice_s=60.0)
            first = solved_pointings(night)[0]
            assert sun_at(first.t_utc_ns) == pytest.approx(-9.98, abs=0.8)
            assert (
                sun_at(first.t_utc_ns) < -6.31 - 2.5
            )  # the adaptive run solved 3.7 degrees earlier
            sky = night.records("sky_quality")
            saturated = [q for q in sky if "saturated_sky" in q.flags]
            # The steps from -5.1 to -9.6 degrees, 3 a degree.
            assert len(saturated) >= 10
            early = [q for q in sky if q.t_utc_ns < first.t_utc_ns]
            assert all(q.cloud_fraction is None and "cloud" not in q.flags for q in early)
            assert all(q.cloud_fraction is None and "cloud" not in q.flags for q in saturated)
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


@pytest.mark.slow
class TestTheVisibilitySummary:
    """The nightly summary of a night that the station watched for its last 20 minutes."""

    def test_the_night_gets_a_summary_with_every_value_set_or_explained(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The split hour falls 20 minutes after the start, in the dark, because a whole night of
        # fast frames would take hours of CPU. A split hour in the dark is no advice for a station,
        # and the summary works the same at any hour. The dark set gives the survey frames a sky
        # brightness, so sky.dark fires after three frames and the verdict three frames later. A
        # cloud hides Polaris after ten minutes, and another one from two minutes before the split
        # to after it, so the last detection falls inside the night.
        night = build_night(
            tmp_path,
            night_split_utc_hour=18.5,
            dark_set=True,
            clouds=[
                cloud(600.0, 120.0, transmission=0.01),
                cloud(1050.0, 600.0, transmission=0.01),
            ],
            config_extra="[survey.darkness]\nframes = 3\nverdict_frames = 3\n",
        )
        split_ns = iso_to_utc_ns("2026-01-01T18:30:00Z")
        try:
            night.run_until(lambda: night.records("visibility_summary"), limit_s=1500.0)
            assert night.clock.utc_ns() > split_ns + 6 * NS_PER_S  # one analysis window after
            (summary,) = night.records("visibility_summary")
            visible = night.events("polaris.visible")
            hidden = night.events("polaris.hidden")
            (dark,) = night.events("sky.dark")
            (verdict,) = night.events("sky.clear_verdict")
            windows = [w for w in windows_with_r0(night) if w.t_utc_ns < split_ns]
        finally:
            night.app.stop()

        assert summary.night == "2025-12-31"
        assert summary.t_utc_ns == split_ns - 24 * 3600 * NS_PER_S
        # Every event falls inside the night.
        assert [e.t_utc_ns < split_ns for e in [*visible, *hidden, dark, verdict]] == [True] * 6
        # The station started 23 h 40 min into the night and found Polaris at once, so the first
        # detection is a bound. The second cloud hid Polaris before the end, while the station
        # watched, so the last detection is measured.
        assert summary.first_visible_utc_ns == visible[0].t_utc_ns
        assert summary.first_visible_sun_deg == visible[0].detail["sun_elevation_deg"]
        assert summary.first_censored is True
        assert [h.detail["reason"] for h in hidden] == ["star_missing", "star_missing"]
        assert summary.last_visible_utc_ns == hidden[-1].t_utc_ns
        assert summary.last_visible_sun_deg == hidden[-1].detail["sun_elevation_deg"]
        assert summary.last_censored is False
        measured_ns = sum(h.t_utc_ns - v.t_utc_ns for v, h in zip(visible, hidden, strict=True))
        assert summary.visible_hours == pytest.approx(measured_ns / 3.6e12, abs=1e-6)
        seeing_s = sum(w.duration_s for w in windows)
        assert summary.seeing_hours == pytest.approx(seeing_s / 3600.0, abs=1e-6)
        assert summary.dark_utc_ns == dark.t_utc_ns
        assert summary.dark_sun_deg == dark.detail["sun_elevation_deg"]
        assert summary.dark_sky_mag_arcsec2 == dark.detail["sky_mag_arcsec2"]
        assert summary.clear_share == verdict.detail["clear_share"] == 1.0
        # A new station has no reference zero point, so no frame has a transparency yet.
        assert summary.transparency_median is None
        missing = {name for name, value in summary.model_dump().items() if value is None}
        assert missing == {"transparency_median"}
        assert set(summary.quality) == missing  # every missing value says why
        assert set(summary.flags) <= {"moon"}

        code = main(["visibility", "stats", "--data-dir", str(tmp_path / "data")])
        out = capsys.readouterr().out
        assert code == 0
        lines = out.splitlines()
        row = re.split(r" {2,}", lines[lines.index("The newest night:") + 2])
        first = utc_ns_to_datetime(summary.first_visible_utc_ns).strftime("%H:%M")
        last = utc_ns_to_datetime(summary.last_visible_utc_ns).strftime("%H:%M")
        assert row[:5] == [
            "2025-12-31",
            f"{first}*",  # censored
            f"{summary.first_visible_sun_deg:+.1f}",
            last,
            f"{summary.last_visible_sun_deg:+.1f}",
        ]
        month = re.split(r" {2,}", lines[lines.index("By month:") + 2])
        # month, nights, unseen; the first: n, median, range, censored, bounds; then the last
        assert month[:3] == ["2025-12", "1", "0"]
        assert month[3:8] == ["0", "-", "-", "1", f"{summary.first_visible_sun_deg:+.1f}"]
        assert month[8] == "1"
        assert month[11:13] == ["0", "-"]
