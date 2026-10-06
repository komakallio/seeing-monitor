"""The rapid focus helper: readings of the star width from the frames of the rapid focus mode.

The tests feed the helper frames of a synthetic star at several widths (a Gaussian that the pixels
integrate, on a sky with photon and read noise, in the counts of the 12-bit ADC of the camera), at
the frame rate of the real stream (82 a second). The widths that the helper reports must fall as
the star comes into focus, the readings must come 20 times a second, and the work for one frame
must stay far below the frame period.
"""

from __future__ import annotations

import functools
import itertools
import json
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from seeingmon.analysis.base import FastUpdate, StarState
from seeingmon.clock import NS_PER_S, VirtualClock
from seeingmon.fastpath import FastPathConfig, create_fast_analyzer
from seeingmon.frames import Frame, Roi
from seeingmon.profile import load_profile
from seeingmon.services.core.alignment.focus import SPIKE_FACTOR
from seeingmon.services.core.alignment.rapid import (
    BEST_MIN,
    BEST_SPAN,
    HISTORY_LENGTH,
    SATURATED_NOTE,
    RapidFocusHelper,
    RapidHistory,
    RapidSnapshot,
    build_view,
)
from seeingmon.services.core.live import PolarisStream
from seeingmon.services.core.polaris import FWHM_PER_SIGMA, PolarisRenderer
from seeingmon.services.core.settings import PolarisSettings
from seeingmon.services.web.contract import (
    MAX_STATE_BYTES,
    PolarisState,
    RapidFocusView,
    RapidReadingsView,
    unpack_polaris_frame,
)
from tests.fastpath.helpers import box_integrated_gaussian, digitize, make_frame

PROFILE = load_profile("asi294mm-gs250")
SCALE = PROFILE.plate_scale_arcsec_per_px("bin1")  # 1.91 arcseconds per pixel
ROI = Roi(2008, 1347, 128, 128)
PERIOD_S = 1.0 / 82.0
STAR_X, STAR_Y = 64.3, 63.7  # in pixels of the ROI
SKY_E = 5.0  # electrons of sky and dark current in a pixel of a 2 ms frame
FLUX_E = 14_000.0  # Polaris at 2 ms
POOL = 24  # the noise of a frame repeats after this many frames of one star
T0_NS = 1_790_000_000 * NS_PER_S


def fast_analyzer() -> Any:
    return create_fast_analyzer(PROFILE, FastPathConfig(), "test")


def expected_fwhm(sigma_px: float) -> float:
    """The width of a Gaussian of `sigma_px` that the pixels integrate, in arcseconds."""
    return FWHM_PER_SIGMA * math.sqrt(sigma_px**2 + 1.0 / 12.0) * SCALE


def star_data(
    sigma_px: float,
    rng: np.random.Generator,
    *,
    flux_e: float = FLUX_E,
    x: float = STAR_X,
    y: float = STAR_Y,
) -> npt.NDArray[np.uint8] | npt.NDArray[np.uint16]:
    electrons = SKY_E + box_integrated_gaussian((128, 128), x, y, sigma_px, flux_e)
    return digitize(electrons, rng=rng)


@functools.cache
def star_pool(
    sigma_px: float, flux_e: float, x: float
) -> list[npt.NDArray[np.uint8] | npt.NDArray[np.uint16]]:
    """Frames of one star with different noise, made once because a frame takes 2 ms to make."""
    rng = np.random.default_rng(round(sigma_px * 1000) + round(x))
    return [star_data(sigma_px, rng, flux_e=flux_e, x=x) for _ in range(POOL)]


class Recorder:
    """A stand-in for the video of Polaris: it keeps what the helper offers."""

    def __init__(self) -> None:
        self.offered: list[tuple[Frame, FastUpdate, bool]] = []

    def offer(self, frame: Frame, update: FastUpdate, rapid: bool = False) -> None:
        self.offered.append((frame, update, rapid))


@dataclass
class Rig:
    helper: RapidFocusHelper
    video: Recorder
    clock: VirtualClock
    rng: np.random.Generator
    seq: int = 0
    stars: list[StarState | None] = field(default_factory=list)

    def feed(
        self,
        count: int,
        sigma_px: float,
        *,
        period_s: float = PERIOD_S,
        flux_e: float = FLUX_E,
        x: float = STAR_X,
        stream_id: int = 1,
        exposure_us: int = 2000,
        dropped_before: int = 0,
    ) -> None:
        """Push `count` frames of a star of one width."""
        pool = star_pool(sigma_px, flux_e, x)
        for _ in range(count):
            data = pool[self.seq % POOL]
            frame = make_frame(
                data,
                seq=self.seq,
                stream_id=stream_id,
                t_ns=T0_NS + round(self.seq * period_s * NS_PER_S),
                roi=ROI,
                exposure_us=exposure_us,
                dropped_before=dropped_before,
            )
            self.seq += 1
            self.stars.append(self.helper.push(frame))

    def readings(self) -> RapidSnapshot:
        return self.helper.snapshot()


@pytest.fixture
def rig() -> Rig:
    video = Recorder()
    clock = VirtualClock(T0_NS)
    helper = RapidFocusHelper(
        profile=PROFILE,
        clock=clock,
        polaris=video,
        kernel_setup=fast_analyzer().kernel_setup,
    )
    helper.begin_session()
    return Rig(helper, video, clock, np.random.default_rng(5))


# --- The history ------------------------------------------------------------------------------


def best_of(history: RapidHistory) -> float | None:
    """The best value, read through a call so that a type checker does not narrow it."""
    return history.best_arcsec


def add_all(history: RapidHistory, values: list[float], **fields: Any) -> list[Any]:
    options = {"peak_fraction": 0.3, "n_frames": 4, "saturated": False, **fields}
    return [
        history.add(1_000_000_000 + 50_000_000 * i, value, **options)
        for i, value in enumerate(values)
    ]


class TestTheHistory:
    def test_a_reading_gets_the_next_index_and_rounded_numbers(self) -> None:
        history = RapidHistory()
        first, second = add_all(history, [2.34567, 2.5])
        assert (first.index, second.index) == (1, 2)
        assert first.fwhm_arcsec == 2.346
        columns = history.columns()
        assert columns.index == [1, 2]
        assert columns.t_utc_ms == [1000, 1050]
        assert columns.fwhm_arcsec == [2.346, 2.5]
        assert columns.n_frames == [4, 4]
        assert columns.spike == [False, False]
        assert len(columns) == 2

    @pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
    def test_a_width_that_is_not_a_positive_number_gives_no_reading(self, value: float) -> None:
        history = RapidHistory()
        assert add_all(history, [value]) == [None]
        assert len(history.columns()) == 0

    def test_the_history_keeps_the_newest_600_readings(self) -> None:
        history = RapidHistory()
        add_all(history, [3.0] * (HISTORY_LENGTH + 100))
        columns = history.columns()
        assert len(columns) == HISTORY_LENGTH == 600
        assert columns.index[0] == 101
        assert columns.index[-1] == 700
        assert columns.index == list(range(101, 701))  # no gap, and the order holds

    def test_the_columns_are_copies(self) -> None:
        history = RapidHistory()
        add_all(history, [3.0])
        columns = history.columns()
        columns.fwhm_arcsec.append(9.0)
        assert history.columns().fwhm_arcsec == [3.0]

    def test_clear_starts_a_new_session(self) -> None:
        history = RapidHistory()
        add_all(history, [3.0] * 30)
        assert best_of(history) is not None
        session = history.session
        history.clear()
        assert history.session == session + 1
        assert len(history.columns()) == 0
        assert best_of(history) is None
        (reading,) = add_all(history, [3.0])
        assert reading.index == 1  # the numbers start again

    def test_a_history_holds_at_least_one_reading(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            RapidHistory(0)


class TestTheSpikeRule:
    def test_a_reading_above_twice_the_median_of_the_preceding_ten_is_a_spike(self) -> None:
        history = RapidHistory()
        readings = add_all(history, [3.0] * 12 + [3.0 * SPIKE_FACTOR + 0.01, 3.0 * SPIKE_FACTOR])
        assert [r.spike for r in readings[:12]] == [False] * 12
        assert readings[12].spike is True  # just above twice the median
        # The reading at exactly twice the median is not a spike: the median has moved by now.
        assert readings[13].spike is False

    def test_the_first_three_readings_of_a_session_are_never_spikes(self) -> None:
        history = RapidHistory()
        first = add_all(history, [1.0, 50.0, 90.0])
        assert [r.spike for r in first] == [False, False, False]
        history.clear()
        after = add_all(history, [1.0, 1.0, 1.0, 50.0])
        assert after[3].spike is True  # three readings precede it

    def test_a_lasting_change_stops_being_a_spike_when_it_fills_half_of_the_window(self) -> None:
        history = RapidHistory()
        readings = add_all(history, [2.0] * 10 + [6.0] * 10)
        flags = [r.spike for r in readings[10:]]
        # The median of the preceding ten follows: five values at the new level move it above
        # the threshold, and the values count again.
        assert flags == [True] * 5 + [False] * 5

    def test_the_rule_is_the_one_of_the_focus_history_of_the_normal_view(self) -> None:
        from seeingmon.services.core.alignment.focus import FocusHistory

        values = [2.0, 2.1, 2.0, 2.2, 4.6, 2.1, 2.0, 9.0, 2.2, 2.1, 2.0, 4.3, 2.0, 2.1]
        rapid = add_all(RapidHistory(), values)
        focus = FocusHistory()
        normal = [focus.add(i, i, value, 5) for i, value in enumerate(values)]
        assert [r.spike for r in rapid] == [p.spike for p in normal if p is not None]


class TestTheBestValue:
    def test_the_best_value_needs_ten_readings_that_count(self) -> None:
        history = RapidHistory()
        add_all(history, [3.0] * (BEST_MIN - 1))
        assert best_of(history) is None
        add_all(history, [3.0])
        assert best_of(history) == 3.0

    def test_the_best_value_is_the_lowest_median_of_the_last_twenty_readings(self) -> None:
        history = RapidHistory()
        add_all(history, [4.0] * 20)
        assert best_of(history) == 4.0
        add_all(history, [3.0] * 20)  # the focus improves
        assert best_of(history) == 3.0
        add_all(history, [5.0] * 40)  # and gets worse again
        assert best_of(history) == 3.0  # the best of the session stays

    def test_one_low_reading_does_not_set_the_best_value(self) -> None:
        history = RapidHistory()
        add_all(history, [3.0] * 15 + [1.0] + [3.0] * 15)
        assert best_of(history) == 3.0  # the median of the span ignores the single reading

    def test_a_spike_never_sets_the_best_value(self) -> None:
        history = RapidHistory()
        add_all(history, [3.0] * 12)
        # Twenty readings of a width that is far above the median are spikes only while they are
        # few, and they are never low, so build the case with a low value that is a spike:
        # a value of 0.2 is not above the median, so it is no spike. A spike is always high.
        readings = add_all(history, [9.0] * 3)
        assert [r.spike for r in readings] == [True, True, True]
        assert best_of(history) == 3.0

    def test_a_saturated_reading_never_sets_the_best_value(self) -> None:
        history = RapidHistory()
        add_all(history, [1.2] * 40, saturated=True)  # a saturated star reads too narrow
        assert best_of(history) is None
        # The first five readings of the real width are spikes, because the saturated ones before
        # them are so low, and then ten readings count.
        add_all(history, [3.0] * 15)
        assert best_of(history) == 3.0

    def test_a_restart_forgets_the_best_value_and_the_history_stays(self) -> None:
        history = RapidHistory()
        add_all(history, [2.0] * 20)
        assert best_of(history) == 2.0
        history.reset_best()
        assert best_of(history) is None
        assert len(history.columns()) == 20
        add_all(history, [5.0] * 20)  # a refocus to a worse place: five spikes, and then it counts
        assert best_of(history) == 5.0

    def test_the_best_value_is_reachable_with_noisy_readings(self) -> None:
        """The lowest single reading of 600 sits far below the star; the smoothed best does not."""
        rng = np.random.default_rng(3)
        values = list(3.0 * (1.0 + 0.06 * rng.standard_normal(600)))
        history = RapidHistory()
        add_all(history, values)
        assert min(values) < 2.6  # the lowest single reading is a poor best value
        best = best_of(history)
        assert best is not None
        assert best > 2.85  # the smoothed best is close to the star's value
        assert best <= float(np.median(values)) + 0.02
        assert BEST_SPAN == 20


# --- The readings from frames -----------------------------------------------------------------


class TestTheReadings:
    def test_there_are_twenty_readings_a_second_with_about_four_frames_each(self, rig: Rig) -> None:
        rig.feed(82 * 3, 0.8)
        snapshot = rig.readings()
        columns = snapshot.columns
        assert 56 <= len(columns) <= 60  # three seconds, and the last interval is still open
        assert set(columns.n_frames) <= {4, 5}
        assert sum(columns.n_frames) <= 246
        gaps = np.diff(columns.t_utc_ms)
        assert gaps.min() >= 35
        assert gaps.max() <= 65
        assert gaps.mean() == pytest.approx(50.0, abs=3.0)
        assert columns.index == list(range(1, len(columns) + 1))

    @pytest.mark.parametrize(
        ("fps", "frames_per_reading"), [(250, (12, 13)), (41, (2, 3)), (20, (1, 2))]
    )
    def test_the_rate_of_readings_holds_for_other_frame_rates(
        self, rig: Rig, fps: int, frames_per_reading: tuple[int, int]
    ) -> None:
        rig.feed(fps * 3, 0.8, period_s=1.0 / fps)
        columns = rig.readings().columns
        assert 56 <= len(columns) <= 60
        assert set(columns.n_frames) <= set(range(frames_per_reading[0], frames_per_reading[1] + 1))

    def test_a_slow_camera_gives_a_reading_for_each_frame(self, rig: Rig) -> None:
        rig.feed(30, 0.8, period_s=0.5)  # two frames a second
        columns = rig.readings().columns
        assert len(columns) == 29  # the last frame waits for the next one
        assert set(columns.n_frames) == {1}

    def test_the_width_falls_as_the_focus_improves(self, rig: Rig) -> None:
        sigmas = [3.0, 2.2, 1.6, 1.1, 0.8, 0.6]
        medians: list[float] = []
        for sigma in sigmas:
            first = len(rig.readings().columns)
            rig.feed(82, sigma)  # one second of frames at each width
            values = rig.readings().columns.fwhm_arcsec[
                first + 1 :
            ]  # skip the interval of the change
            medians.append(float(np.median(values)))
        assert medians == sorted(medians, reverse=True)  # strictly falling
        assert all(a > b for a, b in itertools.pairwise(medians))
        for sigma, measured in zip(sigmas, medians, strict=True):
            assert measured == pytest.approx(expected_fwhm(sigma), rel=0.07)

    def test_the_width_is_in_arcseconds_through_the_plate_scale_of_the_readout_mode(
        self, rig: Rig
    ) -> None:
        rig.feed(120, 1.5)
        value = float(np.median(rig.readings().columns.fwhm_arcsec))
        assert pytest.approx(1.91, abs=0.005) == SCALE  # bin1, and not the 3.82 of bin2
        assert value == pytest.approx(expected_fwhm(1.5), rel=0.05)
        assert 6.0 < value < 7.5  # a star of 1.5 pixels sigma is about 6.8 arcseconds wide

    def test_a_reading_is_the_median_of_the_frames_of_its_interval(self, rig: Rig) -> None:
        """One bad frame in four does not move a reading."""
        rig.feed(40, 1.0)
        base = float(np.median(rig.readings().columns.fwhm_arcsec))
        before = len(rig.readings().columns)
        for index in range(80):
            sigma = 4.0 if index % 5 == 2 else 1.0  # every fifth frame is far too wide
            rig.feed(1, sigma)
        values = rig.readings().columns.fwhm_arcsec[before:]
        assert float(np.median(values)) == pytest.approx(base, rel=0.08)

    def test_the_peak_is_the_brightest_pixel_as_a_share_of_the_full_scale(self, rig: Rig) -> None:
        rig.feed(60, 0.7)
        peaks = rig.readings().columns.peak_fraction
        assert all(0.1 < p < 0.6 for p in peaks)
        wide = Rig(rig.helper, rig.video, rig.clock, rig.rng)
        rig.helper.begin_session()
        wide.seq = 0
        wide.feed(60, 2.0)
        assert float(np.median(wide.readings().columns.peak_fraction)) < float(np.median(peaks))

    def test_a_frame_without_a_star_gives_no_reading_and_no_star(self, rig: Rig) -> None:
        rig.feed(60, 0.8)
        count = len(rig.readings().columns)
        sky = np.full((128, 128), 480, dtype=np.uint16)
        for seq in range(60, 120):
            frame = make_frame(sky, seq=seq, t_ns=T0_NS + round(seq * PERIOD_S * NS_PER_S), roi=ROI)
            assert rig.helper.push(frame) == StarState(found=False)
        assert len(rig.readings().columns) <= count + 1  # at most the interval that was open
        view = build_view(rig.readings())
        assert view.n_stars == 0

    def test_a_star_at_the_edge_of_the_roi_gives_no_reading(self, rig: Rig) -> None:
        rig.feed(60, 0.8, x=5.0)  # 5 pixels from the left edge, inside the aperture of the kernel
        assert len(rig.readings().columns) == 0
        star = rig.stars[-1]
        assert star is not None
        assert star.found
        assert star.edge_distance_px == pytest.approx(5.0, abs=0.5)  # the scheduler still sees it

    def test_the_helper_returns_the_star_in_sensor_pixels_for_the_roi(self, rig: Rig) -> None:
        rig.feed(5, 0.8)
        star = rig.stars[-1]
        assert star is not None
        assert star.found
        assert star.x_px == pytest.approx(ROI.x + STAR_X, abs=0.15)
        assert star.y_px == pytest.approx(ROI.y + STAR_Y, abs=0.15)
        assert star.edge_distance_px == pytest.approx(63.7, abs=0.2)
        assert star.peak_fraction is not None
        assert 0.1 < star.peak_fraction < 0.6
        assert rig.helper.last_star() == pytest.approx((star.x_px, star.y_px))

    def test_every_frame_goes_to_the_video_with_its_star(self, rig: Rig) -> None:
        rig.feed(20, 0.8)
        assert len(rig.video.offered) == 20
        assert all(rapid for _, _, rapid in rig.video.offered)
        for (frame, update, _), star in zip(rig.video.offered, rig.stars, strict=True):
            assert star is not None
            assert update.star == star
            assert frame.stream_id == 1
        assert rig.video.offered[3][0].seq == 3  # in order, and the frame itself

    def test_the_helper_gives_nothing_to_the_fast_analyzer_and_needs_no_seeing_value(
        self, rig: Rig
    ) -> None:
        analyzer = fast_analyzer()
        assert rig.helper._kernel_setup is not None
        rig.feed(10, 0.8)
        assert analyzer.frames_pushed == 0  # the helper never touches an analyzer

    def test_a_readout_mode_that_the_profile_does_not_know_gives_no_reading(self) -> None:
        helper = RapidFocusHelper(profile=PROFILE, clock=VirtualClock(T0_NS))
        helper.begin_session()
        rng = np.random.default_rng(1)
        for seq in range(80):
            frame = make_frame(
                star_data(1.0, rng),
                seq=seq,
                t_ns=T0_NS + round(seq * PERIOD_S * NS_PER_S),
                roi=ROI,
                mode="bin3",
            )
            helper.push(frame)
        assert len(helper.snapshot().columns) == 0  # the width needs a plate scale

    def test_the_kernel_setup_is_asked_once_for_each_stream_setting(self) -> None:
        asked: list[tuple[Any, ...]] = []
        real = fast_analyzer().kernel_setup

        def setup(*key: Any) -> Any:
            asked.append(key)
            return real(*key)

        helper = RapidFocusHelper(profile=PROFILE, clock=VirtualClock(T0_NS), kernel_setup=setup)
        helper.begin_session()
        rig = Rig(helper, Recorder(), VirtualClock(T0_NS), np.random.default_rng(2))
        rig.feed(30, 0.8)
        rig.feed(30, 0.8, exposure_us=1000)
        rig.feed(30, 0.8, exposure_us=1000)
        assert [key[2] for key in asked] == [2000, 1000]  # the exposure; the others stayed

    def test_the_helper_keeps_the_aperture_when_the_seeing_uses_the_weighted_centroid(
        self,
    ) -> None:
        """A narrow Gaussian weight can miss a defocused image, so the helper never takes it."""
        weighted = create_fast_analyzer(PROFILE, FastPathConfig(centroid="gaussian"), "test")
        asked: list[Any] = []

        def setup(*key: Any) -> Any:
            params, calibration = weighted.kernel_setup(*key)
            asked.append(params)
            return params, calibration

        helper = RapidFocusHelper(profile=PROFILE, clock=VirtualClock(T0_NS), kernel_setup=setup)
        helper.begin_session()
        rig = Rig(helper, Recorder(), VirtualClock(T0_NS), np.random.default_rng(2))
        rig.feed(30, 2.5)  # a defocused star
        assert asked
        assert asked[0].centroid_fwhm_px is not None  # the seeing would weight the centroid
        assert helper._setup is not None
        assert helper._setup.params.centroid_fwhm_px is None
        assert all(star is not None for star in rig.stars)

    def test_without_a_kernel_setup_the_helper_uses_the_default_aperture_and_no_correction(
        self,
    ) -> None:
        """A fake analyzer gives no kernel setup. The helper then has no noise model, so it keeps
        the widths of the kernel, which read 9% too wide for this star because of the median."""
        helper = RapidFocusHelper(profile=PROFILE, clock=VirtualClock(T0_NS))
        helper.begin_session()
        rig = Rig(helper, Recorder(), VirtualClock(T0_NS), np.random.default_rng(2))
        rig.feed(120, 1.2)
        value = float(np.median(helper.snapshot().columns.fwhm_arcsec))
        assert value == pytest.approx(expected_fwhm(1.2), rel=0.13)
        assert value > expected_fwhm(1.2)  # the bias of the median background shows

    def test_a_step_back_in_time_starts_the_grid_again_without_an_error(self, rig: Rig) -> None:
        rig.feed(100, 0.8)
        count = len(rig.readings().columns)
        rig.seq = 0
        rig.feed(100, 0.8, period_s=PERIOD_S)  # the clock stepped back to the start
        assert len(rig.readings().columns) > count + 15

    def test_a_gap_of_frames_leaves_no_empty_reading(self, rig: Rig) -> None:
        rig.feed(60, 0.8)
        rig.seq += 820  # ten seconds without a frame
        rig.feed(60, 0.8)
        columns = rig.readings().columns
        assert 0 not in columns.n_frames
        gap = np.diff(columns.t_utc_ms).max()
        assert gap > 9_000  # the curve shows the pause, and no reading fills it


class TestTheSpikesInFrames:
    def test_a_touch_of_the_telescope_makes_spikes_and_never_the_best_value(self, rig: Rig) -> None:
        rig.feed(246, 1.0)  # three seconds of steady focus
        steady = rig.readings()
        assert not any(steady.columns.spike)
        best = steady.best_arcsec
        assert best is not None
        before = len(steady.columns)
        rig.feed(41, 3.2)  # half a second of a bad star
        rig.feed(82, 1.0)
        after = rig.readings()
        spikes = [i for i, flag in enumerate(after.columns.spike) if flag]
        assert len(spikes) >= 3
        assert all(i >= before for i in spikes)
        assert after.best_arcsec is not None
        assert after.best_arcsec == pytest.approx(best, abs=0.15)  # a wide star sets nothing


class TestSaturation:
    def test_a_saturated_star_is_flagged_and_never_sets_the_best_value(self, rig: Rig) -> None:
        rig.feed(246, 0.8, flux_e=150_000.0)  # the core is far above the full scale
        snapshot = rig.readings()
        assert all(snapshot.columns.saturated)
        assert min(snapshot.columns.peak_fraction) > 0.98
        assert snapshot.best_arcsec is None
        view = build_view(snapshot)
        assert view.saturated is True
        assert view.quality["saturated"] == SATURATED_NOTE
        assert "shorten the exposure or lower the gain" in view.quality["saturated"]

    def test_a_star_below_the_saturation_level_is_not_flagged(self, rig: Rig) -> None:
        rig.feed(246, 0.8)
        snapshot = rig.readings()
        assert not any(snapshot.columns.saturated)
        assert snapshot.best_arcsec is not None
        assert "saturated" not in build_view(snapshot).quality

    def test_one_saturated_frame_flags_the_reading_that_holds_it(self, rig: Rig) -> None:
        rig.feed(40, 0.8)
        before = len(rig.readings().columns)
        rig.feed(1, 0.8, flux_e=150_000.0)
        rig.feed(40, 0.8)
        flags = rig.readings().columns.saturated[before:]
        assert flags.count(True) == 1


# --- The session ------------------------------------------------------------------------------


class TestTheSession:
    def test_frames_outside_a_session_are_ignored(self) -> None:
        video = Recorder()
        helper = RapidFocusHelper(profile=PROFILE, clock=VirtualClock(T0_NS), polaris=video)
        rig = Rig(helper, video, VirtualClock(T0_NS), np.random.default_rng(1))
        rig.feed(10, 0.8)
        assert rig.stars == [None] * 10  # no star, and the scheduler learns nothing
        assert video.offered == []
        assert helper.frames == 0
        assert not helper.active

    def test_a_new_session_starts_with_an_empty_history_and_a_new_grid(self, rig: Rig) -> None:
        rig.feed(120, 0.8)
        first = rig.readings()
        assert len(first.columns) > 20
        rig.helper.end_session("you stopped rapid focus")
        rig.helper.begin_session()
        assert len(rig.readings().columns) == 0
        rig.feed(120, 0.8)
        second = rig.readings()
        assert second.session == first.session + 1
        assert second.columns.index[0] == 1
        assert second.since_utc is not None

    def test_the_end_of_a_session_keeps_the_reason_and_stops_the_work(self, rig: Rig) -> None:
        rig.feed(120, 0.8)
        count = len(rig.readings().columns)
        rig.helper.end_session("the star was not in the window for 450 frames")
        snapshot = rig.readings()
        assert (snapshot.active, snapshot.ended) == (
            False,
            "the star was not in the window for 450 frames",
        )
        rig.feed(120, 0.8)  # frames that arrive late are ignored
        assert len(rig.readings().columns) == count
        view = build_view(snapshot)
        assert (view.active, view.ended_reason, view.readings) == (
            False,
            "the star was not in the window for 450 frames",
            None,
        )
        rig.helper.begin_session()
        assert rig.readings().ended is None

    def test_ending_twice_keeps_the_first_reason(self, rig: Rig) -> None:
        rig.helper.end_session("first")
        rig.helper.end_session("second")
        assert rig.readings().ended == "first"

    def test_the_best_value_can_restart_while_the_history_stays(self, rig: Rig) -> None:
        rig.feed(246, 1.0)
        assert rig.readings().best_arcsec is not None
        count = len(rig.readings().columns)
        rig.helper.reset_best()
        snapshot = rig.readings()
        assert snapshot.best_arcsec is None
        assert len(snapshot.columns) == count
        rig.feed(100, 1.6)  # a refocus to a worse place
        best = rig.readings().best_arcsec
        assert best is not None
        assert best == pytest.approx(expected_fwhm(1.6), rel=0.1)

    def test_a_reader_on_another_thread_sees_whole_snapshots(self, rig: Rig) -> None:
        seen: list[int] = []
        errors: list[BaseException] = []
        stop = threading.Event()

        def read() -> None:
            try:
                while not stop.is_set():
                    columns = rig.helper.snapshot().columns
                    lengths = {
                        len(columns.index),
                        len(columns.t_utc_ms),
                        len(columns.fwhm_arcsec),
                        len(columns.peak_fraction),
                        len(columns.n_frames),
                        len(columns.spike),
                        len(columns.saturated),
                    }
                    assert len(lengths) == 1
                    seen.append(len(columns))
            except BaseException as error:  # the test reports it below
                errors.append(error)

        reader = threading.Thread(target=read)
        reader.start()
        try:
            rig.feed(820, 1.0)
        finally:
            stop.set()
            reader.join(10.0)
        assert errors == []
        assert seen
        assert max(seen) > 100


# --- The view ---------------------------------------------------------------------------------


class TestTheView:
    def test_a_running_session_gives_the_numbers_of_the_newest_reading(self, rig: Rig) -> None:
        rig.feed(164, 1.0)
        snapshot = rig.readings()
        view = build_view(snapshot)
        columns = snapshot.columns
        assert (view.available, view.active, view.reason) == (True, True, None)
        assert view.fwhm_arcsec == columns.fwhm_arcsec[-1]
        assert view.best_fwhm_arcsec == snapshot.best_arcsec
        assert view.peak_fraction == columns.peak_fraction[-1]
        assert view.n_stars == 1
        assert (view.mode, view.exposure_us, view.gain) == ("bin1", 2000, 0)
        assert view.roi is not None
        assert (view.roi.width, view.roi.height) == (128, 128)
        assert view.scale_arcsec_px == pytest.approx(1.91, abs=0.005)
        assert view.since_utc is not None
        assert view.since_utc.endswith("Z")
        assert (view.spike, view.saturated) == (columns.spike[-1], False)
        assert view.quality == {}
        readings = view.readings
        assert readings is not None
        assert (readings.session, readings.reset) == (snapshot.session, True)
        assert readings.index == columns.index

    def test_before_the_first_reading_the_view_says_why_the_numbers_are_missing(
        self, rig: Rig
    ) -> None:
        view = build_view(rig.readings())
        assert view.active is True
        assert view.fwhm_arcsec is None
        assert "no reading yet" in view.quality["fwhm_arcsec"]
        assert "ten readings" in view.quality["best_fwhm_arcsec"]
        assert view.readings is not None
        assert view.readings.index == []
        assert view.n_stars == 0  # no frame has shown the star yet

    def test_the_view_survives_the_round_trip_through_json(self, rig: Rig) -> None:
        rig.feed(246, 1.0)
        view = build_view(rig.readings())
        again = RapidFocusView.model_validate_json(view.model_dump_json())
        assert again == view
        assert isinstance(again.readings, RapidReadingsView)

    def test_the_whole_history_fits_a_polaris_message(self, rig: Rig) -> None:
        rig.feed(82 * 31, 1.0)  # 31 seconds: the history is full
        view = build_view(rig.readings())
        assert view.readings is not None
        assert len(view.readings.index) == 600
        state = PolarisState.model_validate(
            {
                "seq": 1,
                "t_utc": "2026-10-01T21:00:00.123Z",
                "t_utc_ns": T0_NS,
                "stream_id": 1,
                "mode": "bin1",
                "exposure_us": 2000,
                "gain": 0,
                "roi": {"x": 2008, "y": 1347, "width": 128, "height": 128},
                "image_width": 128,
                "image_height": 128,
                "star": {"found": True},
                "stretch": {"black_dn": 480.0, "white_dn": 9000.0},
                "rapid_focus": view.model_dump(mode="json"),
            }
        )
        size = len(state.model_dump_json())
        assert size < MAX_STATE_BYTES // 2  # about 28 KB for 600 readings
        assert 20_000 < size < 40_000


# --- The video of Polaris ---------------------------------------------------------------------


class FakeSender:
    """A `StreamSender` that needs no connection: it keeps the payloads that it is given."""

    def __init__(self) -> None:
        self.payloads: list[bytes] = []
        self.closed = False

    def pump(self, timeout_s: float = 0.0) -> None:
        return None

    def has_credit(self, nbytes: int) -> bool:
        return True

    def send(self, payload: bytes | bytearray | memoryview) -> int:
        self.payloads.append(bytes(payload))
        return len(self.payloads)

    def close(self, reason: str = "") -> None:
        self.closed = True


def watched_stream(rig: Rig) -> tuple[PolarisStream, FakeSender]:
    stream = PolarisStream(
        PolarisSettings(),
        clock=rig.clock,
        renderer=PolarisRenderer(PolarisSettings(), scale_for=PROFILE.plate_scale_arcsec_per_px),
        live=None,
        rapid=lambda: build_view(rig.helper.snapshot()),
    )
    sender = FakeSender()
    stream.attach(sender)  # type: ignore[arg-type]
    rig.helper._polaris = stream
    return stream, sender


class TestTheVideoOfPolaris:
    def test_the_frames_of_the_mode_reach_the_video_with_the_state_of_the_mode(
        self, rig: Rig
    ) -> None:
        stream, sender = watched_stream(rig)
        rig.feed(120, 1.0)
        assert stream.frames_kept > 10  # thinned to 20 a second
        slot = stream._slot
        assert slot is not None
        assert slot.rapid is True
        frame = unpack_polaris_frame(stream.process(slot))
        state = frame.state
        assert state.rapid_focus is not None
        assert state.rapid_focus.active is True
        assert state.rapid_focus.readings is not None
        assert len(state.rapid_focus.readings.index) > 15
        assert state.live_seeing is None
        assert "rapid focus" in state.quality["live_seeing"]
        assert state.star.found is True
        assert state.star.fwhm_arcsec == pytest.approx(expected_fwhm(1.0), rel=0.15)
        assert state.scale_arcsec_px == pytest.approx(1.91, abs=0.005)
        assert len(sender.payloads) == 1

    def test_a_frame_of_the_fast_stream_carries_no_state_of_the_mode(self, rig: Rig) -> None:
        stream, _ = watched_stream(rig)
        data = star_data(1.0, rig.rng)
        frame = make_frame(data, seq=0, t_ns=T0_NS, roi=ROI)
        stream.offer(frame, FastUpdate(star=StarState(True, 2072.0, 1411.0, 0.3, 60.0)))
        slot = stream._slot
        assert slot is not None
        assert slot.rapid is False
        state = unpack_polaris_frame(stream.process(slot)).state
        assert state.rapid_focus is None
        assert "no rolling seeing value yet" in state.quality["live_seeing"]

    def test_a_stream_without_a_source_of_the_state_still_shows_the_video(self, rig: Rig) -> None:
        stream = PolarisStream(
            PolarisSettings(),
            clock=rig.clock,
            renderer=PolarisRenderer(PolarisSettings(), scale_for=lambda mode: SCALE),
        )
        stream.attach(FakeSender())  # type: ignore[arg-type]
        rig.helper._polaris = stream
        rig.feed(10, 1.0)
        slot = stream._slot
        assert slot is not None
        state = unpack_polaris_frame(stream.process(slot)).state
        assert state.rapid_focus is None

    def test_the_message_with_a_full_history_stays_small(self, rig: Rig) -> None:
        stream, _ = watched_stream(rig)
        rig.feed(82 * 31, 1.0)
        slot = stream._slot
        assert slot is not None
        payload = stream.process(slot)
        assert len(payload) < MAX_STATE_BYTES
        state = unpack_polaris_frame(payload).state.model_dump_json()
        assert len(state) < 40_000
        # The same state as JSON: the message that `core` sends 20 times a second.
        assert json.loads(state)["rapid_focus"]["readings"]["index"][-1] > 600


# --- The cost ---------------------------------------------------------------------------------


def best_cost_us(helper: RapidFocusHelper, frames: list[Frame], rounds: int = 5) -> float:
    """The best mean CPU time of one `push` over several rounds, in microseconds.

    The CPU time of the thread leaves out the time that other jobs of a busy machine take from
    it, which the wall clock counts. The wall time of the same rounds is printed too.
    """
    best, best_wall = float("inf"), float("inf")
    for _ in range(rounds):
        helper.begin_session()
        wall, cpu = time.perf_counter(), time.thread_time()
        for frame in frames:
            helper.push(frame)
        best = min(best, (time.thread_time() - cpu) / len(frames))
        best_wall = min(best_wall, (time.perf_counter() - wall) / len(frames))
    print(f"the wall time of a frame was {best_wall * 1e6:.1f} microseconds")
    return best * 1e6


class TestTheCost:
    @staticmethod
    def frames(count: int = 4000) -> list[Frame]:
        """Frames of one star; a round of 4000 is long enough for the clock of the thread."""
        rng = np.random.default_rng(9)
        pool = [star_data(1.1, rng) for _ in range(40)]  # the noise differs from frame to frame
        return [
            make_frame(
                pool[seq % 40],
                seq=seq,
                t_ns=T0_NS + round(seq * PERIOD_S * NS_PER_S),
                roi=ROI,
            )
            for seq in range(count)
        ]

    def test_the_work_for_one_frame_stays_under_150_microseconds(self) -> None:
        """The scheduler thread pays this for every frame, 82 times a second (CPU time).

        The helper runs the kernel, the interval bookkeeping, and the offer to a video that a page
        watches (the slot copy of every fourth frame). A development machine takes about 50
        microseconds, and a Raspberry Pi 4 four to six times as much.
        """
        stream = PolarisStream(
            PolarisSettings(),
            clock=VirtualClock(T0_NS),
            renderer=PolarisRenderer(PolarisSettings(), scale_for=lambda mode: SCALE),
        )
        stream.attach(FakeSender())  # type: ignore[arg-type]
        helper = RapidFocusHelper(
            profile=PROFILE,
            clock=VirtualClock(T0_NS),
            polaris=stream,
            kernel_setup=fast_analyzer().kernel_setup,
        )
        cost = best_cost_us(helper, self.frames())
        print(f"the helper takes {cost:.1f} microseconds of CPU time for a frame")
        assert cost < 150.0

    def test_the_cost_without_a_viewer_is_the_kernel_and_a_few_microseconds(self) -> None:
        stream = PolarisStream(
            PolarisSettings(),
            clock=VirtualClock(T0_NS),
            renderer=PolarisRenderer(PolarisSettings(), scale_for=lambda mode: SCALE),
        )
        helper = RapidFocusHelper(
            profile=PROFILE,
            clock=VirtualClock(T0_NS),
            polaris=stream,  # no viewer: `offer` returns at once
            kernel_setup=fast_analyzer().kernel_setup,
        )
        cost = best_cost_us(helper, self.frames())
        print(f"the helper takes {cost:.1f} microseconds of CPU time for a frame without a viewer")
        assert cost < 150.0
