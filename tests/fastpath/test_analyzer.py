"""The `FastPathAnalyzer`: the behavior tests of `FakeFastAnalyzer`, adapted, and more.

The scripted fake in `tests/test_analysis_fakes.py` tracks the brightest pixel of a one-pixel star.
A one-pixel star is what a hot pixel looks like, so the frames here carry a Gaussian star of 1 px
sigma, and the numbers that depend on the star (the peak, the width) differ from the fake's.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import numpy.typing as npt
import pytest

from seeingmon.analysis import NO_STAR, FastAnalyzer, FastContext
from seeingmon.clock import VirtualClock
from seeingmon.fastpath import (
    ALGORITHM_REVISION,
    FastPathAnalyzer,
    FastPathConfig,
    create_fast_analyzer,
    models,
)
from seeingmon.frames import (
    ActiveStream,
    Frame,
    FrameFlag,
    PixelFormat,
    Roi,
    StreamConfig,
    TimeQuality,
)
from seeingmon.profile import Profile
from seeingmon.records import SeeingWindowRecord
from seeingmon.records.segments import segment_dtype
from seeingmon.testing import FakeCameraDriver
from tests.fastpath.helpers import box_integrated_gaussian, digitize, make_frame

ROI = Roi(200, 300, 64, 64)
CONFIG = StreamConfig(
    mode="bin1", exposure_us=1_000_000, gain=120, roi=ROI, pixel_format=PixelFormat.RAW16
)
STAR_PEAK_DN = 5000.0  # of 16 bits: a third of a 14-bit full scale


def gaussian_star(config: StreamConfig, roi: Roi, seq: int) -> npt.NDArray[np.uint16]:
    """A star of 1 px sigma at sensor position (232, 332), the center of `ROI`, on a flat sky."""
    electrons = box_integrated_gaussian(
        (roi.height, roi.width), 232.0 - roi.x, 332.0 - roi.y, 1.0, 30_000.0
    )
    counts = digitize(electrons * 1.0, e_per_adu=1.0, adc_bits=14, offset_adu=25.0)
    return np.asarray(counts, dtype=np.uint16)


def flat(config: StreamConfig, roi: Roi, seq: int) -> npt.NDArray[np.uint16]:
    return np.full((roi.height, roi.width), 100, dtype=np.uint16)


def stream_of(driver: FakeCameraDriver, config: StreamConfig = CONFIG) -> ActiveStream:
    active = driver.configure(config)
    driver.start()
    return active


def read(driver: FakeCameraDriver, count: int) -> list[Frame]:
    return [driver.read_frame(timeout_s=5.0) for _ in range(count)]


@pytest.fixture
def driver() -> FakeCameraDriver:
    fake = FakeCameraDriver(VirtualClock(), frame_factory=gaussian_star)
    fake.open()
    return fake


@pytest.fixture
def analyzer(profile: Profile) -> FastPathAnalyzer:
    return FastPathAnalyzer(profile, station_id="s1")


def test_the_analyzer_satisfies_the_protocol(profile: Profile) -> None:
    assert isinstance(FastPathAnalyzer(profile), FastAnalyzer)
    assert isinstance(create_fast_analyzer(profile), FastAnalyzer)


class TestMeasure:
    """`measure` serves the search bursts: the star and its SNRs, and nothing else changes."""

    def test_measure_finds_the_star_with_its_snr_and_leaves_no_trace(
        self, analyzer: FastPathAnalyzer, driver: FakeCameraDriver
    ) -> None:
        stream_of(driver)
        frames = read(driver, 5)
        stars = [analyzer.measure(frame, (232.0, 332.0), 20.0) for frame in frames]
        for star in stars:
            assert star.found
            # 0.02 px: the matched filter places the star on a grid of a quarter pixel, refined.
            assert star.x_px == pytest.approx(232.0, abs=0.02)
            assert star.y_px == pytest.approx(332.0, abs=0.02)
            assert star.snr is not None
            assert star.snr > 50.0  # 30,000 e- at gain 120 on a dark sky
            assert star.matched_snr is not None
            # The star (1 px sigma) is wider than the filter of the Airy FWHM (0.57 px sigma).
            assert star.matched_snr > 40.0
        assert analyzer.frames_measured == 5
        assert analyzer.frames_pushed == 0
        assert analyzer.drain_metrics() is None  # no metric row
        assert analyzer.flush() == ()  # no window
        assert analyzer.star == NO_STAR  # the state of `push` stays
        assert analyzer.live is None

    def test_measure_looks_only_within_the_radius(
        self, analyzer: FastPathAnalyzer, driver: FakeCameraDriver
    ) -> None:
        stream_of(driver)
        (frame,) = read(driver, 1)
        assert analyzer.measure(frame, (232.0 + 25.0, 332.0), 20.0) == NO_STAR
        assert analyzer.measure(frame, (232.0 + 15.0, 332.0), 20.0).found
        assert analyzer.measure(frame, (232.0 + 25.0, 332.0)).found  # no radius: the whole frame

    def test_measure_of_a_frame_without_a_star_is_no_star(self, profile: Profile) -> None:
        fake = FakeCameraDriver(VirtualClock(), frame_factory=flat)
        fake.open()
        stream_of(fake)
        analyzer = FastPathAnalyzer(profile)
        assert analyzer.measure(fake.read_frame(5.0), (232.0, 332.0)) == NO_STAR

    def test_measure_between_pushes_changes_neither_the_window_nor_the_tracking(
        self, analyzer: FastPathAnalyzer, driver: FakeCameraDriver
    ) -> None:
        analyzer.begin_stream(stream_of(driver))
        first, probed, second = read(driver, 3)
        found = analyzer.push(first).star
        assert analyzer.measure(probed, None).found
        assert analyzer.push(second).star == found  # the same star, tracked as before
        (window,) = analyzer.flush()
        assert window.n_frames == 2  # the measured frame is in no window

    def test_push_reports_the_snr_of_a_found_star(
        self, analyzer: FastPathAnalyzer, driver: FakeCameraDriver
    ) -> None:
        analyzer.begin_stream(stream_of(driver))
        star = analyzer.push(driver.read_frame(5.0)).star
        assert star.snr is not None
        assert star.snr > 50.0
        assert star.matched_snr is not None
        assert star.matched_snr > 40.0  # the star is wider than the filter, as above


class TestBehaviorOfTheFake:
    def test_closes_a_window_every_sixty_seconds_of_frame_time(
        self, profile: Profile, driver: FakeCameraDriver
    ) -> None:
        analyzer = create_fast_analyzer(profile, station_id="s1")
        stream = stream_of(driver)
        analyzer.begin_stream(stream)
        windows: list[SeeingWindowRecord] = []
        for frame in read(driver, 140):
            windows += analyzer.push(frame).windows
        windows += analyzer.flush()
        assert [w.n_frames for w in windows] == [60, 60, 20]
        first, _, last = windows
        assert (first.station_id, first.profile_id, first.stream_id) == (
            "s1",
            profile.id,
            stream.stream_id,
        )
        assert first.readout_mode == "bin1"
        assert first.exposure_us == 1_000_000
        assert first.duration_s == pytest.approx(60.0)
        assert first.frame_rate_hz == pytest.approx(1.0, rel=0.02)
        assert first.valid_fraction == 1.0
        assert first.flags == []
        assert windows[1].t_utc_ns - first.t_utc_ns == 60 * 1_000_000_000
        assert last.flags == ["partial"]
        assert first.sensor_temperature_c == pytest.approx(18.0)

    def test_tracks_the_star_in_sensor_coordinates(
        self, analyzer: FastPathAnalyzer, driver: FakeCameraDriver
    ) -> None:
        analyzer.begin_stream(stream_of(driver))
        update = analyzer.push(driver.read_frame(5.0))
        assert update.star.found
        assert update.star.x_px == pytest.approx(ROI.x + 32, abs=0.01)
        assert update.star.y_px == pytest.approx(ROI.y + 32, abs=0.01)
        assert update.star.edge_distance_px == pytest.approx(32, abs=0.01)
        # The peak pixel of a 1 px Gaussian of 30,000 counts: 30,000 x 0.1466 counts of a 14-bit
        # ADC, shifted by 2 bits in the container, over the full scale of 65,532.
        assert update.star.peak_fraction == pytest.approx(
            (30_000 * 0.1466 + 25) * 4 / 65_532, rel=0.03
        )

    def test_a_missing_star_is_reported_and_the_state_resets_on_a_new_stream(
        self, profile: Profile
    ) -> None:
        fake = FakeCameraDriver(VirtualClock(), frame_factory=flat)
        fake.open()
        analyzer = FastPathAnalyzer(profile)
        analyzer.begin_stream(stream_of(fake))
        assert analyzer.push(fake.read_frame(5.0)).star == NO_STAR

    def test_begin_stream_closes_the_open_window_as_partial(
        self, analyzer: FastPathAnalyzer, driver: FakeCameraDriver
    ) -> None:
        analyzer.begin_stream(stream_of(driver))
        for frame in read(driver, 5):
            analyzer.push(frame)
        closed = analyzer.begin_stream(stream_of(driver))
        assert [(w.n_frames, w.flags) for w in closed] == [(5, ["partial"])]
        assert analyzer.star == NO_STAR
        assert analyzer.begin_stream(stream_of(driver)) == ()

    def test_a_frame_from_another_stream_starts_that_stream(
        self, analyzer: FastPathAnalyzer, driver: FakeCameraDriver
    ) -> None:
        first = stream_of(driver)
        analyzer.begin_stream(first)
        for frame in read(driver, 3):
            analyzer.push(frame)
        second = stream_of(driver)
        update = analyzer.push(driver.read_frame(5.0))
        assert [(w.stream_id, w.n_frames) for w in update.windows] == [(first.stream_id, 3)]
        assert second.stream_id != first.stream_id

    def test_the_context_adds_flags_and_values_to_closing_windows(
        self, analyzer: FastPathAnalyzer, driver: FakeCameraDriver
    ) -> None:
        analyzer.begin_stream(stream_of(driver))
        analyzer.set_context(
            FastContext(
                flags=frozenset({"cloud", "heater_on"}), heater_duty=0.25, zenith_angle_deg=35.5
            )
        )
        for frame in read(driver, 3):
            analyzer.push(frame)
        (window,) = analyzer.flush()
        assert window.flags == ["cloud", "heater_on", "partial"]
        assert (window.heater_duty, window.zenith_angle_deg) == (0.25, 35.5)

    def test_dropped_frames_make_a_window_degraded(
        self, analyzer: FastPathAnalyzer, driver: FakeCameraDriver
    ) -> None:
        analyzer.begin_stream(stream_of(driver))
        frames = read(driver, 10)
        analyzer.push(frames[0])
        analyzer.push(replace(frames[1], dropped_before=4))
        for frame in frames[2:]:
            analyzer.push(frame)
        (window,) = analyzer.flush()
        assert window.n_dropped == 4
        assert window.valid_fraction == pytest.approx(10 / 14)
        assert "degraded" in window.flags

    def test_metrics_use_the_frame_record_dtype_and_drain_once(
        self, analyzer: FastPathAnalyzer, driver: FakeCameraDriver
    ) -> None:
        analyzer.begin_stream(stream_of(driver))
        assert analyzer.drain_metrics() is None
        for frame in read(driver, 7):
            analyzer.push(frame)
        rows = analyzer.drain_metrics()
        assert rows is not None
        assert rows.dtype == segment_dtype("frame")
        assert rows["seq"].tolist() == list(range(7))
        assert rows["cx_px"][0] == pytest.approx(ROI.x + 32, abs=0.01)
        assert rows["peak_dn"][0] == pytest.approx((30_000 * 0.1466 + 25) * 4, rel=0.03)
        assert analyzer.drain_metrics() is None

    def test_windows_validate_as_records(
        self, analyzer: FastPathAnalyzer, driver: FakeCameraDriver
    ) -> None:
        analyzer.begin_stream(stream_of(driver))
        analyzer.push(driver.read_frame(5.0))
        (window,) = analyzer.flush()
        assert window.record_type == "seeing_window"
        assert type(window).from_row(window.to_row()) == window


class TestMetricsRows:
    def test_a_row_holds_the_star_in_sensor_coordinates_with_nan_for_a_missing_star(
        self, profile: Profile
    ) -> None:
        analyzer = FastPathAnalyzer(profile)
        fake = FakeCameraDriver(VirtualClock(), frame_factory=gaussian_star)
        fake.open()
        analyzer.begin_stream(stream_of(fake))
        analyzer.push(fake.read_frame(5.0))
        blank = FakeCameraDriver(VirtualClock(), frame_factory=flat)
        blank.open()
        stream_of(blank)
        later = read(blank, 1)[0]
        analyzer.push(replace(later, seq=1, t_utc_ns=later.t_utc_ns + 1_000_000_000))
        rows = analyzer.drain_metrics()
        assert rows is not None
        assert np.isfinite(rows["cx_px"][0])
        assert np.isnan(rows["cx_px"][1])
        assert rows["flags"][1] & 128  # no_star
        assert rows["flux_e"][0] > 0.0
        assert rows["bg_dn"][0] == pytest.approx(25 * 4, abs=2)

    def test_the_frame_flags_and_the_drops_go_into_the_row(
        self, analyzer: FastPathAnalyzer, driver: FakeCameraDriver
    ) -> None:
        analyzer.begin_stream(stream_of(driver))
        frame = driver.read_frame(5.0)
        analyzer.push(
            replace(frame, flags=FrameFlag.REPLAYED | FrameFlag.TIME_INVALID, dropped_before=3)
        )
        rows = analyzer.drain_metrics()
        assert rows is not None
        assert rows["flags"][0] & 0x1F == int(FrameFlag.REPLAYED | FrameFlag.TIME_INVALID)
        assert rows["dropped_before"][0] == 3
        assert rows["t_err_us"][0] == 1

    def test_a_buffer_that_nobody_drains_stays_bounded(self, profile: Profile) -> None:
        analyzer = FastPathAnalyzer(profile, FastPathConfig(max_buffered_metrics=100))
        fake = FakeCameraDriver(VirtualClock(), frame_factory=gaussian_star)
        fake.open()
        analyzer.begin_stream(stream_of(fake))
        for frame in read(fake, 300):
            analyzer.push(frame)
        rows = analyzer.drain_metrics()
        assert rows is not None
        assert 50 <= len(rows) <= 100
        assert rows["seq"][-1] == 299
        assert analyzer.metrics_dropped == 300 - len(rows)


class TestWindowFlags:
    def test_a_stream_with_an_invalid_clock_carries_time_invalid(
        self, analyzer: FastPathAnalyzer, driver: FakeCameraDriver
    ) -> None:
        analyzer.begin_stream(stream_of(driver))
        frames = read(driver, 6)
        analyzer.push(replace(frames[0], t_quality=TimeQuality.INVALID))
        for frame in frames[1:]:
            analyzer.push(frame)
        (window,) = analyzer.flush()
        assert "time_invalid" in window.flags

    def test_a_flag_in_the_frame_marks_the_window_time_invalid_too(
        self, analyzer: FastPathAnalyzer, driver: FakeCameraDriver
    ) -> None:
        analyzer.begin_stream(stream_of(driver))
        analyzer.push(replace(driver.read_frame(5.0), flags=FrameFlag.TIME_INVALID))
        (window,) = analyzer.flush()
        assert "time_invalid" in window.flags

    def test_a_saturated_star_sets_the_flag_and_the_fraction(self, profile: Profile) -> None:
        def bright(config: StreamConfig, roi: Roi, seq: int) -> npt.NDArray[np.uint16]:
            return np.asarray(
                np.minimum(gaussian_star(config, roi, seq).astype(np.int64) * 8, 65_532),
                dtype=np.uint16,
            )

        fake = FakeCameraDriver(VirtualClock(), frame_factory=bright)
        fake.open()
        analyzer = FastPathAnalyzer(profile)
        analyzer.begin_stream(stream_of(fake))
        for frame in read(fake, 5):
            update = analyzer.push(frame)
            assert update.star.peak_fraction == pytest.approx(1.0, abs=0.01)
        (window,) = analyzer.flush()
        assert "saturated" in window.flags
        assert window.saturated_fraction == 1.0

    def test_a_few_saturated_frames_do_not_set_the_flag(
        self, profile: Profile, driver: FakeCameraDriver
    ) -> None:
        analyzer = FastPathAnalyzer(profile, FastPathConfig(window_s=200.0))
        analyzer.begin_stream(stream_of(driver))
        for index, frame in enumerate(read(driver, 100)):
            if index == 0:
                data = np.asarray(frame.data, dtype=np.uint16).copy()
                data[32, 32] = 65_530
                frame = replace(frame, data=data)
            analyzer.push(frame)
        (window,) = analyzer.flush()
        assert window.saturated_fraction == pytest.approx(0.01)
        assert "saturated" not in window.flags

    def test_unknown_context_flags_are_rejected_at_the_call(
        self, analyzer: FastPathAnalyzer
    ) -> None:
        with pytest.raises(ValueError, match="unknown window flags: moon"):
            analyzer.set_context(FastContext(flags=frozenset({"moon", "cloud"})))


class TestStreams:
    def test_the_roi_moves_without_aliasing_into_motion(self, profile: Profile) -> None:
        """A ROI move changes the pixel coordinates in the frame and not the sensor coordinates."""
        fake = FakeCameraDriver(VirtualClock(), frame_factory=gaussian_star)
        fake.open()
        analyzer = FastPathAnalyzer(profile)
        analyzer.begin_stream(stream_of(fake))
        first = analyzer.push(fake.read_frame(5.0))
        fake.move_roi(ROI.x - 10, ROI.y + 6)
        second = analyzer.push(fake.read_frame(5.0))
        assert first.star.x_px == pytest.approx(second.star.x_px, abs=0.01)
        assert first.star.y_px == pytest.approx(second.star.y_px, abs=0.01)
        # The star is now 42 px from the left edge of the ROI and 22 px from the right edge.
        assert second.star.edge_distance_px == pytest.approx(22.0, abs=0.01)

    def test_eight_bit_frames_work(self, profile: Profile) -> None:
        def eight_bit(config: StreamConfig, roi: Roi, seq: int) -> npt.NDArray[np.uint8]:
            electrons = box_integrated_gaussian(
                (roi.height, roi.width), 232.0 - roi.x, 332.0 - roi.y, 1.0, 30_000.0
            )
            return np.asarray(
                digitize(electrons, e_per_adu=1.0, adc_bits=14, container_bits=8), dtype=np.uint8
            )

        fake = FakeCameraDriver(VirtualClock(), frame_factory=eight_bit)
        fake.open()
        analyzer = FastPathAnalyzer(profile)
        config = replace(CONFIG, pixel_format=PixelFormat.RAW8)
        analyzer.begin_stream(stream_of(fake, config))
        update = analyzer.push(fake.read_frame(5.0))
        assert update.star.found
        assert update.star.x_px == pytest.approx(232.0, abs=0.05)
        # Saturation means 255 in an 8-bit container.
        assert update.star.peak_fraction == pytest.approx(
            (30_000 * 0.1466 + 25) / 64 / 255, rel=0.1
        )
        rows = analyzer.drain_metrics()
        assert rows is not None
        assert rows["peak_dn"][0] <= 255

    def test_a_readout_mode_that_the_profile_does_not_know_still_tracks_the_star(
        self, profile: Profile
    ) -> None:
        fake = FakeCameraDriver(
            VirtualClock(), frame_factory=gaussian_star, full_frames={"bin9": (4000, 3000)}
        )
        fake.open()
        analyzer = FastPathAnalyzer(profile)
        analyzer.begin_stream(stream_of(fake, replace(CONFIG, mode="bin9")))
        for frame in read(fake, 5):
            update = analyzer.push(frame)
            assert update.star.found
        (window,) = analyzer.flush()
        assert window.r0_cm is None
        assert window.quality is not None
        assert window.quality["r0_cm"] == "the readout mode is not in the profile"

    def test_a_short_window_gets_no_seeing_statistics_and_says_why(
        self, analyzer: FastPathAnalyzer, driver: FakeCameraDriver
    ) -> None:
        analyzer.begin_stream(stream_of(driver))
        for frame in read(driver, 10):
            analyzer.push(frame)
        (window,) = analyzer.flush()
        assert window.seeing_fwhm_arcsec is None
        assert window.scintillation_index is None
        assert window.quality is not None
        assert window.quality["seeing_fwhm_arcsec"] == "too few usable frames"
        # The star statistics need no minimum.
        assert window.peak_mean_dn is not None
        assert window.flux_mean_e is not None
        assert window.background_mean_dn == pytest.approx(25 * 4, abs=2)

    def test_the_window_length_is_configurable(
        self, profile: Profile, driver: FakeCameraDriver
    ) -> None:
        analyzer = FastPathAnalyzer(profile, FastPathConfig(window_s=10.0, min_window_s=5.0))
        analyzer.begin_stream(stream_of(driver))
        windows: list[SeeingWindowRecord] = []
        for frame in read(driver, 35):
            windows += analyzer.push(frame).windows
        assert [w.n_frames for w in windows] == [10, 10, 10]


def bright_sky_window(
    profile: Profile, sky_e: float, flux: float, centroid: str, max_noise_bias: float = 0.05
) -> SeeingWindowRecord:
    """The first window of a star of constant flux whose position jumps from frame to frame.

    The frames hold the simulator's image of Polaris (1.33 px FWHM in bin1) at a position that
    jumps by a white Gaussian of 0.25 px per axis, about the image motion of an `r0` of 10 cm. A
    window of 6 s holds 531 frames.
    """
    config = FastPathConfig(
        window_s=6.0, min_window_s=3.0, centroid=centroid, max_noise_bias=max_noise_bias
    )
    analyzer = FastPathAnalyzer(profile, config)
    rng = np.random.default_rng(31)
    sigma_px = 1.333 / 2.3548
    closed: list[SeeingWindowRecord] = []
    seq = 0
    while not closed:
        x, y = 64.0 + rng.normal(0.0, 0.25), 63.5 + rng.normal(0.0, 0.25)
        electrons = box_integrated_gaussian((128, 128), x, y, sigma_px, flux) + sky_e
        frame = make_frame(digitize(electrons, rng=rng), seq=seq, exposure_us=1226)
        closed += analyzer.push(frame).windows
        seq += 1
    return closed[0]


class TestTheNoisyFlag:
    """`noisy` marks a window whose noise share lets the error of the noise model bias `r0`.

    The star has 8,000 e- on 4,300 e- of sky in the model's daylight, and 14,000 e- on a dark sky
    (`bright_sky_window`).
    """

    def window(
        self, profile: Profile, sky_e: float, flux: float, centroid: str
    ) -> SeeingWindowRecord:
        return bright_sky_window(profile, sky_e, flux, centroid)

    def test_the_aperture_in_daylight_is_noisy(self, profile: Profile) -> None:
        """The aperture's noise is about three times the motion, so a model error of 6% predicts a
        bias of `r0` of about 9%, above the limit of 5%."""
        window = self.window(profile, 4_300.0, 8_000.0, "aperture")
        assert "noisy" in window.flags
        assert window.r0_cm is not None
        assert window.centroid_noise_px is not None
        assert window.centroid_noise_px**2 > 0.15  # px^2, against 0.06 of motion

    def test_the_weighted_centroid_in_daylight_is_not(self, profile: Profile) -> None:
        window = self.window(profile, 4_300.0, 8_000.0, "gaussian")
        assert "noisy" not in window.flags
        assert window.centroid_noise_px is not None
        assert window.centroid_noise_px**2 < 0.003  # px^2
        # 3 Airy FWHM of bin1. The record names the centroid, because it changes the estimate.
        assert "centroid=gaussian 4.0 px" in window.provenance["assumptions"]

    def test_the_limit_comes_from_the_configuration(self, profile: Profile) -> None:
        """The aperture's daylight window predicts a bias of about 12%, under a limit of 0.5."""
        window = bright_sky_window(profile, 4_300.0, 8_000.0, "aperture", max_noise_bias=0.5)
        assert "noisy" not in window.flags

    def test_each_centroid_takes_the_error_of_its_own_model(
        self, profile: Profile, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An error of 100 for the weighted centroid's model makes its small share of noise, about
        0.03, predict a bias of about 0.6, while the aperture keeps its own error."""
        monkeypatch.setitem(models.NOISE_MODEL_ERROR, "gaussian", 100.0)
        assert "noisy" in self.window(profile, 4_300.0, 8_000.0, "gaussian").flags
        assert "noisy" not in self.window(profile, 0.0, 14_000.0, "aperture").flags

    def test_the_aperture_in_a_dark_sky_is_not(self, profile: Profile) -> None:
        window = self.window(profile, 0.0, 14_000.0, "aperture")
        assert "noisy" not in window.flags

    def test_a_motion_below_the_noise_is_noisy(self, profile: Profile) -> None:
        """At 600 us in daylight (Polaris 3,900 e- on 2,100 e-) the noise share passes 20."""
        window = self.window(profile, 2_100.0, 3_900.0, "aperture")
        assert "noisy" in window.flags


class TestTheScintillationFloorInABrightSky:
    """The star of `bright_sky_window` has a constant flux, so its true index is 0.

    In the model's daylight, the sky's photons in the aperture and the noise of the background
    level that each flux subtracts make the flux scatter by about 15% of its mean, against 1% from
    the star's own photons. A floor of the read noise alone (`fast-1`) leaves an index of about
    0.020 on these frames, and a floor without the noise of the level about 0.008.
    """

    def test_a_constant_star_in_daylight_has_no_index(self, profile: Profile) -> None:
        window = bright_sky_window(profile, 4_300.0, 8_000.0, "aperture")
        assert window.scintillation_index is not None
        # The raw variance of 531 frames scatters by about 0.0013 (6% of 0.021).
        assert window.scintillation_index < 0.004

    def test_a_constant_star_at_a_short_exposure_has_no_index(self, profile: Profile) -> None:
        """At 600 us in daylight (3,900 e- on 2,100 e-), the floor is about 0.044 of a raw
        variance of 0.046. A floor without the noise of the level leaves 0.020."""
        window = bright_sky_window(profile, 2_100.0, 3_900.0, "aperture")
        assert window.scintillation_index is not None
        assert window.scintillation_index < 0.008  # 3 times the scatter of the raw variance


class TestTheBackgroundAndTheSnr:
    """The window's background as a share of saturation and the star's SNR, for the scheduler's
    adaptive exposure and for a reader who judges the noise of a reading."""

    def test_the_background_is_a_share_of_the_profiles_saturation_level(
        self, profile: Profile, analyzer: FastPathAnalyzer, driver: FakeCameraDriver
    ) -> None:
        analyzer.begin_stream(stream_of(driver))
        for frame in read(driver, 10):
            analyzer.push(frame)
        (window,) = analyzer.flush()
        saturation = profile.saturation("bin1", 120).container_dn
        background = window.background_mean_dn
        assert background is not None
        assert background == pytest.approx(25 * 4, abs=2)  # the offset
        assert window.background_fraction == pytest.approx(background / saturation)

    def test_the_snr_is_the_median_over_the_frames_with_a_star(
        self, analyzer: FastPathAnalyzer, driver: FakeCameraDriver
    ) -> None:
        analyzer.begin_stream(stream_of(driver))
        snrs = []
        for frame in read(driver, 11):
            star = analyzer.push(frame).star
            assert star.snr is not None
            snrs.append(star.snr)
        (window,) = analyzer.flush()
        assert window.star_snr is not None
        assert window.star_snr == pytest.approx(float(np.median(snrs)))
        assert window.star_snr > 50.0  # 30,000 e- on a dark sky
        assert "star_snr" not in (window.quality or {})

    def test_without_a_star_the_snr_is_missing_and_says_why(self, profile: Profile) -> None:
        fake = FakeCameraDriver(VirtualClock(), frame_factory=flat)
        fake.open()
        analyzer = FastPathAnalyzer(profile)
        analyzer.begin_stream(stream_of(fake))
        for frame in read(fake, 3):
            analyzer.push(frame)
        (window,) = analyzer.flush()
        assert window.star_snr is None
        assert window.quality is not None
        assert window.quality["star_snr"] == "no frame had a usable centroid"
        saturation = profile.saturation("bin1", 120).container_dn
        assert window.background_fraction == pytest.approx(100 / saturation)

    def test_an_eight_bit_background_is_a_share_of_its_full_scale(self, profile: Profile) -> None:
        def eight_bit(config: StreamConfig, roi: Roi, seq: int) -> npt.NDArray[np.uint8]:
            return np.full((roi.height, roi.width), 51, dtype=np.uint8)

        fake = FakeCameraDriver(VirtualClock(), frame_factory=eight_bit)
        fake.open()
        analyzer = FastPathAnalyzer(profile)
        analyzer.begin_stream(stream_of(fake, replace(CONFIG, pixel_format=PixelFormat.RAW8)))
        analyzer.push(fake.read_frame(5.0))
        (window,) = analyzer.flush()
        assert window.background_fraction == pytest.approx(51 / 255)

    def test_a_mode_that_the_profile_does_not_know_has_no_background_share(
        self, profile: Profile
    ) -> None:
        fake = FakeCameraDriver(
            VirtualClock(), frame_factory=gaussian_star, full_frames={"bin9": (4000, 3000)}
        )
        fake.open()
        analyzer = FastPathAnalyzer(profile)
        analyzer.begin_stream(stream_of(fake, replace(CONFIG, mode="bin9")))
        analyzer.push(fake.read_frame(5.0))
        (window,) = analyzer.flush()
        assert window.background_mean_dn is not None
        assert window.background_fraction is None
        assert window.star_snr is None
        assert window.quality is not None
        assert (
            window.quality["background_fraction"] == "the saturation level of the mode is unknown"
        )
        assert window.quality["star_snr"] == "the electron scale of the readout mode is unknown"


class TestRecords:
    def test_the_assumptions_are_stored_with_the_window(
        self, analyzer: FastPathAnalyzer, driver: FakeCameraDriver
    ) -> None:
        analyzer.begin_stream(stream_of(driver))
        analyzer.push(driver.read_frame(5.0))
        (window,) = analyzer.flush()
        assert window.provenance["algo"] == ALGORITHM_REVISION
        assert "L0=20 m" in window.provenance["assumptions"]
        assert "wind=10 m/s" in window.provenance["assumptions"]
        assert "centroid=aperture" in window.provenance["assumptions"]
        assert window.outer_scale_m == 20.0
        assert window.assumed_wind_ms == 10.0

    def test_a_configured_outer_scale_of_infinity_is_stored_as_missing(
        self, profile: Profile, driver: FakeCameraDriver
    ) -> None:
        analyzer = FastPathAnalyzer(profile, FastPathConfig(outer_scale_m=float("inf")))
        analyzer.begin_stream(stream_of(driver))
        analyzer.push(driver.read_frame(5.0))
        (window,) = analyzer.flush()
        assert window.outer_scale_m is None

    def test_the_station_and_profile_tag_every_record(
        self, profile: Profile, driver: FakeCameraDriver
    ) -> None:
        analyzer = create_fast_analyzer(profile, FastPathConfig(), "station-7")
        analyzer.begin_stream(stream_of(driver))
        analyzer.push(driver.read_frame(5.0))
        (window,) = analyzer.flush()
        assert (window.station_id, window.profile_id) == ("station-7", profile.id)
