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
from tests.fastpath.helpers import box_integrated_gaussian, digitize

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
