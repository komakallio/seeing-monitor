"""The sim driver: lifecycle, timing, faults, and the wire format.

The first group repeats the conformance tests of the fake camera in `tests/test_fakes.py`, so the
simulator honors the same contract. The cheap Gaussian mixture and a small screen keep these
tests fast. Tests of the wave optics and the statistics are in the other files of this folder.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from seeingmon.clock import DEFAULT_START_UTC_NS, NS_PER_S, Clock, ScaledClock, VirtualClock
from seeingmon.drivers import (
    CameraConfigError,
    CameraDisconnectedError,
    CameraDriver,
    CameraError,
    CameraStateError,
    CameraTimeoutError,
    RecoveryLevel,
    create_driver,
)
from seeingmon.drivers.sim import (
    GeometryChange,
    PsfConfig,
    SimDriver,
    SimFaults,
    SimOptions,
    SimParams,
    TurbulenceConfig,
    create,
    sim_camera,
)
from seeingmon.frames import (
    FrameFlag,
    PixelFormat,
    Roi,
    StreamConfig,
    StreamKind,
    TimeQuality,
    decode_frame,
    encode_frame,
    frames_equal,
)
from seeingmon.profile import load_profile

# Polaris sits at the centre of the full bin1 frame (8288 x 5644) at the reference time.
FAST = StreamConfig(mode="bin1", exposure_us=2000, gain=0, roi=Roi(4080, 2758, 128, 128))


def make(clock: Clock | None = None, **kwargs: object) -> SimDriver:
    """A cheap simulated camera: one layer, Gaussian mixture images, and a small screen."""
    defaults: dict[str, object] = {"psf_mode": "gaussian", "screen_points": 128}
    defaults.update(kwargs)
    return sim_camera(clock or VirtualClock(), **defaults)  # type: ignore[arg-type]


@pytest.fixture
def clock() -> VirtualClock:
    return VirtualClock()


@pytest.fixture
def driver(clock: VirtualClock) -> SimDriver:
    camera = make(clock)
    camera.open()
    return camera


def test_the_driver_satisfies_the_protocol(clock: VirtualClock) -> None:
    camera: CameraDriver = make(clock)
    assert isinstance(camera, CameraDriver)
    assert camera.name == "sim"


class TestLifecycle:
    def test_streams_frames_and_advances_virtual_time(
        self, driver: SimDriver, clock: VirtualClock
    ) -> None:
        active = driver.configure(FAST)
        driver.start()
        started = clock.utc_ns()
        frames = [driver.read_frame(timeout_s=1.0) for _ in range(5)]
        assert [f.seq for f in frames] == [0, 1, 2, 3, 4]
        assert all(f.stream_id == active.stream_id for f in frames)
        period_ns = round(active.frame_period_s * NS_PER_S)  # type: ignore[operator]
        assert clock.utc_ns() - started == 5 * period_ns
        assert frames[1].t_arrival_ns - frames[0].t_arrival_ns == period_ns
        assert all(
            f.t_quality is TimeQuality.EXACT and f.flags & FrameFlag.SIMULATED for f in frames
        )
        assert frames[0].data.shape == (128, 128)
        assert frames[0].t_utc_ns < frames[0].t_arrival_ns

    def test_frames_survive_the_wire_format(self, driver: SimDriver) -> None:
        driver.configure(FAST)
        driver.start()
        frame = driver.read_frame(timeout_s=1.0)
        assert frames_equal(decode_frame(encode_frame(frame)), frame)

    def test_roi_follows_the_vendor_rules_and_stays_inside_the_frame(
        self, driver: SimDriver
    ) -> None:
        odd = StreamConfig(mode="bin1", exposure_us=2000, gain=1, roi=Roi(8200, 5600, 131, 125))
        roi = driver.configure(odd).config.roi
        assert roi is not None
        assert (roi.width, roi.height) == (128, 124)
        assert roi.x_end <= 8288
        assert roi.y_end <= 5644

    def test_the_full_frame_is_the_default_roi(self, driver: SimDriver) -> None:
        active = driver.configure(StreamConfig(mode="bin2", exposure_us=1000, gain=1))
        assert active.frame_shape == (2822, 4144)

    def test_each_configure_starts_a_new_stream(self, driver: SimDriver) -> None:
        first = driver.configure(FAST)
        driver.start()
        driver.read_frame(1.0)
        second = driver.configure(FAST)
        assert second.stream_id == first.stream_id + 1
        with pytest.raises(CameraStateError):  # configure stops capture
            driver.read_frame(1.0)
        driver.start()
        assert driver.read_frame(1.0).seq == 0

    def test_configure_reports_what_the_camera_applied(self, driver: SimDriver) -> None:
        active = driver.configure(
            StreamConfig(mode="bin1", exposure_us=2000, gain=0, roi=FAST.roi, offset=None)
        )
        assert active.config.offset == 30  # the default offset of the profile
        assert active.adc_bits == 12
        fast = driver.configure(
            StreamConfig(mode="bin1", exposure_us=2000, gain=0, roi=FAST.roi, high_speed=True)
        )
        assert fast.adc_bits == 10

    def test_move_roi_keeps_the_stream_and_clamps(self, driver: SimDriver) -> None:
        active = driver.configure(FAST)
        driver.start()
        moved = driver.move_roi(-50, 10_000)
        assert moved == Roi(0, 5644 - 128, 128, 128)
        frame = driver.read_frame(1.0)
        assert frame.roi == moved
        assert frame.stream_id == active.stream_id

    def test_snapshot_returns_one_frame_per_start(self, driver: SimDriver) -> None:
        driver.configure(
            StreamConfig(
                mode="bin2",
                exposure_us=30_000_000,
                gain=120,
                kind=StreamKind.SNAPSHOT,
                roi=Roi(0, 0, 64, 64),
            )
        )
        driver.start()
        frame = driver.read_frame(timeout_s=60.0)
        assert frame.exposure_us == 30_000_000
        with pytest.raises(CameraStateError):
            driver.read_frame(1.0)
        driver.start()
        assert driver.read_frame(60.0).seq == 1

    def test_lifecycle_errors(self, clock: VirtualClock) -> None:
        camera = make(clock)
        with pytest.raises(CameraStateError):
            camera.configure(FAST)
        camera.open()
        with pytest.raises(CameraStateError):
            camera.start()
        with pytest.raises(CameraStateError):
            camera.read_frame(1.0)
        with pytest.raises(CameraConfigError):
            camera.configure(StreamConfig(mode="bin9", exposure_us=1, gain=0))
        with pytest.raises(CameraConfigError):
            camera.configure(StreamConfig(mode="bin1", exposure_us=2000, gain=900))
        with pytest.raises(CameraConfigError):
            camera.configure(StreamConfig(mode="bin1", exposure_us=1, gain=0))
        with pytest.raises(CameraConfigError):
            camera.configure(
                StreamConfig(mode="bin1", exposure_us=2000, gain=0, roi=Roi(0, 0, 9000, 8))
            )

    def test_stop_close_and_info(self, driver: SimDriver) -> None:
        driver.configure(FAST)
        driver.start()
        driver.read_frame(1.0)
        driver.stop()
        driver.stop()
        with pytest.raises(CameraStateError):
            driver.read_frame(1.0)
        driver.close()
        driver.close()
        info = driver.open()
        assert (info.driver, info.has_temperature, info.max_width) == ("sim", True, 8288)
        caps = driver.capabilities()
        assert caps.bins == (1, 2)
        assert PixelFormat.RAW8 in caps.pixel_formats
        temperature = driver.read_temperature_c()
        assert temperature is not None
        assert 15.0 < temperature < 25.0

    def test_a_short_timeout_expires_after_waiting_the_timeout(
        self, driver: SimDriver, clock: VirtualClock
    ) -> None:
        driver.configure(
            StreamConfig(mode="bin1", exposure_us=500_000, gain=1, roi=Roi(4000, 2700, 8, 2))
        )
        driver.start()
        started = clock.monotonic_ns()
        with pytest.raises(CameraTimeoutError):
            driver.read_frame(timeout_s=0.1)
        assert clock.monotonic_ns() - started == round(0.1 * NS_PER_S)
        # The frame stays due: a longer wait gets it.
        assert driver.read_frame(timeout_s=1.0).seq == 0

    def test_a_simulated_night_runs_in_seconds(
        self, driver: SimDriver, clock: VirtualClock
    ) -> None:
        driver.configure(
            StreamConfig(
                mode="bin2", exposure_us=100_000_000, gain=120, roi=Roi(2000, 1400, 64, 64)
            )
        )
        driver.start()
        started = clock.utc_ns()
        wall = time.perf_counter()
        count = 0
        while clock.utc_ns() - started < 12 * 3600 * NS_PER_S:
            driver.read_frame(timeout_s=1000.0)
            count += 1
        assert count == 12 * 36
        assert time.perf_counter() - wall < 60

    def test_the_factory_builds_a_driver_by_name(self, clock: VirtualClock) -> None:
        camera = create_driver("sim", profile=None, clock=clock, options={"seed": 4})
        assert isinstance(camera, CameraDriver)
        assert camera.options.seed == 4  # type: ignore[attr-defined]
        with pytest.raises(ValueError, match="unknown driver"):
            create_driver("nothing", profile=None, clock=clock)
        with pytest.raises(ValueError, match="not a driver name"):
            create_driver("base.x", profile=None, clock=clock)
        assert create(profile=SimParams.reference("bin1"), clock=clock).modes.keys() == {"bin1"}

    def test_the_factory_reads_a_profile_and_a_bandwidth(self, clock: VirtualClock) -> None:
        profile = load_profile("asi294mm-gs250")
        options = SimOptions(
            psf=PsfConfig(bandwidth_fraction=0.2),
            turbulence=TurbulenceConfig(screen_points=128),
        )
        camera = create(profile=profile, clock=clock, options=options)
        assert set(camera.modes) == {"bin1", "bin2"}
        for name in ("bin1", "bin2"):
            assert camera.modes[name].plate_scale_arcsec_per_px == pytest.approx(
                profile.plate_scale_arcsec_per_px(name)
            )
        camera.open()
        active = camera.configure(StreamConfig("bin1", 2000, 0, roi=FAST.roi))
        assert active.frame_period_s == pytest.approx(profile.frame_period_s("bin1", 128, 2000))
        camera.start()
        frame = camera.read_frame(1.0)  # three wavelengths through the wave optics
        assert frame.data.shape == (128, 128)
        assert camera.truth.frames[-1].flux_e > 1000


class TestTiming:
    def test_the_frame_period_is_the_larger_of_the_exposure_and_the_readout(
        self, driver: SimDriver
    ) -> None:
        # Bin1, 128 rows: 6.5 ms + 128 x 37.6 us = 11.3 ms, so a 2 ms exposure runs at 88 fps.
        assert driver.configure(FAST).frame_period_s == pytest.approx(11.3128e-3, rel=1e-6)
        slow = StreamConfig(mode="bin1", exposure_us=20_000, gain=0, roi=FAST.roi)
        assert driver.configure(slow).frame_period_s == pytest.approx(20e-3)
        # Bin2, 64 rows at 2 ms: 1.4 ms + 64 x 21.3 us = 2.76 ms, so the readout sets the rate.
        bin2 = StreamConfig(mode="bin2", exposure_us=1000, gain=0, roi=Roi(2000, 1400, 64, 64))
        assert 1 / (driver.configure(bin2).frame_period_s or 1.0) == pytest.approx(362, rel=0.01)

    def test_a_snapshot_period_adds_the_readout(self, driver: SimDriver) -> None:
        config = StreamConfig(
            mode="bin2",
            exposure_us=1_000_000,
            gain=120,
            kind=StreamKind.SNAPSHOT,
            roi=Roi(0, 0, 64, 64),
        )
        period = driver.configure(config).frame_period_s
        assert period == pytest.approx(1.0 + 1.4e-3 + 64 * 21.3e-6)

    def test_timestamps_follow_the_definition_in_frame(
        self, driver: SimDriver, clock: VirtualClock
    ) -> None:
        driver.configure(FAST)
        driver.start()
        started = clock.utc_ns()
        first = driver.read_frame(1.0)
        second = driver.read_frame(1.0)
        # `t_utc_ns` is the middle of the exposure of the first row, 1 ms after the start.
        assert first.t_utc_ns == started + 1_000_000
        assert second.t_utc_ns - first.t_utc_ns == first.t_arrival_ns - started
        assert first.t_err_ns >= 0

    def test_the_rolling_shutter_delays_a_star_by_its_row(self, driver: SimDriver) -> None:
        # Polaris at row 64 of the ROI exposes 64 row times (2.4 ms in bin1) after row 0.
        driver.configure(FAST)
        driver.start()
        frame = driver.read_frame(1.0)
        truth = driver.truth.frames[-1]
        row = round(truth.catalog_y_px) - frame.roi.y
        assert row == 63
        shift_ns = truth.t_ref_utc_ns - frame.t_utc_ns
        assert shift_ns == pytest.approx(row * 37.6e3, abs=2)
        # Move the ROI and the star's row changes, and so does its time.
        driver.move_roi(4080, 2758 + 32)
        later = driver.read_frame(1.0)
        later_truth = driver.truth.frames[-1]
        later_row = round(later_truth.catalog_y_px) - later.roi.y
        assert later_row == row - 32
        assert later_truth.t_ref_utc_ns - later.t_utc_ns == pytest.approx(later_row * 37.6e3, abs=2)

    def test_a_slow_reader_loses_frames_like_a_camera_without_a_buffer(
        self, driver: SimDriver, clock: VirtualClock
    ) -> None:
        driver.configure(FAST)
        driver.start()
        driver.read_frame(1.0)
        clock.advance(0.5)  # 44 frames pass while the reader works elsewhere
        late = driver.read_frame(1.0)
        assert late.dropped_before > 30
        assert driver.dropped_frames() == late.dropped_before
        assert driver.read_frame(1.0).dropped_before == 0

    def test_the_driver_paces_in_real_time(self) -> None:
        """With a clock that runs in real time, `read_frame` sleeps until each frame is due.

        The exposure is 80 ms, so a loaded machine still renders a frame well inside its period.
        """
        scaled = ScaledClock(
            start_utc_ns=DEFAULT_START_UTC_NS, origin_real_ns=time.time_ns(), speed=1.0
        )
        camera = make(scaled)
        camera.open()
        camera.configure(StreamConfig("bin1", 80_000, 0, roi=FAST.roi))
        camera.start()
        first = camera.read_frame(1.0)
        wall = time.perf_counter()
        last = first
        for _ in range(10):
            last = camera.read_frame(1.0)
        elapsed_s = (last.t_arrival_ns - first.t_arrival_ns) / NS_PER_S
        assert elapsed_s == pytest.approx(10 * 0.080, rel=0.15)
        assert time.perf_counter() - wall == pytest.approx(elapsed_s, rel=0.25)
        assert last.dropped_before == 0


class TestPixels:
    def test_16_bit_data_carries_the_adc_value_in_the_high_bits(self) -> None:
        camera = make(seed=2)
        camera.open()
        camera.configure(
            StreamConfig("bin1", 2000, 0, roi=FAST.roi, pixel_format=PixelFormat.RAW16)
        )
        camera.start()
        frame = camera.read_frame(1.0)
        assert frame.data.dtype == np.uint16
        assert frame.adc_bits == 12
        assert np.all(frame.data % 16 == 0)  # the low 4 bits are zero at 12 bit
        bin2 = make(seed=2)
        bin2.open()
        bin2.configure(StreamConfig("bin2", 2000, 0, roi=Roi(2000, 1400, 64, 64)))
        bin2.start()
        data = bin2.read_frame(1.0).data
        assert np.all(data % 4 == 0)  # the low 2 bits are zero at 14 bit

    def test_8_bit_data_keeps_the_top_8_bits_of_the_adc_value(self) -> None:
        frames = {}
        for pixel_format in (PixelFormat.RAW16, PixelFormat.RAW8):
            camera = make(seed=6)
            camera.open()
            camera.configure(StreamConfig("bin1", 2000, 0, roi=FAST.roi, pixel_format=pixel_format))
            camera.start()
            frames[pixel_format] = camera.read_frame(1.0)
        wide, narrow = frames[PixelFormat.RAW16], frames[PixelFormat.RAW8]
        assert narrow.data.dtype == np.uint8
        # The same seed gives the same noise, so the two frames carry the same ADC values.
        assert np.array_equal(narrow.data, (wide.data >> 8).astype(np.uint8))
        # One 8-bit level is 16 ADU at 12 bit, which is about 56 electrons at gain 0.
        assert int(narrow.data.max()) > 20

    def test_frames_clip_at_the_adc_and_the_full_well(self) -> None:
        camera = make(seed=3)
        camera.open()
        camera.configure(StreamConfig("bin1", 10_000, 0, roi=FAST.roi))
        camera.start()
        frame = camera.read_frame(1.0)
        # At gain 0, 3.5 e-/ADU and a 12-bit ADC clip at 4095, a little below the 14,417 e- well.
        assert int(frame.data.max()) == 4095 << 4

    def test_black_level_and_read_noise(self) -> None:
        camera = make(seed=8, stars=_empty_field(), twilight=False, sky_mag_arcsec2=30.0)
        camera.open()
        camera.configure(StreamConfig("bin1", 100, 0, roi=Roi(1000, 1000, 256, 256), offset=30))
        camera.start()
        data = camera.read_frame(1.0).data.astype(np.float64) / 16.0
        assert data.mean() == pytest.approx(30.0, abs=0.1)  # offset 30 at 12 bit
        # 2.65 e- of read noise at 3.5 e-/ADU is 0.76 ADU, plus the 0.29 ADU of rounding.
        assert data.std() == pytest.approx(math.sqrt((2.65 / 3.5) ** 2 + 1 / 12), rel=0.05)


def _empty_field() -> object:
    from seeingmon.drivers.sim import StarField

    return StarField.from_arrays([], [], [])


class TestFaults:
    def faulty(self, faults: SimFaults, clock: Clock | None = None) -> SimDriver:
        camera = make(clock, faults=faults)
        camera.open()
        camera.configure(FAST)
        camera.start()
        return camera

    def test_scripted_drops_are_reported_once(self) -> None:
        camera = self.faulty(SimFaults(scripted_drops=((2, 3),)))
        frames = [camera.read_frame(1.0) for _ in range(5)]
        assert [f.dropped_before for f in frames] == [0, 0, 3, 0, 0]
        assert camera.dropped_frames() == 3
        # The lost frames took time: frame 2 arrives 3 periods late.
        period = round(11.3128e-3 * NS_PER_S)
        assert frames[2].t_arrival_ns - frames[1].t_arrival_ns == 4 * period

    def test_random_drops_repeat_for_a_seed_and_follow_the_probability(self) -> None:
        def run(seed: int) -> list[int]:
            camera = self.faulty(SimFaults(drop_probability=0.2, seed=seed))
            return [camera.read_frame(1.0).dropped_before for _ in range(200)]

        first = run(1)
        assert first == run(1)
        assert first != run(2)
        assert 20 < sum(1 for lost in first if lost) < 60

    def test_a_timeout_waits_for_the_timeout_and_the_next_read_succeeds(
        self, clock: VirtualClock
    ) -> None:
        camera = self.faulty(SimFaults(scripted_timeouts=frozenset({1})), clock)
        assert camera.read_frame(1.0).seq == 0
        started = clock.monotonic_ns()
        with pytest.raises(CameraTimeoutError):
            camera.read_frame(timeout_s=0.25)
        assert clock.monotonic_ns() - started == round(0.25 * NS_PER_S)
        assert camera.read_frame(1.0).seq == 1

    def test_slow_reads_deliver_late_with_the_true_exposure_time(self, clock: VirtualClock) -> None:
        camera = self.faulty(
            SimFaults(slow_read_s=0.050, scripted_slow_reads=frozenset({1})), clock
        )
        first = camera.read_frame(1.0)
        slow = camera.read_frame(1.0)
        assert slow.t_arrival_ns - first.t_arrival_ns == pytest.approx(
            (11.3128e-3 + 0.050) * NS_PER_S, abs=2
        )
        assert slow.t_utc_ns - first.t_utc_ns == pytest.approx(11.3128e-3 * NS_PER_S, abs=2)

    def test_a_disconnect_persists_until_a_strong_enough_recovery(self) -> None:
        camera = self.faulty(
            SimFaults(disconnect_at_frame=2, disconnect_clears_at=RecoveryLevel.USB_RESET)
        )
        camera.read_frame(1.0)
        camera.read_frame(1.0)
        with pytest.raises(CameraDisconnectedError):
            camera.read_frame(1.0)
        with pytest.raises(CameraDisconnectedError):
            camera.read_frame(1.0)
        with pytest.raises(CameraDisconnectedError):
            camera.recover(RecoveryLevel.REOPEN)  # not strong enough
        with pytest.raises(CameraDisconnectedError):
            camera.open()
        camera.recover(RecoveryLevel.USB_RESET)
        camera.start()
        frame = camera.read_frame(1.0)
        assert frame.flags & FrameFlag.RECOVERED
        assert camera.read_frame(1.0).flags & FrameFlag.RECOVERED == 0

    def test_a_stall_clears_only_after_recovery_at_the_chosen_level(
        self, clock: VirtualClock
    ) -> None:
        camera = self.faulty(
            SimFaults(stall_at_frame=1, stall_clears_at=RecoveryLevel.REOPEN), clock
        )
        camera.read_frame(1.0)
        for _ in range(3):
            with pytest.raises(CameraTimeoutError):
                camera.read_frame(timeout_s=0.5)
        with pytest.raises(CameraError):
            camera.recover(RecoveryLevel.RESTART_CAPTURE)
        camera.recover(RecoveryLevel.REOPEN)
        camera.start()
        frame = camera.read_frame(1.0)
        assert frame.flags & FrameFlag.RECOVERED
        # The stall lasted 1.5 s of virtual time, and the camera counts those frames as lost.
        assert frame.dropped_before > 100

    def test_a_silent_geometry_change_shows_in_the_read_back(self) -> None:
        change = GeometryChange(dx=8, d_width=-16, times=1)
        camera = make(faults=SimFaults(geometry_change=change))
        camera.open()
        applied = camera.configure(FAST)
        assert applied.config.roi == Roi(4088, 2758, 112, 128)
        assert applied.frame_shape == (128, 112)
        camera.start()
        assert camera.read_frame(1.0).data.shape == (128, 112)
        # The next configure behaves, because the change had `times=1`.
        assert camera.configure(FAST).config.roi == FAST.roi

    def test_fault_options_validate(self) -> None:
        with pytest.raises(ValueError, match="drop_probability"):
            SimFaults(drop_probability=2.0)
        with pytest.raises(ValueError, match="scripted drop"):
            SimFaults(scripted_drops=((1, 0),))

    def test_faults_read_from_a_table(self) -> None:
        options = SimOptions.from_mapping(
            {
                "seed": 9,
                "psf_mode": "gaussian",
                "faults": {
                    "scripted_drops": {"3": 2},
                    "stall_at_frame": 10,
                    "stall_clears_at": "usb_reset",
                    "geometry_change": {"dx": 8, "times": 2},
                },
                "turbulence": {
                    "r0_m": 0.07,
                    "outer_scale_m": "inf",
                    "layers": [{"wind_speed_m_s": 5}],
                },
                "clouds": [{"start_s": 10, "duration_s": 20, "transmission": 0.3}],
            }
        )
        assert options.faults.scripted_drops == ((3, 2),)
        assert options.faults.stall_clears_at is RecoveryLevel.USB_RESET
        assert options.turbulence is not None
        assert math.isinf(options.turbulence.outer_scale_m)
        assert options.clouds.events[0].start_utc_ns == DEFAULT_START_UTC_NS + 10 * NS_PER_S
        with pytest.raises(ValueError, match="unknown sim option"):
            SimOptions.from_mapping({"sede": 1})
        with pytest.raises(ValueError, match="unknown key"):
            SimOptions.from_mapping({"faults": {"drop_probabilty": 0.1}})


class TestDeterminism:
    def test_the_same_seed_gives_the_same_frames(self) -> None:
        def run(seed: int) -> list[bytes]:
            camera = make(seed=seed)
            camera.open()
            camera.configure(FAST)
            camera.start()
            return [encode_frame(camera.read_frame(1.0)) for _ in range(3)]

        assert run(5) == run(5)
        assert run(5) != run(6)

    def test_truth_is_a_pure_function_of_time(self) -> None:
        camera = make(seed=11)
        a = camera.truth.tilt_arcsec(DEFAULT_START_UTC_NS + 5 * NS_PER_S, 0.002)
        camera.truth.tilt_arcsec(DEFAULT_START_UTC_NS, 0.002)
        b = camera.truth.tilt_arcsec(DEFAULT_START_UTC_NS + 5 * NS_PER_S, 0.002)
        assert a == b
