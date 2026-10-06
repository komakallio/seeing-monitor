"""The daylight gate, the twilight flag, the cloud tracker, and the background measurement.

The gate decides from the background that the fast stream would have at its shortest exposure:
`read_sky` derives it from a brightness frame through the profile, and a burst or a window of the
fast stream gives it scaled to that exposure. A frame that clipped gives only a lower bound. The
Sun gates nothing.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from seeingmon.frames import Frame
from seeingmon.profile import load_profile
from seeingmon.scheduler.config import CloudConfig, DaylightConfig
from seeingmon.scheduler.gates import (
    REASON_BRIGHT_SKY,
    REASON_NO_MEASUREMENT,
    CloudTracker,
    DaylightGate,
    FastSky,
    SkyReading,
    combine,
    read_sky,
    saturation_level_dn,
    sky_background_fraction,
)
from tests.scheduler.helpers import make_frame

PROFILE = load_profile("asi294mm-gs250")
# Saturation 0.5 and resume at 0.35. A frame clips at 0.9 of its saturation.
CONFIG = DaylightConfig()
SHORTEST_US = PROFILE.limits.exposure_us_range[0]  # 32 us


class TestBackgroundMeasurement:
    def test_a_flat_frame_reads_its_level_as_a_share_of_saturation(self) -> None:
        saturation = PROFILE.saturation("bin2", 0).container_dn
        frame = make_frame(
            np.full((64, 64), round(saturation / 4), dtype=np.uint16), mode="bin2", gain=0
        )
        assert saturation_level_dn(frame, PROFILE) == saturation
        assert sky_background_fraction(frame, PROFILE) == pytest.approx(0.25, abs=1e-4)

    def test_the_level_comes_from_the_readout_mode_of_the_frame(self) -> None:
        data = np.full((32, 32), 8000, dtype=np.uint16)
        bin1 = make_frame(data, mode="bin1", gain=0, adc_bits=12)
        bin2 = make_frame(data, mode="bin2", gain=0)
        assert saturation_level_dn(bin1, PROFILE) == PROFILE.saturation("bin1", 0).container_dn
        assert saturation_level_dn(bin2, PROFILE) == PROFILE.saturation("bin2", 0).container_dn
        assert sky_background_fraction(bin1, PROFILE) == pytest.approx(8000 / 65520)

    def test_an_8_bit_frame_uses_its_full_scale(self) -> None:
        frame = make_frame(np.full((16, 16), 51, dtype=np.uint8), adc_bits=8)
        assert saturation_level_dn(frame, PROFILE) == 255.0
        assert sky_background_fraction(frame, PROFILE) == pytest.approx(0.2)

    def test_stars_do_not_move_the_median(self) -> None:
        data = np.full((128, 128), 1000, dtype=np.uint16)
        data[60:68, 60:68] = 60_000  # a bright star on 64 of 16,384 pixels
        frame = make_frame(data, mode="bin2", gain=0)
        assert sky_background_fraction(frame, PROFILE) == pytest.approx(
            1000 / saturation_level_dn(frame, PROFILE)
        )

    def test_a_large_frame_is_measured_on_a_stride_and_still_reads_the_median(self) -> None:
        rng = np.random.default_rng(3)
        data = rng.normal(5000, 100, size=(2000, 2000)).clip(0, 65535).astype(np.uint16)
        frame = make_frame(data, mode="bin2", gain=0)
        expected = float(np.median(data)) / saturation_level_dn(frame, PROFILE)
        assert sky_background_fraction(frame, PROFILE) == pytest.approx(expected, rel=0.01)


class TestTheSkyOfTheFastStream:
    """`read_sky` derives the background of the fast stream at its shortest exposure."""

    @staticmethod
    def brightness(level: int, *, exposure_us: int = 1000) -> Frame:
        return make_frame(
            np.full((64, 64), level, dtype=np.uint16), mode="bin2", gain=0, exposure_us=exposure_us
        )

    @staticmethod
    def read(
        frame: Frame,
        *,
        fast_exposure_us: float = SHORTEST_US,
        fast_gain: int = 0,
        offset_dn: float = 0.0,
    ) -> SkyReading:
        return read_sky(
            frame,
            PROFILE,
            fast_mode="bin1",
            fast_exposure_us=fast_exposure_us,
            fast_gain=fast_gain,
            clip_fraction=CONFIG.brightness_clip_fraction,
            offset_dn=offset_dn,
        )

    def test_the_fast_background_follows_the_profile(self) -> None:
        """A 1 ms bin2 frame at half of saturation: 33,176 e- in a bin2 pixel at gain 0.

        A bin1 pixel has a quarter of the area, and 32 us is 0.032 of the exposure, so it holds
        265.4 e-, which is 0.0185 of the bin1 full well at gain 0 (14,332 e-).
        """
        saturation = PROFILE.saturation("bin2", 0).container_dn
        sky = self.read(self.brightness(round(saturation / 2)))
        e_per_dn = PROFILE.e_per_adu("bin2", 0) / 4.0  # 14 bits in a 16-bit container
        fast_e = round(saturation / 2) * e_per_dn * 0.25 * SHORTEST_US / 1000
        assert sky.frame_fraction == pytest.approx(0.5, abs=1e-4)
        assert sky.fast_fraction == pytest.approx(
            fast_e / PROFILE.saturation("bin1", 0).full_well_e, rel=1e-9
        )
        assert sky.fast_fraction == pytest.approx(0.0185, abs=1e-4)
        assert sky.clipped is False
        assert sky.median_dn == round(saturation / 2)
        assert FastSky.from_reading(sky) == FastSky(sky.fast_fraction, bound=False)

    def test_the_fast_background_scales_with_the_exposures(self) -> None:
        frame = self.brightness(20_000)
        base = self.read(frame).fast_fraction
        longer = self.read(frame, fast_exposure_us=2000).fast_fraction
        assert longer == pytest.approx(base * 2000 / SHORTEST_US, rel=1e-9)
        shorter_frame = self.brightness(20_000, exposure_us=500)
        assert self.read(shorter_frame).fast_fraction == pytest.approx(2 * base, rel=1e-9)

    def test_the_offset_of_the_brightness_frame_is_not_sky(self) -> None:
        frame = self.brightness(2000)
        assert self.read(frame, offset_dn=2000).fast_fraction == 0.0
        half = self.read(frame, offset_dn=1000).fast_fraction
        assert half == pytest.approx(0.5 * self.read(frame).fast_fraction)

    def test_a_clipped_brightness_frame_gives_only_a_lower_bound(self) -> None:
        """A 1 ms frame at its saturation says only that the fast stream would see at least 3.7%.

        That bound lies far below the limit, so it decides nothing: a running scheduler keeps
        running, and a stopped one stays stopped. Just below the clip, the frame measures.
        """
        saturation = PROFILE.saturation("bin2", 0).container_dn
        sky = self.read(self.brightness(round(saturation)))
        assert sky.clipped is True
        assert sky.fast_fraction == pytest.approx(0.037, abs=0.001)
        bound = FastSky.from_reading(sky)
        assert bound == FastSky(sky.fast_fraction, bound=True)
        gate = DaylightGate(CONFIG)
        assert gate.evaluate(bound, running=True).allowed
        decision = gate.evaluate(bound, running=False)
        assert (decision.allowed, decision.reason) == (False, REASON_BRIGHT_SKY)
        just_below = self.read(self.brightness(round(0.89 * saturation)))
        assert just_below.clipped is False
        assert gate.evaluate(FastSky.from_reading(just_below), running=False).allowed

    def test_the_watch_frame_of_32_us_reads_daylight_without_clipping(self) -> None:
        """At 32 us in bin2, the frame reads 86% of what the fast stream sees at 32 us in bin1.

        So it clips (0.9 of its saturation) only where the fast stream would see 104%, twice the
        limit of the gate.
        """
        saturation = PROFILE.saturation("bin2", 0).container_dn
        frame = self.brightness(round(0.5 * saturation), exposure_us=SHORTEST_US)
        sky = self.read(frame)
        assert sky.frame_fraction / sky.fast_fraction == pytest.approx(0.864, abs=0.001)
        clip = self.read(self.brightness(round(0.9 * saturation), exposure_us=SHORTEST_US))
        assert clip.clipped is True
        assert clip.fast_fraction == pytest.approx(1.04, abs=0.01)

    def test_the_fast_gain_shrinks_the_full_well(self) -> None:
        frame = self.brightness(20_000)
        gain0 = self.read(frame).fast_fraction
        gain300 = self.read(frame, fast_gain=300).fast_fraction
        assert gain300 > 10 * gain0


class TestTheFastStreamItself:
    """A burst or a window reports its own background, which scales to the shortest exposure."""

    def test_the_background_scales_with_the_exposure(self) -> None:
        sky = FastSky.from_fast(0.3, 1500.0, SHORTEST_US, clipped=False)
        assert sky.fraction == pytest.approx(0.3 * 32 / 1500, rel=1e-12)
        assert sky.bound is False

    def test_a_clipped_burst_gives_a_lower_bound(self) -> None:
        sky = FastSky.from_fast(1.0, 2000.0, SHORTEST_US, clipped=True)
        assert sky.bound is True
        assert sky.fraction == pytest.approx(0.016)

    def test_a_sunny_sky_where_the_fast_stream_is_fine_stays_in_auto(self) -> None:
        """The 1 ms frame clipped, and the last burst at 150 us read 0.3 of saturation.

        At 32 us the fast stream would see 0.064, far below the limit, so the scheduler measures.
        """
        clipped = FastSky(0.037, bound=True)
        burst = FastSky.from_fast(0.3, 150.0, SHORTEST_US, clipped=False)
        sky = combine([clipped, burst])
        assert sky is not None
        assert sky.fraction == pytest.approx(0.064, rel=1e-12)
        assert sky.bound is False
        assert DaylightGate(CONFIG).evaluate(sky, running=True).allowed

    def test_a_sky_that_even_the_shortest_exposure_cannot_take_stops_auto(self) -> None:
        """A burst at 32 us that reads 0.55 of saturation: nothing can be measured."""
        burst = FastSky.from_fast(0.55, SHORTEST_US, SHORTEST_US, clipped=False)
        decision = DaylightGate(CONFIG).evaluate(
            combine([FastSky(0.037, bound=True), burst]), running=True
        )
        assert (decision.allowed, decision.reason) == (False, REASON_BRIGHT_SKY)


class TestCombine:
    def test_nothing_known_gives_nothing(self) -> None:
        assert combine([]) is None
        assert combine([None, None]) is None

    def test_the_largest_measurement_stands(self) -> None:
        assert combine([FastSky(0.1), FastSky(0.2), None]) == FastSky(0.2)

    def test_a_bound_above_every_measurement_wins_as_a_bound(self) -> None:
        assert combine([FastSky(0.1), FastSky(0.4, bound=True)]) == FastSky(0.4, bound=True)

    def test_a_measurement_above_every_bound_stands_as_a_measurement(self) -> None:
        assert combine([FastSky(0.2), FastSky(0.04, bound=True)]) == FastSky(0.2)

    def test_bounds_alone_give_the_largest_bound(self) -> None:
        sky = combine([FastSky(0.04, bound=True), FastSky(0.01, bound=True)])
        assert sky == FastSky(0.04, bound=True)


class TestDaylightGate:
    gate = DaylightGate(CONFIG)

    def test_a_dark_sky_allows_auto(self) -> None:
        decision = self.gate.evaluate(FastSky(0.01), running=False)
        assert (decision.allowed, decision.reason) == (True, None)

    def test_the_measured_sky_alone_decides(self) -> None:
        decision = self.gate.evaluate(FastSky(0.6), running=True)
        assert (decision.allowed, decision.reason) == (False, REASON_BRIGHT_SKY)

    def test_a_running_scheduler_stops_at_the_limit(self) -> None:
        run = self.gate.evaluate
        assert run(FastSky(0.49), running=True).allowed
        assert not run(FastSky(0.5), running=True).allowed
        assert self.gate.threshold(running=True) == 0.5

    def test_a_stopped_scheduler_resumes_only_below_the_stricter_value(self) -> None:
        run = self.gate.evaluate
        assert run(FastSky(0.34), running=False).allowed
        assert not run(FastSky(0.36), running=False).allowed
        assert self.gate.threshold(running=False) == 0.35

    def test_a_bound_at_the_limit_stops_and_a_bound_under_it_decides_nothing(self) -> None:
        run = self.gate.evaluate
        assert not run(FastSky(1.04, bound=True), running=True).allowed
        assert run(FastSky(0.04, bound=True), running=True).allowed
        assert not run(FastSky(0.04, bound=True), running=False).allowed

    def test_a_missing_measurement_keeps_a_running_scheduler_and_blocks_a_stopped_one(self) -> None:
        assert self.gate.evaluate(None, running=True).allowed
        decision = self.gate.evaluate(None, running=False)
        assert (decision.allowed, decision.reason) == (False, REASON_NO_MEASUREMENT)

    @pytest.mark.parametrize(
        ("elevation", "flags"),
        [
            (60.0, {"daylight"}),
            (10.0, {"daylight"}),
            (0.1, {"daylight"}),
            (0.0, {"twilight"}),
            (-3.0, {"twilight"}),
            (-17.9, {"twilight"}),
            (-18.0, set()),
            (-18.1, set()),
            (None, set()),
        ],
    )
    def test_daylight_above_the_horizon_and_twilight_from_there_to_the_twilight_limit(
        self, elevation: float | None, flags: set[str]
    ) -> None:
        assert self.gate.sun_flags(elevation) == flags
        assert self.gate.is_daylight(elevation) is ("daylight" in flags)
        assert self.gate.is_twilight(elevation) is ("twilight" in flags)

    @given(st.floats(0, 1))
    def test_a_stopped_scheduler_that_may_start_may_also_keep_running(
        self, fraction: float
    ) -> None:
        """The resume value is stricter than the stop value, so resuming implies running."""
        if self.gate.evaluate(FastSky(fraction), running=False).allowed:
            assert self.gate.evaluate(FastSky(fraction), running=True).allowed

    @given(st.floats(0, 1), st.floats(0, 1), st.booleans())
    def test_a_darker_sky_never_turns_an_allowed_decision_into_a_refusal(
        self, fraction: float, scale: float, running: bool
    ) -> None:
        bright = self.gate.evaluate(FastSky(fraction), running=running)
        darker = self.gate.evaluate(FastSky(fraction * scale), running=running)
        assert darker.allowed or not bright.allowed


class TestCloudTracker:
    def test_the_response_starts_at_the_threshold_and_ends_at_the_clear_threshold(self) -> None:
        tracker = CloudTracker(CloudConfig())  # threshold 0.5, clear 0.3
        assert not tracker.active
        assert tracker.fraction is None
        # Each pair is what `update` returns (the response changed) and whether it is active now.
        outcomes = [(tracker.update(fraction), tracker.active) for fraction in (0.2, 0.5, 0.4, 0.9)]
        assert outcomes == [(False, False), (True, True), (False, True), (False, True)]
        outcomes = [(tracker.update(fraction), tracker.active) for fraction in (0.3, 0.4, 0.0)]
        assert outcomes == [(True, False), (False, False), (False, False)]
        assert tracker.fraction == 0.0

    def test_a_result_that_cannot_tell_changes_nothing(self) -> None:
        tracker = CloudTracker(CloudConfig())
        tracker.update(0.8)
        assert tracker.update(None) is False
        assert tracker.active
        assert tracker.fraction == 0.8

    def test_the_response_shortens_the_window_and_changes_the_cadence(self) -> None:
        tracker = CloudTracker(CloudConfig(fast_window_s=45.0, survey_cadence_s=75.0))
        assert tracker.fast_window_s(120.0) == 120.0
        assert tracker.survey_cadence_s(180.0) == 180.0
        tracker.update(0.7)
        assert tracker.fast_window_s(120.0) == 45.0
        assert tracker.survey_cadence_s(180.0) == 75.0
