"""The daylight gate, the twilight flag, the cloud tracker, and the background measurement.

The gate decides from the background that the fast stream would have at its shortest exposure,
which `read_sky` derives from a brightness frame through the profile. The Sun gates nothing.
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
    SkyReading,
    read_sky,
    saturation_level_dn,
    sky_background_fraction,
)
from tests.scheduler.helpers import make_frame

PROFILE = load_profile("asi294mm-gs250")
# Saturation 0.5 and resume at 0.35. A brightness frame clips at 0.9 and resumes below 0.6.
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
        assert sky.gate_fraction == sky.fast_fraction

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

    def test_a_clipped_brightness_frame_counts_as_too_bright(self) -> None:
        """A clipped frame says only that the sky is at least that bright.

        Its fast background would be 3.7% of saturation, far below the limit, so the gate
        compares 1 instead.
        """
        saturation = PROFILE.saturation("bin2", 0).container_dn
        sky = self.read(self.brightness(round(saturation)))
        assert sky.clipped is True
        assert sky.fast_fraction == pytest.approx(0.037, abs=0.001)
        assert sky.gate_fraction == 1.0
        just_below = self.read(self.brightness(round(0.89 * saturation)))
        assert just_below.clipped is False
        gate = DaylightGate(CONFIG)
        # A running scheduler keeps running just below the clip. A stopped one waits for the
        # resume level of the clip.
        assert gate.evaluate(
            background_fraction=just_below.gate_fraction,
            running=True,
            frame_fraction=just_below.frame_fraction,
        ).allowed
        assert not gate.evaluate(
            background_fraction=just_below.gate_fraction,
            running=False,
            frame_fraction=just_below.frame_fraction,
        ).allowed

    def test_the_fast_gain_shrinks_the_full_well(self) -> None:
        frame = self.brightness(20_000)
        gain0 = self.read(frame).fast_fraction
        gain300 = self.read(frame, fast_gain=300).fast_fraction
        assert gain300 > 10 * gain0


class TestDaylightGate:
    gate = DaylightGate(CONFIG)

    def test_a_dark_sky_allows_auto(self) -> None:
        decision = self.gate.evaluate(background_fraction=0.01, running=False)
        assert (decision.allowed, decision.reason) == (True, None)

    def test_the_measured_sky_alone_decides(self) -> None:
        decision = self.gate.evaluate(background_fraction=0.6, running=True)
        assert (decision.allowed, decision.reason) == (False, REASON_BRIGHT_SKY)

    def test_a_running_scheduler_stops_at_the_limit(self) -> None:
        run = self.gate.evaluate
        assert run(background_fraction=0.49, running=True).allowed
        assert not run(background_fraction=0.5, running=True).allowed

    def test_a_stopped_scheduler_resumes_only_below_the_stricter_value(self) -> None:
        run = self.gate.evaluate
        assert run(background_fraction=0.34, running=False).allowed
        assert not run(background_fraction=0.36, running=False).allowed
        assert not run(background_fraction=1.0, running=False).allowed  # a clipped frame

    def test_a_stopped_scheduler_resumes_only_below_the_resume_level_of_the_clip(self) -> None:
        """A frame between the resume level (0.6) and the clip (0.9) keeps `safe`, not `auto`.

        Its fast background (2.5% at 0.6) is far below both fast limits, so without this level a
        sky that hovers at the clip would flip the state at every brightness frame.
        """
        run = self.gate.evaluate
        assert CONFIG.brightness_resume_fraction == 0.6
        assert run(background_fraction=0.025, running=True, frame_fraction=0.75).allowed
        decision = run(background_fraction=0.025, running=False, frame_fraction=0.75)
        assert (decision.allowed, decision.reason) == (False, REASON_BRIGHT_SKY)
        assert not run(background_fraction=0.025, running=False, frame_fraction=0.6).allowed
        assert run(background_fraction=0.025, running=False, frame_fraction=0.59).allowed

    def test_a_missing_measurement_keeps_a_running_scheduler_and_blocks_a_stopped_one(self) -> None:
        assert self.gate.evaluate(background_fraction=None, running=True).allowed
        decision = self.gate.evaluate(background_fraction=None, running=False)
        assert (decision.allowed, decision.reason) == (False, REASON_NO_MEASUREMENT)

    @pytest.mark.parametrize(
        ("elevation", "twilight"),
        [(10.0, True), (-3.0, True), (-17.9, True), (-18.0, False), (-18.1, False), (None, False)],
    )
    def test_the_twilight_flag_applies_above_the_twilight_limit(
        self, elevation: float | None, twilight: bool
    ) -> None:
        assert self.gate.is_twilight(elevation) is twilight

    @given(st.floats(0, 1))
    def test_a_stopped_scheduler_that_may_start_may_also_keep_running(
        self, fraction: float
    ) -> None:
        """The resume value is stricter than the stop value, so resuming implies running."""
        if self.gate.evaluate(background_fraction=fraction, running=False).allowed:
            assert self.gate.evaluate(background_fraction=fraction, running=True).allowed

    @given(st.floats(0, 1), st.floats(0, 1), st.booleans())
    def test_a_darker_sky_never_turns_an_allowed_decision_into_a_refusal(
        self, fraction: float, scale: float, running: bool
    ) -> None:
        bright = self.gate.evaluate(background_fraction=fraction, running=running)
        darker = self.gate.evaluate(background_fraction=fraction * scale, running=running)
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
