"""The daylight gate, the twilight flag, the cloud tracker, and the background measurement."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from seeingmon.profile import load_profile
from seeingmon.scheduler.config import CloudConfig, DaylightConfig
from seeingmon.scheduler.gates import (
    REASON_BRIGHT_SKY,
    REASON_DAYLIGHT,
    REASON_NO_MEASUREMENT,
    CloudTracker,
    DaylightGate,
    saturation_level_dn,
    sky_background_fraction,
)
from tests.scheduler.helpers import make_frame

PROFILE = load_profile("asi294mm-gs250")
CONFIG = DaylightConfig()  # limit -3 deg, resume at -4 deg, saturation 0.5, resume at 0.35


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


class TestDaylightGate:
    gate = DaylightGate(CONFIG)

    def test_a_dark_sky_and_a_low_sun_allow_auto(self) -> None:
        decision = self.gate.evaluate(
            sun_elevation_deg=-30.0, background_fraction=0.01, running=False
        )
        assert (decision.allowed, decision.reason, decision.twilight) == (True, None, False)

    def test_the_sun_gates_the_attempt(self) -> None:
        decision = self.gate.evaluate(sun_elevation_deg=10.0, background_fraction=0.0, running=True)
        assert (decision.allowed, decision.reason) == (False, REASON_DAYLIGHT)

    def test_the_measured_sky_overrides_a_low_sun(self) -> None:
        decision = self.gate.evaluate(
            sun_elevation_deg=-40.0, background_fraction=0.6, running=True
        )
        assert (decision.allowed, decision.reason) == (False, REASON_BRIGHT_SKY)

    def test_a_running_scheduler_stops_at_the_limits(self) -> None:
        run = self.gate.evaluate
        assert run(sun_elevation_deg=-3.0, background_fraction=0.49, running=True).allowed
        assert not run(sun_elevation_deg=-2.9, background_fraction=0.0, running=True).allowed
        assert not run(sun_elevation_deg=-40.0, background_fraction=0.5, running=True).allowed

    def test_a_stopped_scheduler_resumes_only_below_the_stricter_values(self) -> None:
        run = self.gate.evaluate
        assert run(sun_elevation_deg=-4.0, background_fraction=0.34, running=False).allowed
        assert not run(sun_elevation_deg=-3.5, background_fraction=0.0, running=False).allowed
        assert not run(sun_elevation_deg=-40.0, background_fraction=0.36, running=False).allowed

    def test_without_a_site_the_measurement_decides(self) -> None:
        run = self.gate.evaluate
        assert run(sun_elevation_deg=None, background_fraction=0.1, running=False).allowed
        assert not run(sun_elevation_deg=None, background_fraction=0.6, running=True).allowed

    def test_a_missing_measurement_keeps_a_running_scheduler_and_blocks_a_stopped_one(self) -> None:
        assert self.gate.evaluate(
            sun_elevation_deg=-30.0, background_fraction=None, running=True
        ).allowed
        decision = self.gate.evaluate(
            sun_elevation_deg=-30.0, background_fraction=None, running=False
        )
        assert (decision.allowed, decision.reason) == (False, REASON_NO_MEASUREMENT)

    @pytest.mark.parametrize(
        ("elevation", "twilight"),
        [(10.0, True), (-3.0, True), (-17.9, True), (-18.0, False), (-18.1, False), (None, False)],
    )
    def test_the_twilight_flag_applies_above_the_twilight_limit(
        self, elevation: float | None, twilight: bool
    ) -> None:
        assert self.gate.is_twilight(elevation) is twilight
        decision = self.gate.evaluate(
            sun_elevation_deg=elevation, background_fraction=0.0, running=True
        )
        assert decision.twilight is twilight

    @given(st.floats(-90, 90), st.floats(0, 1))
    def test_a_stopped_scheduler_that_may_start_may_also_keep_running(
        self, elevation: float, fraction: float
    ) -> None:
        """The resume values are stricter than the stop values, so resuming implies running."""
        if self.gate.evaluate(
            sun_elevation_deg=elevation, background_fraction=fraction, running=False
        ).allowed:
            assert self.gate.evaluate(
                sun_elevation_deg=elevation, background_fraction=fraction, running=True
            ).allowed

    @given(st.floats(-90, 90), st.floats(0, 1), st.floats(0, 30), st.floats(0, 1), st.booleans())
    def test_a_darker_world_never_turns_an_allowed_decision_into_a_refusal(
        self, elevation: float, fraction: float, lower_by: float, scale: float, running: bool
    ) -> None:
        bright = self.gate.evaluate(
            sun_elevation_deg=elevation, background_fraction=fraction, running=running
        )
        darker = self.gate.evaluate(
            sun_elevation_deg=elevation - lower_by,
            background_fraction=fraction * scale,
            running=running,
        )
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
