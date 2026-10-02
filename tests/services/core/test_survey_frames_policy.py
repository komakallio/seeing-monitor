"""Which survey frames get a preview and which a FITS file: the every-tenth rule and the events."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from seeingmon.analysis import SurveyOutput
from seeingmon.clock import NS_PER_S
from seeingmon.frames import Frame
from seeingmon.profile import Profile, load_profile
from seeingmon.records import Record, SurveyFrameRecord
from seeingmon.records.samples import sample_record
from seeingmon.services.core.settings import SurveyFrameSettings
from seeingmon.services.core.survey_frames import (
    KIND_EVENT,
    KIND_SHORT,
    KIND_SURVEY,
    LONG,
    SHORT,
    Decision,
    KeepPolicy,
)
from tests.scheduler.helpers import make_frame

START_NS = 1_790_000_000 * NS_PER_S
STEP_NS = 180 * NS_PER_S
LONG_US = 30_000_000
SHORT_US = 1_000
SATURATION_DN = 16_383  # the ADC full scale of the reference camera in bin2 at gain 120


@pytest.fixture(scope="module")
def profile() -> Profile:
    return load_profile("asi294mm-gs250")


def make_policy(profile: Profile | None, **settings: Any) -> KeepPolicy:
    return KeepPolicy(SurveyFrameSettings(**settings), profile, long_min_exposure_s=5.0)


@dataclass
class Clockwork:
    """A run of frames at the cadence of the survey, with a way to make each frame's result."""

    policy: KeepPolicy
    t_ns: int = START_NS

    def frame(self, exposure_us: int = LONG_US, *, step: int = 0) -> Frame:
        gain = 120 if exposure_us >= 5_000_000 else 0
        data = np.zeros((4, 4), dtype=np.uint16)
        t_ns = self.t_ns + step * STEP_NS
        return make_frame(data, mode="bin2", gain=gain, exposure_us=exposure_us, t_utc_ns=t_ns)

    def decide(
        self,
        frame: Frame,
        *,
        solved: bool = True,
        cloud: float | None = 0.0,
        moved: bool = False,
        background: float | None = 900.0,
        gate: Callable[[], bool] | None = None,
    ) -> Decision:
        records: list[Record] = [
            SurveyFrameRecord(
                station_id="t",
                t_utc_ns=frame.t_utc_ns,
                profile_id="p",
                provenance={"algo": "test"},
                exposure_s=frame.exposure_us / 1e6,
                gain=frame.gain,
                readout_mode=frame.mode,
                background_dn=background,
            )
        ]
        if moved:
            records.append(
                sample_record(
                    "pointing",
                    station_id="t",
                    profile_id="p",
                    t_utc_ns=frame.t_utc_ns,
                    flags=["moved"],
                )
            )
        output = SurveyOutput(
            t_utc_ns=frame.t_utc_ns,
            records=tuple(records),
            solved=solved,
            cloud_fraction=cloud,
        )
        return self.policy.decide(frame, output, capture_allowed=gate)


def run(policy: KeepPolicy) -> Clockwork:
    return Clockwork(policy)


class TestTheClasses:
    def test_an_exposure_of_the_sky_quality_minimum_is_long(self) -> None:
        policy = make_policy(None)
        assert policy.classify(run(policy).frame(5_000_000)) == LONG
        assert policy.classify(run(policy).frame(4_999_999)) == SHORT
        assert policy.classify(run(policy).frame(SHORT_US)) == SHORT


class TestEveryTenthFrame:
    def test_every_tenth_long_frame_is_kept_and_the_first_one_too(self) -> None:
        work = run(make_policy(None))  # keep_every is 10 by default
        kept = [i for i in range(25) if work.decide(work.frame(step=i)).fits]
        assert kept == [0, 10, 20]

    def test_the_reason_names_the_rule(self) -> None:
        work = run(make_policy(None, keep_every=4))
        first = work.decide(work.frame(step=0))
        assert first.reasons == ("every_4",)
        assert work.decide(work.frame(step=1)).reasons == ()

    def test_a_short_frame_does_not_count_and_is_not_kept_by_the_rule(self) -> None:
        work = run(make_policy(None, keep_every=2))
        shorts = [work.decide(work.frame(SHORT_US, step=i)) for i in range(6)]
        assert not any(d.fits or d.preview for d in shorts)
        # the short frames in between did not move the count of the long ones
        longs = [work.decide(work.frame(step=10 + i)).fits for i in range(4)]
        assert longs == [True, False, True, False]

    def test_keeping_every_frame_is_a_setting(self) -> None:
        work = run(make_policy(None, keep_every=1))
        assert all(work.decide(work.frame(step=i)).fits for i in range(5))


class TestThePreview:
    def test_every_long_frame_gets_one_and_a_plain_short_frame_gets_none(self) -> None:
        work = run(make_policy(None))
        long_ = work.decide(work.frame(step=1))
        short = work.decide(work.frame(SHORT_US, step=1))
        assert long_.preview is True
        assert short.preview is False

    def test_the_kind_says_what_the_frame_is(self, profile: Profile) -> None:
        work = run(make_policy(profile, keep_every=1))
        assert work.decide(work.frame(step=0)).kind == KIND_SURVEY
        short = work.decide(work.frame(SHORT_US, step=0), background=SATURATION_DN * 0.9)
        assert short.kind == KIND_EVENT  # a bright short frame
        plain = work.decide(work.frame(SHORT_US, step=1), background=900.0)
        assert plain.kind == KIND_SHORT


class TestEventFrames:
    def test_the_first_unsolved_frame_is_an_event_and_the_ones_after_it_are_not(
        self, profile: Profile
    ) -> None:
        work = run(make_policy(profile, keep_every=1000))
        work.decide(work.frame(step=0))  # the first long frame: kept by the rule
        outcome = [work.decide(work.frame(step=i), solved=False) for i in range(1, 5)]
        assert [d.fits for d in outcome] == [True, False, False, False]
        assert outcome[0].reasons == ("event:unsolved",)
        assert outcome[0].kind == KIND_EVENT

    def test_a_new_start_inside_the_interval_is_not_an_event_and_after_it_is(
        self, profile: Profile
    ) -> None:
        work = run(make_policy(profile, keep_every=1000, event_min_interval_s=3600.0))
        work.decide(work.frame(step=0))
        assert work.decide(work.frame(step=1), solved=False).fits
        work.decide(work.frame(step=2), solved=True)  # the pointing came back
        assert not work.decide(work.frame(step=3), solved=False).fits  # 6 min on
        work.decide(work.frame(step=4), solved=True)
        late = work.decide(work.frame(step=21), solved=False)  # 63 minutes after the last event
        assert late.fits is True
        assert late.reasons == ("event:unsolved",)

    @pytest.mark.parametrize(
        ("cloud", "event"), [(0.0, False), (0.49, False), (0.5, True), (1.0, True)]
    )
    def test_clouds_start_an_event_at_the_configured_fraction(
        self, profile: Profile, cloud: float, event: bool
    ) -> None:
        work = run(make_policy(profile, keep_every=1000, event_cloud_fraction=0.5))
        work.decide(work.frame(step=0))
        outcome = work.decide(work.frame(step=1), cloud=cloud)
        assert outcome.fits is event
        if event:
            assert outcome.reasons == ("event:cloud",)

    def test_a_missing_cloud_fraction_is_no_cloud(self, profile: Profile) -> None:
        work = run(make_policy(profile, keep_every=1000))
        work.decide(work.frame(step=0))
        assert not work.decide(work.frame(step=1), cloud=None).fits

    def test_a_pointing_that_moved_starts_an_event(self, profile: Profile) -> None:
        work = run(make_policy(profile, keep_every=1000))
        work.decide(work.frame(step=0))
        outcome = work.decide(work.frame(step=1), moved=True)
        assert outcome.reasons == ("event:moved",)

    def test_a_bright_sky_starts_an_event_on_a_long_and_on_a_short_frame(
        self, profile: Profile
    ) -> None:
        bright = 0.5 * SATURATION_DN
        work = run(make_policy(profile, keep_every=1000))
        work.decide(work.frame(step=0))
        assert work.decide(work.frame(step=1), background=bright - 1).fits is False
        outcome = work.decide(work.frame(step=2), background=bright + 1)
        assert outcome.reasons == ("event:bright_sky",)
        short = work.decide(work.frame(SHORT_US, step=2), background=bright + 1)
        assert short.reasons == ("event:bright_sky",)
        assert short.fits is True

    def test_a_short_frame_starts_no_pointing_or_cloud_event(self, profile: Profile) -> None:
        work = run(make_policy(profile, keep_every=1000))
        outcome = work.decide(work.frame(SHORT_US), solved=False, cloud=1.0, moved=True)
        assert outcome.fits is False
        assert outcome.reasons == ()

    def test_two_conditions_that_start_together_make_one_frame_that_names_both(
        self, profile: Profile
    ) -> None:
        work = run(make_policy(profile, keep_every=1000))
        work.decide(work.frame(step=0))
        outcome = work.decide(work.frame(step=1), solved=False, cloud=0.9)
        assert outcome.reasons == ("event:cloud+unsolved",)

    def test_a_second_condition_inside_the_interval_waits_for_it(self, profile: Profile) -> None:
        work = run(make_policy(profile, keep_every=1000, event_min_interval_s=3600.0))
        work.decide(work.frame(step=0))
        assert work.decide(work.frame(step=1), cloud=0.9).fits
        assert not work.decide(work.frame(step=2), cloud=0.9, solved=False).fits

    def test_each_kind_of_frame_has_its_own_interval(self, profile: Profile) -> None:
        work = run(make_policy(profile, keep_every=1000, event_min_interval_s=3600.0))
        work.decide(work.frame(step=0))
        bright = 0.9 * SATURATION_DN
        assert work.decide(work.frame(SHORT_US, step=1), background=bright).fits
        assert work.decide(work.frame(step=1), cloud=0.9).fits

    def test_an_interval_of_zero_allows_every_start(self, profile: Profile) -> None:
        work = run(make_policy(profile, keep_every=1000, event_min_interval_s=0.0))
        work.decide(work.frame(step=0))
        starts = [
            work.decide(work.frame(step=step), cloud=0.9 if step % 2 else 0.0).fits
            for step in range(1, 7)
        ]
        assert starts == [True, False, True, False, True, False]

    def test_a_clock_that_stepped_back_does_not_hold_back_an_event(self, profile: Profile) -> None:
        work = run(make_policy(profile, keep_every=1000, event_min_interval_s=3600.0))
        work.decide(work.frame(step=100))
        assert work.decide(work.frame(step=101), cloud=0.9).fits
        work.decide(work.frame(step=102), cloud=0.0)
        assert work.decide(work.frame(step=2), cloud=0.9).fits  # hours earlier

    def test_without_a_profile_a_bright_sky_cannot_be_judged(self) -> None:
        work = run(make_policy(None, keep_every=1000))
        work.decide(work.frame(step=0))
        assert not work.decide(work.frame(step=1), background=1e9).fits


class TestTheCaptureGate:
    def test_a_closed_gate_keeps_the_preview_and_skips_the_fits(self, profile: Profile) -> None:
        work = run(make_policy(profile))
        outcome = work.decide(work.frame(step=0), gate=lambda: False)
        assert outcome.fits is False
        assert outcome.preview is True
        assert outcome.skipped == "low_space"
        assert outcome.reasons == ("every_10",)  # the wish stays on record

    def test_an_open_gate_changes_nothing(self, profile: Profile) -> None:
        work = run(make_policy(profile))
        outcome = work.decide(work.frame(step=0), gate=lambda: True)
        assert outcome.fits is True
        assert outcome.skipped is None

    def test_the_gate_is_not_asked_when_no_file_is_wanted(self, profile: Profile) -> None:
        asked: list[int] = []

        def gate() -> bool:
            asked.append(1)
            return False

        work = run(make_policy(profile))
        work.decide(work.frame(step=0), gate=gate)  # the first long frame: asked
        work.decide(work.frame(step=1), gate=gate)  # a plain long frame: not asked
        work.decide(work.frame(SHORT_US, step=1), gate=gate)  # a plain short frame: not asked
        assert asked == [1]

    def test_a_gate_that_fails_counts_as_open(self, profile: Profile) -> None:
        def broken() -> bool:
            raise OSError("the disk probe failed")

        work = run(make_policy(profile))
        assert work.decide(work.frame(step=0), gate=broken).fits
