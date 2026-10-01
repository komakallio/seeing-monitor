"""The sim driver of `acquire`: the options that put Polaris where the survey code predicts it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sep", reason="the survey path needs the survey extra")

from seeingmon.clock import VirtualClock, iso_to_utc_ns
from seeingmon.drivers.base import CameraConfigError
from seeingmon.drivers.sim.stars import Pointing
from seeingmon.frames import PixelFormat, StreamConfig, StreamKind
from seeingmon.profile import load_profile
from seeingmon.services.acquire.factory import create_camera_driver
from seeingmon.services.simsky import seed_solution, sim_field, write_small_profile
from seeingmon.survey.tracker import PointingTracker

START = iso_to_utc_ns("2026-01-01T19:00:00Z")


@pytest.fixture
def profile(tmp_path: Path) -> object:
    return load_profile(str(write_small_profile(tmp_path)))


def make(profile: object, **options: object) -> object:
    return create_camera_driver(
        "sim", profile=profile, clock=VirtualClock(START), options={"seed": 1, **options}
    )


class TestTheOptions:
    def test_without_the_options_it_is_the_plain_simulator(self, profile: object) -> None:
        assert make(profile).name == "sim"  # type: ignore[attr-defined]

    def test_a_pointing_table_and_the_real_polaris_build_a_driver(self, profile: object) -> None:
        driver = make(
            profile,
            pointing={"t_ref_utc_ns": START, "roll_deg": 0.0},
            polaris="real",
            polaris_mag=6.0,
        )
        assert driver.name == "sim"  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        "options",
        [
            {"polaris": "imaginary"},
            {"pointing": {"roll_deg": 1.0}},  # no reference time
            {"pointing": {"t_ref_utc_ns": 1, "colour": 2}},  # an unknown key
            {"pointing": "now"},
        ],
    )
    def test_a_bad_option_is_a_configuration_error(
        self, profile: object, options: dict[str, object]
    ) -> None:
        with pytest.raises(CameraConfigError):
            make(profile, **options)

    def test_an_unknown_option_of_the_simulator_still_fails(self, profile: object) -> None:
        with pytest.raises((CameraConfigError, ValueError, TypeError)):
            make(profile, colour="red")


class TestTheStarIsWhereTheSurveyExpectsIt:
    def test_the_frame_shows_a_star_at_the_pixel_that_the_tracker_predicts(
        self, tmp_path: Path
    ) -> None:
        small = load_profile(str(write_small_profile(tmp_path)))
        pointing = Pointing(t_ref_utc_ns=START)
        fit = seed_solution(small, sim_field(1, polaris_mag=6.0), pointing, START)
        tracker = PointingTracker(small)
        tracker.update(fit.solution)

        clock = VirtualClock(START)
        driver = create_camera_driver(
            "sim",
            profile=small,
            clock=clock,
            options={
                "seed": 1,
                "psf_mode": "gaussian",
                "polaris": "real",
                "polaris_mag": 6.0,
                "pointing": {"t_ref_utc_ns": START},
            },
        )
        driver.open()
        driver.configure(
            StreamConfig(
                "bin2",
                2_000_000,
                120,
                pixel_format=PixelFormat.RAW16,
                roi=None,
                kind=StreamKind.SNAPSHOT,
            )
        )
        driver.start()
        frame = driver.read_frame(30.0)
        predicted = tracker.polaris_position(frame.t_utc_ns, "bin2")
        assert predicted is not None
        x, y = round(predicted[0]), round(predicted[1])
        window = frame.data[y - 5 : y + 6, x - 5 : x + 6].astype(float)
        background = float(np.median(frame.data))
        noise = float(np.median(np.abs(frame.data - background)) * 1.48)
        assert window.max() > background + 20 * noise  # a star sits within five pixels of the spot
        # The synthetic Polaris of the simulator has moved: the optical axis points at its old
        # place, and the frame shows no second Polaris there.
        center = frame.data[235:245, 315:325].astype(float)
        assert center.max() < background + 20 * noise
        driver.close()
