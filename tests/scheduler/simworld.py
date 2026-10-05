"""Helpers for scheduler tests on the `sim` driver: a small sensor, a logging camera, and fakes.

The scheduler runs against the simulated camera, which renders the stars, the turbulence, and the
twilight sky. The pointing provider and the survey analysis read the simulator's truth, so the
tests need no plate solver. Import this module only after a check that the simulator imports,
because the simulator needs SciPy (the `fast` extra):

    sim = pytest.importorskip("seeingmon.drivers.sim")
    from tests.scheduler.simworld import LoggingSim
"""

from __future__ import annotations

from typing import Any

import numpy as np

import seeingmon.drivers.sim as sim
from seeingmon.clock import Clock
from seeingmon.frames import ActiveStream, Frame, StreamConfig
from seeingmon.profile import Profile, parse_profile
from seeingmon.scheduler import Scheduler, SiteConfig
from seeingmon.testing import FakeSurveyAnalyzer
from tests.profile.builders import reference_data

# The synthetic site of the simulator and of the scheduler: 55 degrees north on the prime meridian.
SITE = SiteConfig(latitude_deg=55.0, longitude_deg=0.0)


def small_profile() -> Profile:
    """The reference camera with one eighth of the width and height. The optics stay the same."""
    data = reference_data()
    data["id"] = "asi294mm-gs250-small"
    for mode in data["readout_modes"]:
        mode["width_px"] = mode["width_px"] // 8
        mode["height_px"] = mode["height_px"] // 8
    return parse_profile(data)


class LoggingSim(sim.SimDriver):
    """The simulator, with the time of every `configure` call, and the purpose of each stream.

    The purpose comes from the scheduler (`scheduler`, set after both exist) when the stream
    starts, so that a search burst and a fast stream with the same settings tell apart.
    """

    def __init__(self, profile: Profile, clock: Clock, options: Any) -> None:
        super().__init__(sim.SimParams.modes_from_profile(profile), clock, options)
        self.configure_log: list[tuple[int, StreamConfig]] = []
        self.purposes: dict[int, str] = {}  # the index in `configure_log` to the purpose
        self.stream_ids: dict[int, int] = {}  # the index in `configure_log` to the stream ID
        self.scheduler: Scheduler | None = None

    def configure(self, config: StreamConfig) -> ActiveStream:
        self.configure_log.append((self.clock.utc_ns(), config))
        active = super().configure(config)
        self.stream_ids[len(self.configure_log) - 1] = active.stream_id
        return active

    def start(self) -> None:
        super().start()
        stream = None if self.scheduler is None else self.scheduler.stream
        if stream is not None:
            self.purposes.setdefault(len(self.configure_log) - 1, stream.purpose)

    def streams(self, purpose: str) -> list[tuple[int, int, StreamConfig]]:
        """The time, the stream ID, and the settings of each stream of a purpose, in order."""
        return [
            (t, self.stream_ids[index], config)
            for index, (t, config) in enumerate(self.configure_log)
            if self.purposes.get(index) == purpose
        ]


class SimPointing:
    """The Polaris position from the simulator's truth: its brightest star, projected to pixels."""

    def __init__(self, truth: Any) -> None:
        self._truth = truth

    def polaris_position(self, t_utc_ns: int, mode: str) -> tuple[float, float] | None:
        stars = self._truth.star_positions(t_utc_ns, mode)
        brightest = int(np.argmin(stars.mag))
        return float(stars.x[brightest]), float(stars.y[brightest])


class SimSurvey(FakeSurveyAnalyzer):
    """A survey analysis that reads the cloud fraction from the simulator's transparency."""

    def __init__(self, truth: Any, profile_id: str) -> None:
        super().__init__(station_id="test", profile_id=profile_id)
        self._truth = truth

    def submit(self, frame: Frame) -> None:
        long_exposure = frame.exposure_us >= 1_000_000
        transparency = float(self._truth.transparency(frame.t_utc_ns))
        self.cloud_fraction = 1.0 - transparency if long_exposure else None
        self.solved = long_exposure and transparency > 0.15
        super().submit(frame)
