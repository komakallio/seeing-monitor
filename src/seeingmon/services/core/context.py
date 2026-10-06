"""The context that only `core` knows, for the fast analysis: heater duty, zenith angle, clock.

The scheduler asks a `context_provider` for a `FastContext` every few seconds, and it hands the
result to the fast analyzer, which puts it on the windows that close. The scheduler adds the flags
that it owns (`cloud`, and `twilight` or `daylight`, from its own gates). The provider here adds
what the scheduler cannot know:

- **The heater duty.** The share of the last analysis window that the dew heater was on, from the
  controller's log, and the `heater_on` flag while it is above zero. Heater plumes can add local
  turbulence, so every seeing window carries the duty.
- **The zenith angle of Polaris**, from the latitude and longitude of the site in the local
  configuration and the time of the window. The analysis converts r0 and the seeing to the zenith
  with it. The provider gives none while the clock is not synchronized, because the angle depends on
  the UTC time, and then the window carries `time_invalid`.
- **The clock.** A clock that is not synchronized adds `time_invalid`, so that a record never passes
  for a measurement at a trusted time.
"""

from __future__ import annotations

from typing import Protocol

from seeingmon.analysis import FastContext
from seeingmon.clock import Clock
from seeingmon.scheduler.config import SiteConfig
from seeingmon.scheduler.ephemeris import polaris_zenith_angle_deg


class HeaterDuty(Protocol):
    """What the provider needs from the heater controller."""

    def recent_duty(self, window_s: float) -> float | None: ...


class ContextProvider:
    """Builds the `FastContext` for the time `t_utc_ns`. Pass an instance as `context_provider`."""

    def __init__(
        self,
        *,
        clock: Clock,
        site: SiteConfig | None,
        window_s: float,
        heater: HeaterDuty | None = None,
    ) -> None:
        self._clock = clock
        self._site = site
        self._window_s = window_s
        self._heater = heater

    def __call__(self, t_utc_ns: int) -> FastContext:
        flags: set[str] = set()
        duty = None if self._heater is None else self._heater.recent_duty(self._window_s)
        if duty is not None and duty > 0.0:
            flags.add("heater_on")
        synchronized = self._clock.status().synchronized
        if synchronized is False:
            flags.add("time_invalid")
        zenith = None
        if self._site is not None and synchronized is not False:
            zenith = polaris_zenith_angle_deg(
                t_utc_ns, self._site.latitude_deg, self._site.longitude_deg
            )
        return FastContext(flags=frozenset(flags), heater_duty=duty, zenith_angle_deg=zenith)
