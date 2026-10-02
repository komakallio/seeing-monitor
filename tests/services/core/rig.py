"""A rig for `CoreApp`: a small profile, a local configuration, fakes, and a way to read the store.

The rig builds a `CoreApp` on a `VirtualClock` with `threads=False`, so a test steps the scheduler
and calls `tick`, and a night runs on one thread in seconds. The storage gets a clock whose `sleep`
does nothing, so the housekeeping never moves virtual time.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from seeingmon.clock import NS_PER_S, Clock, ClockStatus, VirtualClock, iso_to_utc_ns
from seeingmon.config import Config, load_config
from seeingmon.drivers.base import CameraDriver
from seeingmon.frames import Frame
from seeingmon.hardware.events import HardwareEvent
from seeingmon.profile import Profile
from seeingmon.records import Record
from seeingmon.scheduler.levels import EscalationLevel
from seeingmon.services.acquire.events import EventBatch, LoggedEvent
from seeingmon.services.config import ServicesConfig
from seeingmon.services.core.app import CoreApp, CoreParts
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.notify import SystemdNotifier
from seeingmon.services.simsky import write_small_profile
from seeingmon.store.db import StoreReader, record_from_row
from seeingmon.testing import (
    FakeCameraDriver,
    FakeFastAnalyzer,
    FakePointingProvider,
    FakeSurveyAnalyzer,
)
from tests.scheduler.helpers import make_frame
from tests.services.addresses import unique_address

# A clear autumn evening at a synthetic site (55 degrees north on the prime meridian).
NIGHT = iso_to_utc_ns("2026-01-01T22:00:00Z")
SMALL_BIN2 = (640, 480)


def local_config_text(
    data_dir: Path,
    profile: Path | None = None,
    *,
    extra: str = "",
    site: bool = True,
    analysis_window_s: float = 10.0,
) -> str:
    """The text of a local configuration for a test: short windows, a synthetic site, no sinks."""
    lines = [
        'station_id = "test"',
        f'profile = "{profile.as_posix()}"' if profile is not None else "",
        "[paths]",
        f'data_dir = "{data_dir.as_posix()}"',
        "[fastpath]",
        "window_s = 10.0",
        "min_window_s = 5.0",
        "[scheduler.fast]",
        "window_s = 20.0",
        f"analysis_window_s = {analysis_window_s}",
    ]
    if site:
        lines += ["[site]", "latitude_deg = 55.0", "longitude_deg = 0.0", "elevation_m = 0.0"]
    return "\n".join(lines) + "\n" + extra


def make_config(
    tmp_path: Path,
    *,
    profile: Path | None = None,
    extra: str = "",
    analysis_window_s: float = 10.0,
) -> Config:
    local = tmp_path / "local.toml"
    local.write_text(
        local_config_text(
            tmp_path / "data", profile, extra=extra, analysis_window_s=analysis_window_s
        ),
        encoding="utf-8",
    )
    return load_config(local_file=local, env={})


class NoSleepClock:
    """A view of a clock whose `sleep` does nothing, for the storage of a stepped run."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock

    def utc_ns(self) -> int:
        return self._clock.utc_ns()

    def monotonic_ns(self) -> int:
        return self._clock.monotonic_ns()

    def sleep(self, seconds: float) -> None:
        return None

    def status(self) -> ClockStatus:
        return self._clock.status()


class FakeRemote:
    """The camera side of `core` for a test: a driver that is not behind `acquire`.

    It plays the part of `RemoteCameraDriver` for the escalation, the health, and the event pump.
    """

    def __init__(self) -> None:
        self.instance: str | None = "acquire-one"
        self.connected = True
        self.restarts: list[str] = []
        self.log: list[HardwareEvent] = []
        self.queue_frames = 3

    def health(self) -> dict[str, Any]:
        return {"state": "streaming", "threads_alive": True, "queue_frames": self.queue_frames}

    def request_restart(self, reason: str = "") -> None:
        self.restarts.append(reason)
        self.connected = False

    def events(self, after: int = 0) -> EventBatch:
        items = tuple(LoggedEvent(i + 1, e) for i, e in enumerate(self.log) if i + 1 > after)
        return EventBatch(items, len(self.log), 0)


@dataclass
class CoreRig:
    app: CoreApp
    clock: VirtualClock
    camera: FakeCameraDriver
    fast: FakeFastAnalyzer
    survey: FakeSurveyAnalyzer
    pointing: FakePointingProvider
    remote: FakeRemote
    tmp_path: Path
    extra: dict[str, Any] = field(default_factory=dict)

    def run_for(self, seconds: float, *, tick_every_s: float = 1.0) -> None:
        """Step the scheduler for `seconds` of virtual time, and do the periodic work."""
        end = self.clock.monotonic_ns() + round(seconds * NS_PER_S)
        next_tick = 0
        while self.clock.monotonic_ns() < end:
            self.app.scheduler.step()
            now = self.clock.monotonic_ns()
            if now >= next_tick:
                self.app.tick()
                next_tick = now + round(tick_every_s * NS_PER_S)

    def records(self, record_type: str) -> list[Record]:
        """Every record of a type in the store, in row order."""
        assert self.app.storage is not None
        return read_all(self.app.storage.store, record_type)

    def events(self, kind: str | None = None) -> list[Any]:
        found = self.records("event")
        return [e for e in found if kind is None or e.kind == kind]  # type: ignore[attr-defined]


def read_all(store: StoreReader, record_type: str) -> list[Record]:
    """Read every record of a type from a store."""
    rows: list[Record] = []
    after = 0
    while True:
        batch = store.after(record_type, after, 500)
        if not batch:
            return rows
        rows += [record_from_row(record_type, row) for row in batch]
        after = batch[-1].row_id


def events_of(records: Iterable[Record], kind: str) -> list[Any]:
    return [r for r in records if r.record_type == "event" and r.kind == kind]  # type: ignore[attr-defined]


def build_rig(
    tmp_path: Path,
    *,
    start_utc_ns: int = NIGHT,
    config_extra: str = "",
    analysis_window_s: float = 10.0,
    parts: dict[str, Any] | None = None,
    key: ConnectionKey | None = None,
    endpoint: Any = None,
    threads: bool = False,
    clock: Clock | None = None,
    profile: Path | None = None,
    driver_factory: Callable[[Clock, Profile], CameraDriver] | None = None,
) -> CoreRig:
    """Build a `CoreApp` on fakes, a virtual clock, and a real store in `tmp_path`.

    `profile` names a profile file instead of the default one. `driver_factory` builds the camera
    from the clock of the app and the profile, for a test that needs a simulated camera.
    """
    virtual = VirtualClock(start_utc_ns)
    use_clock: Clock = clock or virtual
    config = make_config(
        tmp_path, profile=profile, extra=config_extra, analysis_window_s=analysis_window_s
    )
    services = config.section("services", ServicesConfig).model_copy(
        update={
            "acquire_address": unique_address("acquire"),
            "core_address": unique_address("core"),
        }
    )
    camera = FakeCameraDriver(use_clock, full_frames={"bin1": (8288, 5644), "bin2": SMALL_BIN2})
    fast = FakeFastAnalyzer(station_id="test", profile_id=config.profile.id, window_s=10.0)
    survey = FakeSurveyAnalyzer(station_id="test", profile_id=config.profile.id)
    pointing = FakePointingProvider()
    pointing.set_solution("bin1", 4144.0, 2822.0)
    pointing.set_solution("bin2", 320.0, 240.0)
    remote = FakeRemote()
    core_parts = CoreParts(
        driver=camera,
        remote=remote,
        fast=fast,
        survey=survey,
        pointing=pointing,
        notifier=SystemdNotifier(env={}),
        storage_clock=NoSleepClock(use_clock),
        sinks=[],
    )
    if driver_factory is not None:
        core_parts.driver = driver_factory(use_clock, config.profile)
    for name, value in (parts or {}).items():
        setattr(core_parts, name, value)
    app = CoreApp(
        config,
        services,
        use_clock,
        key or ConnectionKey.from_text("a-test-key-of-more-than-32-characters"),
        parts=core_parts,
        endpoint=endpoint or services.endpoint("core"),
        threads=threads,
    )
    return CoreRig(app, virtual, camera, fast, survey, pointing, remote, tmp_path)


def sky_frame(
    seq: int = 1, *, t_utc_ns: int = NIGHT, saturate: float = 0.0, seed: int = 1
) -> Frame:
    """A 16-bit bin2 frame of 480 by 640 pixels: sky, one star, and a saturated patch."""
    rng = np.random.default_rng(seed)
    data = rng.normal(2000.0, 20.0, (480, 640))
    yy, xx = np.mgrid[0:480, 0:640]
    data += 20000.0 * np.exp(-((xx - 300) ** 2 + (yy - 200) ** 2) / (2 * 1.8**2))
    if saturate:
        data[: int(480 * saturate), :] = 65532.0
    return make_frame(
        np.clip(data, 0, 65535).astype(np.uint16),
        mode="bin2",
        gain=120,
        exposure_us=500_000,
        adc_bits=14,
        t_utc_ns=t_utc_ns,
        seq=seq,
    )


def escalation_names(app: CoreApp) -> list[str]:
    return [level.name for level in app.escalator.performed]


__all__ = [
    "NIGHT",
    "SMALL_BIN2",
    "CoreRig",
    "EscalationLevel",
    "FakeRemote",
    "NoSleepClock",
    "build_rig",
    "escalation_names",
    "events_of",
    "make_config",
    "read_all",
    "sky_frame",
    "write_small_profile",
]
