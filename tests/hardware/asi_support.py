"""Shared setup for the tests of the `asi` driver: a driver on a fake SDK and a virtual clock."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from seeingmon.clock import VirtualClock
from seeingmon.drivers.asi import AsiDriver, AsiOptions
from seeingmon.frames import Roi, StreamConfig
from seeingmon.hardware.asi.fake import FakeAsiSdk, FakeUsbResetter
from seeingmon.hardware.asi.watchdog import CallWatchdog, HangReport
from seeingmon.hardware.events import HardwareEvent
from seeingmon.profile import Profile, load_profile

FAST = StreamConfig(mode="bin1", exposure_us=2000, gain=120, roi=Roi(100, 200, 128, 128))
TINY = StreamConfig(mode="bin1", exposure_us=2000, gain=120, roi=Roi(8, 4, 16, 8))


def reference_profile() -> Profile:
    return load_profile("asi294mm-gs250")


@dataclass
class Rig:
    """A driver, its fake SDK, and everything that a test observes."""

    clock: VirtualClock
    sdk: FakeAsiSdk
    driver: AsiDriver
    watchdog: CallWatchdog
    resetter: FakeUsbResetter
    hangs: list[HangReport] = field(default_factory=list)
    events: list[HardwareEvent] = field(default_factory=list)

    def event_kinds(self) -> list[str]:
        return [event.kind for event in self.events]

    def opened(self) -> Rig:
        self.driver.open()
        return self

    def streaming(self, config: StreamConfig = FAST) -> Rig:
        """Open, configure, and start."""
        self.driver.open()
        self.driver.configure(config)
        self.driver.start()
        return self


def make_rig(
    *,
    sdk: Mapping[str, object] | None = None,
    api: Callable[[FakeAsiSdk], Any] | None = None,
    reappear_after_s: float = 2.0,
    **options: object,
) -> Rig:
    """Build a rig. `sdk` holds `FakeAsiSdk` arguments, `options` holds `AsiOptions` fields, and
    `api` wraps the SDK that the driver sees, such as a proxy that advances the clock."""
    clock = VirtualClock()
    fake = FakeAsiSdk(clock, **(sdk or {}))  # type: ignore[arg-type]
    hangs: list[HangReport] = []
    events: list[HardwareEvent] = []
    watchdog = CallWatchdog(clock, hangs.append)
    resetter = FakeUsbResetter(fake, reappear_after_s=reappear_after_s)
    driver = AsiDriver(
        api=api(fake) if api is not None else fake,
        profile=reference_profile(),
        clock=clock,
        options=AsiOptions(**options),
        usb_resetter=resetter,
        watchdog=watchdog,
        on_event=events.append,
    )
    return Rig(clock, fake, driver, watchdog, resetter, hangs, events)


def wait_until(condition: Callable[[], bool], timeout_s: float = 5.0) -> bool:
    """Wait in real time for a condition that another thread makes true."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return condition()


def run_in_thread(function: Callable[[], object]) -> tuple[threading.Thread, list[object]]:
    """Run `function` in a thread. The list receives its result or the exception it raised."""
    outcome: list[object] = []

    def target() -> None:
        try:
            outcome.append(function())
        except BaseException as error:
            outcome.append(error)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, outcome
